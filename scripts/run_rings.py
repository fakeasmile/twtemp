"""全概念（环1a/1b/2/3/4）+ S1 提示词 v1.4 推理脚本（vLLM 加速，自包含）。

v1.5 起：run_ring1a.py 的全部机制（vLLM 模型加载、verbalizer 候选展开、首 token
受限解码、chat 渲染与首 token 位置修正、数据加载）并入本文件，不再 import——
只跑本脚本即可得到完整概念向量。测量机制与原 1a 脚本逐字一致：
  单概念两阶段（存在性 是/否 + 强度 A/B）+ 首 token logprobs 提取 + 受限解码；
  CAT1a 模板为 1a 先导实验 v2.11 定稿原文；环1a 模板/阈值同源。

每次调用 = 一条文本 × 一个判断项：
  阶段 1  存在性（是/否）：全部概念（环1a 6、环1b 3、环2 6、环3 12、环4 8）
          0-1 值域概念只走阶段 1；0-2 值域概念在判"是"后触发阶段 2（A/B 对称定义）
  环4     概念模板首句替换为角色框架（熟悉网络社群用语的标注助手），其余与 1a 一字不差
  CAT     每条文本 6 问：CAT1a / CAT1b / CAT2a（含"话题提及不算"排除语）/ CAT2b（不含）/ CAT3 / CAT4
  S1      严重度 ordinal 0-5，单次调用，首 token 在数字 0-5 上提质量，双变体：
          S1A 无例词 / S1B 定义带例词（决策例词去留）
映射：阶段1"否"→0，阶段2 A→1、B→2；0-1 概念 是→1；S1 取 argmax 数字。
环确认聚合（DINO 析取 / guard 压制逻辑）已不在本脚本（读出层直接用 soft p_yes 编码）。

输出命名（防覆盖）：输出文件名带数据来源后缀——
  data_path 指向单个 json 文件 → rings_{model_name}_{文件名去后缀}.json / ..._flat.csv
  data_path 指向目录（--split test|train）→ rings_{model_name}_{split}.json / ..._flat.csv
  目录模式分两次跑（先 test 后 train）会得到两个文件，天然不覆盖；不提供 all 模式。

使用示例：
  python scripts/run_rings.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/CCC/ccc.json
  python scripts/run_rings.py --model_path D:/models/Qwen3.5-9B --data_path data/raw/TOXICN --split test

自动处理说明（无需命令行指定）：
  首 token 位置修正：优先查 MODEL_PROMPT_SUFFIX 表（按模型目录名登记后缀）；
  未登记的模型走自动修正——生成前缀不以换行结尾则补 '\n'。
  思考模式探测：chat template 含 enable_thinking 变量（Qwen3 系混合思考模型）时自动
  传 enable_thinking=False——思考前缀会占据首 token，破坏 verbalizer 提取。
  不启用量化：量化改变 logits 分布，影响概率读数效度（测量类任务不可妥协）。
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent


# =============================================================================
# 模型加载与生成前缀修正
# =============================================================================
def load_vllm_model(llm_path: Path, gpu_memory_utilization: float):
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {llm_path}")
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,     # 同文本的多个概念调用共享指令前缀，prefill 复用
        max_model_len=2048,             # 指令+定义+文本最长约 600-700 token，2048 留足余量
        max_num_seqs=64,
        max_num_batched_tokens=16384,   # 批量以 prefill 为主（每请求只解码 1 token），大 batched token 提高 GPU 利用率
        enforce_eager=True,             # 只解码 1 个 token，CUDA graph 主要加速多步 decode 循环，此处禁用可省启动时间与显存（腾给 KV cache）
        max_logprobs=50,                # 与 SamplingParams(logprobs=50) 配套；默认上限 20，候选词可能被截断
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    """chat template 支持 enable_thinking 时关闭思考模式，否则返回空。"""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


# 生成前缀后缀手工指定表（按模型目录名）。不同模型 chat template 结尾不一致，
# 需要特定后缀的模型在此登记；未登记的模型走自动修正（前缀不以换行结尾则补 '\n'）。
# Qwen3.5-9B 用 '\n'：首 token 落在裸字母 token（'B'=33 / 'A'=32），A+B 概率质量 0.973，
# <think> 仅 0.004；改空格后缀会把标签并成 ' B'(417) / ' A'(357)，且 1 空格时 <think> 高达 0.151。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}

# 阶段 1 判是阈值手工表（按模型目录名）。p(是) 是连续量，但不同模型的先验偏移不同
# （CCC 实测：GLM 偏宽，τ=0.5 时 toxic1 是率 74.6% / toxic0 33.9%；Qwen 整体下移，
# τ=0.5 只剩 15.0%/1.1%，需 τ=0.1 才回到可比工作点 49.2%/14.7%）。
# 未登记的模型用默认值 0.5；实际所用 τ 记录在 meta.stage1_tau。
STAGE1_THRESHOLD = {
    "glm-4-9b-chat": 0.50,
    "Qwen3.5-9B": 0.10,
}
STAGE1_THRESHOLD_DEFAULT = 0.50


# =============================================================================
# 提示词模板（rings v1.4 定稿；v1.3→v1.4 环1a 并入：CSV 环1a 6 概念与 CAT1a 进入本脚本，
# CAT1A 模板为 1a 先导实验 v2.11 定稿原文，模板与定义未动。
# v1.2→v1.3 模板与定义未动，仅登记阈值表：
# CAT_TAU / GUARD_TAU = analyze_rings.py 6c 统一校准（Youden's J）在 CCC 试点上的程序输出，
# 落 meta.cat_taus / meta.guard_tau。
# v1.1→v1.2 仅改 CSV 环3 guard 角色为 descriptive——ring_spec v3.4：环3 C5a/b/c、C6a/b 降为
# 描述性维度，不参与阻断，环3 确认 = evidence≥1。
# v1.0→v1.1 仅改 CSV 定义：环3 及 U4a/U4b 定义补"贬损存在"前提，C5b/C5c/U4a/E6a 加排除语，
# U1/U2 删首token架构不可执行的括号注）
# =============================================================================
STAGE1_TEMPLATE = """你是语言学标注助手。判断文本中是否出现下述判断项。只依据文本本身判断，不评价内容对错，不解释。判断项以讽刺、委婉、谐音、缩写等变形出现同样算。

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

该判断项在文本中出现了吗？只回答“是/否”。回答："""

STAGE2_TEMPLATE = """你是语言学标注助手。判断项已确认出现在文本中，现在判断它是怎样出现的。只依据文本本身判断，不评价内容对错，不解释。
A = 间接出现：需要结合上下文才能读出判断项的含义（讽刺、委婉、暗示）
B = 直接出现：文本字面内容本身就是判断项，无需结合上下文

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

判断项是直接出现（B）还是间接出现（A）？只回答 A/B。回答："""

# 环4 概念模板：仅首句换成角色框架（ring_spec 4.5 的固定组件），其余与上面一字不差。
STAGE1_TEMPLATE_U = STAGE1_TEMPLATE.replace(
    "你是语言学标注助手。", "你是熟悉网络社群用语的标注助手。", 1)
STAGE2_TEMPLATE_U = STAGE2_TEMPLATE.replace(
    "你是语言学标注助手。", "你是熟悉网络社群用语的标注助手。", 1)

CAT_TEMPLATES = {
    # CAT1a = 环1a 先导实验 v2.11 定稿原文，此处不重复定义于他处。
    "CAT1A": """以下文本是否包含对某人或某群体的负面态度表达（鄙夷、厌恶、敌意、威胁、道德定罪）。
文本：{text}
文本中有这类表达吗？
只回答 是/否。答：""",
    # 原 CSV 问句"该文本是否通过保护、角色限定或地位贬低，把人或群体固定到从属/较低的位置？"
    # 为机制归因式，改出现性锚定。
    "CAT1B": """以下文本是否包含把人或群体固定到从属、较低位置的表达（以保护、角色限定或地位贬低的方式）。
文本：{text}
文本中有这类表达吗？
只回答 是/否。答：""",
    # 环2 双变体：CAT2A 含"话题提及≠锚定"排除语，CAT2B 不含，跑完 ccc.json 对比后定去留。
    "CAT2A": """以下文本是否把对人的评价指向某个身份群体（性别、地域、民族、宗教、性取向、残障、年龄、职业等），而非只针对具名个人。话题提到某群体但评价不落在群体上的，不算。
文本：{text}
文本把评价指向身份群体了吗？
只回答 是/否。答：""",
    "CAT2B": """以下文本是否把对人的评价指向某个身份群体（性别、地域、民族、宗教、性取向、残障、年龄、职业等），而非只针对具名个人。
文本：{text}
文本把评价指向身份群体了吗？
只回答 是/否。答：""",
    "CAT3": """以下文本中的贬损内容，是否是说话人自己的表态（不是引用他人说法、不是假设情境、不是在反驳这种观点）。
文本：{text}
文本中有说话人自己的贬损表态吗？
只回答 是/否。答：""",
    "CAT4": """假设你是熟悉该文本社区用语的网络社群成员。
以下文本会被你的社群读作对某人或某群体的攻击或贬损吗。
文本：{text}
会吗？
只回答 是/否。答：""",
}

S1_TEMPLATES = {
    # S1 严重度双变体：S1A 纯档位名，S1B 每档带例词（更可执行 vs 更简洁，跑完对比定去留）。
    "S1A": """你是语言学标注助手。按文本中出现的最高伤害档位评级。只依据文本本身判断，不评价内容对错，不解释。
0 = 无伤害内容
1 = 口头贬低
2 = 疏远回避
3 = 排斥歧视
4 = 人身攻击或煽动
5 = 清除灭绝

文本：{text}

档位：只回答一个数字。答：""",
    "S1B": """你是语言学标注助手。按文本中出现的最高伤害档位评级。只依据文本本身判断，不评价内容对错，不解释。
0 = 无伤害内容
1 = 口头贬低（骂人话、贬低性评价）
2 = 疏远回避（不理、拉黑、远离他们）
3 = 排斥歧视（不配享有同等权利、滚回去）
4 = 人身攻击或煽动（威胁、号召动手、去死）
5 = 清除灭绝（灭绝、清除、不该存在）

文本：{text}

档位：只回答一个数字。答：""",
}

DIGITS = "012345"
FULLWIDTH_DIGITS = {"0": "０", "1": "１", "2": "２", "3": "３", "4": "４", "5": "５"}

# CAT/GUARD 判是阈值登记表（按模型目录名）。数值 = analyze_rings.py 6c 统一校准（Youden's J）
# 在 CCC 试点上的程序输出（2026-11-03 登记），非手工拍定；未登记的模型/通道用默认 0.50。
# 实际所用值记录在 meta.cat_taus / meta.guard_tau；正式实验需在独立校准集上重新校准（挂账）。
CAT_TAU = {
    # CAT1A = 环1a 时期登记值（原 run_ring1a.CAT1A_THRESHOLD），非 6c 输出
    "glm-4-9b-chat": {"CAT1A": 0.50, "CAT1B": 0.94, "CAT2A": 0.92, "CAT2B": 0.94, "CAT3": 0.96, "CAT4": 0.92},
    "Qwen3.5-9B":    {"CAT1A": 0.10, "CAT1B": 0.14, "CAT2A": 0.12, "CAT2B": 0.50, "CAT3": 0.34, "CAT4": 0.22},
}
CAT_TAU_DEFAULT = 0.50
# 环4 guard 阻断阈值：任一 guard 概念 p(是) ≥ τ_guard 即阻断（环3 guard 已降级 descriptive，不阻断）。
GUARD_TAU = {"glm-4-9b-chat": 0.74, "Qwen3.5-9B": 0.10}
GUARD_TAU_DEFAULT = 0.50


# =============================================================================
# 概念词典
# =============================================================================
RING_KEY = {
    "环1a 负面评价": "ring1a",
    "环1b 限制性/等级化评价": "ring1b",
    "环2 身份锚定": "ring2",
    "环3 立场承诺": "ring3",
    "环4 语境取效": "ring4",
}


def load_ring_concepts(csv_path: Path):
    """读取除 S1 外的全部概念（含环1a），保留类别/角色/值域（决定是否走阶段 2）。"""
    concepts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 传输环节可能在表头引入多余 BOM/NUL（曾出现双重 BOM 致 KeyError），统一剥掉
        reader.fieldnames = [(fn or "").lstrip("\ufeff\x00").strip() for fn in reader.fieldnames]
        for row in reader:
            cat = row["类别"].strip()
            if cat not in RING_KEY:
                continue
            concepts.append({
                "id": row["id"].strip(),
                "name": row["概念"].strip(),
                "definition": row["定义"].strip(),
                "category": cat,
                "ring": RING_KEY[cat],
                "role": row["角色"].strip(),
                "range": row["值域"].strip(),
            })
    if not concepts:
        raise ValueError(f"未在 {csv_path} 中找到环1a/1b/2/3/4 的概念")
    return concepts


# =============================================================================
# Verbalizer 工具（原 run_ring1a.py 已验证机制，逐字并入）
# =============================================================================
# 候选词的表层变体：不同 tokenizer 会把同一字母切成 "A" / " A" / "Ａ" / "a" 等
# 不同 token；只取裸形式会在 Qwen 等词表上漏掉实际被采样的那个 token。
LETTER_VARIANTS = {
    "A": ["A", " A", "Ａ", "a", " a"],
    "B": ["B", " B", "Ｂ", "b", " b"],
    "是": ["是"],
    "否": ["否"],
}

# 数字候选的表层变体（与字母同理：裸形式 / 前导空格 / 全角）。
for _d, _fw in FULLWIDTH_DIGITS.items():
    LETTER_VARIANTS[_d] = [_d, f" {_d}", _fw, f" {_fw}"]


def get_candidate_ids(word_list, tokenizer):
    """把若干标签词展开成候选 token id 列表（只接受单 token 形式）。"""
    token_ids = []
    for word in word_list:
        for variant in LETTER_VARIANTS.get(word, [word]):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            # 多 token 的变体无法由首 token 表达，跳过（否则会把整词首个 token 当标签）
            if len(encoded) == 1:
                token_ids.append(encoded[0])
    token_ids = list(dict.fromkeys(token_ids))
    if not token_ids:
        raise ValueError("get_candidate_ids ERROR: 词表中无有效候选 token")
    return token_ids


def candidate_mass(first_token_logprobs, id_groups):
    """把候选按标签分组求和：返回 (每组总概率质量, 归一化后各组概率)。"""
    masses = [sum(math.exp(first_token_logprobs[t].logprob) for t in g if t in first_token_logprobs)
              for g in id_groups]
    total = sum(masses) + 1e-8
    return masses, [m / total for m in masses]


def make_mask_processor(allowed_ids):
    """受限解码：把候选集之外的 logits 全部掩为 -inf，保证首 token 落在候选上。

    这一步让"首 token 概率"真正等于模型在二选一条件下的后验概率，
    而不是"模型想说什么"与候选词的偶然重合。

    vLLM 各版本对 logits_processor 的入参签名不一致（(token_ids, logits) 或
    多参数新式签名），故用 *args 取其中唯一的 logits 张量，避免版本差异导致报错。
    """
    allowed = sorted(set(allowed_ids))

    def _processor(*args, **kwargs):
        import torch
        logits = kwargs.get("logits")
        if logits is None:
            logits = next(a for a in args if isinstance(a, torch.Tensor))
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked

    return _processor


def build_sampling_params(allowed_ids):
    """构造"首 token 只能是候选"的采样参数，按 vLLM 版本自动降级：

    1) logits_processors（自定义掩码，各版本入参签名已用 *args 兼容）
    2) allowed_token_ids（vLLM 原生的候选白名单，新版移除 logits_processors 后的替代）
    3) 都不支持时不加约束——此时候选若不在 top-k，判定记为无效（不猜）

    2) 与 1) 在数学上等价：都等价于在候选集上做条件归一化。
    """
    allowed = sorted(set(allowed_ids))
    candidates = [
        ("logits_processors", {"logits_processors": [make_mask_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束", {}),
    ]
    for name, extra in candidates:
        try:
            sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra)
        except TypeError as e:
            print(f"受限解码 {name} 不可用（{e}），尝试下一种")
            continue
        print(f"受限解码方式: {name}，候选 id = {allowed}")
        return sp
    raise RuntimeError("当前 vLLM 既不支持 logits_processors 也不支持 allowed_token_ids，"
                       "无法约束首 token 到候选集")


def render_prompts(tokenizer, texts, chat_kwargs, model_name):
    """texts: 已填槽的完整模板字符串列表 → chat template 后的 prompt 列表。

    后缀优先取 MODEL_PROMPT_SUFFIX[model_name]；未登记则自动修正
    （生成前缀不以换行结尾时补 '\\n'，GLM-4-9B-chat 等模型不加换行会扭曲首 token 分布）。
    """
    manual_suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    prompts = []
    for t in texts:
        messages = [{"role": "user", "content": t}]
        p = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if manual_suffix is not None:
            p += manual_suffix
        elif not p.endswith("\n"):
            p += "\n"
        prompts.append(p)
    return prompts


# =============================================================================
# 数据加载
# =============================================================================
def _normalize_sample(item, source_name, split, idx):
    """统一成内部结构，兼容不同数据集字段差异（COLD 无 platform/expression）。"""
    return {
        "id": item.get("id") or f"{source_name}_{split}_{idx:04d}",
        "source": item.get("source") or source_name,
        "split": split,
        "content": item["content"],
        "toxic": item.get("toxic"),
        "topic": item.get("topic"),
        "expression": item.get("expression"),
        "platform": item.get("platform"),
    }


def load_dataset(data_path: Path, split: str):
    """data_path 支持两种形态：
      1) 单个 json 文件（如 data/raw/CCC/ccc.json）——split 置为文件名去后缀；
      2) 数据集目录（如 data/raw/TOXICN），内含 train.json / test.json；
         split = test | train。
    返回 (样本列表, 实际读取的文件名列表, 数据集标签)。
    """
    if not data_path.exists():
        raise ValueError(f"data_path 不存在: {data_path}")

    if data_path.is_file():
        items = json.load(open(data_path, "r", encoding="utf-8"))
        name = data_path.stem
        return ([_normalize_sample(x, name, name, i) for i, x in enumerate(items)],
                [data_path.name], data_path.parent.name)

    p = data_path / f"{split}.json"
    if not p.exists():
        raise ValueError(f"{data_path} 下缺少 {split}.json")
    items = json.load(open(p, "r", encoding="utf-8"))
    samples = [_normalize_sample(x, data_path.name, p.stem, i) for i, x in enumerate(items)]
    return samples, [p.name], data_path.name


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="环1a/1b/2/3/4 + S1 + CAT vLLM 推理")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="目录模式下的划分（单文件模式忽略；单文件模式文件名进输出名）")
    parser.add_argument("--concepts_csv", type=str,
                        default=str(project_root / "data" / "concepts" / "cbm_concept_set_v3.csv"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    concepts = load_ring_concepts(Path(args.concepts_csv))
    n_by_ring = {}
    for c in concepts:
        n_by_ring[c["ring"]] = n_by_ring.get(c["ring"], 0) + 1
    print(f"数据集 {dataset_tag}（{', '.join(files)}）样本 {len(dataset)} 条，概念 "
          + " ".join(f"{k}:{v}" for k, v in n_by_ring.items())
          + f"，另加 CAT×{len(CAT_TEMPLATES)}、S1×{len(S1_TEMPLATES)} 每条文本")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    a_ids = get_candidate_ids(["A"], tokenizer)
    b_ids = get_candidate_ids(["B"], tokenizer)
    digit_groups = [get_candidate_ids([d], tokenizer) for d in DIGITS]
    print(f"verbalizer: 是={yes_ids} 否={no_ids} A={a_ids} B={b_ids} "
          f"数字={[g for g in digit_groups]}")

    sp_yn = build_sampling_params(yes_ids + no_ids)
    sp_ab = build_sampling_params(a_ids + b_ids)
    sp_digit = build_sampling_params([t for g in digit_groups for t in g])

    # ---------- 批量 1：阶段 1（35 概念）+ CAT（6 问/条） ----------
    yn_prompts, yn_keys = [], []
    for sample in dataset:
        for c in concepts:
            tmpl = STAGE1_TEMPLATE_U if c["ring"] == "ring4" else STAGE1_TEMPLATE
            yn_prompts.append(tmpl.format(text=sample["content"],
                                          concept_name=c["name"], definition=c["definition"]))
            yn_keys.append(("s1", sample["id"], c["id"]))
        for cat_name, tmpl in CAT_TEMPLATES.items():
            yn_prompts.append(tmpl.format(text=sample["content"]))
            yn_keys.append(("cat", sample["id"], cat_name))
    n_s1_prompts = sum(1 for k in yn_keys if k[0] == "s1")
    print(f"阶段 1 + CAT 推理（合并批量 {n_s1_prompts}+{len(yn_keys) - n_s1_prompts} 条）...")
    yn_outputs = llm.generate(render_prompts(tokenizer, yn_prompts, chat_kwargs, model_name),
                              sp_yn, use_tqdm=True)

    s1_results, cat_results = {}, {}
    for key, out in zip(yn_keys, yn_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, [yes_ids, no_ids])
        valid = sum(masses) > 0
        rec = {"p_yes": norm[0], "p_no": norm[1],
               "raw": [round(masses[0], 6), round(masses[1], 6)],
               "valid": valid,
               "topk_hit": [any(t in lp for t in yes_ids), any(t in lp for t in no_ids)],
               "gen_token": gen_tok}
        if key[0] == "s1":
            s1_results[(key[1], key[2])] = rec
        else:
            cat_results[(key[1], key[2])] = rec

    # ---------- 阶段 2：仅 0-2 值域概念且阶段 1 判"是" ----------
    tau = STAGE1_THRESHOLD.get(model_name, STAGE1_THRESHOLD_DEFAULT)
    print(f"阶段 1 判是阈值 τ = {tau}（模型 {model_name}，表内登记值或默认值）")
    range2 = {c["id"] for c in concepts if c["range"] == "0-2"}
    concept_by_id = {c["id"]: c for c in concepts}
    sample_by_id = {s["id"]: s for s in dataset}
    s2_tasks = []
    for (sid, cid), rec in s1_results.items():
        if cid in range2 and rec["valid"] and rec["p_yes"] >= tau:
            c = concept_by_id[cid]
            tmpl = STAGE2_TEMPLATE_U if c["ring"] == "ring4" else STAGE2_TEMPLATE
            s2_tasks.append((sid, cid, tmpl.format(text=sample_by_id[sid]["content"],
                                                   concept_name=c["name"],
                                                   definition=c["definition"])))
    print(f"阶段 2 触发 {len(s2_tasks)}/{len(s1_results)} 对"
          f"（0-2 概念 {len(range2)} 个，0-1 概念不触发）")
    s2_results = {}
    if s2_tasks:
        s2_outputs = llm.generate(
            render_prompts(tokenizer, [t[2] for t in s2_tasks], chat_kwargs, model_name),
            sp_ab, use_tqdm=True)
        for (sid, cid, _), out in zip(s2_tasks, s2_outputs):
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            masses, norm = candidate_mass(lp, [a_ids, b_ids])
            valid = sum(masses) > 0
            s2_results[(sid, cid)] = {"p_A": norm[0], "p_B": norm[1],
                                      "pred_B": (masses[1] > masses[0]) if valid else False,
                                      "raw": [round(masses[0], 6), round(masses[1], 6)],
                                      "valid": valid,
                                      "topk_hit": [any(t in lp for t in a_ids),
                                                   any(t in lp for t in b_ids)],
                                      "gen_token": gen_tok}

    # ---------- S1：双变体 ----------
    sev_prompts, sev_keys = [], []
    for sample in dataset:
        for name, tmpl in S1_TEMPLATES.items():
            sev_prompts.append(tmpl.format(text=sample["content"]))
            sev_keys.append((sample["id"], name))
    print(f"S1 推理（批量 {len(sev_prompts)} 条）...")
    sev_outputs = llm.generate(render_prompts(tokenizer, sev_prompts, chat_kwargs, model_name),
                               sp_digit, use_tqdm=True)
    sev_results = {}
    for (sid, name), out in zip(sev_keys, sev_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, digit_groups)
        valid = sum(masses) > 0
        pred = int(max(range(len(DIGITS)), key=lambda i: masses[i])) if valid else None
        sev_results[(sid, name)] = {
            "probs": {d: round(norm[i], 6) for i, d in enumerate(DIGITS)},
            "pred": pred,
            "value": pred if pred is not None else 0,
            "expected": round(sum(i * norm[i] for i in range(len(DIGITS))), 4),
            "valid": valid,
            "topk_hit": [any(t in lp for t in g) for g in digit_groups],
            "gen_token": gen_tok,
        }

    # ---------- 汇总 ----------
    cat_taus = {name: CAT_TAU.get(model_name, {}).get(name, CAT_TAU_DEFAULT)
                for name in CAT_TEMPLATES}
    guard_tau = GUARD_TAU.get(model_name, GUARD_TAU_DEFAULT)
    print("CAT 判是阈值 τ = " + ", ".join(f"{k}:{v}" for k, v in cat_taus.items())
          + f" | 环4 guard 阻断 τ = {guard_tau}")
    results = []
    for sample in dataset:
        sid = sample["id"]
        entry_concepts, ring_vals = {}, {}
        for c in concepts:
            s1r = s1_results[(sid, c["id"])]
            s2r = s2_results.get((sid, c["id"]))
            pred_yes = (s1r["p_yes"] >= tau) if s1r["valid"] else False
            if not pred_yes:
                value = 0
            elif c["range"] == "0-1":
                value = 1
            elif s2r is None or not s2r["valid"]:
                value = 1  # 阶段 1 判是但强度没测出，保守取 1
            else:
                value = 2 if s2r["pred_B"] else 1
            entry_concepts[c["id"]] = {"stage1": s1r, "stage2": s2r, "value": value}
            ring_vals.setdefault(c["ring"], {}).setdefault(c["role"], []).append(value)

        rings = {}
        for ring, roles in ring_vals.items():
            rings[ring] = {
                "evidence_max": max(roles.get("evidence", [0])),
                "guard_max": max(roles.get("guard", [0])),
            }
        results.append({
            "id": sid,
            "source": sample["source"],
            "split": sample["split"],
            "content": sample["content"],
            "toxic": sample["toxic"],
            "topic": sample.get("topic"),
            "expression": sample.get("expression"),
            "platform": sample.get("platform"),
            "concepts": entry_concepts,
            "rings": rings,
            **{name: cat_results[(sid, name)] for name in CAT_TEMPLATES},
            **{name: sev_results[(sid, name)] for name in S1_TEMPLATES},
        })

    # ---------- 保存 ----------
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / dataset_tag / model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 输出名带数据来源后缀，防两次运行互相覆盖：
    #   单文件模式 → 文件名去后缀（如 ccc）；目录模式 → split（test/train）
    name_suffix = (Path(args.data_path).stem if Path(args.data_path).is_file()
                   else args.split)
    json_path = out_dir / f"rings_{model_name}_{name_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "data_path": str(args.data_path),
                "files": files,
                "split": args.split,
                "n_samples": len(dataset),
                "prompt_version": "rings v1.4",
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "stage1_tau": tau,
                "cat_taus": cat_taus,
                "guard_tau": guard_tau,
                "concepts": [{"id": c["id"], "ring": c["ring"], "role": c["role"],
                              "range": c["range"]} for c in concepts],
                "stage2_trigger_rate": round(len(s2_tasks) / max(1, len(s1_results)), 4),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=4)

    # 扁平 CSV（WPS 审阅用）
    csv_path = out_dir / f"rings_{model_name}_{name_suffix}_flat.csv"
    cat_names = list(CAT_TEMPLATES)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "toxic", "expression", "concept", "ring", "role",
                    "value", "p_yes", "stage1_pred", "p_B", "stage2_valid", "stage2_gen",
                    "S1A_pred", "S1B_pred",
                    *[f"{n}_p" for n in cat_names], "content"])
        for r in results:
            cat_p = [round(r[n]["p_yes"], 4) for n in cat_names]
            for cid, e in r["concepts"].items():
                meta_c = concept_by_id[cid]
                w.writerow([r["id"], r["source"], r["split"], r["toxic"], r.get("expression"),
                            cid, meta_c["ring"], meta_c["role"], e["value"],
                            round(e["stage1"]["p_yes"], 4), int(e["stage1"]["p_yes"] >= tau),
                            round(e["stage2"]["p_B"], 4) if e["stage2"] else "",
                            int(e["stage2"]["valid"]) if e["stage2"] else "",
                            e["stage2"]["gen_token"] if e["stage2"] else "",
                            r["S1A"]["pred"], r["S1B"]["pred"], *cat_p, r["content"]])

    # ---------- 概览统计 ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(results)}")
    for c in concepts:
        vals = [r["concepts"][c["id"]]["value"] for r in results]
        yes = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["p_yes"] >= tau)
        dist = {v: vals.count(v) for v in (0, 1, 2) if vals.count(v)}
        print(f"  {c['id']}({c['ring']},{c['role']},{c['range']}): 阶段1'是' {yes}/{len(results)}"
              f"  值分布 {dist}")
    for name in cat_names:
        pm = sum(r[name]["p_yes"] for r in results) / len(results)
        y = sum(1 for r in results if r[name]["p_yes"] >= cat_taus[name])
        print(f"  {name}: 判是 {y}/{len(results)}  p(是)均 {pm:.4f}")
    for name in S1_TEMPLATES:
        preds = [r[name]["pred"] for r in results]
        dist = {d: preds.count(d) for d in range(6) if preds.count(d)}
        e_mean = sum(r[name]["expected"] for r in results) / len(results)
        print(f"  {name}: 档位分布 {dist}  E[S1]均 {e_mean:.3f}")
    off_yn = sum(1 for v in list(s1_results.values()) + list(cat_results.values())
                 if v["gen_token"].strip() not in ("是", "否"))
    off_s2 = sum(1 for v in s2_results.values() if v["gen_token"].strip() not in ("A", "B"))
    off_d = sum(1 for v in sev_results.values() if v["gen_token"].strip() not in DIGITS)
    inv_yn = sum(1 for v in list(s1_results.values()) + list(cat_results.values())
                 if not v["valid"])
    inv_s2 = sum(1 for v in s2_results.values() if not v["valid"])
    inv_d = sum(1 for v in sev_results.values() if not v["valid"])
    print(f"首token越界: 阶段1+CAT {off_yn}  阶段2 {off_s2}  S1 {off_d}")
    print(f"无效判定: 阶段1+CAT {inv_yn}  阶段2 {inv_s2}  S1 {inv_d}（受限解码后应为 0）")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
