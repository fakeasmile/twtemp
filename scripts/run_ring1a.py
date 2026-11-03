"""环1a 提示词 v2.11 两阶段推理脚本（vLLM 加速）。

流程（每次调用 = 一条文本 × 一个判断项，模板取自 data/concepts/prompt_ring1a_v1.md）：
  阶段 1  存在性二值判断（候选 是/否），全部 文本×概念 调用
  阶段 2  强度两档判断（候选 A/B），仅阶段 1 判"是"的对触发
  CAT1a   类别整体单问（候选 是/否），每条文本一次
提取：首 token logprobs → verbalizer 概率归一化。
数据层映射：阶段1"否"→0，阶段2 A→1、B→2；结果含 ring1a 聚合（max 值与确认位）。

概念来源：data/concepts/cbm_concept_set_v3.csv 中 类别=="环1a 负面评价" 的全部概念。

使用示例：
  python scripts/run_ring1a.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/CCC/ccc.json
  python scripts/run_ring1a.py --model_path D:/models/Qwen3.5-9B --data_path data/raw/TOXICN --split test
data_path 支持单个 json 文件，或含 train.json/test.json 的目录（--split test|train|all）。

说明（均自动处理，无需命令行指定）：
  首 token 位置修正：优先查代码内 MODEL_PROMPT_SUFFIX 表（按模型目录名手工登记后缀，
  如 glm-4-9b-chat: '\n'）；未登记的模型走自动修正——生成前缀不以换行结尾则补 '\n'。
  思考模式探测：chat template 含 enable_thinking 变量（Qwen3 系混合思考模型）时自动传
  enable_thinking=False——思考前缀会占据首 token，破坏 verbalizer 提取。脚本不判定支持哪些模型。

性能设定（针对 32G vGPU + 只要首 token 的负载）：
  每请求 max_tokens=1，负载以 prefill 为主：enforce_eager 省 CUDA graph 显存与启动时间、
  prefix caching 开、阶段 1 与 CAT1a 合并单次批量调用、阶段 2 一次批量调用。
  全程共 2 次 llm.generate 往返。9B bf16 权重约 18G，0.85 利用率下 KV 约 9G，
  单模型一次装不下可降 --gpu_memory_utilization 前先确认无其他进程占卡。
  不启用量化：量化改变 logits 分布，影响概率读数效度（测量类任务不可妥协）。
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# 模型加载
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

# CAT1a 判是阈值手工表（按模型目录名）。CAT 问句与概念问句措辞独立，但同样受模型级
# 概率偏移影响（CCC 实测 5b 扫描：Qwen τ=0.2 的工作点 66%/87.5%/51.7% 与 GLM τ=0.5
# 的 69%/95%/51.7% 逐项对齐；Qwen 的 CAT 连续分离度仅 1.7×，弱于其阶段 1 的 4×）。
CAT1A_THRESHOLD = {
    "glm-4-9b-chat": 0.50,
    "Qwen3.5-9B": 0.20,
}
CAT1A_THRESHOLD_DEFAULT = 0.50


# =============================================================================
# 提示词模板（与 prompt_ring1a_v1.md v2.11 定稿逐字一致）
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

CAT1A_TEMPLATE = """以下文本是否包含对某人或某群体的负面态度表达（鄙夷、厌恶、敌意、威胁、道德定罪）。
文本：{text}
文本中有这类表达吗？
只回答 是/否。答："""


# =============================================================================
# 概念词典
# =============================================================================
def load_ring1a_concepts(csv_path: Path):
    concepts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["类别"].strip() == "环1a 负面评价":
                concepts.append({
                    "id": row["id"].strip(),
                    "name": row["概念"].strip(),
                    "definition": row["定义"].strip(),
                })
    if not concepts:
        raise ValueError(f"未在 {csv_path} 中找到 类别==环1a 负面评价 的概念")
    return concepts


# =============================================================================
# Verbalizer 工具
# =============================================================================
# 候选词的表层变体：不同 tokenizer 会把同一字母切成 "A" / " A" / "Ａ" / "a" 等
# 不同 token；只取裸形式会在 Qwen 等词表上漏掉实际被采样的那个 token。
LETTER_VARIANTS = {
    "A": ["A", " A", "Ａ", "a", " a"],
    "B": ["B", " B", "Ｂ", "b", " b"],
    "是": ["是"],
    "否": ["否"],
}


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
    """把候选按标签分组求和：返回 (每组总概率质量, 归一化后两组概率)。"""
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
         split = test | train | all（all 合并两份）。
    返回 (样本列表, 实际读取的文件名列表, 数据集标签)。
    """
    if not data_path.exists():
        raise ValueError(f"data_path 不存在: {data_path}")

    if data_path.is_file():
        items = json.load(open(data_path, "r", encoding="utf-8"))
        name = data_path.stem
        return ([_normalize_sample(x, name, name, i) for i, x in enumerate(items)],
                [data_path.name], data_path.parent.name)

    names = ([n for n in ("train.json", "test.json") if (data_path / n).exists()]
             if split == "all" else [f"{split}.json"])
    if not names:
        raise ValueError(f"{data_path} 下未找到 train.json / test.json")
    samples = []
    for n in names:
        p = data_path / n
        if not p.exists():
            raise ValueError(f"{data_path} 下缺少 {n}")
        items = json.load(open(p, "r", encoding="utf-8"))
        samples += [_normalize_sample(x, data_path.name, p.stem, i) for i, x in enumerate(items)]
    return samples, names, data_path.name


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="环1a 两阶段 + CAT1a vLLM 推理")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录（如 xxx/TOXICN）")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "all"],
                        help="目录模式下的划分（单文件模式忽略）")
    parser.add_argument("--concepts_csv", type=str, default=str(project_root / "data" / "concepts" / "cbm_concept_set_v3.csv"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name  # 输出目录/文件名用模型目录名
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    concepts = load_ring1a_concepts(Path(args.concepts_csv))
    print(f"数据集 {dataset_tag}（{', '.join(files)}）样本 {len(dataset)} 条 × 概念 {len(concepts)} 个 "
          f"({'/'.join(c['id'] for c in concepts)})，另加 CAT1a {len(dataset)} 次")

    tokenizer, llm = load_vllm_model(
        Path(args.model_path), args.gpu_memory_utilization
    )
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    a_ids = get_candidate_ids(["A"], tokenizer)
    b_ids = get_candidate_ids(["B"], tokenizer)
    print(f"verbalizer: 是={yes_ids} 否={no_ids} A={a_ids} B={b_ids}")

    # 受限解码：每次调用都把候选集之外的 token 屏蔽，保证首 token 只可能是候选词，
    # 首 token 概率因而直接就是二选一条件下的后验概率（具体走哪条 API 见打印）。
    sp_yn = build_sampling_params(yes_ids + no_ids)
    sp_ab = build_sampling_params(a_ids + b_ids)

    # ---------- 阶段 1 + CAT1a：合并为一次批量调用 ----------
    # 两者互不依赖（阶段 2 才依赖阶段 1），合并成一个 generate 请求批次，
    # 让 vLLM 调度器统一拼 batch，省一次引擎往返、提高 prefill 吞吐。
    s1_prompts, s1_keys = [], []
    for sample in dataset:
        for c in concepts:
            s1_prompts.append(STAGE1_TEMPLATE.format(
                text=sample["content"], concept_name=c["name"], definition=c["definition"]))
            s1_keys.append((sample["id"], c["id"]))
    cat_prompts = [CAT1A_TEMPLATE.format(text=s["content"]) for s in dataset]
    n_s1 = len(s1_prompts)
    print(f"阶段 1 + CAT1a 推理（合并批量 {n_s1}+{len(cat_prompts)} 条）...")
    all_outputs = llm.generate(
        render_prompts(tokenizer, s1_prompts + cat_prompts, chat_kwargs, model_name),
        sp_yn, use_tqdm=True)
    s1_outputs = all_outputs[:n_s1]
    cat_outputs = all_outputs[n_s1:]

    # ---------- 阶段 2：仅阶段 1 判"是" ----------
    tau = STAGE1_THRESHOLD.get(model_name, STAGE1_THRESHOLD_DEFAULT)
    print(f"阶段 1 判是阈值 τ = {tau}（模型 {model_name}，表内登记值或默认值）")
    s1_results = {}
    s2_tasks = []
    for (sid, cid), out in zip(s1_keys, s1_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, [yes_ids, no_ids])
        valid = sum(masses) > 0
        pred_yes = (norm[0] >= tau) if valid else False
        s1_results[(sid, cid)] = {"p_yes": norm[0], "p_no": norm[1], "pred_yes": pred_yes,
                                  "raw": [round(masses[0], 6), round(masses[1], 6)],
                                  "valid": valid,
                                  "topk_hit": [any(t in lp for t in yes_ids),
                                               any(t in lp for t in no_ids)],
                                  "gen_token": gen_tok}
        if pred_yes:
            sample = next(s for s in dataset if s["id"] == sid)
            concept = next(c for c in concepts if c["id"] == cid)
            s2_tasks.append((sid, cid, STAGE2_TEMPLATE.format(
                text=sample["content"], concept_name=concept["name"],
                definition=concept["definition"])))
    trigger_rate = len(s2_tasks) / max(1, len(s1_keys))
    print(f"阶段 2 触发 {len(s2_tasks)}/{len(s1_keys)} 对（{trigger_rate:.1%}）")

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

    # ---------- 汇总与映射 ----------
    results = []
    cat_tau = CAT1A_THRESHOLD.get(model_name, CAT1A_THRESHOLD_DEFAULT)
    print(f"CAT1a 判是阈值 τ = {cat_tau}（模型 {model_name}，表内登记值或默认值）")
    for sample, cat_out in zip(dataset, cat_outputs):
        lp = cat_out.outputs[0].logprobs[0]
        cat_masses, cat_norm = candidate_mass(lp, [yes_ids, no_ids])
        cat = {"p_yes": cat_norm[0], "p_no": cat_norm[1],
               "pred_yes": (cat_norm[0] >= cat_tau) if sum(cat_masses) > 0 else False,
               "raw": [round(cat_masses[0], 6), round(cat_masses[1], 6)],
               "valid": sum(cat_masses) > 0,
               "topk_hit": [any(t in lp for t in yes_ids), any(t in lp for t in no_ids)],
               "gen_token": tokenizer.decode(list(cat_out.outputs[0].token_ids))}

        concept_values, entry_concepts = {}, {}
        for c in concepts:
            s1 = s1_results[(sample["id"], c["id"])]
            s2 = s2_results.get((sample["id"], c["id"]))
            if not s1["pred_yes"]:
                value = 0
            elif s2 is None or not s2["valid"]:
                # 阶段 1 已确认存在，强度没测出时保守取 1，不默认成 B
                value = 1
            else:
                value = 2 if s2["pred_B"] else 1
            concept_values[c["id"]] = value
            entry_concepts[c["id"]] = {
                "stage1": s1,
                "stage2": s2,
                "value": value,
            }

        max_value = max(concept_values.values())
        results.append({
            "id": sample["id"],
            "source": sample["source"],
            "split": sample["split"],
            "content": sample["content"],
            "toxic": sample["toxic"],
            "topic": sample.get("topic"),
            "expression": sample.get("expression"),
            "platform": sample.get("platform"),
            "concepts": entry_concepts,
            "ring1a_max_value": max_value,
            "ring1a_confirmed": max_value >= 1,
            "CAT1a": cat,
        })

    # ---------- 保存 ----------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "data" / "processed" / dataset_tag / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ring1a_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "data_path": str(args.data_path),
                "files": files,
                "split": args.split,
                "n_samples": len(dataset),
                "prompt_version": "ring1a v2.11",
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "stage2_trigger_rate": round(trigger_rate, 4),
                "stage1_tau": tau,
                "cat_tau": cat_tau,
                "concepts": [c["id"] for c in concepts],
            },
            "results": results,
        }, f, ensure_ascii=False, indent=4)

    # 扁平 CSV（WPS 审阅用）
    csv_path = out_dir / f"ring1a_{model_name}_flat.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "toxic", "expression", "concept",
                    "p_yes", "stage1_pred", "stage1_gen", "p_B", "stage2_gen", "stage2_valid", "value",
                    "ring1a_max", "ring1a_confirmed", "CAT1a_p_yes", "CAT1a_gen", "content"])
        for r in results:
            for cid, e in r["concepts"].items():
                w.writerow([r["id"], r["source"], r["split"], r["toxic"], r.get("expression"), cid,
                            round(e["stage1"]["p_yes"], 4), int(e["stage1"]["pred_yes"]),
                            e["stage1"]["gen_token"],
                            round(e["stage2"]["p_B"], 4) if e["stage2"] else "",
                            e["stage2"]["gen_token"] if e["stage2"] else "",
                            int(e["stage2"]["valid"]) if e["stage2"] else "",
                            e["value"], r["ring1a_max_value"], int(r["ring1a_confirmed"]),
                            round(r["CAT1a"]["p_yes"], 4), r["CAT1a"]["gen_token"], r["content"]])

    # ---------- 概览统计 ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(results)}")
    print(f"阶段2触发率: {trigger_rate:.1%}")
    for c in concepts:
        vs = [r["concepts"][c["id"]]["value"] for r in results]
        s1y = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["pred_yes"])
        print(f"  {c['id']}: 阶段1'是' {s1y}/{len(results)}  值分布 0/1/2 = "
              f"{vs.count(0)}/{vs.count(1)}/{vs.count(2)}")
    confirmed = sum(1 for r in results if r["ring1a_confirmed"])
    print(f"环1a 确认: {confirmed}/{len(results)}")
    cat_mean = sum(r["CAT1a"]["p_yes"] for r in results) / len(results)
    print(f"CAT1a p(是) 均值: {cat_mean:.4f}")
    # 首 token 越界监控：生成 token 不在候选集（是/否 或 A/B）的调用数
    off_s1 = sum(1 for v in s1_results.values() if v["gen_token"].strip() not in ("是", "否"))
    off_s2 = sum(1 for v in s2_results.values() if v["gen_token"].strip() not in ("A", "B"))
    off_cat = sum(1 for r in results if r["CAT1a"]["gen_token"].strip() not in ("是", "否"))
    print(f"首token越界: 阶段1 {off_s1}/{len(s1_results)}  阶段2 {off_s2}/{len(s2_results)}  CAT1a {off_cat}/{len(results)}")
    # 候选未进 top-k 监控：任一候选缺失则说明归一化读数失真
    miss_s1 = sum(1 for v in s1_results.values() if not all(v["topk_hit"]))
    miss_s2 = sum(1 for v in s2_results.values() if not all(v["topk_hit"]))
    miss_cat = sum(1 for r in results if not all(r["CAT1a"]["topk_hit"]))
    # 无效判定监控：受限解码后应为 0；非 0 说明 logits_processors 未生效
    inv_s1 = sum(1 for v in s1_results.values() if not v["valid"])
    inv_s2 = sum(1 for v in s2_results.values() if not v["valid"])
    inv_cat = sum(1 for r in results if not r["CAT1a"]["valid"])
    print(f"无效判定: 阶段1 {inv_s1}/{len(s1_results)}  阶段2 {inv_s2}/{len(s2_results)}  CAT1a {inv_cat}/{len(results)}")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
