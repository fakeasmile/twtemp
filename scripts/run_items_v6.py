"""概念集 v6 条目测量脚本（vLLM 加速，自包含）。

v6 心理测量学架构的测量层：读 cbm_concept_set_v6.csv 的 152 条条目（19 构念 × 8），
对每条文本逐条目单阶段问询（是/否问句，首 token logprobs 提 p_yes）。
与 run_rings.py 的测量机制逐字同源（vLLM 加载、verbalizer 候选展开、首 token
受限解码、chat 渲染与后缀修正、数据加载），仅测量对象与流程改为：
  单阶段：每条目一次调用，p_yes ∈ [0,1] 直接落矩阵（soft 分数进下游，
          不做阈值二值化——二值化是条目分析/读出层的可选变换，测量层只存概率）；
  无阶段 2（v3 两阶段取消，直接性已是 D1 构念）；
  无 CAT / S1 通道（其信号已条目化为 D1/D2/D3）。

测量协议（继承协议 M，冻结）：
  出现性锚定模板首句 + "变形出现同样算"条款（与 v3 阶段 1 一字不差）；
  temperature=0；受限解码保证首 token 落在 是/否 候选上；
  不启用量化（量化改变 logits 分布，影响概率读数效度）；
  反向条目不在此反转——测量层存原始 p_yes，反向计分（1−p）在聚合层做。

提示词只含条目问句本身，不含构念操作化定义（MMPI 传统：被测者不见量表定义；
条目问句 + 例词已构成完整操作性定义，见设计文档 §3 规则 5）。

输出（防覆盖）：
  data/processed/<数据集>/<model_name>_seed{N}/items_{model_name}_{split}.json
    —— 全量明细：每样本 × 每条目的 p_yes；无效判定单独登记（含首 token）。
  同目录 items_{model_name}_{split}_matrix.csv
    —— 宽表矩阵：每行一个样本（id/toxic/content + 152 列条目 p_yes），
       条目分析与聚合读出脚本直接读这张表。
双教师 = 同一脚本换 --model_path 跑两次（GLM-4-9B-Chat 主测 + Qwen3.5-9B 复测）。

使用示例：
  python scripts/run_items_v6.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/TOXICN --split test
  python scripts/run_items_v6.py --model_path D:/models/Qwen3.5-9B --data_path data/raw/TOXICN --split train

自动处理说明（与 run_rings.py 一致）：
  首 token 位置修正：优先查 MODEL_PROMPT_SUFFIX 表（按模型目录名登记后缀）；
  未登记的模型走自动修正——生成前缀不以换行结尾则补 '\n'。
  思考模式探测：chat template 含 enable_thinking 变量时自动传 enable_thinking=False。
  TRANSFORMERS_OFFLINE=1 强制离线，禁远程下载。
"""

import os

# 离线约束：测量环境禁远程下载（模型路径显式给出，权重本地已就位）
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import json
import math
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
# 模型加载与生成前缀修正（与 run_rings.py 逐字同源）
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
        enable_prefix_caching=True,     # 同文本的 152 条条目共享"模板头+文本"前缀，prefill 复用
        max_model_len=2048,             # 模板+问句+文本最长约 400-600 token，2048 留足余量
        max_num_seqs=64,
        max_num_batched_tokens=16384,   # 批量以 prefill 为主（每请求只解码 1 token）
        enforce_eager=True,             # 只解码 1 个 token，禁 CUDA graph 省显存（腾给 KV cache）
        max_logprobs=50,                # 与 SamplingParams(logprobs=50) 配套
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    """chat template 支持 enable_thinking 时关闭思考模式，否则返回空。"""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


# 生成前缀后缀手工指定表（按模型目录名，与 run_rings.py 同源）。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}


# =============================================================================
# 提示词模板（v6.0 定稿）
# =============================================================================
# 模板首句与"变形出现同样算"条款 = run_rings.py STAGE1_TEMPLATE 逐字继承（协议 M）。
# 问句槽直接填 v6 csv 的"条目问句"列（问句自带例词与排除条款）。
ITEM_TEMPLATE = """你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。问句指出的现象以讽刺、委婉、谐音、缩写等变形出现同样算。

文本：{text}

问句：{question}

只回答“是/否”。回答："""


# =============================================================================
# 条目词典加载
# =============================================================================
def load_v6_items(csv_path: Path):
    """读 v6 条目池：152 条条目，字段含构念归属、题式、方向、问句。"""
    items = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 传输环节可能在表头引入多余 BOM/NUL（曾出现双重 BOM 致 KeyError），统一剥掉
        reader.fieldnames = [(fn or "").lstrip("\ufeff\x00").strip() for fn in reader.fieldnames]
        for row in reader:
            items.append({
                "item_id": row["条目id"].strip(),
                "construct_id": row["构念id"].strip(),
                "construct_name": row["构念名"].strip(),
                "block": row["区块"].strip(),
                "form": row["题式"].strip(),
                "direction": row["方向"].strip(),
                "question": row["条目问句"].strip(),
            })
    if not items:
        raise ValueError(f"{csv_path} 中无条目")
    ids = [it["item_id"] for it in items]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"条目 id 重复: {dup}")
    return items


# =============================================================================
# Verbalizer 工具（run_rings.py 已验证机制，逐字并入）
# =============================================================================
LETTER_VARIANTS = {
    "是": ["是"],
    "否": ["否"],
}


def get_candidate_ids(word_list, tokenizer):
    """把若干标签词展开成候选 token id 列表（只接受单 token 形式）。"""
    token_ids = []
    for word in word_list:
        for variant in LETTER_VARIANTS.get(word, [word]):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
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
    """受限解码：把候选集之外的 logits 全部掩为 -inf，保证首 token 落在候选上。"""
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
    logits_processors → allowed_token_ids → 无约束（无效判定不猜）。
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
    """texts: 已填槽的完整模板字符串列表 → chat template 后的 prompt 列表。"""
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
# 数据加载（与 run_rings.py 同源）
# =============================================================================
def _normalize_sample(item, source_name, split, idx):
    return {
        "id": item.get("id") or f"{source_name}_{split}_{idx:04d}",
        "source": item.get("source") or source_name,
        "split": split,
        "content": item["content"],
        "toxic": item.get("toxic"),
    }


def load_dataset(data_path: Path, split: str):
    """data_path 支持单个 json 文件或含 train.json/test.json 的目录。"""
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
    parser = argparse.ArgumentParser(description="v6 条目池单阶段测量（vLLM）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="目录模式下的划分（单文件模式忽略）")
    parser.add_argument("--items_csv", type=str,
                        default=str(project_root / "data" / "concepts" / "cbm_concept_set_v6.csv"))
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（temperature=0 下主要作运行批次登记，进输出目录名防覆盖）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>_seed{N}/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    items = load_v6_items(Path(args.items_csv))

    n_by_construct = {}
    for it in items:
        n_by_construct[it["construct_id"]] = n_by_construct.get(it["construct_id"], 0) + 1
    n_reverse = sum(1 for it in items if it["direction"] == "反向")
    print(f"数据集 {dataset_tag}（{', '.join(files)}）样本 {len(dataset)} 条，"
          f"条目 {len(items)} 条（{len(n_by_construct)} 构念，反向 {n_reverse}），"
          f"总问询 {len(dataset) * len(items)} 次")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    print(f"verbalizer: 是={yes_ids} 否={no_ids}")
    sp_yn = build_sampling_params(yes_ids + no_ids)

    # ---------- 单阶段批量推理：样本外层 × 条目内层（同样本的 152 问共享前缀缓存） ----------
    prompts, keys = [], []
    for sample in dataset:
        for it in items:
            prompts.append(ITEM_TEMPLATE.format(text=sample["content"], question=it["question"]))
            keys.append((sample["id"], it["item_id"]))
    print(f"条目测量推理（批量 {len(prompts)} 条）...")
    outputs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                           sp_yn, use_tqdm=True)

    # ---------- 提取 p_yes ----------
    # matrix: sid -> {item_id: p_yes(4位小数)}；invalid: 无效判定登记（受限解码后应为空）
    matrix = {s["id"]: {} for s in dataset}
    invalid = []
    for (sid, iid), out in zip(keys, outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, [yes_ids, no_ids])
        valid = sum(masses) > 0
        if valid:
            matrix[sid][iid] = round(norm[0], 4)
        else:
            matrix[sid][iid] = None
            invalid.append({"id": sid, "item": iid, "gen_token": gen_tok})

    # ---------- 保存 ----------
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / dataset_tag / f"{model_name}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = (Path(args.data_path).stem if Path(args.data_path).is_file()
                   else args.split)
    json_path = out_dir / f"items_{model_name}_{name_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "data_path": str(args.data_path),
                "files": files,
                "split": args.split,
                "seed": args.seed,
                "n_samples": len(dataset),
                "prompt_version": "v6.0",
                "items_csv": str(args.items_csv),
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "items": [{"item_id": it["item_id"], "construct_id": it["construct_id"],
                           "form": it["form"], "direction": it["direction"]}
                          for it in items],
                "n_invalid": len(invalid),
            },
            "matrix": matrix,
            "invalid": invalid,
        }, f, ensure_ascii=False)  # 紧凑输出（train × 152 体量大，不用 indent）

    # 宽表矩阵 CSV（条目分析/聚合读出直接读这张表）
    csv_path = out_dir / f"items_{model_name}_{name_suffix}_matrix.csv"
    item_ids = [it["item_id"] for it in items]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "toxic", "content", *item_ids])
        for s in dataset:
            row = [s["id"], s["toxic"], s["content"]]
            for iid in item_ids:
                v = matrix[s["id"]].get(iid)
                row.append("" if v is None else v)
            w.writerow(row)

    # ---------- 概览统计 ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(dataset)}  条目: {len(items)}")
    def _label(s):
        try:
            return int(s["toxic"])
        except (TypeError, ValueError):
            return None

    toxic_yes = {s["id"] for s in dataset if _label(s) == 1}
    toxic0_yes = {s["id"] for s in dataset if _label(s) == 0}
    for cid in n_by_construct:
        c_items = [it["item_id"] for it in items if it["construct_id"] == cid]
        pos = [it["item_id"] for it in items if it["construct_id"] == cid and it["direction"] == "正向"]
        rev = [it["item_id"] for it in items if it["construct_id"] == cid and it["direction"] == "反向"]

        def construct_p(sids_subset):
            vals = []
            for sid in sids_subset:
                pv = []
                for iid in c_items:
                    v = matrix[sid].get(iid)
                    if v is None:
                        continue
                    pv.append((1 - v) if iid in set(rev) else v)
                if pv:
                    vals.append(sum(pv) / len(pv))
            return sum(vals) / len(vals) if vals else float("nan")

        m1 = construct_p(toxic_yes)
        m0 = construct_p(toxic0_yes)
        print(f"  {cid}: 构念均分 toxic1={m1:.4f} toxic0={m0:.4f} 差={m1 - m0:+.4f} "
              f"（正向{len(pos)} 反向{len(rev)}，反向已按 1−p 计入，仅概览参考）")
    off = sum(1 for _, out in zip(keys, outputs)
              if tokenizer.decode(list(out.outputs[0].token_ids)).strip() not in ("是", "否"))
    print(f"首token越界: {off}  无效判定: {len(invalid)}（受限解码后应为 0）")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
