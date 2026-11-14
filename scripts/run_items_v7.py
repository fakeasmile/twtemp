"""概念集 v7 条目测量脚本（vLLM 加速，完全自包含）。

v7 词典（180 条 = 19 构念×8 + G2/G3/E1/E2/E4 新构念 28 条）的测量层。
本脚本不 import 项目内任何其他脚本；LLM 加载、verbalizer、受限解码、
chat 渲染机制参照 run_items_v6.py 的已验证实现内联复制，仅测量对象与
输出改为 v7：

  输出三件套（文件名带 v7 前缀，不覆盖 v6 历史矩阵）：
    items_v7_{model}_{split}.json
      —— meta（含受限解码方式、首token质量统计）+ p_yes 矩阵 + 无效判定明细
    items_v7_{model}_{split}_matrix.csv
      —— 宽表：id/toxic/content + 180 列条目 p_yes（下游聚合直接读）
    items_v7_{model}_{split}_firsttoken.csv
      —— 每次推理一行：sample_id,item_id,first_token,p_yes,cand_mass
         cand_mass = 首 token 分布中"是+否"候选的原始概率质量之和。
         受限解码正常时 cand_mass=1.000、first_token∈{是,否}，
         覆盖率应≈100%；若 cand_mass 明显<1 或首token越界，
         说明该次推理的受限解码降级/失效，可据此逐条审计。

  首 token 记录协议（本轮新增，硬性要求）：
    全部推理逐条记录 first_token（解码出的词）与 cand_mass（是+否原始质量），
    不做抽样、不做省略；无效判定另在 json 的 invalid 列表登记。

测量协议（继承协议 M，冻结）：
  出现性锚定模板首句 + "变形出现同样算"条款（与 v3/v6 一字不差）；
  temperature=0；受限解码保证首 token 落在 是/否 候选上；
  不启用量化（量化改变 logits 分布，影响概率读数效度）；
  反向条目不在此反转——测量层存原始 p_yes，反向计分（1−p）在聚合层做。

使用示例（远端 /root/autodl-tmp/twtemp，conda env toxic_env）：
  冒烟（8 样本验证首token覆盖）:
    python scripts/run_items_v7.py --model_path /root/autodl-tmp/glm-4-9b-chat \
        --data_path data/raw/TOXICN --split test --limit 8
  全量:
    python scripts/run_items_v7.py --model_path /root/autodl-tmp/glm-4-9b-chat \
        --data_path data/raw/TOXICN --split test
    python scripts/run_items_v7.py --model_path /root/autodl-tmp/glm-4-9b-chat \
        --data_path data/raw/TOXICN --split train
  （运行前需把 cbm_concept_set_v7.csv 同步到远端 data/concepts/ 下）
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
# 模型加载与生成前缀修正（参照 run_items_v6.py 已验证实现）
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
        enable_prefix_caching=True,     # 同文本的 180 条条目共享"模板头+文本"前缀，prefill 复用
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


# 生成前缀后缀手工指定表（按模型目录名）。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}


# =============================================================================
# 提示词模板（协议 M 冻结，与 v3/v6 逐字一致）
# =============================================================================
ITEM_TEMPLATE = """你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。问句指出的现象以讽刺、委婉、谐音、缩写等变形出现同样算。

文本：{text}

问句：{question}

只回答“是/否”。回答："""


# =============================================================================
# 条目词典加载（v7 8 列 schema；管线必需 7 列 + 设计依据列不参与测量）
# =============================================================================
def load_v7_items(csv_path: Path):
    """读 v7 条目池，校验必需列与 id 唯一性。"""
    required = ["条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句"]
    items = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 传输环节可能在表头引入多余 BOM/NUL（曾出现双重 BOM 致 KeyError），统一剥掉
        reader.fieldnames = [(fn or "").lstrip("\ufeff\x00").strip() for fn in reader.fieldnames]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} 缺少必需列: {missing}")
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
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"条目 id 重复: {dup}")
    bad_dir = sorted({it["direction"] for it in items} - {"正向", "反向"})
    if bad_dir:
        raise ValueError(f"方向列含非法取值: {bad_dir}")
    return items


# =============================================================================
# Verbalizer 与受限解码（参照 run_items_v6.py 已验证实现）
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
    返回 (SamplingParams, 实际生效方式名)——方式名进 meta 供首token审计。
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
        return sp, name
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
# 数据加载
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
    parser = argparse.ArgumentParser(description="v7 条目池单阶段测量（vLLM，首token逐条登记）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 /root/autodl-tmp/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="目录模式下的划分（单文件模式忽略）")
    parser.add_argument("--items_csv", type=str,
                        default=str(project_root / "data" / "concepts" / "cbm_concept_set_v7.csv"))
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（temperature=0 下主要作运行批次登记，进输出目录名防覆盖）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>_seed{N}/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None,
                        help="只测前 N 个样本（冒烟/试点用；全量运行不传）")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    if args.limit is not None and args.limit > 0:
        dataset = dataset[:args.limit]
        print(f"--limit 生效：仅测量前 {len(dataset)} 个样本")
    items = load_v7_items(Path(args.items_csv))

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
    sp_yn, restrict_mode = build_sampling_params(yes_ids + no_ids)

    # ---------- 单阶段批量推理：样本外层 × 条目内层（同样本的 180 问共享前缀缓存） ----------
    prompts, keys = [], []
    for sample in dataset:
        for it in items:
            prompts.append(ITEM_TEMPLATE.format(text=sample["content"], question=it["question"]))
            keys.append((sample["id"], it["item_id"]))
    print(f"条目测量推理（批量 {len(prompts)} 条）...")
    outputs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                           sp_yn, use_tqdm=True)

    # ---------- 提取 p_yes + 逐条登记首token ----------
    # matrix: sid -> {item_id: p_yes(4位小数)}；无效判定（候选质量=0）单独登记。
    # 首token逐条流式写入 firsttoken CSV（全量推理一条不漏）。
    matrix = {s["id"]: {} for s in dataset}
    invalid = []
    n_total = len(outputs)
    n_in_range = 0        # 首 token 解码后 ∈ {是, 否}
    mass_sum = 0.0        # 是+否 原始概率质量之和（受限解码正常时应=1）
    mass_ge_999 = 0       # cand_mass >= 0.999 的推理数
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / dataset_tag / f"{model_name}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = (Path(args.data_path).stem if Path(args.data_path).is_file()
                   else args.split)
    ft_path = out_dir / f"items_v7_{model_name}_{name_suffix}_firsttoken.csv"
    with open(ft_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "item_id", "first_token", "p_yes", "cand_mass"])
        for (sid, iid), out in zip(keys, outputs):
            lp = out.outputs[0].logprobs[0]
            tok_ids = list(out.outputs[0].token_ids)
            gen_tok = tokenizer.decode(tok_ids)
            masses, norm = candidate_mass(lp, [yes_ids, no_ids])
            raw = sum(masses)
            valid = raw > 0
            p_yes = round(norm[0], 4) if valid else None
            matrix[sid][iid] = p_yes
            w.writerow([sid, iid, gen_tok, "" if p_yes is None else p_yes, round(raw, 4)])
            if gen_tok.strip() in ("是", "否"):
                n_in_range += 1
            mass_sum += raw
            if raw >= 0.999:
                mass_ge_999 += 1
            if not valid:
                invalid.append({"id": sid, "item": iid, "gen_token": gen_tok})

    # ---------- 首token质量统计（覆盖审计） ----------
    mass_mean = mass_sum / n_total if n_total else float("nan")
    ft_stats = {
        "total": n_total,
        "in_range": n_in_range,
        "in_range_pct": round(n_in_range / n_total, 6) if n_total else float("nan"),
        "cand_mass_mean": round(mass_mean, 6),
        "cand_mass_ge_999_pct": round(mass_ge_999 / n_total, 6) if n_total else float("nan"),
    }

    # ---------- 保存明细 JSON（文件名带 v7 前缀，不覆盖 v6） ----------
    json_path = out_dir / f"items_v7_{model_name}_{name_suffix}.json"
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
                "prompt_version": "v7.0",
                "items_csv": str(args.items_csv),
                "restricted_decoding": restrict_mode,
                "first_token_stats": ft_stats,
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "items": [{"item_id": it["item_id"], "construct_id": it["construct_id"],
                           "form": it["form"], "direction": it["direction"]}
                          for it in items],
                "n_invalid": len(invalid),
            },
            "matrix": matrix,
            "invalid": invalid,
        }, f, ensure_ascii=False)  # 紧凑输出（train × 180 体量大，不用 indent）

    # 宽表矩阵 CSV（条目分析/聚合读出直接读这张表）
    csv_path = out_dir / f"items_v7_{model_name}_{name_suffix}_matrix.csv"
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
    print(f"首token质量: 总问询 {ft_stats['total']}  首∈{{是,否}} {n_in_range} "
          f"({ft_stats['in_range_pct']*100:.2f}%)  cand_mass均值 {ft_stats['cand_mass_mean']:.6f}  "
          f"mass>=0.999占比 {ft_stats['cand_mass_ge_999_pct']*100:.2f}%")
    print(f"无效判定: {len(invalid)}（受限解码后应为 0）")

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
        rev_set = set(rev)

        def construct_p(sids_subset):
            vals = []
            for sid in sids_subset:
                pv = []
                for iid in c_items:
                    v = matrix[sid].get(iid)
                    if v is None:
                        continue
                    pv.append((1 - v) if iid in rev_set else v)
                if pv:
                    vals.append(sum(pv) / len(pv))
            return sum(vals) / len(vals) if vals else float("nan")

        m1 = construct_p(toxic_yes)
        m0 = construct_p(toxic0_yes)
        print(f"  {cid}: 构念均分 toxic1={m1:.4f} toxic0={m0:.4f} 差={m1 - m0:+.4f} "
              f"（正向{len(pos)} 反向{len(rev)}，反向已按 1−p 计入，仅概览参考）")
    print(f"输出: {json_path}\n      {csv_path}\n      {ft_path}")


if __name__ == "__main__":
    main()
