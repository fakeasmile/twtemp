# -*- coding: utf-8 -*-
"""v9 意图维度测量脚本（vLLM 加速，自包含）。

背景（v8.3 审计链，见 results/v8/diag_v83_audit/metrics.txt）：
  教师按"表面毒性轴"回答一切问题——TP>FP>TN，FN（意图攻击但表面弱）与
  FP（表面攻击形状但只是谈论）同处中带。v4 的意图类直问（SA/Q 通道）经
  增强审计证实不携带中带信息（增量≈0）：教师对意图主题的回答坍缩回表面轴。
  因此 v9 的关键不是换问题主题，而是换测量模式——强迫教师做表面阅读无法
  完成的加工：情景替换（把文本放进"当面对你说"的实施情景）、听者反应意图、
  严肃性判断。禁止 v6 协议 M 的出现性锚定首句（"变形出现同样算"会把教师
  重新锚回表面匹配）。

与 run_items_v6.py 的关系：测量机制逐字同源（vLLM 加载、verbalizer 候选展开、
首 token 受限解码、chat 渲染与后缀修正、数据加载、p_yes 提取），仅模板与
条目池不同：
  - 模板：语用指令版（考虑说话人真实意图与言语情境），无出现性锚定条款；
  - 条目：6 条意图条目（内置，见 V9_ITEMS），全部是/否题式；
  - 单阶段：每条目一次调用，p_yes ∈ [0,1] 直接落矩阵；
  - 测量层存原始 p_yes，极性由 train AUC 在条目门槛层裁决（同 v6/v3 约定）。

条目级预登记门槛（analyze_items_v9.py 裁决，读出实验前杀掉不合格条目）：
  G1 expr2-FN vs expr0-FP 单条目 AUC ≥ 0.70（对全题最难边界有效）；
  G2 与现有 232 维特征 max|corr| < 0.70（不得是表面毒性轴的镜像）；
  G3 train 全局毒性-无毒 AUC ≥ 0.55。
若全灭 → 结论升级为"GLM-4-9B 教师能力边界"，不再设计新测量。

输出（防覆盖，同 run_items_v6.py 结构）：
  data/processed/<数据集>/<model_name>_v9items/items_{model_name}_{split}.json
  data/processed/<数据集>/<model_name>_v9items/items_{model_name}_{split}_matrix.csv

使用示例：
  python scripts/run_items_v9.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/TOXICN --split train
  python scripts/run_items_v9.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/TOXICN --split test
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
# 模型加载与生成前缀修正（与 run_items_v6.py 逐字同源）
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
        enable_prefix_caching=True,     # 同文本的 6 条条目共享"模板头+文本"前缀，prefill 复用
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


# 生成前缀后缀手工指定表（按模型目录名，与 run_items_v6.py 同源）。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}


# =============================================================================
# 提示词模板（v9.0 语用指令版，新定稿）
# =============================================================================
# 与 v6 协议 M 的关键区别：无出现性锚定首句与"变形出现同样算"条款（该条款
# 把教师锚在表面匹配上，是 v8.3 审计证明的中带盲区成因之一）；改为显式的
# 语用加工指令（考虑说话人真实意图与言语情境）。
V9_TEMPLATE = """你是语用学标注助手。回答问题时请考虑说话人的真实意图与言语情境（对谁说、为什么说、期望对方怎么反应），不要只依据字面词语。只依据文本本身判断，不评价内容对错，不解释。

文本：{text}

问题：{question}

只回答“是/否”。回答："""


# =============================================================================
# v9 意图条目（内置 6 条，是/否题式；测量层存原始 p_yes，极性由门槛层裁决）
# =============================================================================
V9_ITEMS = [
    {"item_id": "V9-A", "tag": "情景替换冒犯",
     "question": "如果这段话是有人当面对你说出的，你被冒犯或被攻击的可能性大吗？"},
    {"item_id": "V9-B", "tag": "实施vs谈论",
     "question": "这段话中的攻击性词语或内容，是说话人在实际实施对他人的攻击，而不是在谈论、引用、转述或评论他人的言论吗？"},
    {"item_id": "V9-C", "tag": "目标指向",
     "question": "这段话针对的是某个具体的人或具体的群体，而不是某个现象、事件或抽象事物吗？"},
    {"item_id": "V9-D", "tag": "听者反应意图",
     "question": "说话人期望的听众反应，更接近让听众受伤、屈辱或愤怒，而不是认同其观点、觉得好笑或中立地了解信息吗？"},
    {"item_id": "V9-E", "tag": "严肃恶意判断",
     "question": "说话人是在严肃地表达贬损或恶意，而不是在开玩笑、反讽、玩梗或进行文艺创作吗？"},
    {"item_id": "V9-F", "tag": "反讽检测",
     "question": "说话人所说的字面内容与其真实态度相反，或故意以夸张反语表达嘲讽吗？"},
]


# =============================================================================
# Verbalizer 工具（run_items_v6.py 已验证机制，逐字并入）
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
# 数据加载（与 run_items_v6.py 同源）
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
    parser = argparse.ArgumentParser(description="v9 意图条目测量（vLLM）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="目录模式下的划分（单文件模式忽略）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（temperature=0 下主要作运行批次登记，进输出目录名防覆盖）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>_v9items/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    items = V9_ITEMS

    print(f"数据集 {dataset_tag}（{', '.join(files)}）样本 {len(dataset)} 条，"
          f"意图条目 {len(items)} 条，总问询 {len(dataset) * len(items)} 次"
          f"（约为 v6 全量测量的 1/{len(dataset) * 152 // (len(dataset) * len(items))}）")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    print(f"verbalizer: 是={yes_ids} 否={no_ids}")
    sp_yn = build_sampling_params(yes_ids + no_ids)

    # ---------- 批量推理：样本外层 × 条目内层（同样本的 6 问共享前缀缓存） ----------
    prompts, keys = [], []
    for sample in dataset:
        for it in items:
            prompts.append(V9_TEMPLATE.format(text=sample["content"], question=it["question"]))
            keys.append((sample["id"], it["item_id"]))
    print(f"意图条目推理（批量 {len(prompts)} 条）...")
    outputs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                           sp_yn, use_tqdm=True)

    # ---------- 提取 p_yes ----------
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
               else project_root / "data" / "processed" / dataset_tag / f"{model_name}_v9items")
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
                "prompt_version": "v9.0",
                "template_note": "语用指令版：无出现性锚定条款，显式要求考虑说话人意图与言语情境",
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "items": [{"item_id": it["item_id"], "tag": it["tag"],
                           "question": it["question"]} for it in items],
                "n_invalid": len(invalid),
            },
            "matrix": matrix,
            "invalid": invalid,
        }, f, ensure_ascii=False)  # 紧凑输出（train 体量大，不用 indent）

    # 宽表矩阵 CSV（条目门槛分析脚本直接读这张表）
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

    # ---------- 概览统计（方向未定，p_yes 原始均值仅参考） ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(dataset)}  条目: {len(items)}")

    def _label(s):
        try:
            return int(s["toxic"])
        except (TypeError, ValueError):
            return None

    for it in items:
        iid = it["item_id"]
        vals1, vals0 = [], []
        for s in dataset:
            v = matrix[s["id"]].get(iid)
            if v is None:
                continue
            if _label(s) == 1:
                vals1.append(v)
            elif _label(s) == 0:
                vals0.append(v)
        m1 = sum(vals1) / len(vals1) if vals1 else float("nan")
        m0 = sum(vals0) / len(vals0) if vals0 else float("nan")
        print(f"  {iid}（{it['tag']}）: p_yes均值 toxic1={m1:.4f} toxic0={m0:.4f} "
              f"差={m1 - m0:+.4f}（极性未定，仅概览）")
    off = sum(1 for _, out in zip(keys, outputs)
              if tokenizer.decode(list(out.outputs[0].token_ids)).strip() not in ("是", "否"))
    print(f"首token越界: {off}  无效判定: {len(invalid)}（受限解码后应为 0）")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
