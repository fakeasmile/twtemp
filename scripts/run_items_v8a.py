#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8a 概念测量脚本：使用 GLM-4-9B-Chat + vLLM 生成概念概率向量。

基于 scripts/run_items_v7.py 的模型加载、chat template、是/否 verbalizer
和受限首 token 解码实现。支持三种 prompt 条件，便于完成 P0 配对消融：
  v7_global       v7 原始全局“变形出现同样算”规则（复现基线）
  p1_no_global    删除全局变形规则
  p2_local_boundary 删除全局规则，强调每条概念自身的定义/排除边界（默认）

测量输出：
  items_<tag>_<model>_<split>_<prompt>.json
  items_<tag>_<model>_<split>_<prompt>_matrix.csv
  items_<tag>_<model>_<split>_<prompt>_firsttoken.csv

说明：
- 矩阵存原始 p_yes；反向条目不在测量层做 1-p，沿用 v7 协议。
- p_yes 是“是/否”候选 token 的概率质量归一化值。
- cand_mass 保留候选在完整首 token 分布中的原始质量，供审计。
- 分批生成，避免 train × 180 的所有 prompt 一次性驻留内存。
- 默认只读本地模型，不访问 Hugging Face 下载。
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

PROMPT_TEMPLATES = {
    "v7_global": (
        "你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。"
        "问句指出的现象以讽刺、委婉、谐音、缩写等变形出现同样算。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答“是/否”。回答："
    ),
    "p1_no_global": (
        "你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答“是/否”。回答："
    ),
    "p2_local_boundary": (
        "你是语言学标注助手。请判断文本是否符合问句所描述的具体现象。"
        "只依据文本及其语境线索判断，不评价内容对错，不解释。"
        "只有当文本满足该问句的核心定义时才回答“是”；仅出现相关话题、身份标签、情绪词、相似词语或相关表达形式，不足以回答“是”。"
        "若问句列出排除条件，必须遵守；不能把其他概念的现象自动算作本问句现象。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答“是/否”。回答："
    ),
}

MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}


def load_vllm_model(model_path: Path, gpu_memory_utilization: float,
                    max_model_len: int, max_num_seqs: int,
                    max_num_batched_tokens: int):
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {model_path}")
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=True,
        max_logprobs=50,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def load_items(csv_path: Path):
    required = ["条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句"]
    if not csv_path.exists():
        raise FileNotFoundError(f"概念 CSV 不存在：{csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(x or "").lstrip("\ufeff\x00").strip() for x in reader.fieldnames]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"概念 CSV 缺少列：{missing}")
        items = []
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
        raise ValueError("概念 CSV 中没有条目")
    ids = [x["item_id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("概念 CSV 中存在重复条目 id")
    bad = sorted(set(x["direction"] for x in items) - {"正向", "反向"})
    if bad:
        raise ValueError(f"方向列存在非法值：{bad}")
    return items


def load_dataset(data_path: Path, split: str):
    if not data_path.exists():
        raise FileNotFoundError(f"数据路径不存在：{data_path}")
    if data_path.is_file():
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        source = data_path.stem
        samples = [_normalize_sample(x, source, source, i) for i, x in enumerate(raw)]
        return samples, [data_path.name], data_path.parent.name or source
    json_path = data_path / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"数据目录中缺少 {split}.json：{json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = [_normalize_sample(x, data_path.name, split, i) for i, x in enumerate(raw)]
    return samples, [json_path.name], data_path.name


def _normalize_sample(item, source_name, split, idx):
    if "content" not in item:
        raise KeyError(f"样本索引 {idx} 缺少 content 字段")
    return {
        "id": str(item.get("id") or f"{source_name}_{split}_{idx:04d}"),
        "source": item.get("source") or source_name,
        "split": split,
        "content": str(item["content"]),
        "toxic": item.get("toxic"),
    }


def get_candidate_ids(tokenizer, word):
    ids = []
    encoded = tokenizer.encode(word, add_special_tokens=False)
    if len(encoded) == 1:
        ids.append(encoded[0])
    if not ids:
        raise RuntimeError(f"verbalizer“{word}”无法编码为单 token；停止以避免错误测量")
    return list(dict.fromkeys(ids))


def candidate_mass(first_token_logprobs, id_groups):
    masses = [
        sum(math.exp(first_token_logprobs[t].logprob)
            for t in group if t in first_token_logprobs)
        for group in id_groups
    ]
    raw_total = sum(masses)
    norm_total = raw_total + 1e-12
    return masses, [m / norm_total for m in masses]


def make_mask_processor(allowed_ids):
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
    allowed = sorted(set(allowed_ids))
    candidates = [
        ("logits_processors", {"logits_processors": [make_mask_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束", {}),
    ]
    for name, extra in candidates:
        try:
            params = SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra)
            print(f"首 token 解码方式：{name}；候选 token ids={allowed}")
            return params, name
        except TypeError as exc:
            print(f"当前 vLLM 不支持 {name}（{exc}），尝试下一种方式")
    raise RuntimeError("当前 vLLM 版本无法建立首 token 解码参数")


def render_prompts(tokenizer, texts, chat_kwargs, model_name):
    suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    prompts = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if suffix is not None:
            prompt += suffix
        elif not prompt.endswith("\n"):
            prompt += "\n"
        prompts.append(prompt)
    return prompts


def parse_args():
    p = argparse.ArgumentParser(description="v8a 概念测量（GLM/vLLM，首 token 审计）")
    p.add_argument("--model_path", required=True, help="本地 GLM-4-9B-Chat 权重目录")
    p.add_argument("--data_path", required=True, help="TOXICN 数据目录，或单个 JSON 文件")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--items_csv", default=str(PROJECT_ROOT / "data" / "concepts" / "cbm_concept_set_v8a_P0_180D.csv"))
    p.add_argument("--run_tag", default="v8a", help="输出版本标签；v7 对照运行时可设为 v7")
    p.add_argument("--prompt_variant", choices=list(PROMPT_TEMPLATES), default="p2_local_boundary")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=32)
    p.add_argument("--max_num_batched_tokens", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=256, help="每次送入 vLLM 的文本-概念问询数")
    p.add_argument("--limit", type=int, default=None, help="试点样本数；全量运行不传")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if not (0 < args.gpu_memory_utilization < 1):
        raise ValueError("gpu_memory_utilization 必须在 0 和 1 之间")

    model_path = Path(args.model_path)
    model_name = model_path.name
    data_path = Path(args.data_path)
    items_path = Path(args.items_csv)
    dataset, files, dataset_tag = load_dataset(data_path, args.split)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit 必须为正整数")
        # 固定随机抽样而非只取文件开头；相同 seed 可在不同 prompt 条件间复用同一批样本。
        k = min(args.limit, len(dataset))
        chosen = sorted(random.Random(args.seed).sample(range(len(dataset)), k))
        dataset = [dataset[i] for i in chosen]
        print(f"--limit：按 seed={args.seed} 固定随机抽取 {len(dataset)} 条样本")
    items = load_items(items_path)

    ids = [s["id"] for s in dataset]
    if len(ids) != len(set(ids)):
        raise ValueError("数据集中存在重复 sample id；请先保证样本 id 唯一")

    run_name = f"{args.run_tag}_{model_name}_{args.split}_{args.prompt_variant}"
    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data" / "processed" / dataset_tag / run_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"items_{run_name}.json"
    matrix_path = out_dir / f"items_{run_name}_matrix.csv"
    firsttoken_path = out_dir / f"items_{run_name}_firsttoken.csv"

    print(f"数据集={dataset_tag}/{args.split} 样本={len(dataset)} 概念={len(items)} 问询={len(dataset)*len(items)}")
    print(f"Prompt={args.prompt_variant}；输出目录={out_dir}")
    if json_path.exists() or matrix_path.exists() or firsttoken_path.exists():
        raise FileExistsError(f"输出文件已存在，为防止覆盖旧结果而停止：{out_dir}（请换 output_dir 或移走旧结果）")

    tokenizer, llm = load_vllm_model(
        model_path, args.gpu_memory_utilization, args.max_model_len,
        args.max_num_seqs, args.max_num_batched_tokens,
    )
    chat_kwargs = detect_chat_kwargs(tokenizer)
    yes_ids = get_candidate_ids(tokenizer, "是")
    no_ids = get_candidate_ids(tokenizer, "否")
    sampling_params, restrict_mode = build_sampling_params(yes_ids + no_ids)
    print(f"verbalizer: 是={yes_ids} 否={no_ids}")

    matrix = {s["id"]: {} for s in dataset}
    invalid = []
    total = len(dataset) * len(items)
    in_range = 0
    mass_sum = 0.0
    mass_ge_999 = 0
    done = 0

    # first-token 文件先创建；逐批落盘，长任务中断时至少保留已完成批次的审计记录。
    with open(firsttoken_path, "w", encoding="utf-8-sig", newline="") as ft:
        writer = csv.writer(ft)
        writer.writerow(["sample_id", "item_id", "first_token", "p_yes", "cand_mass"])

        # 按样本-概念顺序分批，保证输出顺序稳定。
        pending_prompts, pending_keys = [], []
        for sample in dataset:
            for item in items:
                prompt_text = PROMPT_TEMPLATES[args.prompt_variant].format(
                    text=sample["content"], question=item["question"]
                )
                pending_prompts.append(prompt_text)
                pending_keys.append((sample["id"], item["item_id"]))
                if len(pending_prompts) >= args.batch_size:
                    done, in_range, mass_sum, mass_ge_999 = run_batch(
                        llm, tokenizer, chat_kwargs, model_name, sampling_params,
                        yes_ids, no_ids, pending_prompts, pending_keys, matrix,
                        invalid, writer, done, in_range, mass_sum, mass_ge_999,
                    )
                    pending_prompts, pending_keys = [], []
        if pending_prompts:
            done, in_range, mass_sum, mass_ge_999 = run_batch(
                llm, tokenizer, chat_kwargs, model_name, sampling_params,
                yes_ids, no_ids, pending_prompts, pending_keys, matrix,
                invalid, writer, done, in_range, mass_sum, mass_ge_999,
            )
        ft.flush()

    mass_mean = mass_sum / total if total else float("nan")
    stats = {
        "total": total,
        "in_range": in_range,
        "in_range_pct": round(in_range / total, 6) if total else None,
        "cand_mass_mean": round(mass_mean, 6) if total else None,
        "cand_mass_ge_999_pct": round(mass_ge_999 / total, 6) if total else None,
    }

    with open(matrix_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        item_ids = [x["item_id"] for x in items]
        writer.writerow(["id", "toxic", "content", *item_ids])
        for sample in dataset:
            vals = matrix[sample["id"]]
            writer.writerow([
                sample["id"], sample["toxic"], sample["content"],
                *["" if vals.get(iid) is None else vals.get(iid) for iid in item_ids]
            ])

    sha256 = hashlib.sha256(items_path.read_bytes()).hexdigest()
    meta = {
        "run_tag": args.run_tag,
        "model": model_name,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "files": files,
        "split": args.split,
        "seed": args.seed,
        "n_samples": len(dataset),
        "n_items": len(items),
        "prompt_variant": args.prompt_variant,
        "prompt_template": PROMPT_TEMPLATES[args.prompt_variant],
        "items_csv": str(items_path),
        "items_csv_sha256": sha256,
        "restricted_decoding": restrict_mode,
        "verbalizer_ids": {"是": yes_ids, "否": no_ids},
        "first_token_stats": stats,
        "n_invalid": len(invalid),
        "batch_size": args.batch_size,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "items": [{"item_id": x["item_id"], "construct_id": x["construct_id"],
                   "form": x["form"], "direction": x["direction"]} for x in items],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "matrix": matrix, "invalid": invalid}, f, ensure_ascii=False)

    print("\n" + "=" * 68)
    print(f"完成：{run_name}")
    print(f"问询数={done}/{total}；首 token 是/否={in_range}/{total} ({stats['in_range_pct']})")
    print(f"cand_mass 均值={stats['cand_mass_mean']}；mass>=0.999 比例={stats['cand_mass_ge_999_pct']}")
    print(f"无效判定={len(invalid)}；受限解码={restrict_mode}")
    print(f"JSON: {json_path}\n矩阵: {matrix_path}\n首 token: {firsttoken_path}")


def run_batch(llm, tokenizer, chat_kwargs, model_name, sampling_params,
              yes_ids, no_ids, prompt_texts, keys, matrix, invalid, writer,
              done, in_range, mass_sum, mass_ge_999):
    prompts = render_prompts(tokenizer, prompt_texts, chat_kwargs, model_name)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(keys):
        raise RuntimeError(f"vLLM 返回数量不匹配：{len(outputs)} != {len(keys)}")

    for (sample_id, item_id), output in zip(keys, outputs):
        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError(f"缺少首 token logprobs：sample={sample_id}, item={item_id}")
        logprob_map = output.outputs[0].logprobs[0]
        masses, normalized = candidate_mass(logprob_map, [yes_ids, no_ids])
        cand_mass = sum(masses)
        valid = cand_mass > 0
        p_yes = round(normalized[0], 6) if valid else None
        token_ids = list(output.outputs[0].token_ids)
        first_token = tokenizer.decode(token_ids) if token_ids else ""
        matrix[sample_id][item_id] = p_yes
        writer.writerow([sample_id, item_id, first_token,
                         "" if p_yes is None else p_yes, round(cand_mass, 6)])

        if first_token.strip() in ("是", "否"):
            in_range += 1
        mass_sum += cand_mass
        if cand_mass >= 0.999:
            mass_ge_999 += 1
        if not valid:
            invalid.append({"id": sample_id, "item": item_id, "gen_token": first_token})
        done += 1
    return done, in_range, mass_sum, mass_ge_999


if __name__ == "__main__":
    main()
