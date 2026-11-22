#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGSCC Stage 1: locally measure semantic counter-evidence anchor candidates.

This script does NOT train a classifier or modify the v8 concept matrix.
It selects samples solely by the mean activation of the 16 v8 A/B/C/D concepts,
then measures N03/N04/N05 with the same local GLM/vLLM verbalizer path as v8.
Labels are excluded from selection and prompts; they are used only for post-hoc
summary tables. All model loading is local-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Enforce offline mode before importing the v8 helper / Transformers / vLLM.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

SCRIPT_VERSION = "agscc-stage1-anchor-screen-1.0.0"
EXPECTED_V8_HELPER_SHA256 = "a627177200516dc272f438cef994f7859aa69db2dd6bf5eb0fba2f57b45cae8e"
ABCD_COLUMNS = [
    "A01", "A02", "A03", "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "C01", "C02", "C03", "C04", "D01", "D02",
]
ALL_V8_CONCEPTS = ABCD_COLUMNS + ["E01", "E02", "E03", "E04", "E05", "T01", "T02", "T03", "T04", "T05"]
ANCHOR_IDS = ["N03", "N04", "N05"]
PROMPT_VERSION = "agscc-anchor-prompt-v1"

PROMPT_TEMPLATE = """你是严格的中文文本语境判定助手。请判断文本中是否出现指定的语境反证锚点。
只依据文本中可观察到的表达判断，不推测作者没有表达的意图，不补充外部事实。
判断当前文本中是否存在定义所要求的明确语义证据；不能仅因文本整体看起来友善、中立或没有明显攻击，就回答“是”。
如果文本同时包含独立的群体攻击/歧视主张，且该锚点不能明确限定到相关内容片段，则保守回答“否”。
引用或转述他人的攻击内容不自动构成反对攻击；仅在明确把攻击性词语作为语言对象讨论时，才可能符合元语言讨论锚点。

文本：{text}

锚点：{name}
判定问题：{question}
锚点定义：{definition}
边界说明：{boundary}

只回答“是”或“否”。回答："""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_v8_helper(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少 v8 推理辅助脚本：{path}")
    actual = sha256_file(path)
    if actual != EXPECTED_V8_HELPER_SHA256:
        raise RuntimeError(
            "run_items_v8.py 与已审查版本不一致，拒绝执行。\n"
            f"expected={EXPECTED_V8_HELPER_SHA256}\nactual={actual}"
        )
    spec = importlib.util.spec_from_file_location("agscc_run_items_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 v8 辅助脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidates(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到锚点候选词典：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ANCHOR_IDS):
        raise ValueError(f"候选词典应有 {len(ANCHOR_IDS)} 项，实际 {len(rows)} 项")
    ids = [r.get("条目id", "") for r in rows]
    if ids != ANCHOR_IDS:
        raise ValueError(f"候选项顺序/ID不符合预期：{ids}")
    required = ["条目id", "构念名", "条目问句", "概念定义", "边界说明", "候选纠错目标"]
    for r in rows:
        missing = [k for k in required if not r.get(k, "").strip()]
        if missing:
            raise ValueError(f"候选项 {r.get('条目id')} 缺少字段：{missing}")
    return rows


def load_and_select(data_json: Path, matrix_csv: Path, split: str,
                    activation_threshold: float, limit: int | None):
    if not data_json.is_file():
        raise FileNotFoundError(f"JSON 数据不存在：{data_json}")
    if not matrix_csv.is_file():
        raise FileNotFoundError(f"v8 概念矩阵不存在：{matrix_csv}")
    with data_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError("data_json 顶层必须是非空 JSON list")
    matrix = pd.read_csv(matrix_csv, encoding="utf-8-sig")
    if len(raw) != len(matrix):
        raise ValueError(f"JSON/矩阵样本数不一致：{len(raw)} vs {len(matrix)}")
    required = ["id", "toxic", "content"] + ALL_V8_CONCEPTS
    missing = [c for c in required if c not in matrix.columns]
    if missing:
        raise ValueError(f"v8 矩阵缺少列：{missing}")
    expected_ids = [f"TOXICN_{split}_{i:04d}" for i in range(len(raw))]
    if matrix["id"].astype(str).tolist() != expected_ids:
        raise ValueError("矩阵 ID 不符合 TOXICN_{split}_{index:04d} 顺序，拒绝错位拼接")

    labels = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise ValueError(f"JSON 第 {i} 条不是 object")
        if str(obj.get("content", "")) != str(matrix.iloc[i]["content"]):
            raise ValueError(f"第 {i} 条 JSON/矩阵 content 不一致：{expected_ids[i]}")
        y_json = int(obj.get("toxic", -999))
        y_matrix = int(matrix.iloc[i]["toxic"])
        if y_json != y_matrix or y_json not in (0, 1):
            raise ValueError(f"第 {i} 条 JSON/矩阵标签不一致或非法：{expected_ids[i]}")
        labels.append(y_json)

    x = matrix[ABCD_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("A/B/C/D 16维概念中存在 NaN/Inf")
    if ((x < -1e-8) | (x > 1 + 1e-8)).any():
        raise ValueError("A/B/C/D 概念值不在 [0,1]")
    activation = x.mean(axis=1)
    # IMPORTANT: sample selection depends only on concept activations, never on labels.
    selected_idx = np.flatnonzero(activation >= activation_threshold).tolist()
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0")
        selected_idx = selected_idx[:limit]
    if not selected_idx:
        raise ValueError("阈值筛选后没有样本；请检查 --activation_threshold")

    samples = []
    for i in selected_idx:
        obj = raw[i]
        row = matrix.iloc[i]
        samples.append({
            "id": str(row["id"]),
            "content": str(row["content"]),
            "toxic": int(row["toxic"]),
            "toxic_type": obj.get("toxic_type", ""),
            "expression": obj.get("expression", ""),
            "topic": obj.get("topic", ""),
            "platform": obj.get("platform", ""),
            "length": obj.get("length", len(str(row["content"]))),
            "activation_score": float(activation[i]),
            "concepts": {c: float(row[c]) for c in ALL_V8_CONCEPTS},
            "source_index": int(i),
        })
    return samples, matrix, raw, activation


def build_prompt(sample: dict[str, Any], item: dict[str, str]) -> str:
    return PROMPT_TEMPLATE.format(
        text=sample["content"],
        name=item["构念名"],
        question=item["条目问句"],
        definition=item["概念定义"],
        boundary=item["边界说明"],
    )


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def summarize(samples: list[dict[str, Any]], results: dict[str, dict[str, Any]],
              output_dir: Path, seed: int = 20261122) -> None:
    rows = []
    for sample in samples:
        rec = results[sample["id"]]
        for aid in ANCHOR_IDS:
            rows.append({
                "id": sample["id"], "anchor_id": aid,
                "p_yes": float(rec["p_yes"][aid]),
                "first_token": rec["first_token"][aid],
                "first_token_id": int(rec["first_token_id"][aid]),
                "cand_mass": float(rec["cand_mass"][aid]),
            })
    long_df = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "anchor_firsttoken_audit.csv", long_df)

    mat_rows = []
    for sample in samples:
        rec = results[sample["id"]]
        row = {
            "id": sample["id"], "split": sample.get("split", ""),
            "source_index": sample["source_index"], "toxic": sample["toxic"],
            "toxic_type": sample["toxic_type"], "expression": sample["expression"],
            "topic": sample["topic"], "platform": sample["platform"],
            "length": sample["length"], "abcd_mean_activation": sample["activation_score"],
            "content": sample["content"],
        }
        row.update(sample["concepts"])
        for aid in ANCHOR_IDS:
            row[f"p_{aid}"] = float(rec["p_yes"][aid])
            row[f"first_token_{aid}"] = rec["first_token"][aid]
            row[f"cand_mass_{aid}"] = float(rec["cand_mass"][aid])
        mat_rows.append(row)
    anchor_matrix = pd.DataFrame(mat_rows)
    atomic_write_csv(output_dir / "anchor_matrix.csv", anchor_matrix)

    # E03 is an existing v8 stance concept and serves as an internal reference,
    # not as a newly measured anchor. It is evaluated on the same selected samples.
    ref_rows = []
    for sample in samples:
        for aid in ANCHOR_IDS:
            ref_rows.append({"id": sample["id"], "anchor_id": aid,
                             "p_yes": float(results[sample["id"]]["p_yes"][aid])})
        ref_rows.append({"id": sample["id"], "anchor_id": "E03_baseline",
                         "p_yes": float(sample["concepts"]["E03"])})
    summary_long = pd.DataFrame(ref_rows)

    group_rows = []
    group_meta = pd.DataFrame(samples)[["id", "toxic", "toxic_type", "expression"]]
    for aid in ANCHOR_IDS + ["E03_baseline"]:
        sub = summary_long[summary_long.anchor_id == aid]
        joined = sub.merge(group_meta, on="id", how="left", validate="many_to_one")
        for field in ["toxic", "toxic_type", "expression"]:
            values = sorted(joined[field].dropna().astype(str).unique().tolist())
            for value in values:
                g = joined[joined[field].astype(str) == value]
                group_rows.append({
                    "anchor_id": aid, "group_field": field, "group_value": value,
                    "n": int(len(g)), "mean_p_yes": float(g.p_yes.mean()),
                    "median_p_yes": float(g.p_yes.median()),
                    "rate_ge_0.5": float((g.p_yes >= 0.5).mean()),
                    "rate_ge_0.8": float((g.p_yes >= 0.8).mean()),
                    "rate_ge_0.9": float((g.p_yes >= 0.9).mean()),
                })
    atomic_write_csv(output_dir / "anchor_summary_by_group.csv", pd.DataFrame(group_rows))

    screen_rows = []
    rng = np.random.default_rng(seed)
    # Diagnostic only: AUC asks whether an anchor distinguishes non-toxic from toxic
    # WITHIN the high-ABCD-activation subset. Labels are used only here, post-inference.
    meta = group_meta[["id", "toxic"]]
    for aid in ANCHOR_IDS + ["E03_baseline"]:
        sub = summary_long[summary_long.anchor_id == aid].merge(meta, on="id", validate="many_to_one")
        y_non_toxic = (sub.toxic.to_numpy(dtype=int) == 0).astype(int)
        p = sub.p_yes.to_numpy(dtype=float)
        neg = p[y_non_toxic == 1]
        pos = p[y_non_toxic == 0]
        auc = float(roc_auc_score(y_non_toxic, p)) if len(np.unique(y_non_toxic)) == 2 else float("nan")
        ap = float(average_precision_score(y_non_toxic, p)) if len(np.unique(y_non_toxic)) == 2 else float("nan")
        diffs = []
        if len(neg) and len(pos):
            for _ in range(1000):
                bneg = rng.choice(neg, size=len(neg), replace=True)
                bpos = rng.choice(pos, size=len(pos), replace=True)
                diffs.append(float(bneg.mean() - bpos.mean()))
            lo, hi = np.quantile(diffs, [0.025, 0.975]).tolist()
        else:
            lo, hi = float("nan"), float("nan")
        screen_rows.append({
            "anchor_id": aid, "n_non_toxic": int(len(neg)), "n_toxic": int(len(pos)),
            "reference_type": "existing_v8_E03" if aid == "E03_baseline" else "new_candidate",
            "mean_p_yes_non_toxic": float(neg.mean()) if len(neg) else float("nan"),
            "mean_p_yes_toxic": float(pos.mean()) if len(pos) else float("nan"),
            "mean_difference_non_toxic_minus_toxic": float(neg.mean() - pos.mean()) if len(neg) and len(pos) else float("nan"),
            "difference_bootstrap_95ci_low": float(lo), "difference_bootstrap_95ci_high": float(hi),
            "auc_non_toxic_higher_is_better": auc, "ap_non_toxic_higher_is_better": ap,
            "interpretation": "exploratory screening within high-ABCD-activation subset; not a test-set score or proof of causality",
        })
    atomic_write_csv(output_dir / "anchor_screen_metrics.csv", pd.DataFrame(screen_rows))


def self_test(args) -> None:
    helper_path = args.v8_runner or Path(__file__).with_name("run_items_v8.py")
    helper = load_v8_helper(helper_path)
    candidates_path = args.candidates_csv or Path(__file__).with_name("anchor_candidates_v1.csv")
    candidates = load_candidates(candidates_path)
    assert len(candidates) == 3
    sample = {"content": "不能把个别人的行为归咎于整个群体。"}
    prompt = build_prompt(sample, candidates[0])
    assert "当前发言者" in prompt and "只能回答" not in prompt and "只回答“是”或“否”" in prompt
    required_helper_functions = ["load_vllm_model", "get_candidate_ids", "build_sampling_params", "render_prompts", "get_probability_result", "read_checkpoint"]
    assert all(hasattr(helper, name) for name in required_helper_functions)
    print("SELF_TEST_OK")
    print(f"v8_helper_sha256={sha256_file(helper_path)}")
    print(f"candidate_count={len(candidates)}; candidate_ids={','.join(ANCHOR_IDS)}")
    print(f"prompt_sha256={sha256_text(PROMPT_TEMPLATE)}")


def parse_args():
    p = argparse.ArgumentParser(description="AGSCC Stage 1: local GLM anchor candidate screening")
    p.add_argument("--model_path", type=Path, default=None, help="本地 GLM-4-9B-Chat 模型目录；禁止远程模型名")
    p.add_argument("--data_json", type=Path, default=None, help="与 v8 矩阵严格对应的 train.json 或 test.json")
    p.add_argument("--matrix_csv", type=Path, default=None, help="对应的 v8 train/test matrix CSV")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--output_dir", type=Path, default=Path("AGSCC_stage1_outputs"))
    p.add_argument("--candidates_csv", type=Path, default=None)
    p.add_argument("--v8_runner", type=Path, default=None)
    p.add_argument("--activation_threshold", type=float, default=0.28)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.78)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=16)
    p.add_argument("--max_num_batched_tokens", type=int, default=8192)
    p.add_argument("--batch_samples", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="仅取筛选后前 N 条样本；用于技术冒烟测试，不能与全量结果混用")
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--overwrite", action="store_true", help="删除旧 checkpoint，从头测量")
    p.add_argument("--self_test", action="store_true", help="不加载模型的代码/词典/提示词检查")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test(args)
        return
    if args.model_path is None or args.data_json is None or args.matrix_csv is None:
        raise ValueError("正式运行必须提供 --model_path、--data_json 和 --matrix_csv")
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"本地模型目录不存在或缺少 config.json：{args.model_path}")
    if not (0.05 <= args.gpu_memory_utilization <= 0.98):
        raise ValueError("gpu_memory_utilization 必须在 0.05 到 0.98 之间")
    if args.batch_samples <= 0 or args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("batch/序列/token 参数必须为正整数")
    if not (0.0 <= args.activation_threshold <= 1.0):
        raise ValueError("activation_threshold 必须在 [0,1]")

    helper_path = args.v8_runner or Path(__file__).with_name("run_items_v8.py")
    helper = load_v8_helper(helper_path)
    candidate_path = args.candidates_csv or Path(__file__).with_name("anchor_candidates_v1.csv")
    candidates = load_candidates(candidate_path)
    samples, matrix, raw, activation = load_and_select(
        args.data_json.resolve(), args.matrix_csv.resolve(), args.split,
        args.activation_threshold, args.limit,
    )
    for s in samples:
        s["split"] = args.split

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "anchor_checkpoint.jsonl"
    candidate_ids = [x["条目id"] for x in candidates]
    helper_sha = sha256_file(helper_path)
    candidate_sha = sha256_file(candidate_path)
    config = {
        "script_version": SCRIPT_VERSION,
        "split": args.split,
        "model_path": str(args.model_path.expanduser().resolve()),
        "data_json_sha256": sha256_file(args.data_json.resolve()),
        "matrix_csv_sha256": sha256_file(args.matrix_csv.resolve()),
        "v8_runner_sha256": helper_sha,
        "candidate_csv_sha256": candidate_sha,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(PROMPT_TEMPLATE),
        "anchor_ids": candidate_ids,
        "activation_columns": ABCD_COLUMNS,
        "activation_threshold": args.activation_threshold,
        "selected_samples": len(samples),
        "selected_ids_sha256": sha256_text("\n".join(s["id"] for s in samples)),
        "limit": args.limit,
        "batch_samples": args.batch_samples,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.seed,
        "labels_used_for_selection": False,
        "labels_used_in_prompts": False,
        "labels_used_for_model_training": False,
        "labels_used_for_posthoc_summary_only": True,
        "test_split_is_exploratory_only": args.split == "test",
    }
    existing = helper.read_checkpoint(
        checkpoint, config, {s["id"] for s in samples}, candidate_ids, args.overwrite
    )
    pending = [s for s in samples if s["id"] not in existing]
    decode_mode = "reused_existing_checkpoint"

    if pending:
        tokenizer, llm = helper.load_vllm_model(
            args.model_path.expanduser().resolve(), args.gpu_memory_utilization,
            args.max_model_len, args.max_num_seqs, args.max_num_batched_tokens,
        )
        yes_ids = helper.get_candidate_ids(["是"], tokenizer)
        no_ids = helper.get_candidate_ids(["否"], tokenizer)
        if set(yes_ids) & set(no_ids):
            raise RuntimeError("‘是’与‘否’ verbalizer token ID 冲突")
        allowed_ids = helper.get_candidate_ids(["是", "否"], tokenizer)
        sampling_params, decode_mode = helper.build_sampling_params(allowed_ids)
        chat_kwargs = helper.detect_chat_kwargs(tokenizer)
        model_name = args.model_path.name
        for start in tqdm(range(0, len(pending), args.batch_samples), desc="GLM anchor measurement", dynamic_ncols=True):
            batch = pending[start:start + args.batch_samples]
            prompts = []
            ordering = []
            for s in batch:
                for item in candidates:
                    prompts.append(build_prompt(s, item))
                    ordering.append((s["id"], item["条目id"]))
            rendered = helper.render_prompts(tokenizer, prompts, chat_kwargs, model_name)
            outputs = llm.generate(rendered, sampling_params)
            if len(outputs) != len(ordering):
                raise RuntimeError(f"vLLM输出数量不匹配：{len(outputs)} vs {len(ordering)}")
            batch_result: dict[str, dict[str, dict[str, Any]]] = {
                s["id"]: {"p_yes": {}, "first_token": {}, "cand_mass": {}, "first_token_id": {}}
                for s in batch
            }
            for (sid, aid), out in zip(ordering, outputs):
                parsed = helper.get_probability_result(out, tokenizer, yes_ids, no_ids)
                if parsed["first_token"] not in ("是", "否") or parsed["p_yes"] is None:
                    raise RuntimeError(f"{sid}/{aid} 首 token 非是/否或概率无效：{parsed}")
                batch_result[sid]["p_yes"][aid] = float(parsed["p_yes"])
                batch_result[sid]["first_token"][aid] = parsed["first_token"]
                batch_result[sid]["cand_mass"][aid] = float(parsed["cand_mass"])
                batch_result[sid]["first_token_id"][aid] = int(parsed["first_token_id"])
            for sid, res in batch_result.items():
                if any(set(res[k]) != set(candidate_ids) for k in ["p_yes", "first_token", "cand_mass", "first_token_id"]):
                    raise RuntimeError(f"样本 {sid} 的锚点输出不完整")
                append_jsonl(checkpoint, {"record_type": "sample", "id": sid, **res})
                existing[sid] = {"record_type": "sample", "id": sid, **res}
        del llm
        del tokenizer

    # Revalidate result completeness before producing downstream files.
    if set(existing) != {s["id"] for s in samples}:
        missing = sorted({s["id"] for s in samples} - set(existing))[:10]
        raise RuntimeError(f"checkpoint仍缺少样本结果：{missing}")
    results = {sid: existing[sid] for sid in existing}
    summarize(samples, results, output_dir, args.seed)

    meta = {
        **config,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decode_mode": decode_mode,
        "n_input_samples": len(raw),
        "n_selected_samples": len(samples),
        "n_selected_non_toxic": int(sum(s["toxic"] == 0 for s in samples)),
        "n_selected_toxic": int(sum(s["toxic"] == 1 for s in samples)),
        "n_measurements_expected": len(samples) * len(candidates),
        "n_measurements_observed": sum(len(existing[s["id"]]["p_yes"]) for s in samples),
        "local_model_inference": True,
        "remote_download_allowed": False,
        "notes": [
            "This is anchor candidate screening only; no classifier is trained and no concept is corrected here.",
            "Original v8 concept probabilities are retained unchanged in anchor_matrix.csv.",
            "toxic/toxic_type/expression labels are post-hoc grouping variables only.",
            "Do not interpret an anchor probability as a probability that the whole sample is non-toxic.",
        ],
    }
    tmp = output_dir / "run_meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_dir / "run_meta.json")
    print("完成：", output_dir)
    print(f"samples={len(samples)}; anchors={len(candidates)}; measurements={len(samples)*len(candidates)}")
    print("下一步请先核对 run_meta.json，再打包整个输出目录。")


if __name__ == "__main__":
    main()
