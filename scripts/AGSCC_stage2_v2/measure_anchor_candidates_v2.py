#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGSCC Stage 2: one-shot screening of N06-N08 using local GLM-4-9B-Chat.

This script measures anchors only. It does not train a classifier, alter v8
concept probabilities, or use labels in prompts/sample selection.
"""
from __future__ import annotations

import os
# Enforce offline mode before importing Transformers/vLLM.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

VERSION = "agscc-stage2-bounded-anchor-screen-2.0.0"
PROMPT_VERSION = "agscc-anchor-prompt-v2"
PROMPT_TEMPLATE = """你是严格的中文文本现象判定助手。请根据文本判断指定概念是否出现。
只依据文本中可观察到的内容判断，不推测作者未表达的意图，不补充外部事实。
对内容现象概念，文本中被引用或转述的表达也按其实际内容判断；对发言者立场概念，
只判断能够归属于当前发言者的立场，不能因为文本没有反对就推定其支持。

文本：{text}

概念：{concept_name}
判定问题：{question}
概念定义：{definition}
边界说明：{boundary}

只回答“是”或“否”。回答："""

ANCHOR_IDS = ["N06", "N07", "N08"]
ABCD_COLS = [
    "A01", "A02", "A03", "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04", "D01", "D02",
]
V8_COLS = [
    "A01", "A02", "A03", "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04", "D01", "D02", "E01", "E02", "E03", "E04",
    "E05", "T01", "T02", "T03", "T04", "T05",
]
EXPECTED = {
    "data_json_sha256": "1616ce7edef507f972dbd6038f5e0209afde4c391369e7e3fe582e883d35353f",
    "matrix_csv_sha256": "7cf43c4409bb3c67c04e48d4963d82f45e64f02ce818763ae624de0dead53551",
    "v8_runner_sha256": "a627177200516dc272f438cef994f7859aa69db2dd6bf5eb0fba2f57b45cae8e",
    "v8_dictionary_sha256": "18b61749eeaf78285e8e4af20680521535651b8b35e734ac739967de8f741743",
    "stage1_reference_sha256": "a732f2578a1b867bf8411686ed7c3e87f0f757f8d16aed3f4ef6066b92bd5d9a",
    "candidate_csv_sha256": "d465bbfb1c27c95acc856461aa9e695d1eeff05769babbba4b4f9b340d4818e7",
}
ACTIVATION_THRESHOLD = 0.28
EXPECTED_N_ALL = 9600
EXPECTED_N_SELECTED = 5356
BOOTSTRAP_REPS = 1000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_helper(helper_path: Path):
    expected = EXPECTED["v8_runner_sha256"]
    actual = sha256_file(helper_path)
    if actual != expected:
        raise ValueError(f"v8 helper SHA256 不匹配：expected={expected}, actual={actual}")
    spec = importlib.util.spec_from_file_location("agscc_stage2_v8_helper", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v8 helper：{helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidates(path: Path) -> List[dict]:
    required = ["条目id", "构念名", "条目问句", "概念定义", "边界说明", "候选纠错目标", "设计定位"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"候选词典缺少列：{missing}")
    if frame["条目id"].tolist() != ANCHOR_IDS:
        raise ValueError(f"候选条目必须且只能按顺序为 {ANCHOR_IDS}；实际为 {frame['条目id'].tolist()}")
    if frame[required].apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("候选词典存在空字段")
    return frame.to_dict("records")


def load_and_validate_inputs(args, package_dir: Path):
    data_path = args.data_json.expanduser().resolve()
    matrix_path = args.matrix_csv.expanduser().resolve()
    ref_path = (args.previous_anchor_matrix or (package_dir / "stage1_reference_anchor_matrix.csv")).expanduser().resolve()
    candidate_path = package_dir / "anchor_candidates_v2.csv"
    helper_path = package_dir / "run_items_v8.py"
    dictionary_path = package_dir / "ToxiCN_compact_concept_dictionary_v8_revised.csv"

    for p in (data_path, matrix_path, ref_path, candidate_path, helper_path, dictionary_path):
        if not p.is_file():
            raise FileNotFoundError(f"必需文件不存在：{p}")

    hashes = {
        "data_json_sha256": sha256_file(data_path),
        "matrix_csv_sha256": sha256_file(matrix_path),
        "v8_runner_sha256": sha256_file(helper_path),
        "v8_dictionary_sha256": sha256_file(dictionary_path),
        "stage1_reference_sha256": sha256_file(ref_path),
        "candidate_csv_sha256": sha256_file(candidate_path),
    }
    for key, expected in EXPECTED.items():
        if hashes[key] != expected:
            raise ValueError(
                f"输入文件哈希不匹配（{key}）。\n期望：{expected}\n实际：{hashes[key]}\n"
                "请确认使用的是本实验对应的 ToxiCN/v8/Stage1 文件，不要混用其他版本。"
            )

    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) != EXPECTED_N_ALL:
        raise ValueError(f"训练 JSON 必须为 {EXPECTED_N_ALL} 条 list，实际类型/长度：{type(raw)}/{len(raw) if isinstance(raw, list) else 'NA'}")
    for i, sample in enumerate(raw):
        if not isinstance(sample, dict) or not isinstance(sample.get("content"), str) or not sample["content"].strip():
            raise ValueError(f"训练 JSON 第 {i} 条缺少有效 content")

    matrix = pd.read_csv(matrix_path, encoding="utf-8-sig")
    needed = ["id", "toxic", "content", *V8_COLS]
    missing = [c for c in needed if c not in matrix.columns]
    if missing:
        raise ValueError(f"v8 矩阵缺少列：{missing}")
    if len(matrix) != EXPECTED_N_ALL:
        raise ValueError(f"v8 矩阵应为 {EXPECTED_N_ALL} 行，实际 {len(matrix)}")
    expected_ids = [f"TOXICN_train_{i:04d}" for i in range(EXPECTED_N_ALL)]
    if matrix["id"].astype(str).tolist() != expected_ids:
        raise ValueError("v8 矩阵 id 顺序/格式与本轮预期不一致")
    for col in V8_COLS:
        matrix[col] = pd.to_numeric(matrix[col], errors="raise")
        if not np.isfinite(matrix[col].to_numpy(dtype=float)).all():
            raise ValueError(f"v8 概念列 {col} 含 NaN/Inf")
    matrix["toxic"] = pd.to_numeric(matrix["toxic"], errors="raise").astype(int)
    if not set(matrix["toxic"].unique()).issubset({0, 1}):
        raise ValueError("toxic 标签不是二分类 0/1")

    # Confirm the raw JSON and v8 matrix are row-aligned before any model work.
    for i, sample in enumerate(raw):
        if sample["content"] != str(matrix.at[i, "content"]):
            raise ValueError(f"JSON 与 v8 矩阵第 {i} 行 content 不一致：index={i}")
        if int(sample.get("toxic")) != int(matrix.at[i, "toxic"]):
            raise ValueError(f"JSON 与 v8 矩阵第 {i} 行 toxic 标签不一致：index={i}")

    # Selection reads only v8 A/B/C/D probabilities, never labels.
    matrix["abcd_mean_activation"] = matrix[ABCD_COLS].mean(axis=1)
    selected = matrix.loc[matrix["abcd_mean_activation"] >= ACTIVATION_THRESHOLD].copy()
    if len(selected) != EXPECTED_N_SELECTED:
        raise ValueError(
            f"固定阈值 {ACTIVATION_THRESHOLD} 应筛出 {EXPECTED_N_SELECTED} 条，实际 {len(selected)}。"
            "拒绝自动改阈值，以免结果不可比。"
        )
    selected["source_index"] = selected.index.astype(int)
    selected = selected.reset_index(drop=True)

    # Validate prior Stage-1 result and retrieve prior N05/reference values.
    reference = pd.read_csv(ref_path, encoding="utf-8-sig")
    ref_required = ["id", "source_index", "content", "p_N05", *V8_COLS]
    ref_missing = [c for c in ref_required if c not in reference.columns]
    if ref_missing:
        raise ValueError(f"Stage1 参照矩阵缺少列：{ref_missing}")
    if len(reference) != EXPECTED_N_SELECTED:
        raise ValueError(f"Stage1 参照矩阵行数异常：{len(reference)}")
    if reference["id"].astype(str).tolist() != selected["id"].astype(str).tolist():
        raise ValueError("Stage1 参照矩阵样本 ID/顺序与当前固定筛选结果不一致")
    if reference["source_index"].astype(int).tolist() != selected["source_index"].astype(int).tolist():
        raise ValueError("Stage1 参照矩阵 source_index 与当前筛选结果不一致")
    if reference["content"].astype(str).tolist() != selected["content"].astype(str).tolist():
        raise ValueError("Stage1 参照矩阵文本与当前源数据不一致")
    for col in V8_COLS:
        a = reference[col].to_numpy(dtype=float)
        b = selected[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-6):
            raise ValueError(f"Stage1 参照矩阵的原始 v8 概念列 {col} 与源矩阵不一致")
    reference["p_N05"] = pd.to_numeric(reference["p_N05"], errors="raise")
    if not np.isfinite(reference["p_N05"].to_numpy(dtype=float)).all():
        raise ValueError("Stage1 N05 参照概率含 NaN/Inf")
    selected["p_N05_reference"] = reference["p_N05"].to_numpy(dtype=float)
    selected["p_E03_reference"] = selected["E03"].to_numpy(dtype=float)

    # Attach labels/metadata only after sample selection; never pass them to the LLM.
    selected_records = []
    for _, row in selected.iterrows():
        idx = int(row["source_index"])
        src = raw[idx]
        selected_records.append({
            "id": str(row["id"]),
            "source_index": idx,
            "content": str(row["content"]),
            "toxic": int(row["toxic"]),
            "toxic_type": src.get("toxic_type"),
            "expression": src.get("expression"),
            "topic": src.get("topic"),
            "platform": src.get("platform"),
            "length": src.get("length"),
            "target": src.get("target"),
            "abcd_mean_activation": float(row["abcd_mean_activation"]),
            "p_N05_reference": float(row["p_N05_reference"]),
            "p_E03_reference": float(row["p_E03_reference"]),
            "v8": {c: float(row[c]) for c in V8_COLS},
        })

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须是正整数")
        selected_records = selected_records[:args.limit]
    helper = load_helper(helper_path)
    candidates = load_candidates(candidate_path)
    return data_path, matrix_path, ref_path, hashes, selected_records, candidates, helper


def model_artifact_hashes(model_path: Path) -> dict:
    """Hash tokenizer/config artifacts without hashing multi-GB weight shards."""
    if not model_path.is_dir():
        return {}
    candidates = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "special_tokens_map.json", "added_tokens.json",
        "tokenizer.model", "vocab.json", "merges.txt",
    ]
    out = {}
    for name in candidates:
        path = model_path / name
        if path.is_file():
            out[name] = sha256_file(path)
    return out


def build_run_config(args, model_path: Path, model_config_path: Path, hashes: dict,
                     selected_records: list, script_path: Path) -> dict:
    return {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(PROMPT_TEMPLATE),
        "script_sha256": sha256_file(script_path),
        "model_path": str(model_path),
        "model_config_sha256": (sha256_file(model_config_path) if model_path.is_dir() else None),
        "model_tokenizer_artifact_hashes": model_artifact_hashes(model_path),
        "model_weight_shards_hashed": False,
        "input_hashes": hashes,
        "split": "train",
        "selection_rule": f"mean({','.join(ABCD_COLS)}) >= {ACTIVATION_THRESHOLD}",
        "selection_threshold": ACTIVATION_THRESHOLD,
        "n_selected_before_limit": EXPECTED_N_SELECTED,
        "limit": args.limit,
        "selected_count": len(selected_records),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(x["id"] for x in selected_records).encode("utf-8")
        ).hexdigest(),
        "anchor_ids": ANCHOR_IDS,
        "anchor_count": len(ANCHOR_IDS),
        "labels_used_for_selection": False,
        "labels_used_in_prompts": False,
        "labels_used_for_training": False,
        "labels_used_for_posthoc_metrics_only": True,
        "test_split_used": False,
        "local_model_only": True,
        "remote_download_allowed": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "batch_samples": args.batch_samples,
        "temperature": 0,
        "max_tokens": 1,
    }


def read_checkpoint(path: Path, config: dict, selected_ids: set, overwrite: bool) -> dict:
    if overwrite and path.exists():
        path.unlink()
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", "config": config}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed, valid_end = [], 0
    truncated_partial = False
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line.decode("utf-8"))
            parsed.append(obj)
            valid_end += len(line)
        except Exception as exc:
            if i == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                truncated_partial = True
                break
            raise ValueError(f"checkpoint 第 {i + 1} 行损坏：{exc}") from exc
    if raw and not raw.endswith(b"\n") and parsed and not truncated_partial and len(parsed) == len(lines):
        # A complete JSON record may have been flushed immediately before its newline.
        with path.open("ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    if not parsed or parsed[0].get("record_type") != "meta":
        raise ValueError("checkpoint 缺少 meta 首行")
    if parsed[0].get("config") != config:
        raise ValueError("checkpoint 配置/输入/模型/prompt/样本集合不一致，拒绝续跑；请换 output_dir 或 --overwrite")
    results = {}
    for line_no, rec in enumerate(parsed[1:], start=2):
        if rec.get("record_type") != "sample":
            raise ValueError(f"checkpoint 第 {line_no} 行记录类型非法")
        sid = rec.get("id")
        if sid not in selected_ids or sid in results:
            raise ValueError(f"checkpoint 样本 ID 不存在或重复：{sid}")
        for key in ("p_yes", "first_token", "cand_mass", "first_token_id"):
            if set(rec.get(key, {}).keys()) != set(ANCHOR_IDS):
                raise ValueError(f"checkpoint 样本 {sid} 的 {key} 字段不完整")
        if any(rec["first_token"][a] not in {"是", "否"} for a in ANCHOR_IDS):
            raise ValueError(f"checkpoint 样本 {sid} 存在非‘是/否’首 token")
        if any(rec["p_yes"][a] is None for a in ANCHOR_IDS):
            raise ValueError(f"checkpoint 样本 {sid} 存在无效 p_yes")
        results[sid] = rec
    return results


def append_checkpoint(path: Path, records: list) -> None:
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def bootstrap_mean_difference(y: np.ndarray, p: np.ndarray, reps: int, seed: int) -> Tuple[float, float]:
    # Positive class here is non-toxic (toxic == 0).
    a = p[y == 0]
    b = p[y == 1]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps, dtype=float)
    for i in range(reps):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        diffs[i] = aa.mean() - bb.mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def make_reports(records: list, results: dict, output_dir: Path, config: dict):
    # Delayed sklearn import; inference does not depend on sklearn's model code.
    from sklearn.metrics import average_precision_score, roc_auc_score

    candidate_names = {"N06": "明确拒斥被引述的攻击/歧视主张",
                       "N07": "明确切断身份与行为的因果归属",
                       "N08": "攻击性词形的非攻击性字面或术语用法"}
    names = [*ANCHOR_IDS, "N05_reference", "E03_reference"]
    rows, audit_rows, matrix_rows, group_rows, example_rows = [], [], [], [], []
    for sample in records:
        rec = results[sample["id"]]
        row = {
            "id": sample["id"], "source_index": sample["source_index"],
            "toxic": sample["toxic"], "toxic_type": sample["toxic_type"],
            "expression": sample["expression"], "topic": sample["topic"],
            "platform": sample["platform"], "length": sample["length"],
            "abcd_mean_activation": sample["abcd_mean_activation"],
            "content": sample["content"],
            "target": json.dumps(sample.get("target"), ensure_ascii=False),
        }
        for c in V8_COLS:
            row[c] = sample["v8"][c]
        for anchor in ANCHOR_IDS:
            row[f"p_{anchor}"] = rec["p_yes"][anchor]
            row[f"first_token_{anchor}"] = rec["first_token"][anchor]
            row[f"cand_mass_{anchor}"] = rec["cand_mass"][anchor]
            audit_rows.append({
                "id": sample["id"], "source_index": sample["source_index"],
                "anchor_id": anchor, "anchor_name": candidate_names[anchor],
                "first_token": rec["first_token"][anchor],
                "first_token_id": rec["first_token_id"][anchor],
                "p_yes": rec["p_yes"][anchor], "cand_mass": rec["cand_mass"][anchor],
            })
        row["p_N05_reference"] = sample["p_N05_reference"]
        row["p_E03_reference"] = sample["p_E03_reference"]
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    y = matrix_df["toxic"].to_numpy(dtype=int)
    values = {
        "N06": matrix_df["p_N06"].to_numpy(dtype=float),
        "N07": matrix_df["p_N07"].to_numpy(dtype=float),
        "N08": matrix_df["p_N08"].to_numpy(dtype=float),
        "N05_reference": matrix_df["p_N05_reference"].to_numpy(dtype=float),
        "E03_reference": matrix_df["p_E03_reference"].to_numpy(dtype=float),
    }
    label_names = {"N06": candidate_names["N06"], "N07": candidate_names["N07"],
                   "N08": candidate_names["N08"], "N05_reference": "Stage1已有N05", "E03_reference": "v8已有E03"}
    for idx, key in enumerate(names):
        p = values[key]
        non = p[y == 0]
        tox = p[y == 1]
        lo, hi = bootstrap_mean_difference(y, p, BOOTSTRAP_REPS, 20261122 + idx)
        rows.append({
            "anchor_id": key, "anchor_name": label_names[key],
            "n": int(len(p)), "n_non_toxic": int((y == 0).sum()), "n_toxic": int((y == 1).sum()),
            "mean_p_non_toxic": float(non.mean()), "mean_p_toxic": float(tox.mean()),
            "mean_difference_non_toxic_minus_toxic": float(non.mean() - tox.mean()),
            "difference_bootstrap_ci95_low": lo, "difference_bootstrap_ci95_high": hi,
            "auc_non_toxic_positive": float(roc_auc_score((y == 0).astype(int), p)),
            "ap_non_toxic_positive": float(average_precision_score((y == 0).astype(int), p)),
            "non_toxic_prevalence": float((y == 0).mean()),
            "rate_ge_0_5_non_toxic": float((non >= 0.5).mean()),
            "rate_ge_0_5_toxic": float((tox >= 0.5).mean()),
        })

    # Descriptive breakdowns: labels are used post-hoc only.
    for group_col in ("toxic", "toxic_type", "expression"):
        for group_value, subset in matrix_df.groupby(group_col, dropna=False):
            for key in names:
                p_col = {"N06": "p_N06", "N07": "p_N07", "N08": "p_N08",
                         "N05_reference": "p_N05_reference", "E03_reference": "p_E03_reference"}[key]
                arr = subset[p_col].to_numpy(dtype=float)
                group_rows.append({
                    "group_variable": group_col, "group_value": group_value,
                    "anchor_id": key, "n": int(len(arr)), "mean_p": float(arr.mean()),
                    "rate_ge_0_5": float((arr >= 0.5).mean()),
                })

    # A small, deterministic qualitative inspection aid: top 8 per label per anchor.
    for key in names:
        p_col = {"N06": "p_N06", "N07": "p_N07", "N08": "p_N08",
                 "N05_reference": "p_N05_reference", "E03_reference": "p_E03_reference"}[key]
        for toxic_value, label in ((0, "non_toxic"), (1, "toxic")):
            sub = matrix_df[matrix_df["toxic"] == toxic_value].nlargest(8, p_col)
            for _, r in sub.iterrows():
                example_rows.append({"anchor_id": key, "label_group": label,
                    "probability": float(r[p_col]), "id": r["id"], "toxic_type": r["toxic_type"],
                    "expression": r["expression"], "topic": r["topic"], "content": r["content"]})

    # Correlation audit among new candidates and prior anchors.
    corr_rows = []
    for group_name, mask in (("all", np.ones(len(y), dtype=bool)),
                             ("non_toxic", y == 0), ("toxic", y == 1)):
        group_values = {k: v[mask] for k, v in values.items()}
        corr_df = pd.DataFrame(group_values).corr(method="pearson")
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                corr = corr_df.loc[left, right]
                corr_rows.append({"group": group_name, "n": int(mask.sum()),
                                  "anchor_left": left, "anchor_right": right,
                                  "pearson_r": (float(corr) if pd.notna(corr) else None)})

    atomic_csv(output_dir / "anchor_matrix.csv", matrix_df)
    atomic_csv(output_dir / "anchor_firsttoken_audit.csv", pd.DataFrame(audit_rows))
    atomic_csv(output_dir / "anchor_screen_metrics.csv", pd.DataFrame(rows))
    atomic_csv(output_dir / "anchor_summary_by_group.csv", pd.DataFrame(group_rows))
    atomic_csv(output_dir / "anchor_high_score_examples.csv", pd.DataFrame(example_rows))
    atomic_csv(output_dir / "anchor_pairwise_correlations.csv", pd.DataFrame(corr_rows))

    # Integrity assertions on saved analytical table.
    if len(matrix_df) != len(records) or matrix_df[[f"p_{a}" for a in ANCHOR_IDS]].isna().any().any():
        raise RuntimeError("最终矩阵行数或概率完整性检查失败")
    if not matrix_df[[f"first_token_{a}" for a in ANCHOR_IDS]].isin(["是", "否"]).all().all():
        raise RuntimeError("最终矩阵存在非‘是/否’首 token")

    result_hashes = {p.name: sha256_file(p) for p in output_dir.glob("*.csv")}
    return rows, result_hashes


def parse_args():
    p = argparse.ArgumentParser(description="AGSCC Stage2 bounded anchor screen (local-only GLM)")
    p.add_argument("--model_path", type=Path, required=False, help="本地 GLM-4-9B-Chat 目录；目录名须为 glm-4-9b-chat")
    p.add_argument("--data_json", type=Path, required=False, help="ToxiCN train.json 原文件")
    p.add_argument("--matrix_csv", type=Path, required=False, help="v8 train concept matrix CSV")
    p.add_argument("--previous_anchor_matrix", type=Path, default=None, help="可选；默认使用包内 Stage1 reference matrix")
    p.add_argument("--output_dir", type=Path, default=Path("AGSCC_stage2_train_outputs"))
    p.add_argument("--gpu_memory_utilization", type=float, default=0.78)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=16)
    p.add_argument("--max_num_batched_tokens", type=int, default=8192)
    p.add_argument("--batch_samples", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="技术冒烟测试前 N 条筛选样本；必须使用独立输出目录")
    p.add_argument("--preflight_only", action="store_true", help="验证输入/哈希/筛选结果，不加载模型")
    p.add_argument("--self_test", action="store_true", help="运行不加载模型的内部逻辑自检")
    p.add_argument("--overwrite", action="store_true", help="删除当前输出目录 checkpoint 并从头运行")
    return p.parse_args()


def self_test(package_dir: Path):
    candidates = load_candidates(package_dir / "anchor_candidates_v2.csv")
    assert [x["条目id"] for x in candidates] == ANCHOR_IDS
    assert len(set(ANCHOR_IDS)) == 3
    assert sha256_file(package_dir / "run_items_v8.py") == EXPECTED["v8_runner_sha256"]
    assert sha256_file(package_dir / "stage1_reference_anchor_matrix.csv") == EXPECTED["stage1_reference_sha256"]
    assert sha256_file(package_dir / "anchor_candidates_v2.csv") == EXPECTED["candidate_csv_sha256"]
    assert PROMPT_TEMPLATE.count("{text}") == 1
    # Unit test the fixed selector and bootstrap statistic without touching labels.
    toy = pd.DataFrame({c: [0.1, 0.2, 0.4, 0.6] for c in ABCD_COLS})
    assert int((toy[ABCD_COLS].mean(axis=1) >= 0.28).sum()) == 2
    lo, hi = bootstrap_mean_difference(np.array([0, 0, 1, 1]), np.array([0.8, 0.7, 0.2, 0.3]), 100, 17)
    assert lo <= 0.5 <= hi
    # Check safe resume when the final complete JSON line lacks its newline.
    with tempfile.TemporaryDirectory(prefix="agscc_ckpt_test_") as td:
        cp = Path(td) / "checkpoint.jsonl"
        cfg = {"test": "fixed"}
        first = {"record_type": "sample", "id": "s1",
                 "p_yes": {a: 0.1 for a in ANCHOR_IDS},
                 "first_token": {a: "否" for a in ANCHOR_IDS},
                 "cand_mass": {a: 0.99 for a in ANCHOR_IDS},
                 "first_token_id": {a: 1 for a in ANCHOR_IDS}}
        cp.write_text(json.dumps({"record_type": "meta", "config": cfg}) + "\n" +
                      json.dumps(first), encoding="utf-8")
        loaded = read_checkpoint(cp, cfg, {"s1"}, overwrite=False)
        assert set(loaded) == {"s1"} and cp.read_bytes().endswith(b"\n")
    print("SELF_TEST_OK")
    print(f"version={VERSION}; anchors={','.join(ANCHOR_IDS)}")
    print(f"v8_runner_sha256={EXPECTED['v8_runner_sha256']}")
    print(f"candidate_sha256={EXPECTED['candidate_csv_sha256']}")


def main():
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    if args.self_test:
        self_test(package_dir)
        return
    if args.batch_samples <= 0:
        raise ValueError("--batch_samples 必须为正整数")
    if not (0.05 <= args.gpu_memory_utilization <= 0.98):
        raise ValueError("--gpu_memory_utilization 必须介于 0.05 和 0.98")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("max_model_len/max_num_seqs/max_num_batched_tokens 必须为正数")
    if args.preflight_only and args.model_path is None:
        # Model is not loaded in preflight mode; use a neutral marker in output.
        model_path = Path("LOCAL_MODEL_PATH_NOT_LOADED")
    else:
        if args.model_path is None:
            raise ValueError("正式测量必须提供 --model_path")
        model_path = args.model_path.expanduser().resolve()
        if model_path.name.lower() != "glm-4-9b-chat":
            raise ValueError("模型目录名必须为 glm-4-9b-chat，以确保沿用已审定的 v8 chat-template 后缀")
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"本地模型目录无效或缺少 config.json：{model_path}")
    if args.data_json is None or args.matrix_csv is None:
        raise ValueError("必须提供 --data_json 与 --matrix_csv")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path, matrix_path, ref_path, hashes, records, candidates, helper = load_and_validate_inputs(args, package_dir)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")

    config = build_run_config(args, model_path, model_path / "config.json" if model_path.is_dir() else package_dir / "run_items_v8.py", hashes, records, Path(__file__).resolve())
    print(f"预检通过：训练集固定高激活子集 {EXPECTED_N_SELECTED} 条；本次测量 {len(records)} 条 × 3 个锚点。")
    print(f"输入/参照哈希已匹配；未读取测试集；标签仅用于测量后的分组与探索性指标。")
    print(f"总进度条：AGSCC锚点测量（单条样本完成后更新）；vLLM 内部进度关闭。")
    if args.preflight_only:
        print("PREFLIGHT_OK")
        return

    checkpoint_path = out_dir / "anchor_checkpoint.jsonl"
    selected_ids = {x["id"] for x in records}
    results = read_checkpoint(checkpoint_path, config, selected_ids, args.overwrite)
    todo = [x for x in records if x["id"] not in results]

    if todo:
        tokenizer, llm = helper.load_vllm_model(
            model_path, args.gpu_memory_utilization, args.max_model_len,
            args.max_num_seqs, args.max_num_batched_tokens
        )
        chat_kwargs = helper.detect_chat_kwargs(tokenizer)
        yes_ids = helper.get_candidate_ids(["是"], tokenizer)
        no_ids = helper.get_candidate_ids(["否"], tokenizer)
        if set(yes_ids) & set(no_ids):
            raise ValueError("‘是’与‘否’verbalizer token ID 重叠")
        sampling_params, decoding_mode = helper.build_sampling_params(yes_ids + no_ids)
        prompt_sha = sha256_text(PROMPT_TEMPLATE)
        progress = tqdm(total=len(records), initial=len(results), desc="AGSCC锚点总进度", unit="条", dynamic_ncols=True)
        try:
            for start in range(0, len(todo), args.batch_samples):
                batch = todo[start:start + args.batch_samples]
                prompts, request_keys = [], []
                for sample in batch:
                    for item in candidates:
                        prompt = PROMPT_TEMPLATE.format(
                            text=sample["content"], concept_name=item["构念名"],
                            question=item["条目问句"], definition=item["概念定义"],
                            boundary=item["边界说明"],
                        )
                        prompts.append(prompt)
                        request_keys.append((sample["id"], item["条目id"]))
                rendered = helper.render_prompts(tokenizer, prompts, chat_kwargs, model_path.name)
                outputs = llm.generate(rendered, sampling_params, use_tqdm=False)
                if len(outputs) != len(request_keys):
                    raise RuntimeError(f"vLLM 输出数不一致：{len(outputs)} != {len(request_keys)}")
                batch_records = {
                    s["id"]: {"record_type": "sample", "id": s["id"],
                        "p_yes": {}, "first_token": {}, "cand_mass": {}, "first_token_id": {},
                        "restricted_decoding": decoding_mode}
                    for s in batch
                }
                for (sid, anchor_id), output in zip(request_keys, outputs):
                    value = helper.get_probability_result(output, tokenizer, yes_ids, no_ids)
                    if value["first_token"] not in {"是", "否"}:
                        raise RuntimeError(f"样本 {sid}/{anchor_id} 首 token 非是/否：{value['first_token']!r}")
                    if value["p_yes"] is None or not math.isfinite(float(value["p_yes"])):
                        raise RuntimeError(f"样本 {sid}/{anchor_id} 的 p_yes 无效")
                    if value["cand_mass"] is None or not math.isfinite(float(value["cand_mass"])):
                        raise RuntimeError(f"样本 {sid}/{anchor_id} 的 cand_mass 无效")
                    rec = batch_records[sid]
                    rec["p_yes"][anchor_id] = value["p_yes"]
                    rec["first_token"][anchor_id] = value["first_token"]
                    rec["cand_mass"][anchor_id] = value["cand_mass"]
                    rec["first_token_id"][anchor_id] = value["first_token_id"]
                for sample in batch:
                    rec = batch_records[sample["id"]]
                    if any(set(rec[k]) != set(ANCHOR_IDS) for k in ("p_yes", "first_token", "cand_mass", "first_token_id")):
                        raise RuntimeError(f"样本 {sample['id']} 的锚点测量结果不完整")
                append_checkpoint(checkpoint_path, [batch_records[s["id"]] for s in batch])
                results.update(batch_records)
                progress.update(len(batch))
                progress.set_postfix_str(f"{len(results)}/{len(records)}")
        finally:
            progress.close()
    else:
        print("checkpoint 已包含全部样本；跳过模型加载与推理。")
        decoding_mode = next(iter(results.values())).get("restricted_decoding", "resume") if results else "not_started"

    missing = [x["id"] for x in records if x["id"] not in results]
    if missing:
        raise RuntimeError(f"推理结束后仍缺少 {len(missing)} 条样本，示例：{missing[:5]}")
    if any(results[s["id"]].get("restricted_decoding") != decoding_mode for s in records):
        raise RuntimeError("checkpoint 中受限解码方式不一致")

    metric_rows, output_hashes = make_reports(records, results, out_dir, config)
    firsttoken_df = pd.read_csv(out_dir / "anchor_firsttoken_audit.csv", encoding="utf-8-sig")
    masses = firsttoken_df["cand_mass"].to_numpy(dtype=float)
    tokens_ok = bool(firsttoken_df["first_token"].isin(["是", "否"]).all())
    if not np.isfinite(masses).all() or (masses < 0).any() or (masses > 1.0001).any():
        raise RuntimeError("cand_mass 包含非有限值或越界值")
    output_hashes[checkpoint_path.name] = sha256_file(checkpoint_path)
    meta = {
        "script_version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(records),
        "measurements_expected": len(records) * len(ANCHOR_IDS),
        "measurements_observed": int(len(firsttoken_df)),
        "first_token_yes_no_coverage": float(tokens_ok),
        "cand_mass_mean": float(masses.mean()),
        "cand_mass_min": float(masses.min()),
        "invalid_p_yes": int(firsttoken_df["p_yes"].isna().sum()),
        "restricted_decoding": decoding_mode,
        "labels_used_for_selection": False,
        "labels_used_in_prompts": False,
        "labels_used_for_posthoc_metrics_only": True,
        "test_split_used": False,
        "local_model_inference": True,
        "remote_download_allowed": False,
        "no_classifier_trained": True,
        "no_v8_concept_probability_modified": True,
        "candidate_ids": ANCHOR_IDS,
        "reference_features": ["N05 from Stage1", "E03 from v8"],
        "output_sha256": output_hashes,
    }
    atomic_json(out_dir / "run_meta.json", meta)

    print("\nAGSCC_STAGE2_COMPLETE")
    print(f"样本={len(records)}；测量={len(firsttoken_df)}；首 token 是/否覆盖={tokens_ok}")
    print(f"cand_mass 均值={masses.mean():.6f}；最小值={masses.min():.6f}")
    print(f"结果目录：{out_dir}")
    print("本轮仅完成候选测量与诊断，不训练分类器、不执行概念纠错。")


if __name__ == "__main__":
    main()
