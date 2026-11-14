# -*- coding: utf-8 -*-
"""解析 v3 rings JSON → GLM 教师三通道概念特征宽表。

通道（与 v3 当年 97 维读出的特征形态一致）：
  v3p_<id>  —— stage1 p_yes（存在性概率，0-1）
  v3d_<id>  —— stage2 p_A（直接性概率，未触发填 0，两阶段遗留信息）
  v3v_<id>  —— value（0-2 序数值，如 range 标注）

输出：data/processed/TOXICN/glm-4-9b-chat_v3/v3_full_{train,test}_matrix.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_one(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    order = [c["id"] for c in meta["concepts"]]
    print(f"  {json_path.name}: {meta['n_samples']} 样本, {len(order)} 概念")

    rows = []
    for r in data["results"]:
        row = {"id": r["id"], "toxic": int(r["toxic"])}
        cpts = r["concepts"]
        for cid in order:
            c = cpts.get(cid, {})
            s1 = c.get("stage1")
            s2 = c.get("stage2")
            row[f"v3p_{cid}"] = s1["p_yes"] if s1 else None
            row[f"v3d_{cid}"] = (s2["p_A"] if s2 and s2.get("p_A") is not None else 0.0)
            row[f"v3v_{cid}"] = c.get("value")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="解析 v3 rings JSON 为三通道概念特征宽表")
    ap.add_argument("--v3_dir", default="data/processed/TOXICN/glm-4-9b-chat_v3")
    args = ap.parse_args()
    v3_dir = Path(args.v3_dir)

    for split in ("train", "test"):
        src = v3_dir / f"rings_glm-4-9b-chat_{split}.json"
        dst = v3_dir / f"v3_full_{split}_matrix.csv"
        if not src.exists():
            print(f"跳过（不存在）: {src}")
            continue
        df = parse_one(src)
        feat_cols = [c for c in df.columns if c not in ("id", "toxic")]
        n_const = int(sum(df[c].nunique() <= 1 for c in feat_cols))
        print(f"    特征列 {len(feat_cols)}，恒定列 {n_const}（诊断时按方差过滤）")
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  -> {dst}  shape={df.shape}")


if __name__ == "__main__":
    main()
