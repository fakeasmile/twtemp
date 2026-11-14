# -*- coding: utf-8 -*-
"""解析 v3 rings JSON → GLM 教师概念分宽表（合并题池诊断的数据准备）。

输入：data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_{train,test}.json
输出：data/processed/TOXICN/glm-4-9b-chat_v3/v3_concepts_{train,test}_matrix.csv
      列：id, toxic, v3_<概念id> × N（全部概念的 stage1 p_yes，值域 0-1）

说明：
- 只取 stage1.p_yes（一阶段概率），stage2/CAT/S1 不取（两阶段机制已废弃，
  直接性等信号在 v6 中已条目化）。
- 概念清单以 meta.concepts 为准（含 role/range 信息，打印分布供参考）。
- v6 条目 id 形如 A1-01，与 v3 概念 id（A1/E1/...）加前缀后无冲突。
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
    roles = {c["id"]: c["role"] for c in meta["concepts"]}
    print(f"  {json_path.name}: {meta['n_samples']} 样本, {len(order)} 概念")
    from collections import Counter
    print("    role 分布:", dict(Counter(roles.values())))

    rows = []
    for r in data["results"]:
        row = {"id": r["id"], "toxic": int(r["toxic"])}
        cpts = r["concepts"]
        for cid in order:
            stage1 = cpts.get(cid, {}).get("stage1")
            row[f"v3_{cid}"] = stage1["p_yes"] if stage1 else None
        rows.append(row)
    df = pd.DataFrame(rows)
    n_missing = int(df[[c for c in df.columns if c.startswith("v3_")]].isna().sum().sum())
    print(f"    缺失单元格: {n_missing} / {df.shape[0] * (df.shape[1] - 2)}")
    return df


def main():
    ap = argparse.ArgumentParser(description="解析 v3 rings JSON 为概念分宽表")
    ap.add_argument("--v3_dir", default="data/processed/TOXICN/glm-4-9b-chat_v3")
    args = ap.parse_args()
    v3_dir = Path(args.v3_dir)

    for split in ("train", "test"):
        src = v3_dir / f"rings_glm-4-9b-chat_{split}.json"
        dst = v3_dir / f"v3_concepts_{split}_matrix.csv"
        if not src.exists():
            print(f"跳过（不存在）: {src}")
            continue
        df = parse_one(src)
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  -> {dst}  shape={df.shape}")


if __name__ == "__main__":
    main()
