# -*- coding: utf-8 -*-
"""从 v4 rings JSON 提取 SA_A-E（言语行为五分类）与 Q1-Q7（立场/语境）通道矩阵。

背景：v8.3 审计表明教师缺"意图/字面 vs 转述"维度，而 v4 测量恰好包含
SA_B（引用转述）、Q4（元语言指涉=谈论vs使用）等意图通道。当时"SA 有效但
冗余"的判定是在合并池+秩变换之前的旧特征景观下做出的，需要在当前景观下
重新检验（中带 expr2-FN vs expr0-FP 的可分性）。

本脚本纯 JSON 解析，无任何模型调用。每样本取 12 个通道的 stage1 p_yes
（缺失/无效填 0.5 = 无信息先验）。

产物: data/processed/TOXICN/glm-4-9b-chat_v4/v4_sa_{train,test}_matrix.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SA_KEYS = [f"SA_{c}" for c in "ABCDE"] + [f"Q{i}" for i in range(1, 8)]


def extract(rings_path):
    rings = json.loads(Path(rings_path).read_text(encoding="utf-8"))
    rs = rings["results"] if isinstance(rings, dict) else rings
    rows = []
    for r in rs:
        rec = {"id": r["id"]}
        for k in SA_KEYS:
            c = r.get("concepts", {}).get(k)
            v = np.nan
            if c is not None:
                s1 = c.get("stage1")
                if s1 is not None and s1.get("p_yes") is not None:
                    v = float(s1["p_yes"])
            rec[k] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    n_nan = int(df[SA_KEYS].isna().sum().sum())
    df[SA_KEYS] = df[SA_KEYS].fillna(0.5)  # 无效测量 → 无信息先验
    return df, len(rs), n_nan


def main():
    ap = argparse.ArgumentParser(description="提取 v4 SA/Q 通道矩阵")
    ap.add_argument("--rings_train", required=True)
    ap.add_argument("--rings_test", required=True)
    ap.add_argument("--output_dir", default="data/processed/TOXICN/glm-4-9b-chat_v4")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, path in [("train", args.rings_train), ("test", args.rings_test)]:
        df, n, n_nan = extract(path)
        dst = out / f"v4_sa_{split}_matrix.csv"
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"{split}: n={n}  通道 12  NaN填充数={n_nan}  -> {dst}")


if __name__ == "__main__":
    main()
