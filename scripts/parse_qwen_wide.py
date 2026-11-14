# -*- coding: utf-8 -*-
"""解析 Qwen3.5-9B v3 协议 flat 长表 → 宽表（双教师第二证据源，用户已解冻）。

长表：每行 = 样本 × 概念 × ring。同概念跨 ring 取均值（内置测量去噪）。
列前缀：
  qw_p_<concept>   p_yes 均值（role=evidence）
  qw_b_<concept>   p_B 均值（触发样本，缺失填 0）
  qw_pd_<concept>  P(直接) = p_yes × p_B 均值
  qw_v_<concept>   value 序数均值
  qw_s1a / qw_s1b  样本级 S1A/S1B 序数预测
  qw_cat1a..cat4   样本级 CAT 通道 p_yes

输出：data/processed/TOXICN/Qwen3.5-9B/qwen_wide_{train,test}_matrix.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pivot_one(flat_path: Path) -> pd.DataFrame:
    df = pd.read_csv(flat_path)
    print(f"  {flat_path.name}: {df['id'].nunique()} 样本, {df['concept'].nunique()} 概念, "
          f"rings={sorted(df['ring'].dropna().unique())}")
    ev = df[df["role"] == "evidence"]
    g = ev.groupby(["id", "concept"])
    base = g.agg(p_yes=("p_yes", "mean"), p_b=("p_B", "mean"),
                 value=("value", "mean"), toxic=("toxic", "first")).reset_index()
    base["p_b"] = base["p_b"].fillna(0.0)
    base["pd"] = base["p_yes"] * base["p_b"]
    ids = base[["id", "toxic"]].drop_duplicates("id").set_index("id")
    wide = pd.DataFrame(index=ids.index)
    wide["toxic"] = ids["toxic"]
    for col, pre in [("p_yes", "qw_p"), ("p_b", "qw_b"), ("pd", "qw_pd"), ("value", "qw_v")]:
        piv = base.pivot_table(index="id", columns="concept", values=col, aggfunc="mean")
        piv = piv.fillna(0.0) if col in ("p_b", "pd", "value") else piv
        piv.columns = [f"{pre}_{c}" for c in piv.columns]
        wide = wide.join(piv)
    # 样本级列（每个 id 取第一行）
    samp = df.groupby("id").first()
    wide["qw_s1a"] = samp["S1A_pred"]
    wide["qw_s1b"] = samp["S1B_pred"]
    for k in ("CAT1A", "CAT1B", "CAT2A", "CAT2B", "CAT3", "CAT4"):
        if k + "_p" in samp.columns:
            wide[f"qw_{k.lower()}"] = pd.to_numeric(samp[k + "_p"], errors="coerce")
    wide = wide.reset_index().rename_axis(None, axis=1)
    return wide


def main():
    ap = argparse.ArgumentParser(description="Qwen flat 长表 → 宽表")
    ap.add_argument("--qwen_dir", default="data/processed/TOXICN/Qwen3.5-9B")
    args = ap.parse_args()
    qdir = Path(args.qwen_dir)
    for split in ("train", "test"):
        src = qdir / f"rings_Qwen3.5-9B_{split}_flat.csv"
        if not src.exists():
            print(f"跳过（不存在）: {src}")
            continue
        wide = pivot_one(src)
        dst = qdir / f"qwen_wide_{split}_matrix.csv"
        wide.to_csv(dst, index=False, encoding="utf-8-sig")
        n_const = int(sum(wide[c].nunique() <= 1
                          for c in wide.columns if c not in ("id", "toxic")))
        print(f"  -> {dst}  shape={wide.shape}, 恒定列 {n_const}")


if __name__ == "__main__":
    main()
