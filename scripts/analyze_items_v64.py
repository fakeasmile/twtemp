# -*- coding: utf-8 -*-
"""v6.4 条目分析：经验极性对齐 + 判别加权聚合（train 唯一裁决点，新建脚本）。

修复 v6.1 审计发现的两处程序性失真（用户确认）：
  失真1：反向条目极性异质（21/39 设计极性成立、18/39 教师语义反向），
         按设计意图统一翻转导致每构念恰有一条反极性 → α 崩塌 → 假降级。
         修复：每条条目的使用极性由 train 数据裁决（raw AUC≥0.5 用原始 p，
         否则 1−p），登记为"经验极性对齐"（每条目 1 bit，train 内决定）。
  失真2：饱和 yes-bias 条目（D1/D4/B8 区 benign P50 0.71-0.74）的噪声方差
         污染等权构念均值（D4 构念 AUC 0.594 vs 条目 0.82+）。
         修复：构念分 = 对齐条目的信号加权均值（权重 = 与 y 的点二列相关，
         构念内归一化，train 冻结）；等权均值同时保留供 CV 竞争。

D2 特例不变：主分 = Guttman 取档（D2-01..05，阈值 0.5，单调性 <0.80 回退均值），
D2-06/07/08 不进主分。α/CITC 在对齐矩阵上重算（门槛 0.70 / 0.30）。

产物：results/v6/item_analysis_v64_train/
  v6_4_freeze.json      —— 读出脚本输入（items_per_construct / item_polarity /
                           item_weights / d2 / 诊断摘要）
  item_details_v64.csv  —— 152 条目逐条诊断（极性、对齐 AUC、CITC、权重、保留）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALPHA_THRESHOLD = 0.70
CITC_THRESHOLD = 0.30
GUTTMAN_MONO_THRESHOLD = 0.80
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]


def cronbach_alpha(M):
    k = M.shape[1]
    if k < 2:
        return np.nan
    var_rows = M.sum(axis=1).var(ddof=1)
    if var_rows < 1e-12:
        return np.nan
    return k / (k - 1) * (1 - M.var(ddof=1).sum() / var_rows)


def citc(M):
    tot = M.sum(axis=1)
    out = {}
    for c in M.columns:
        rest = tot - M[c]
        out[c] = float(np.corrcoef(M[c], rest)[0, 1]) if rest.std() > 1e-12 else np.nan
    return out


def guttman_score(M):
    Mv = M.values if hasattr(M, "values") else np.asarray(M)
    hits = Mv >= 0.5
    mono_ok = 0
    for i in range(Mv.shape[0]):
        h = hits[i]
        if h.any():
            top = int(np.where(h)[0][0])
            if h[top:].all():
                mono_ok += 1
        else:
            mono_ok += 1
    if mono_ok / Mv.shape[0] < GUTTMAN_MONO_THRESHOLD:
        return Mv.mean(axis=1), False
    score = np.zeros(Mv.shape[0])
    for i in range(Mv.shape[0]):
        h = hits[i]
        score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
    return score, True


def main():
    ap = argparse.ArgumentParser(description="v6.4 条目分析：经验极性对齐 + 加权聚合")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--output_dir", default="results/v6/item_analysis_v64_train")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    items = items.rename(columns={"条目id": "item", "构念id": "construct",
                                  "构念名": "cname", "区块": "block",
                                  "题式": "form", "方向": "direction"})
    df = pd.read_csv(args.matrix)
    y = df["toxic"].astype(int).values
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    X = df[item_cols]
    n = len(df)

    # ---- 1. 经验极性对齐（train 裁决，每条目 1 bit）----
    polarity, aligned_auc = {}, {}
    A = X.copy()
    n_flip = 0
    for c in item_cols:
        auc_raw = roc_auc_score(y, X[c].values)
        if auc_raw >= 0.5:
            polarity[c] = 1
            aligned_auc[c] = auc_raw
        else:
            polarity[c] = -1
            aligned_auc[c] = 1.0 - auc_raw
            A[c] = 1.0 - X[c]
            n_flip += 1

    # ---- 2. 构念级裁决（对齐矩阵上 α/CITC）----
    freeze_items, weights, details, decisions = {}, {}, [], []
    for cid, g in items.groupby("construct", sort=False):
        g_items = [c for c in g["item"].tolist() if c in A.columns]
        Mall = A[g_items]
        a_all = cronbach_alpha(Mall)
        citc_all = citc(Mall)

        if cid == "D2":
            keep = [c for c in D2_TIERS if c in A.columns]
            cscore_eq = guttman_score(A[keep])[0]
            cscore_w = cscore_eq  # D2 主分固定 Guttman，无加权变体
            d2_mode = "guttman"
        else:
            keep = [c for c in g_items if citc_all.get(c, 0.0) >= CITC_THRESHOLD]
            if len(keep) < 2:
                keep = sorted(g_items, key=lambda c: -aligned_auc[c])[:3]
            Mkeep = A[keep]
            cscore_eq = Mkeep.mean(axis=1).values
            rpb = {c: float(abs(np.corrcoef(A[c].values, y)[0, 1])) for c in keep}
            s = sum(rpb.values())
            w = {c: (rpb[c] / s if s > 1e-9 else 1.0 / len(keep)) for c in keep}
            cscore_w = sum(w[c] * Mkeep[c].values for c in keep)
            d2_mode = "weighted"

        auc_eq = roc_auc_score(y, cscore_eq)
        auc_w = roc_auc_score(y, cscore_w)
        for c in g_items:
            details.append({
                "item": c, "construct": cid,
                "direction_design": g[g["item"] == c]["direction"].iloc[0],
                "polarity_aligned": polarity[c],
                "raw_auc": 1 - aligned_auc[c] if polarity[c] == -1 else aligned_auc[c],
                "aligned_auc": aligned_auc[c],
                "citc": citc_all.get(c),
                "weight": weights.get(c) if cid == "D2" else
                          (w.get(c) if c in keep and cid != "D2" else None),
                "kept": c in keep,
            })
        if cid != "D2":
            weights.update(w)
        for c in keep:
            freeze_items.setdefault(cid, []).append(c)

        decisions.append({
            "construct": cid, "alpha_aligned": a_all,
            "n_kept": len(keep), "auc_equal": auc_eq, "auc_weighted": auc_w,
            "alive": len(keep) >= 2 and max(auc_eq, auc_w) > 0.55,
        })

    detail_df = pd.DataFrame(details)
    detail_df.to_csv(out / "item_details_v64.csv", index=False, encoding="utf-8-sig")

    # ---- 3. 冻结 ----
    dec_df = pd.DataFrame(decisions)
    freeze = {
        "version": "v6.4",
        "protocol": ("经验极性对齐（raw AUC≥0.5 → 原始 p，否则 1−p；train 裁决）+ "
                     "信号加权构念均值（点二列相关归一化）+ CITC≥0.30；"
                     "D2 主分 = Guttman 取档"),
        "items_per_construct": freeze_items,
        "item_polarity": polarity,
        "item_weights": weights,
        "d2": {"tiers": D2_TIERS, "mode": "guttman"},
        "constructs_alive": dec_df[dec_df["alive"]]["construct"].tolist(),
        "summary": {
            "n_train": n, "n_flipped": n_flip,
            "n_kept_items": int(detail_df["kept"].sum()),
            "alpha_lt_threshold": dec_df[dec_df["alpha_aligned"] < ALPHA_THRESHOLD]["construct"].tolist(),
        },
    }
    with open(out / "v6_4_freeze.json", "w", encoding="utf-8") as f:
        json.dump(freeze, f, ensure_ascii=False, indent=2)

    # ---- 4. 报告 ----
    lines = ["=" * 72, "v6.4 条目分析（train 9600，经验极性对齐 + 加权聚合）", "=" * 72,
             f"极性翻转 {n_flip}/152 条（教师语义与设计意图相反，按数据裁决对齐）",
             f"保留条目 {int(detail_df['kept'].sum())}/152（CITC≥{CITC_THRESHOLD}，对齐矩阵上重算）",
             f"α<{ALPHA_THRESHOLD} 的构念: {freeze['summary']['alpha_lt_threshold'] or '无'}",
             "", "构念   α(对齐)  n保留  AUC等权  AUC加权   裁决",
             "-" * 52]
    for r in decisions:
        lines.append(f"{r['construct']:<5} {r['alpha_aligned']:.3f}    {r['n_kept']}     "
                     f"{r['auc_equal']:.3f}    {r['auc_weighted']:.3f}    "
                     f"{'存活' if r['alive'] else '判死'}")
    lines += ["", f"冻结文件: {out / 'v6_4_freeze.json'}"]
    report = "\n".join(lines)
    print(report)
    (out / "analysis_report.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
