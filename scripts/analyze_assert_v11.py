# -*- coding: utf-8 -*-
"""v11.1 Stage3：断言状态测量的预登记门槛判读（CPU）。

特征聚合（每文本，基于句级 p_tox 与断言状态）：
  n_sent                          句数
  p_tox_max                       全句最大内容毒性
  p_tox_max_asserted              断言句（argmax=断言）最大内容毒性（无则 0.0）
  p_tox_max_embedded              非断言句（引用/假设/反问）最大内容毒性（无则 0.0）
  gap_hard                        embedded − asserted（无词表 use/mention 主特征）
  assert_share                    断言句占比
  n_embedded_toxic / n_asserted_toxic
                                  非断言/断言句中 p_tox≥0.5 的句数
  soft_asserted_tox / soft_embedded_tox
                                  soft 加权：mean(p_tox × w_断言) / mean(p_tox × (1−w_断言))
  gap_soft                        soft_embedded − soft_asserted

预登记门槛（train 全量，test 零参与；E0FP=expr0 无毒 n≈4413 vs 有毒 n=5187，全功效）：
  G1 主门：最优单特征 AUC ≥ 0.75 且新特征联合 LR（3 种子 5 折，C=0.01）AUC ≥ 0.75
  G2 正交门：11 个特征与现有概率特征（141 翻转条目 + v3p_ + S1A/S1B/CAT）
     max|corr| < 0.7
  双门皆过 -> 写 readout_v11（M4 + 断言特征，CV 门槛 ≥0.002）

产物：results/v11/assert/analysis_metrics.txt + gate.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ["n_sent", "p_tox_max", "p_tox_max_asserted", "p_tox_max_embedded",
         "gap_hard", "assert_share", "n_embedded_toxic", "n_asserted_toxic",
         "soft_asserted_tox", "soft_embedded_tox", "gap_soft"]


def aggregate(flat: pd.DataFrame, ids):
    """flat: 全部句子记录（train）；ids: train 样本序。返回逐样本特征。"""
    by_id = {k: v for k, v in flat.groupby("id", sort=False)}
    rows = []
    for _id in ids:
        g = by_id.get(_id)
        if g is None or len(g) == 0:
            rows.append({f: 0.0 for f in FEATS})
            continue
        p = g["p_tox"].fillna(0.5).values
        wa = g["w_断言"].values
        asserted = (g["status_argmax"] == "断言").values
        tox = p >= 0.5
        mx_a = p[asserted].max() if asserted.any() else 0.0
        mx_e = p[~asserted].max() if (~asserted).any() else 0.0
        rows.append({
            "n_sent": float(len(g)),
            "p_tox_max": float(p.max()),
            "p_tox_max_asserted": float(mx_a),
            "p_tox_max_embedded": float(mx_e),
            "gap_hard": float(mx_e - mx_a),
            "assert_share": float(asserted.mean()),
            "n_embedded_toxic": float((tox & ~asserted).sum()),
            "n_asserted_toxic": float((tox & asserted).sum()),
            "soft_asserted_tox": float(np.mean(p * wa)),
            "soft_embedded_tox": float(np.mean(p * (1.0 - wa))),
            "gap_soft": float(np.mean(p * (1.0 - wa)) - np.mean(p * wa)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="v11.1 Stage3 门槛判读")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json", required=True)
    ap.add_argument("--assert_flat", default="results/v11/assert/assert_flat_train.csv")
    ap.add_argument("--output_dir", default="results/v11/assert")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.train_matrix)
    y = tr["toxic"].astype(int).values
    ids = tr["id"].values
    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids])

    flat = pd.read_csv(args.assert_flat)
    F = aggregate(flat, ids).values
    assert F.shape == (len(ids), len(FEATS))

    lines = ["=" * 72, "v11.1 句子级断言状态测量——预登记门槛判读", "=" * 72,
             f"train 样本 {len(ids)}，句子 {len(flat)}"
             f"（均 {len(flat) / len(ids):.2f} 句/样本）", ""]

    # ---- 组分布（机制检查）----
    hit_df = pd.DataFrame(F, columns=FEATS)
    groups = [("E0无毒", (y == 0) & (expr == 0)), ("expr1有毒", (y == 1) & (expr == 1)),
              ("expr2有毒", (y == 1) & (expr == 2)), ("expr3有毒", (y == 1) & (expr == 3))]
    lines += [f"{'组':<10}{'n':<7}{'gap_hard均值':<14}{'embedded毒性均值':<16}"
              f"{'asserted毒性均值':<16}", "-" * 72]
    for name, m in groups:
        if m.sum():
            lines.append(f"{name:<10}{m.sum():<7}"
                         f"{hit_df.loc[m, 'gap_hard'].mean():<14.4f}"
                         f"{hit_df.loc[m, 'soft_embedded_tox'].mean():<16.4f}"
                         f"{hit_df.loc[m, 'soft_asserted_tox'].mean():<16.4f}")
    lines.append("")

    # ---- G1 主门：E0FP vs 有毒（全功效）----
    e0fp = np.where((y == 0) & (expr == 0))[0]
    tox = np.where(y == 1)[0]
    lines += ["G1 主门（E0FP vs 有毒，全量，预登记 ≥0.75）", "-" * 72]
    sub = np.concatenate([e0fp, tox])
    ysub = np.where(np.arange(len(sub)) < len(e0fp), 0, 1)
    best = (None, -1.0, 0)
    for j, f in enumerate(FEATS):
        v = F[sub, j]
        if np.std(v) < 1e-9:
            continue
        a = roc_auc_score(ysub, v)
        sign = "+" if a >= 0.5 else "-"
        a = max(a, 1 - a)
        if a > best[1]:
            best = (f, a, sign)
        lines.append(f"  {f:<20} AUC={a:.4f} 方向'{sign}'")
    lines.append(f"  最优单特征: {best[0]} AUC={best[1]:.4f} 方向'{best[2]}'"
                 f"  [门槛 {'PASS' if best[1] >= 0.75 else 'FAIL'}]")

    aucs = []
    Xsub = F[sub]
    for seed in (42, 43, 44):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(sub))
        for trn, va in cv.split(Xsub, ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=0.01, max_iter=2000))])
            m.fit(Xsub[trn], ysub[trn])
            oof[va] = m.predict_proba(Xsub[va])[:, 1]
        aucs.append(max(roc_auc_score(ysub, oof), 1 - roc_auc_score(ysub, oof)))
    joint = float(np.mean(aucs))
    lines.append(f"  联合 LR AUC = {joint:.4f} ± {np.std(aucs):.4f}"
                 f"  [门槛 {'PASS' if joint >= 0.75 else 'FAIL'}]")

    # ---- G2 正交门 ----
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr = tr[item_cols].copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    v3 = pd.read_csv(args.v3_train)
    v3_cols = [c for c in v3.columns
               if c.startswith(("v3p_", "s1a_", "s1b_", "cat_"))]
    exist = np.hstack([Atr[kept_items].values, v3[v3_cols].values])
    maxc = {}
    for j, f in enumerate(FEATS):
        r = np.abs(np.corrcoef(F[:, j], exist.T)[0, 1:])
        maxc[f] = float(np.nanmax(r))
    worst_f, worst_v = max(maxc.items(), key=lambda x: x[1])
    lines += ["", "G2 正交门（与现有概率特征 max|corr| < 0.7）", "-" * 72]
    for f in FEATS:
        lines.append(f"  {f:<20} max|corr|={maxc[f]:.4f}")
    g2_pass = worst_v < 0.7
    lines.append(f"  最大: {worst_f} {worst_v:.4f}  [门槛 {'PASS' if g2_pass else 'FAIL'}]")

    # ---- 裁决 ----
    g1_pass = best[1] >= 0.75 and joint >= 0.75
    lines += ["", "=" * 72,
              f"[裁决] G1 主门 {'PASS' if g1_pass else 'FAIL'}，"
              f"G2 正交门 {'PASS' if g2_pass else 'FAIL'}",
              "双门皆过 -> 写 readout_v11（M4 + 断言特征，CV 门槛 ≥0.002）；"
              "任一 FAIL -> 断言状态通道判死并归档机制结论", "=" * 72]

    (out / "gate.json").write_text(json.dumps(
        {"g1_single": {"feat": best[0], "auc": best[1], "sign": best[2]},
         "g1_joint": joint, "g1_pass": g1_pass, "g2_maxcorr": worst_v,
         "g2_pass": g2_pass}, ensure_ascii=False, indent=1), encoding="utf-8")
    report = "\n".join(lines)
    print(report)
    (out / "analysis_metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
