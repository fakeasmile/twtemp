# -*- coding: utf-8 -*-
"""v11.2 附加快检：攻击目标特征在 FP vs TP 边界上的判别力（CPU）。

背景：readout_v11 全局门 FAIL（增益 -0.0004）。但 v11 的原始靶点是
FP vs TP（v8.3 审计上限 0.7335），全局宏 F1 可能稀释局部效应。
本检查：seed42 锚点 OOF 定错误组 → FP vs TP 子集内 5 折 CV AUC，
比较 11 个目标特征 vs 现有 M4 特征（同子集同折）。
判读（预登记）：目标特征比 M4 参照 +0.05 以上 → 局部裂缝存在；
+0.02 以内 → 粒度/结构假设彻底闭环。

产物：results/v11/target/fp_tp_check.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83
from analyze_target_v11 import FEATS as TGT_FEATS, aggregate as tgt_aggregate

OUT = Path("results/v11/target/fp_tp_check.txt")


def main():
    freeze = json.loads(Path(
        "results/v6/item_analysis_v64_train/v6_4_freeze.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv")
    v3tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_v3/v3_soft_train_matrix.csv")
    ytr = tr["toxic"].astype(int).values

    # ---- M4 装配（readout_v11 同款，seed42 锚点 OOF）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw = tr[item_cols]
    Atr = Xtr_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok if auc[c] >= r83.V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < r83.V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    v3_tr = np.hstack([Ftr_c[kept_c].values, v3tr[kept_s].values])
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rall_tr = np.zeros_like(raw_tr)
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)
    n_items = len(kept_items)
    Atr_items = Atr[kept_items].values
    Mtr, _ = r83.meta_base(tr["content"].tolist(), [""] * len(tr))
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1]) for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    Xtr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    print(f"M4 装配 {Xtr.shape}")

    a_alpha, tau, oof = r83.cv_select(Xtr, ytr, 42, "anchor")
    err = (oof >= tau).astype(int) != ytr
    fp_i = np.where(err & (ytr == 0))[0]
    tp_i = np.where((~err) & (ytr == 1))[0]
    print(f"anchor alpha={a_alpha} tau={tau}  FP={len(fp_i)} TP={len(tp_i)}")

    tf = pd.read_csv("results/v11/target/target_flat_train.csv")
    F_t = tgt_aggregate(tf, tr["id"].values)[TGT_FEATS].values

    idx = np.concatenate([fp_i, tp_i])
    ysub = np.where(np.arange(len(idx)) < len(fp_i), 0, 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    def subset_auc(X):
        oof = np.zeros(len(idx))
        for trn, va in cv.split(X[idx], ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=0.01, max_iter=2000))])
            m.fit(X[idx][trn], ysub[trn])
            oof[va] = m.predict_proba(X[idx][va])[:, 1]
        a = roc_auc_score(ysub, oof)
        return max(a, 1 - a)

    a_m4 = subset_auc(Xtr)
    a_tgt = subset_auc(F_t)
    a_comb = subset_auc(np.hstack([Xtr, F_t]))
    lines = ["=" * 72,
             "v11.2 快检：攻击目标特征在 FP vs TP 边界（seed42 锚点错误组）",
             "=" * 72,
             f"FP={len(fp_i)}  TP={len(tp_i)}  （v8.3 审计参照：M4 线性 0.7335）",
             "",
             f"  M4 264维（同子集复测）   AUC = {a_m4:.4f}",
             f"  目标特征 11 维单独      AUC = {a_tgt:.4f}",
             f"  M4 + 目标特征           AUC = {a_comb:.4f}",
             "",
             f"  目标特征相对 M4 复测: {a_tgt - a_m4:+.4f}"
             f"  （预登记：+0.05 以上=局部裂缝；+0.02 以内=闭环）",
             f"  合并相对 M4 复测:     {a_comb - a_m4:+.4f}",
             "=" * 72]
    report = "\n".join(lines)
    print(report)
    OUT.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
