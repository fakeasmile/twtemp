# -*- coding: utf-8 -*-
"""v11.2 结论验证：'目标特征的正交残差是测量噪声' 的直接检验（CPU）。

用户质疑：零增益是否足以支持"正交残差与标签无关"？是否应在句子结构
方案上再探索几轮？本脚本做三个预登记检查：

【A】同配对基准补测（我方门槛设计的缺口）：
    M4 在 E0无毒 vs 全部有毒 配对上的 AUC（此前从未测过）。
    若 M4 ≫ 目标特征 0.7702 → G1 PASS 只说明"弱维度"，噪声论无需成立。
【B】正交残差的标签信息直接检验（核心）：
    对 11 个目标特征逐一做 5 折 cross-fitted 岭回归残差化
    （residual = 特征 − OOF[M4 对该特征的预测]），然后：
      - 残差联合 LR 在 E0无毒 vs 有毒 配对上的 AUC
      - 残差联合 LR 对全标签的 AUC
    预登记判读：残差联合 AUC < 0.55 → 正交部分确无标签信息（噪声论成立）；
    ≥ 0.60 → 正交部分有信号而线性读出未利用（应继续探索）。
    cross-fitted 保证残差化本身不吞掉信号（in-sample 会高估 R² 偏向噪声论）。
【C】非线性提取检验：
    GBDT（v8.7 参数）3 折 CV AUC：M4 单独 vs M4+目标特征。
    若 GBDT 也提不出（增益 < +0.002）→ 信号确实不存在，任何读出都无效。

产物：results/v11/target/verify_noise.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83
from analyze_target_v11 import FEATS as TGT_FEATS, aggregate as tgt_aggregate

OUT = Path("results/v11/target/verify_noise.txt")


def assemble_m4(tr, v3tr, freeze, ytr):
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
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    return np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])


def cv_pair_auc(X, ysub, idx, seeds=(42, 43, 44), C=0.01):
    """子集内 5 折 CV AUC，3 种子均值（与 v8.3 审计/v11 快检同法）。"""
    vals = []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(idx))
        for trn, va in cv.split(X[idx], ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=C, max_iter=2000))])
            m.fit(X[idx][trn], ysub[trn])
            oof[va] = m.predict_proba(X[idx][va])[:, 1]
        a = roc_auc_score(ysub, oof)
        vals.append(max(a, 1 - a))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    freeze = json.loads(Path(
        "results/v6/item_analysis_v64_train/v6_4_freeze.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv")
    v3tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_v3/v3_soft_train_matrix.csv")
    ytr = tr["toxic"].astype(int).values
    with open("data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json",
              encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])

    X = assemble_m4(tr, v3tr, freeze, ytr)
    print(f"M4 装配 {X.shape}")
    tf = pd.read_csv("results/v11/target/target_flat_train.csv")
    F_t = tgt_aggregate(tf, tr["id"].values)[TGT_FEATS].values
    Xc = np.hstack([X, F_t])

    e0fp = np.where((ytr == 0) & (expr == 0))[0]
    tox = np.where(ytr == 1)[0]
    idx = np.concatenate([e0fp, tox])
    ysub = np.where(np.arange(len(idx)) < len(e0fp), 0, 1)
    lines = ["=" * 76,
             "v11.2 结论验证：正交残差是否为测量噪声（三个预登记检查）",
             "=" * 76, f"配对：E0无毒 n={len(e0fp)} vs 有毒 n={len(tox)}", ""]

    # ---- 【A】同配对基准补测 ----
    a_m4, s1 = cv_pair_auc(X, ysub, idx)
    a_tgt, s2 = cv_pair_auc(F_t, ysub, idx)
    a_comb, s3 = cv_pair_auc(Xc, ysub, idx)
    lines += ["【A】同配对基准补测（M4 在 E0无毒 vs 有毒 上的 AUC，此前未测）",
              "-" * 76,
              f"  M4 264维          AUC = {a_m4:.4f} ± {s1:.4f}",
              f"  目标特征 11 维     AUC = {a_tgt:.4f} ± {s2:.4f}  (Stage3 单种子口径 0.7702)",
              f"  M4+目标 275 维    AUC = {a_comb:.4f} ± {s3:.4f}",
              f"  判读：目标特征比 M4 {'弱' if a_tgt < a_m4 else '强'} "
              f"{abs(a_m4 - a_tgt):.4f}；合并增量 {a_comb - a_m4:+.4f}", ""]

    # ---- 【B】cross-fitted 正交残差检验 ----
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Fs = StandardScaler().fit_transform(F_t)
    kf = KFold(5, shuffle=True, random_state=42)
    F_res = np.zeros_like(Fs)
    r2s = []
    for j in range(Fs.shape[1]):
        pred_oof = np.zeros(len(Fs))
        for trn, va in kf.split(Xs):
            rg = Ridge(alpha=1.0).fit(Xs[trn], Fs[trn, j])
            pred_oof[va] = rg.predict(Xs[va])
        ss = float(np.sum((Fs[:, j] - Fs[:, j].mean()) ** 2))
        r2s.append(1.0 - float(np.sum((Fs[:, j] - pred_oof) ** 2)) / ss)
        F_res[:, j] = Fs[:, j] - pred_oof
    a_res_pair, s4 = cv_pair_auc(F_res, ysub, idx)
    # 残差联合对全标签也要 CV（防 in-sample 高估）
    oof_full = np.zeros(len(ytr))
    cvf = StratifiedKFold(5, shuffle=True, random_state=42)
    for trn, va in cvf.split(F_res, ytr):
        m = Pipeline([("sc", StandardScaler()),
                      ("m", LogisticRegression(C=0.01, max_iter=2000))])
        m.fit(F_res[trn], ytr[trn])
        oof_full[va] = m.predict_proba(F_res[va])[:, 1]
    a_res_full_cv = max(roc_auc_score(ytr, oof_full), 1 - roc_auc_score(ytr, oof_full))
    lines += ["【B】cross-fitted 正交残差的标签信息（核心检验）", "-" * 76,
              "  各特征被 M4 解释的调整 R²（OOF 口径）:",
              "  " + ", ".join(f"{f}={r2:.3f}" for f, r2 in zip(TGT_FEATS, r2s)),
              f"  残差联合 LR @E0无毒vs有毒: AUC = {a_res_pair:.4f} ± {s4:.4f}",
              f"  残差联合 LR @全标签(5折CV): AUC = {a_res_full_cv:.4f}",
              f"  预登记判读: <0.55=噪声论成立; ≥0.60=正交信号存在应继续探索", ""]

    # ---- 【C】非线性提取检验 ----
    def gbdt_auc(Xa, seed=42):
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        oof = np.zeros(len(ytr))
        for trn, va in cv.split(Xa, ytr):
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=1.0,
                random_state=seed).fit(Xa[trn], ytr[trn])
            oof[va] = m.predict_proba(Xa[va])[:, 1]
        a = roc_auc_score(ytr, oof)
        return max(a, 1 - a)

    print("GBDT M4 单独 3 折...")
    g_m4 = gbdt_auc(X)
    print("GBDT M4+目标 3 折...")
    g_comb = gbdt_auc(Xc)
    lines += ["【C】非线性提取检验（GBDT 3 折 CV AUC，v8.7 参数）", "-" * 76,
              f"  GBDT @M4         AUC = {g_m4:.4f}",
              f"  GBDT @M4+目标    AUC = {g_comb:.4f}",
              f"  增量 {g_comb - g_m4:+.4f}（预登记 <+0.002 → 任何读出都提不出）", ""]

    # ---- 总裁决 ----
    noise_ok = a_res_pair < 0.55 and a_res_full_cv < 0.55 and (g_comb - g_m4) < 0.002
    signal = a_res_pair >= 0.60 or a_res_full_cv >= 0.60 or (g_comb - g_m4) >= 0.002
    verdict = ("噪声论成立：正交残差与标签无关，句子结构方向闭环"
               if noise_ok else
               "正交残差含信号（应继续探索句子结构方向）" if signal else
               "灰色地带（0.55-0.60 之间），需人工判读")
    lines += ["=" * 76, f"[总裁决] {verdict}",
              f"  (A) M4同配对 {a_m4:.4f} vs 目标特征 {a_tgt:.4f}",
              f"  (B) 残差配对 {a_res_pair:.4f} / 全标签 {a_res_full_cv:.4f}",
              f"  (C) GBDT 增量 {g_comb - g_m4:+.4f}", "=" * 76]
    report = "\n".join(lines)
    print(report)
    OUT.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
