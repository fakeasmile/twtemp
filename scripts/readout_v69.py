# -*- coding: utf-8 -*-
"""v6.9 读出：完整 soft 通道合并池，三配置 CV 竞争。

背景：train_concept_lr.py 的 99 维冻结配方揭示此前合并池遗漏三类通道：
  1) P(直接) = p_yes × p_B（乘积交互通道，当年权重第 4）
  2) 样本级 S1A/S1B 六档概率（各 6 维）
  3) CAT 六通道 p_yes（当年权重第 2 的 CAT2A:p_yes +0.639）
且当年 alpha 网格下探至 1e-5（本版网格同步扩展）。

v3 特征筛选（train 冻结，预登记）：
  概念通道（v3p_/v3pd_/v3d_）：方差>0 → 经验极性对齐 → 单特征 AUC>=0.55
    → 与 v6 对齐条目 |r|>=0.9 去冗余
  样本通道（s1a_/s1b_/cat_）：仅方差>0（六档相邻档单特征 AUC 无意义，整块保留）

配置：
  H1  v3_soft 单独              —— 检验 v6 基座是否稀释 v3 信息
  H2  基座162 + v3p             —— v6.7 F2 复刻（扩展 alpha 网格锚点）
  H3  基座162 + v3_soft 全通道  —— 主候选
5 折 CV × 3 种子（τ@宏F1）竞争，OOF AUC 并报；胜者 5 种子 test 各评一次，
τ@宏F1（主口径）与 τ@有毒F1（判死线对照口径）双报告。

预登记判读：若 H1 OOF AUC ≥ H3 + 0.002 → v6 基座稀释 v3 信息，转向 v3 为核心。

产物：results/v6/readout_v69/{metrics.txt, cv_competition.csv,
      v3_feature_audit.csv, test_predictions.csv}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
TAU_GRID = np.round(np.arange(0.30, 0.701, 0.01), 3)
CV_SEEDS = [42, 43, 44]
FINAL_SEEDS = [0, 1, 2, 3, 4]
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
V3_MIN_AUC = 0.55
V3_MAX_CORR = 0.90


class Pipeline:
    def __init__(self, model):
        self.scaler = StandardScaler()
        self.model = model

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def cv_select(X, y, seed, tag=""):
    """alpha 扫描（τ@宏F1）→ OOF 概率。返回 (alpha, tau_macro, oof)。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in tqdm(ALPHA_GRID, desc=f"    alpha[{tag} s{seed}]", leave=False, dynamic_ncols=True):
        f1s = []
        for tr, va in cv.split(X, y):
            m = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000))
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[va])[:, 1]
            f1s.append(max(f1_score(y[va], (prob >= t).astype(int), average="macro")
                           for t in TAU_GRID))
        f = float(np.mean(f1s))
        if f > best_f:
            best_a, best_f = a, f
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / best_a, max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    tau = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return best_a, float(tau), oof


def construct_features(df, freeze, A, Xr):
    """v6.5 宏层 21 维（与 readout_v67 同口径）。"""
    wmap = freeze.get("item_weights", {})
    means = {}
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"]:
            continue
        kept = [c for c in kept if c in A.columns]
        if cid == "D2":
            hits = A[D2_TIERS].values >= 0.5
            score = np.zeros(len(df))
            for i in range(len(df)):
                h = hits[i]
                score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
            means[cid] = score
            continue
        ws = np.array([wmap.get(c, 1.0 / len(kept)) for c in kept])
        ws = ws / ws.sum()
        means[cid] = (A[kept].values * ws).sum(axis=1)
    v1 = np.zeros(len(df))
    for cid, kept in freeze["items_per_construct"].items():
        pos = [c for c in kept if c in Xr.columns and freeze["item_polarity"].get(c, 1) == 1]
        neg = [c for c in kept if c in Xr.columns and freeze["item_polarity"].get(c, 1) == -1]
        if pos and neg:
            v1 += Xr[pos].mean(axis=1).values * Xr[neg].mean(axis=1).values
    v1 /= max(len(freeze["items_per_construct"]), 1)
    m, sd = Xr.mean(axis=1).values, Xr.std(axis=1).values
    v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0
    out = pd.DataFrame({"id": df["id"].values})
    for cid, v in means.items():
        out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
    out["V1"], out["V2"] = v1, v2
    return out


def main():
    ap = argparse.ArgumentParser(description="v6.9 读出：完整 soft 通道合并池")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True, help="parse_v3_soft.py 产物（train）")
    ap.add_argument("--v3_test", required=True, help="parse_v3_soft.py 产物（test）")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/readout_v69")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all(), "train id 未对齐"
    assert (te["id"].values == v3te["id"].values).all(), "test id 未对齐"
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    # ---- 基座 162（v6.5 复刻）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    mtr = construct_features(tr, freeze, Atr, Xtr_raw)
    mte = construct_features(te, freeze, Ate, Xte_raw)
    mcols = [c for c in mtr.columns if c != "id"]
    base_tr = np.hstack([mtr[mcols].values, Atr[kept_items].values])
    base_te = np.hstack([mte[mcols].values, Ate[kept_items].values])

    # ---- v3 soft 特征筛选（train 冻结）----
    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]

    var_ok_c = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok_c:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok_c if auc[c] >= V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    Fte_c = v3te[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
            Fte_c[c] = 1.0 - v3te[c]
    v6_ref = Atr[kept_items]
    kept_c, redund_c = [], []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, v6_ref.values.T)[0, 1:])
        if np.nanmax(r) >= V3_MAX_CORR:
            redund_c.append(c)
        else:
            kept_c.append(c)
    var_ok_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    kept_s = var_ok_s

    Ftr = pd.concat([Ftr_c[kept_c], v3tr[kept_s]], axis=1)
    Fte = pd.concat([Fte_c[kept_c], v3te[kept_s]], axis=1)
    kept_all = kept_c + kept_s
    n_p = sum(c.startswith("v3p_") for c in kept_c)
    n_pd = sum(c.startswith("v3pd_") for c in kept_c)
    n_d = sum(c.startswith("v3d_") for c in kept_c)
    print(f"v3 soft: 总 {len(all_cols)} → 概念通道筛选保留 {len(kept_c)} "
          f"(p={n_p} pd={n_pd} d={n_d}, 冗余剔除 {len(redund_c)}) "
          f"+ 样本通道 {len(kept_s)}")

    extra = pd.DataFrame({
        "feature": var_ok_c,
        "aligned_auc": [auc[c] for c in var_ok_c],
        "polarity": [pol[c] for c in var_ok_c],
        "kept_concept": [c in kept_c for c in var_ok_c],
        "redundant": [c in redund_c for c in var_ok_c],
    }).sort_values("aligned_auc", ascending=False)
    extra.to_csv(out / "v3_feature_audit.csv", index=False, encoding="utf-8-sig")

    p_only = [c for c in kept_c if c.startswith("v3p_")]
    cfg = {
        "H1_v3soft单独": (Ftr[kept_all].values, Fte[kept_all].values, kept_all),
        "H2_基座+v3p": (np.hstack([base_tr, Ftr[p_only].values]),
                        np.hstack([base_te, Fte[p_only].values]),
                        mcols + kept_items + p_only),
        "H3_基座+soft全通道": (np.hstack([base_tr, Ftr.values]),
                              np.hstack([base_te, Fte.values]),
                              mcols + kept_items + kept_all),
    }

    lines = ["=" * 72, "v6.9 读出（完整 soft 通道合并池，三配置 CV 竞争，τ@宏F1）", "=" * 72,
             f"v3 soft: 总 {len(all_cols)} → 概念通道 {len(kept_c)} "
             f"(p={n_p} pd={n_pd} d={n_d}, 冗余剔除 {len(redund_c)}) + 样本通道 {len(kept_s)}",
             "维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, (X, _, _) in cfg.items()),
             ""]

    rows = []
    for tag, (X, _, _) in tqdm(cfg.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, oof = cv_select(X, ytr, seed, tag)
            oof_auc = roc_auc_score(ytr, oof)
            oof_f1 = f1_score(ytr, (oof >= t).astype(int), average="macro")
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "oof_macro_f1": oof_f1, "oof_auc": oof_auc, "oof_tau": t})
            lines.append(f"  {tag:<20} seed{seed}  alpha={a:<8} OOF宏F1={oof_f1:.4f}  "
                         f"OOF AUC={oof_auc:.4f}  τ*={t:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    winner = mean_f1.idxmax()
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"胜者配置: {winner}", ""]

    Xw_tr, Xw_te, w_names = cfg[winner]

    per_seed, coefs = [], []
    prob = pred = None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau_m, oof = cv_select(Xw_tr, ytr, seed, winner)
        tau_t = max(TAU_GRID, key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
        prob = model.predict_proba(Xw_te)[:, 1]
        pred_m = (prob >= tau_m).astype(int)
        pred_t = (prob >= tau_t).astype(int)
        per_seed.append({
            "seed": seed, "alpha": a, "tau_macro": tau_m, "tau_toxic": tau_t,
            "macro_f1": f1_score(yte, pred_m, average="macro"),
            "toxic_f1": f1_score(yte, pred_m),
            "benign_f1": f1_score(yte, pred_m, pos_label=0),
            "auc": roc_auc_score(yte, prob),
            "toxic_f1_t": f1_score(yte, pred_t),
            "macro_f1_t": f1_score(yte, pred_t, average="macro"),
        })
        coefs.append(model.model.coef_[0])

    per = pd.DataFrame(per_seed)
    lines += ["test 评估（胜者配置，5 种子，各评一次）:",
              f"  {'seed':<6}{'alpha':<10}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}{'无毒F1':<9}{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}",
              "-" * 72]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['alpha']:<10}{r['tau_macro']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}"
                     f"{r['benign_f1']:<9.4f}{r['auc']:<8.4f}{r['tau_toxic']:<7.2f}"
                     f"{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 72,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}"
              f"（判死线对照口径，历史 0.8224）",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v6.5 混合 0.7889±0.0055 / v6.7 合并 0.7960±0.0028 (有毒 0.8133)",
              "      v3 concept_lr 有毒F1 0.8224 / AUC 0.8849   判死线（有毒F1口径）0.8331"]

    coef = pd.Series(coefs[0], index=w_names)
    nz = coef[coef.abs() > 1e-6]
    v3_nz = nz[nz.index.str.startswith(("v3p_", "v3pd_", "v3d_", "s1a_", "s1b_", "cat_"))]
    lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0），其中 v3 特征非零 {len(v3_nz)}",
              "", "|coef| 前 20："]
    for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<14} {w:+.3f}")

    te_out = te[["id", "toxic"]].copy()
    te_out["prob"] = prob
    te_out["pred"] = pred_m
    te_out.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
