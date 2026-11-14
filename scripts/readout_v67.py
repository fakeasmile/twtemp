# -*- coding: utf-8 -*-
"""v6.7 读出：合并题池 + 宏微混合，三配置 CV 竞争，胜者 test 评估一次。

背景：diag_union 显示合并池（v6 条目 + v3 stage1 概念）宏 F1 +0.0062，
AUC 压线（0.8842 vs 门槛 0.8845）。且 v3 概念池只挖了 stage1 一层，
v3 当年 0.8224 读出用的是 p_yes + p_A(直接性) + value 三通道完整特征。

配置（宏微混合基座 = v6.5 的 21 维宏层 + 141 条对齐条目）：
  F1  基座                      —— v6.5 复刻
  F2  基座 + v3 stage1 特征     —— diag_union 合并池
  F3  基座 + v3 三通道完整特征  —— 主候选
v3 特征处理：train 上方差>0 过滤 + 经验极性对齐（AUC>=0.5 原始，否则翻转），
极性与列清单冻结后投影到 test。全部选择在 train 5 折 CV × 3 种子（τ@宏F1），
胜者 5 种子 test 各评一次。宏 F1 主口径，有毒 F1 并报。

产物：results/v6/readout_v67/{metrics.txt, cv_competition.csv, test_predictions.csv}
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

ALPHA_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
TAU_GRID = np.round(np.arange(0.30, 0.701, 0.01), 3)
CV_SEEDS = [42, 43, 44]
FINAL_SEEDS = [0, 1, 2, 3, 4]
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]


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
    best_t = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return best_a, float(best_t), best_f


def empirical_polarity(X, y):
    pol = {}
    for c in X.columns:
        v = X[c].values
        pol[c] = 1 if (np.std(v) < 1e-9 or roc_auc_score(y, v) >= 0.5) else -1
    return pol


def align(X, pol):
    A = X.copy()
    for c in X.columns:
        if pol[c] == -1:
            A[c] = 1.0 - X[c]
    return A


def v6_macro_features(df, freeze, A, Xr):
    """v6.5 宏层 21 维（与 readout_v65 同口径）。"""
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
    ap = argparse.ArgumentParser(description="v6.7 读出：合并题池三通道配置竞争")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True, help="parse_v3_full.py 产物（train）")
    ap.add_argument("--v3_test", required=True, help="parse_v3_full.py 产物（test）")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/readout_v67")
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

    # v6 对齐条目 + 宏层
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    mtr = v6_macro_features(tr, freeze, Atr, Xtr_raw)
    mte = v6_macro_features(te, freeze, Ate, Xte_raw)
    mcols = [c for c in mtr.columns if c != "id"]

    # v3 三通道特征：train 冻结（方差>0 过滤 + 经验极性），投影 test
    v3_cols = [c for c in v3tr.columns if c.startswith(("v3p_", "v3d_", "v3v_"))]
    var_ok = [c for c in v3_cols if v3tr[c].std() > 1e-9]
    pol3 = empirical_polarity(v3tr[var_ok], ytr)
    F3tr = align(v3tr[var_ok], pol3)
    F3te = align(v3te[var_ok], pol3)
    print(f"v3 特征: 总 {len(v3_cols)}，方差>0 {len(var_ok)}，其中极性翻转 "
          f"{sum(1 for v in pol3.values() if v == -1)}")

    base_tr = np.hstack([mtr[mcols].values, Atr[kept_items].values])
    base_te = np.hstack([mte[mcols].values, Ate[kept_items].values])
    s1_cols = [c for c in var_ok if c.startswith("v3p_")]
    cfg = {
        "F1_基座162": (base_tr, base_te, mcols + kept_items),
        "F2_基座+v3p": (np.hstack([base_tr, F3tr[s1_cols].values]),
                        np.hstack([base_te, F3te[s1_cols].values]),
                        mcols + kept_items + s1_cols),
        "F3_基座+v3全通道": (np.hstack([base_tr, F3tr.values]),
                             np.hstack([base_te, F3te.values]),
                             mcols + kept_items + var_ok),
    }

    lines = ["=" * 72, "v6.7 读出（合并题池，三配置 CV 竞争，τ@宏F1）", "=" * 72,
             "维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, (X, _, _) in cfg.items()), ""]
    rows = []
    for tag, (X, _, _) in tqdm(cfg.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, f1 = cv_select(X, ytr, seed, tag)
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "oof_macro_f1": f1, "oof_tau": t})
            lines.append(f"  {tag:<18} seed{seed}  alpha={a:<8} OOF宏F1={f1:.4f}  τ*={t:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    winner = mean_f1.idxmax()
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              f"胜者配置: {winner}", ""]

    Xw_tr, Xw_te, w_names = cfg[winner]

    per_seed, coefs = [], []
    prob = pred = None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau, _ = cv_select(Xw_tr, ytr, seed, winner)
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
        prob = model.predict_proba(Xw_te)[:, 1]
        pred = (prob >= tau).astype(int)
        per_seed.append({"seed": seed, "alpha": a, "tau": tau,
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "toxic_f1": f1_score(yte, pred),
                         "benign_f1": f1_score(yte, pred, pos_label=0),
                         "auc": roc_auc_score(yte, prob)})
        coefs.append(model.model.coef_[0])

    per = pd.DataFrame(per_seed)
    lines += ["test 评估（胜者配置，5 种子，各评一次）:",
              f"  {'seed':<6}{'alpha':<10}{'τ*':<7}{'宏F1':<9}{'有毒F1':<9}{'无毒F1':<9}{'AUC':<8}",
              "-" * 58]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['alpha']:<10}{r['tau']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}"
                     f"{r['benign_f1']:<9.4f}{r['auc']:<8.4f}")
    lines += ["-" * 58,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v6.5 混合 0.7889±0.0055 / v6.6 剪枝 0.7859±0.0032 / v3平铺 0.8047",
              "      判死线（有毒F1口径）0.8331"]

    coef = pd.Series(coefs[0], index=w_names)
    nz = coef[coef.abs() > 1e-6]
    v3_nz = nz[nz.index.str.startswith(("v3p_", "v3d_", "v3v_"))]
    lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0），其中 v3 特征非零 {len(v3_nz)}",
              "", "|coef| 前 20："]
    for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<14} {w:+.3f}")

    te_out = te[["id", "toxic"]].copy()
    te_out["prob"] = prob
    te_out["pred"] = pred
    te_out.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
