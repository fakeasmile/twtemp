# -*- coding: utf-8 -*-
"""v6.6 读出：构念剪枝配置竞争（新建脚本，单教师）。

依据 diag_block_validity 的 leave-one-block-out Δ 值做三种剪枝配置竞争：
  FULL19  —— 全量 19 构念（v6.5 基线）
  PRUNE6  —— 剔除 6 个负 Δ 构念（实测噪声源：挖掉后性能反升）
  PRUNE10 —— 仅保留 Δ≥0.0005 的构念（预登记标准，最激进）
三配置 train 5 折 CV × 3 种子竞争（宏 F1 口径，test 零参与），
胜者冻结 → 5 种子 test 评估一次。剪枝不掉点才可能上 test。

剪枝实现与 diag_block_validity 一致：被剔构念的微层条目全部移除、
宏层构念分置 0；V1/V2 不变。alpha 网格加密至 0.3。

产物：results/v6/readout_v66/{metrics.txt, cv_competition.csv, test_predictions.csv}
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


def construct_features(df, freeze, A, Xr, dropped):
    """宏层：alive 构念的加权构念分 T 分（dropped 内构念置 0）+ V1 + V2。"""
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
        if cid in dropped:
            out[f"{cid}_mean"] = 0.0
            continue
        out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
    out["V1"], out["V2"] = v1, v2
    return out


def build_sets(train, test, freeze, keep_sets):
    """为每个保留构念集合构建 train/test 特征矩阵。"""
    Xtr_raw = train[[c for c in train.columns if c not in ("id", "toxic", "content")]]
    Xte_raw = test[[c for c in test.columns if c not in ("id", "toxic", "content")]]
    Atr = Xtr_raw.copy()
    Ate = Xte_raw.copy()
    for c in Xtr_raw.columns:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    item_of = {}
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            item_of[c] = cid

    feats = {}
    for tag, dropped in keep_sets.items():
        mtr = construct_features(train, freeze, Atr, Xtr_raw, dropped)
        mte = construct_features(test, freeze, Ate, Xte_raw, dropped)
        mcols = [c for c in mtr.columns if c != "id"]
        micro = [c for c, cc in item_of.items()
                 if c in Atr.columns and cc not in dropped]
        Xtr = np.hstack([mtr[mcols].values, Atr[micro].values])
        Xte = np.hstack([mte[mcols].values, Ate[micro].values])
        feats[tag] = (Xtr, Xte, mcols + micro)
    return feats


def main():
    ap = argparse.ArgumentParser(description="v6.6 读出：构念剪枝配置竞争")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--block_validity_csv", default="results/v6/diag_block_validity/block_validity.csv")
    ap.add_argument("--output_dir", default="results/v6/readout_v66")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    bv = pd.read_csv(args.block_validity_csv)
    alive = set(freeze["constructs_alive"])
    bv = bv[bv["construct"].isin(alive)]
    neg = set(bv.loc[bv["delta"] < 0, "construct"])
    core = set(bv.loc[bv["delta"] >= 0.0005, "construct"])
    keep_sets = {
        "FULL19(基线)": set(),
        "PRUNE6(去负Δ)": neg,
        "PRUNE10(Δ>=0.0005)": alive - core,
    }

    train = pd.read_csv(args.train_matrix)
    test = pd.read_csv(args.test_matrix)
    ytr = train["toxic"].astype(int).values
    yte = test["toxic"].astype(int).values
    feats = build_sets(train, test, freeze, keep_sets)

    lines = ["=" * 72, "v6.6 读出（构念剪枝配置竞争，τ@宏F1）", "=" * 72]
    for tag, dropped in keep_sets.items():
        lines.append(f"  {tag:<22} 剔除 {len(dropped)} 构念: {sorted(dropped) if dropped else '无'}")
    lines.append("")

    comp_rows = []
    for tag, (Xtr, Xte, names) in tqdm(feats.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, f1 = cv_select(Xtr, ytr, seed, tag)
            comp_rows.append({"config": tag, "seed": seed, "alpha": a,
                              "oof_macro_f1": f1, "oof_tau": t})
            lines.append(f"  {tag:<22} seed{seed}  alpha={a:<8} OOF宏F1={f1:.4f}  τ*={t:.2f}")
    comp = pd.DataFrame(comp_rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    winner = mean_f1.idxmax()
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              f"胜者配置: {winner}", ""]

    Xw_tr, Xw_te, w_names = feats[winner]

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
              "历史对照: v6.1 0.7696 / v6.4 0.7815 / v6.5 混合 0.7889±0.0055 / v3平铺 0.8047",
              "          判死线（有毒F1口径）0.8331"]

    coef = pd.Series(coefs[0], index=w_names)
    nz = coef[coef.abs() > 1e-6]
    lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0，|coef| 前 20）："]
    for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<14} {w:+.3f}")

    te_out = test[["id", "toxic"]].copy()
    te_out["prob"] = prob
    te_out["pred"] = pred
    te_out.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
