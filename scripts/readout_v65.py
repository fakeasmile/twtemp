# -*- coding: utf-8 -*-
"""v6.5 读出：宏+微混合架构（新建脚本）。

诊断结论（results/v6/diag_flat_v64）：平铺 OOF AUC 0.8795 > 构念层 0.8674，
信息存在、构念均值是瓶颈（情形 A）。用户批准方案 3：
下游输入 = 宏（19 构念分 + V1 + V2）+ 微（141 条对齐条目分），L1 稀疏选择。

配置竞争（train 5 折 CV × 3 种子，τ@宏F1，test 零参与）：
  A. 构念层 21 维        —— v6.4 胜者基线
  B. 条目平铺 141 维     —— 诊断中的微层
  C. 混合 162 维         —— 宏+微（本方案主体）
胜者冻结 → 5 种子 test 评估一次；宏 F1 为主口径、有毒 F1 并报。
极性与权重沿用 analyze_items_v64.py 的 v6_4_freeze.json（不重裁决）。

产物：results/v6/readout_v65/{metrics.txt, cv_competition.csv, test_predictions.csv}
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

ALPHA_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
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


def cv_select(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in tqdm(ALPHA_GRID, desc="    alpha", leave=False, dynamic_ncols=True):
        f1s = []
        for tr, va in cv.split(X, y):
            m = Pipeline(LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0 / a, max_iter=2000))
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[va])[:, 1]
            f1s.append(max(f1_score(y[va], (prob >= t).astype(int), average="macro")
                           for t in TAU_GRID))
        f = float(np.mean(f1s))
        if f > best_f:
            best_a, best_f = a, f
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = Pipeline(LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0 / best_a, max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    best_t = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return best_a, float(best_t), best_f


def aligned_matrix(df, freeze):
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    X = df[item_cols]
    A = X.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            A[c] = 1.0 - X[c]
    return X, A, item_cols


def construct_features(df, freeze, A, X):
    """宏层 21 维：19 对齐加权构念分 T 分 + V1 + V2（与 readout_v64 同口径）。"""
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
        pos = [c for c in kept if c in X.columns and freeze["item_polarity"].get(c, 1) == 1]
        neg = [c for c in kept if c in X.columns and freeze["item_polarity"].get(c, 1) == -1]
        if pos and neg:
            v1 += X[pos].mean(axis=1).values * X[neg].mean(axis=1).values
    v1 /= max(len(freeze["items_per_construct"]), 1)
    m, sd = X.mean(axis=1).values, X.std(axis=1).values
    v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0
    out = pd.DataFrame({"id": df["id"].values})
    for cid, v in means.items():
        out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
    out["V1"], out["V2"] = v1, v2
    return out


def main():
    ap = argparse.ArgumentParser(description="v6.5 读出：宏+微混合（三配置 CV 竞争）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/readout_v65")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values
    Xtr_raw, Atr, item_cols = aligned_matrix(tr, freeze)
    Xte_raw, Ate, _ = aligned_matrix(te, freeze)

    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    macro_tr = construct_features(tr, freeze, Atr, Xtr_raw)
    macro_te = construct_features(te, freeze, Ate, Xte_raw)
    macro_cols = [c for c in macro_tr.columns if c != "id"]

    micro_tr = Atr[kept_items].reset_index(drop=True)
    micro_te = Ate[kept_items].reset_index(drop=True)

    cfg_macro_tr = macro_tr[macro_cols].values
    cfg_macro_te = macro_te[macro_cols].values
    cfg_micro_tr = micro_tr.values
    cfg_micro_te = micro_te.values
    cfg_hybrid_tr = np.hstack([cfg_macro_tr, cfg_micro_tr])
    cfg_hybrid_te = np.hstack([cfg_macro_te, cfg_micro_te])
    macro_names = macro_cols
    hybrid_names = macro_names + kept_items

    configs = [("构念层21维", cfg_macro_tr, cfg_macro_te, macro_names),
               ("条目平铺141", cfg_micro_tr, cfg_micro_te, kept_items),
               ("宏+微混合162", cfg_hybrid_tr, cfg_hybrid_te, hybrid_names)]

    # ---- 配置竞争（test 零参与）----
    lines = ["=" * 72, "v6.5 读出（宏+微混合，三配置 CV 竞争，τ@宏F1）", "=" * 72,
             f"维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, X, _, _ in configs), ""]
    comp_rows = []
    for tag, X, _, _ in tqdm(configs, desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, f1 = cv_select(X, ytr, seed)
            comp_rows.append({"config": tag, "seed": seed, "alpha": a,
                              "oof_macro_f1": f1, "oof_tau": t})
            lines.append(f"  {tag:<14} seed{seed}  alpha={a:<8} OOF宏F1={f1:.4f}  τ*={t:.2f}")

    comp = pd.DataFrame(comp_rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    winner = mean_f1.idxmax()
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              f"胜者配置: {winner}", ""]

    win_tr = next(X for tag, X, _, _ in configs if tag == winner)
    win_te = next(X for tag, _, X, _ in configs if tag == winner)
    win_names = next(n for tag, _, _, n in configs if tag == winner)

    # ---- 胜者 5 种子 test 评估（各评一次）----
    per_seed, coefs = [], []
    prob = pred = None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau, _ = cv_select(win_tr, ytr, seed)
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(win_tr, ytr)
        prob = model.predict_proba(win_te)[:, 1]
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
              f"  宏F1  mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  AUC   mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史对照: v6.1 宏F1 0.7696 / v6.4 0.7815±0.0013 / v3平铺 宏F1 0.8047",
              "          判死线（有毒F1口径）0.8331"]

    # 权重表（种子 0，宏列与条目列分节，|coef| 前 20 各取）
    coef = pd.Series(coefs[0], index=win_names)
    nz = coef[coef.abs() > 1e-6]
    lines += ["", f"非零权重 {len(nz)}/{len(win_names)}（种子 0）"]
    macro_nz = nz[nz.index.isin(macro_names)]
    item_nz = nz[~nz.index.isin(macro_names)]
    lines += [f"  宏层非零 {len(macro_nz)}/{len(macro_names)}，条目层非零 {len(item_nz)}/{len(kept_items)}",
              "", "宏层 |coef| 前 20："]
    for name, w in macro_nz.reindex(macro_nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<14} {w:+.3f}")
    lines += ["条目层 |coef| 前 20："]
    for name, w in item_nz.reindex(item_nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<10} {w:+.3f}")

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
