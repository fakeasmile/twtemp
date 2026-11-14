# -*- coding: utf-8 -*-
"""v6.4 读出：极性对齐后的构念特征，三配置 CV 竞争（新建脚本）。

消费 analyze_items_v64.py 的 v6_4_freeze.json（经验极性 + 信号权重）。
配置竞争（train 5 折 CV × 3 种子，宏 F1 口径，test 零参与）：
  A. 对齐等权   19 构念均值（D2=Guttman）+ V1 + V2 = 21 维
  B. 对齐加权   信号加权均值 + V1 + V2 = 21 维
  C. 对齐加权+交互  B + 构念均值两两乘积（171 维）= 192 维
胜者冻结 → 5 种子 test 评估一次；宏 F1 为主口径、有毒 F1 并报。
V1 重定义：按经验极性分组的构念内矛盾度（原始 p 均值乘积）。

产物：results/v6/readout_v64/{metrics.txt, test_predictions.csv}
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


def guttman_score(M):
    hits = M >= 0.5
    score = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        h = hits[i]
        score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
    return score


def t_transform(scores, cols, exclude=("id", "toxic")):
    t_cols = [c for c in cols if c not in exclude]
    out = scores[["id"]].copy()
    for c in t_cols:
        mu, sd = scores[c].mean(), scores[c].std()
        out[c] = 50.0 + 10.0 * (scores[c] - mu) / (sd if sd > 1e-9 else 1.0)
    return out


def add_interactions(scores):
    """仅对 19 个构念均值列做两两乘积（V1/V2 不参与，保持 171 交互维）。"""
    cols = [c for c in scores.columns if c.endswith("_mean")]
    Z = StandardScaler().fit_transform(scores[cols].values)
    names, feats = [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            names.append(f"{cols[i]}x{cols[j]}")
            feats.append(Z[:, i] * Z[:, j])
    inter = pd.DataFrame(np.column_stack(feats), columns=names)
    out = pd.concat([scores[["id"]].reset_index(drop=True),
                     scores[cols].reset_index(drop=True),
                     inter], axis=1)
    return out, cols


def build_features(df, freeze):
    """按冻结规则构建三个配置的特征（全部构念层）。"""
    y = df["toxic"].astype(int).values
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    X = df[item_cols]
    pol = freeze["item_polarity"]
    A = X.copy()
    for c in item_cols:
        if pol.get(c, 1) == -1:
            A[c] = 1.0 - X[c]

    means_eq, means_w = {}, {}
    alive = freeze["constructs_alive"]
    wmap = freeze.get("item_weights", {})
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in alive:
            continue
        kept = [c for c in kept if c in A.columns]
        if cid == "D2":
            means_eq[cid] = guttman_score(A[D2_TIERS].values)
            means_w[cid] = means_eq[cid]
            continue
        means_eq[cid] = A[kept].mean(axis=1).values
        ws = np.array([wmap.get(c, 1.0 / len(kept)) for c in kept])
        if abs(ws.sum() - 1.0) > 1e-6:
            ws = ws / ws.sum()
        means_w[cid] = (A[kept].values * ws).sum(axis=1)

    # V1：按经验极性分组的构念内矛盾度（原始 p 分组均值乘积）
    v1 = np.zeros(len(df))
    for cid, kept in freeze["items_per_construct"].items():
        pos = [c for c in kept if c in X.columns and pol.get(c, 1) == 1]
        neg = [c for c in kept if c in X.columns and pol.get(c, 1) == -1]
        if pos and neg:
            v1 += X[pos].mean(axis=1).values * X[neg].mean(axis=1).values
    v1 /= max(len(freeze["items_per_construct"]), 1)

    # V2：全条目原始 p 方差极低且均值极端
    m, sd = X.mean(axis=1).values, X.std(axis=1).values
    v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0

    def assemble(means):
        s = pd.DataFrame({"id": df["id"]})
        for cid, v in means.items():
            s[f"{cid}_mean"] = v
        s["V1"], s["V2"] = v1, v2
        return t_transform(s, list(s.columns), exclude=("id",))

    base_eq = assemble(means_eq)
    base_w = assemble(means_w)
    base_w_int, _ = add_interactions(base_w)
    return base_eq, base_w, base_w_int


def cv_select(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in tqdm(ALPHA_GRID, desc="    alpha", leave=False, dynamic_ncols=True):
        f1s = []
        for tr, va in cv.split(X, y):
            pipe = Pipeline(StandardScaler, LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0 / a, max_iter=2000))
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[va])[:, 1]
            f1s.append(max(f1_score(y[va], (prob >= t).astype(int), average="macro")
                           for t in TAU_GRID))
        f = float(np.mean(f1s))
        if f > best_f:
            best_a, best_f = a, f
    # 最优 alpha 下的 OOF 概率与 τ*
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        pipe = Pipeline(StandardScaler, LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0 / best_a, max_iter=2000))
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
    best_t = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return best_a, float(best_t), best_f, oof


class Pipeline:
    def __init__(self, scaler_cls, model):
        self.scaler = scaler_cls()
        self.model = model

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def main():
    ap = argparse.ArgumentParser(description="v6.4 读出：三配置 CV 竞争（宏 F1 口径）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/readout_v64")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    ytr = tr["toxic"].astype(int).values
    base_eq_tr, base_w_tr, base_w_int_tr = build_features(tr, freeze)
    base_eq_te, base_w_te, base_w_int_te = build_features(te, freeze)

    configs = [("对齐等权", base_eq_tr, base_eq_te),
               ("对齐加权", base_w_tr, base_w_te),
               ("对齐加权+交互", base_w_int_tr, base_w_int_te)]

    # ---- 配置竞争（train CV × 3 种子，test 零参与）----
    lines = ["=" * 72, "v6.4 读出（经验极性对齐 + 三配置 CV 竞争，宏 F1 口径）",
             "=" * 72,
             f"维度: " + ", ".join(f"{tag}={X.shape[1]-1}" for tag, X, _ in configs), ""]
    comp_rows = []
    for tag, X, _ in tqdm(configs, desc="配置竞争", dynamic_ncols=True):
        Xtr = X.drop(columns=["id"]).values
        for seed in CV_SEEDS:
            a, t, f1, _ = cv_select(Xtr, ytr, seed)
            comp_rows.append({"config": tag, "seed": seed, "alpha": a,
                              "oof_macro_f1": f1, "oof_tau": t})
            lines.append(f"  {tag:<12} seed{seed}  alpha={a:<8} OOF宏F1={f1:.4f}  τ*={t:.2f}")

    comp = pd.DataFrame(comp_rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    winner = mean_f1.idxmax()
    lines += ["", f"CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              f"胜者配置: {winner}", ""]

    win_tr = next(X for tag, X, _ in configs if tag == winner)
    win_te = next(X for tag, _, X in configs if tag == winner)
    feat_cols = [c for c in win_tr.columns if c != "id"]
    Xtr_all, Xte_all = win_tr[feat_cols].values, win_te[feat_cols].values
    yte = te["toxic"].astype(int).values

    # ---- 胜者 5 种子 test 评估（各评一次）----
    per_seed, coefs, best_model = [], [], None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau, _, _ = cv_select(Xtr_all, ytr, seed)
        model = Pipeline(StandardScaler, LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0 / a, max_iter=2000))
        model.fit(Xtr_all, ytr)
        prob = model.predict_proba(Xte_all)[:, 1]
        pred = (prob >= tau).astype(int)
        per_seed.append({
            "seed": seed, "alpha": a, "tau": tau,
            "macro_f1": f1_score(yte, pred, average="macro"),
            "toxic_f1": f1_score(yte, pred),
            "benign_f1": f1_score(yte, pred, pos_label=0),
            "auc": roc_auc_score(yte, prob),
        })
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
              "历史对照: v6.1 宏F1 0.7696 / v6.2 0.7647 / v6.3 0.7828±0.0029",
              "          判死线（有毒F1口径）0.8331"]

    # 权重表（种子 0 模型，|coef| 前 25）
    coef = pd.Series(coefs[0], index=feat_cols)
    nz = coef[coef.abs() > 1e-6]
    lines += ["", f"非零权重 {len(nz)}/{len(feat_cols)}（|coef| 前 25，种子 0）："]
    for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(25).items():
        lines.append(f"  {name:<28} {w:+.3f}")

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
