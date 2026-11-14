# -*- coding: utf-8 -*-
"""v6.4 诊断实验：构念层 vs 条目层平铺的信息量对比（train CV 内完成，不碰 test）。

回答用户问题：概念向量是"信息不足"还是"未充分利用"？
同一 CV 协议（5 折 × 3 种子，L1 LR，τ@宏F1）下对比三种表示的 OOF 指标：
  C. 构念层 21 维（v6.4 对齐加权均值 + V1 + V2）—— 当前架构
  F1a. 条目层平铺：141 条保留条目（极性对齐）
  F1b. 条目层平铺：全 152 条（极性对齐）
若平铺 OOF AUC 显著高于构念层 → 信息存在、构念均值是瓶颈（架构决策问题）；
若平铺 ≈ 构念层 → v6 题池本身的信息上限低于 v3（题池回炉问题）。

产物：results/v6/diag_flat_v64/metrics.txt（含条目平铺权重的短表单候选清单）
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
    return best_a, float(best_t), best_f, float(roc_auc_score(y, oof)), oof


def construct_features(df, freeze):
    """v6.4 构念层 21 维（对齐加权均值 + V1 + V2），与 readout_v64 同口径。"""
    y = df["toxic"].astype(int).values
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    X = df[item_cols]
    pol = freeze["item_polarity"]
    A = X.copy()
    for c in item_cols:
        if pol.get(c, 1) == -1:
            A[c] = 1.0 - X[c]
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
        pos = [c for c in kept if c in X.columns and pol.get(c, 1) == 1]
        neg = [c for c in kept if c in X.columns and pol.get(c, 1) == -1]
        if pos and neg:
            v1 += X[pos].mean(axis=1).values * X[neg].mean(axis=1).values
    v1 /= max(len(freeze["items_per_construct"]), 1)
    m, sd = X.mean(axis=1).values, X.std(axis=1).values
    v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0
    out = pd.DataFrame({"id": df["id"]})
    for cid, v in means.items():
        out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
    out["V1"], out["V2"] = v1, v2
    return out.drop(columns=["id"]).values


def main():
    ap = argparse.ArgumentParser(description="诊断：构念层 vs 条目层平铺（train CV）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--output_dir", default="results/v6/diag_flat_v64")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    items = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    df = pd.read_csv(args.train_matrix)
    y = df["toxic"].astype(int).values
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    X = df[item_cols]
    pol = freeze["item_polarity"]
    A = X.copy()
    for c in item_cols:
        if pol.get(c, 1) == -1:
            A[c] = 1.0 - X[c]

    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in A.columns]
    variants = [("构念层21维(当前)", construct_features(df, freeze)),
                ("条目平铺141(对齐保留)", A[kept_items].values),
                ("条目平铺152(对齐全池)", A[item_cols].values)]

    lines = ["=" * 72, "诊断：构念层 vs 条目层平铺（train CV × 3 种子，不碰 test）",
             "=" * 72, "",
             f"{'表示':<24}{'维度':<7}{'OOF宏F1':<10}{'OOF宏F1 std':<13}{'OOF AUC':<10}{'OOF AUC std':<12}",
             "-" * 76]
    results, best_coefs = [], {}
    for tag, F in tqdm(variants, desc="表示对比", dynamic_ncols=True):
        rows = []
        for seed in CV_SEEDS:
            a, t, f1, auc, oof = cv_select(F, y, seed)
            rows.append((a, t, f1, auc))
            best_coefs.setdefault(tag, (a, F))
        f1s = [r[2] for r in rows]
        aucs = [r[3] for r in rows]
        results.append((tag, F.shape[1], np.mean(f1s), np.std(f1s),
                        np.mean(aucs), np.std(aucs)))
        lines.append(f"{tag:<24}{F.shape[1]:<7}{np.mean(f1s):<10.4f}{np.std(f1s):<13.4f}"
                     f"{np.mean(aucs):<10.4f}{np.std(aucs):<12.4f}")

    lines += ["", "-" * 76,
              "判读：",
              "  若平铺 OOF AUC 高于构念层 ≥0.01 → 信息存在，构念均值是瓶颈（架构决策）",
              "  若平铺 ≈ 构念层 → v6 题池信息上限低于 v3 平铺 0.8849（题池回炉）",
              "  参考：v3 条目级平铺（97 概念）test AUC 0.8849 / 有毒F1 0.8224"]

    # 平铺 141 的 L1 权重 → 心理测量学意义上的"短表单"候选清单
    a, _, _, _, _ = cv_select(variants[1][1], y, 42)
    m = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                    C=1.0 / a, max_iter=2000)).fit(variants[1][1], y)
    coef = pd.Series(m.model.coef_[0], index=kept_items)
    nz = coef[coef.abs() > 1e-6].reindex(
        coef[coef.abs() > 1e-6].abs().sort_values(ascending=False).index)
    lines += ["", f"条目平铺141 非零权重 {len(nz)}/{len(kept_items)}（alpha={a}，权重降序前 40）："]
    for name, w in nz.head(40).items():
        lines.append(f"  {name:<10} {w:+.3f}")

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
