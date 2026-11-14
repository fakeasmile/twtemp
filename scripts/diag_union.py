# -*- coding: utf-8 -*-
"""合并题池诊断：v6 条目池 vs v3 概念池 vs 合并池（train 内 CV，test 零参与）。

背景：单教师 v6 池（141 对齐条目）平铺 OOF AUC 0.8795，v3 概念池实测上限
test AUC 0.8849 / 有毒 F1 0.8224。本诊断检验两池合并是否有独占互补信息：
  C1. v6-141 条目（对齐，v6.4 freeze）
  C2. v3-概念（解析自 rings JSON，train 上经验极性对齐）
  C3. 合并池（C1 + C2）
协议：alpha 网格 [0.0003..0.3] 密扫，5 折 CV × 3 种子，τ@宏F1，
报告 OOF 宏 F1 与 OOF AUC。全部 train 内完成，不碰 test。
胜者若为合并池 → 下一步写 readout_v67 冻结合并池 test 评估一次。
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


class Pipeline:
    def __init__(self, model):
        self.scaler = StandardScaler()
        self.model = model

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def cv_oof(X, y, seed, tag=""):
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
    return {
        "macro_f1": f1_score(y, (oof >= best_t).astype(int), average="macro"),
        "auc": roc_auc_score(y, oof),
        "tau": float(best_t), "alpha": best_a,
    }


def empirical_polarity(X, y):
    """train 上按 AUC 裁决每列极性：AUC>=0.5 用原始值，否则取 1-值。"""
    pol = {}
    for c in X.columns:
        v = X[c].values
        if np.std(v) < 1e-9:
            pol[c] = 1
            continue
        auc = roc_auc_score(y, v)
        pol[c] = 1 if auc >= 0.5 else -1
    return pol


def align(X, pol):
    A = X.copy()
    for c in X.columns:
        if pol[c] == -1:
            A[c] = 1.0 - X[c]
    return A


def main():
    ap = argparse.ArgumentParser(description="合并题池诊断（v6 条目 vs v3 概念 vs 合并）")
    ap.add_argument("--train_matrix", required=True, help="v6 条目矩阵（train）")
    ap.add_argument("--v3_train_matrix", required=True, help="parse_v3_matrix.py 产物（train）")
    ap.add_argument("--freeze", required=True, help="v6_4_freeze.json")
    ap.add_argument("--output_dir", default="results/v6/diag_union")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    v6 = pd.read_csv(args.train_matrix)
    v3 = pd.read_csv(args.v3_train_matrix)
    merged = v6[["id", "toxic"]].merge(
        v3.drop(columns=["toxic"]), on="id", how="inner", suffixes=("", "_v3"))
    assert len(merged) == len(v6), f"id 对齐失败: v6 {len(v6)} vs 合并 {len(merged)}"
    assert (merged["toxic"] == merged["toxic_v3"]).all() if "toxic_v3" in merged else True
    y = merged["toxic"].astype(int).values

    # C1: v6 条目（对齐 + freeze 保留清单）
    item_cols_all = [c for c in v6.columns if c not in ("id", "toxic", "content")]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in item_cols_all]
    X6_raw = v6[kept_items]
    pol6 = empirical_polarity(X6_raw, y)
    C1 = align(X6_raw, pol6)

    # C2: v3 概念（对齐）
    v3_cols = [c for c in v3.columns if c.startswith("v3_")]
    n_const_cols = [c for c in v3_cols if merged[c].std() > 1e-9]
    X3_raw = merged[n_const_cols]
    print(f"v3 概念列: {len(v3_cols)}，其中方差>0: {len(n_const_cols)}")
    pol3 = empirical_polarity(X3_raw, y)
    C2 = align(X3_raw, pol3)

    C3 = pd.concat([C1.reset_index(drop=True), C2.reset_index(drop=True)], axis=1)

    configs = [("C1_v6条目141", C1), ("C2_v3概念", C2), ("C3_合并池", C3)]

    lines = ["=" * 72, "合并题池诊断（train CV × 3 种子，τ@宏F1，test 零参与）", "=" * 72,
             f"维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, X in configs), ""]
    rows = []
    for tag, X in tqdm(configs, desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            r = cv_oof(X.values, y, seed, tag)
            rows.append({"config": tag, "seed": seed, **r})
            lines.append(f"  {tag:<14} seed{seed}  alpha={r['alpha']:<8} "
                         f"OOF宏F1={r['macro_f1']:.4f}  OOF AUC={r['auc']:.4f}  τ*={r['tau']:.2f}")
    res = pd.DataFrame(rows)
    agg = res.groupby("config")[["macro_f1", "auc"]].mean()
    lines += ["", "CV 均值:"]
    for tag, r in agg.iterrows():
        lines.append(f"  {tag:<14} 宏F1={r['macro_f1']:.4f}  AUC={r['auc']:.4f}")
    winner = agg["macro_f1"].idxmax()
    lines += ["", f"胜者配置: {winner}",
              "判读（预登记）: 合并池 OOF AUC > v6 单池 0.8795 + 0.005 → v3 概念有独占信息，"
              "下一步冻结合并池做 test 评估；否则合并无增益，回到 v6 池内挖掘"]

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    res.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
