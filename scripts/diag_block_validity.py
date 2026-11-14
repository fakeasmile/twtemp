# -*- coding: utf-8 -*-
"""构念增量效度诊断：leave-one-block-out CV（MMPI 效标计分思想的构念层应用）。

问题背景：v6 构念源自人格/社会心理学量表（原测人，现改写为文本条目），
哪些构念对"有害言论"效标有独占增量、哪些测得再好也与效标无关，用数据裁决。

方法：以 v6.5 宏+微混合读出为基线，每次从特征集中挖掉一个构念的全部
条目（微层）与其构念分/V1 贡献（宏层），train 5 折 CV × 3 种子量 OOF 宏 F1。
Δ = 全量 OOF − 去块 OOF。Δ≈0 的构念按数据判死候选。

输出：
  results/v6/diag_block_validity/block_validity.csv
  results/v6/diag_block_validity/metrics.txt
test 零参与。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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


def cv_oof_macro_f1(X, y, seed, tag=""):
    """alpha 扫描 + OOF 宏 F1（与 readout_v65 同协议）。"""
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
    return f1_score(y, (oof >= best_t).astype(int), average="macro")


def main():
    ap = argparse.ArgumentParser(description="构念增量效度诊断（leave-one-block-out）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/diag_block_validity")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    df = pd.read_csv(args.train_matrix)
    y = df["toxic"].astype(int).values
    X_raw = df[[c for c in df.columns if c not in ("id", "toxic", "content")]]
    A = X_raw.copy()
    for c in X_raw.columns:
        if freeze["item_polarity"].get(c, 1) == -1:
            A[c] = 1.0 - X_raw[c]

    alive = freeze["constructs_alive"]
    item_of = {}  # 条目 -> 构念
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            item_of[c] = cid

    def build(exclude_cid=None):
        """宏+微混合特征（挖掉 exclude_cid 的微层条目；宏层保持 21 维不变，
        因为构念分本身携带信息，leave-one-block-out 的裁决对象是构念的全部证据）。
        但若 exclude_cid 给定，其构念分 T 分列同样置 0（等价于从宏层挖除该构念证据）。"""
        kept_all = [c for lst in freeze["items_per_construct"].values() for c in lst]
        micro = [c for c in kept_all if c in A.columns and item_of.get(c) != exclude_cid]
        macro = construct_features(df, freeze, A, X_raw, zero_cid=exclude_cid)
        mcols = [c for c in macro.columns if c != "id"]
        return np.hstack([macro[mcols].values, A[micro].values]), mcols + micro

    def construct_features(d, fz, Al, Xr, zero_cid=None):
        wmap = fz.get("item_weights", {})
        means = {}
        for cid, kept in fz["items_per_construct"].items():
            if cid not in fz["constructs_alive"]:
                continue
            kept = [c for c in kept if c in Al.columns]
            if cid == "D2":
                hits = Al[D2_TIERS].values >= 0.5
                score = np.zeros(len(d))
                for i in range(len(d)):
                    h = hits[i]
                    score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
                means[cid] = score
                continue
            ws = np.array([wmap.get(c, 1.0 / len(kept)) for c in kept])
            ws = ws / ws.sum()
            means[cid] = (Al[kept].values * ws).sum(axis=1)
        v1 = np.zeros(len(d))
        for cid, kept in fz["items_per_construct"].items():
            pos = [c for c in kept if c in Xr.columns and fz["item_polarity"].get(c, 1) == 1]
            neg = [c for c in kept if c in Xr.columns and fz["item_polarity"].get(c, 1) == -1]
            if pos and neg:
                v1 += Xr[pos].mean(axis=1).values * Xr[neg].mean(axis=1).values
        v1 /= max(len(fz["items_per_construct"]), 1)
        m, sd = Xr.mean(axis=1).values, Xr.std(axis=1).values
        v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0
        out = pd.DataFrame({"id": d["id"].values})
        for cid, v in means.items():
            if cid == zero_cid:
                out[f"{cid}_mean"] = 0.0
                continue
            out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
        out["V1"], out["V2"] = v1, v2
        return out

    # ---- 全量基线 ----
    X_full, names_full = build(None)
    base = float(np.mean([cv_oof_macro_f1(X_full, y, s, "基线") for s in
                          tqdm(CV_SEEDS, desc="  基线 CV", dynamic_ncols=True)]))
    print(f"全量基线 OOF宏F1 = {base:.4f}（{X_full.shape[1]} 维）")

    # ---- 逐构念挖除 ----
    rows = []
    for cid in tqdm(alive, desc="构念挖除", dynamic_ncols=True):
        X_sub, _ = build(exclude_cid=cid)
        f = float(np.mean([cv_oof_macro_f1(X_sub, y, s, cid) for s in
                           tqdm(CV_SEEDS, desc=f"  {cid} CV", leave=False, dynamic_ncols=True)]))
        rows.append({"construct": cid, "oof_macro_f1_without": f, "delta": base - f})
        print(f"  去掉 {cid:<3} OOF宏F1={f:.4f}  Δ={base - f:+.4f}")

    res = pd.DataFrame(rows).sort_values("delta", ascending=False)
    lines = ["=" * 64, "构念增量效度诊断（leave-one-block-out，train 内）", "=" * 64,
             f"全量基线 OOF宏F1 = {base:.4f}", "",
             f"{'构念':<6}{'去掉后OOF宏F1':<16}{'Δ(基线-去掉)':<14}裁决",
             "-" * 48]
    for _, r in res.iterrows():
        verdict = "核心（有独占增量）" if r["delta"] >= 0.002 else (
            "边缘（可考虑并入/删）" if r["delta"] >= 0.0005 else "无增量（效标无关候选）")
        lines.append(f"{r['construct']:<6}{r['oof_macro_f1_without']:<16.4f}{r['delta']:+.4f}     {verdict}")
    n_zero = int((res["delta"] < 0.0005).sum())
    lines += ["-" * 48,
              f"无增量构念数: {n_zero}/19",
              "判读标准（预登记）: Δ≥0.002 核心 / 0.0005-0.002 边缘 / <0.0005 无增量",
              "无增量构念 = 构念测得再好也与二值效标无关，v6.7 词典候选剔除"]

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    res.to_csv(out / "block_validity.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
