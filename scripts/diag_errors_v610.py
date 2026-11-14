# -*- coding: utf-8 -*-
"""v6.10 诊断：胜者配置（H3 基座+soft全通道）train OOF 误差分析。test 零参与。

目的：特征侧增量已到顶（AUC 0.8878），定位剩余错误集中在哪类样本，
决定是否让教师跑一批针对性新条目（单教师）。

方法：
  1) 复刻 v6.9 H3 特征（基座162 + v3_soft 全通道，同筛选冻结规则）
  2) train 5 折 CV × 3 种子收集 OOF 概率与各自 τ*（宏F1 协议）
  3) 多数投票定义稳定错误：FN = ≥2 种子漏判的有毒样本；FP = ≥2 种子误报的无毒
  4) 按 expression 组（0=隐式 / 1,2=显式）与文本长度分桶统计 FN/FP
  5) 概念轮廓对比：FN 有毒 vs 正确检出的有毒，头部 v3 特征均值差
  6) 输出典型漏判案例（prob 最低 15 例）

expression 取自 v3 rings JSON（样本级字段），id 对齐。

产物：results/v6/diag_errors_v610/{metrics.txt, oof_errors.csv}
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

ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
TAU_GRID = np.round(np.arange(0.30, 0.701, 0.01), 3)
CV_SEEDS = [42, 43, 44]
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
V3_MIN_AUC = 0.55
V3_MAX_CORR = 0.90
LEN_BINS = [0, 10, 20, 40, 1000]
LEN_LABELS = ["<10", "10-20", "20-40", ">=40"]
PROFILE_FEATURES = ["cat_CAT2A", "v3pd_A1", "v3p_U2", "v3p_E1", "v3pd_A2",
                    "v3pd_C7", "v3p_E2", "cat_CAT4", "s1b_p1", "s1a_p3"]


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
    tau = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return float(tau), oof


def construct_features(df, freeze, A, Xr):
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
    ap = argparse.ArgumentParser(description="v6.10 train OOF 误差分析（H3 胜者配置）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_rings_json", required=True,
                    help="rings_glm-4-9b-chat_train.json（取 expression 字段）")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v6/diag_errors_v610")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    v3tr = pd.read_csv(args.v3_train)
    assert (tr["id"].values == v3tr["id"].values).all(), "train id 未对齐"
    ytr = tr["toxic"].astype(int).values

    # expression（rings JSON 样本级字段）
    print("加载 rings JSON 取 expression ...")
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map[i] for i in tr["id"].values])
    content = tr["content"].astype(str).values if "content" in tr.columns else np.array([""] * len(tr))
    length = np.array([len(s) for s in content])

    # ---- H3 特征复刻 ----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw = tr[item_cols]
    Atr = Xtr_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    mtr = construct_features(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in mtr.columns if c != "id"]
    base_tr = np.hstack([mtr[mcols].values, Atr[kept_items].values])

    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok_c = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    from sklearn.metrics import roc_auc_score
    pol, auc = {}, {}
    for c in var_ok_c:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok_c if auc[c] >= V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
    kept_c = []
    v6_ref = Atr[kept_items]
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, v6_ref.values.T)[0, 1:])
        if np.nanmax(r) < V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    Ftr = pd.concat([Ftr_c[kept_c], v3tr[kept_s]], axis=1)
    X = np.hstack([base_tr, Ftr.values])
    print(f"H3 特征: {X.shape[1]} 维（基座 {base_tr.shape[1]} + v3 {len(kept_c) + len(kept_s)}）")

    # ---- OOF × 3 种子 ----
    taus, oofs = [], []
    for seed in tqdm(CV_SEEDS, desc="OOF CV", dynamic_ncols=True):
        t, oof = cv_select(X, ytr, seed, "H3")
        taus.append(t)
        oofs.append(oof)
        print(f"  seed{seed}: τ*={t:.2f} OOF宏F1={f1_score(ytr, (oof >= t).astype(int), average='macro'):.4f}")
    preds = np.stack([(o >= t).astype(int) for o, t in zip(oofs, taus)])
    prob_mean = np.mean(oofs, axis=0)
    n_fn = ((preds == 0) & (ytr == 1)[None, :]).sum(axis=0)
    n_fp = ((preds == 1) & (ytr == 0)[None, :]).sum(axis=0)
    fn_mask = n_fn >= 2
    fp_mask = n_fp >= 2

    # ---- 分组统计 ----
    grp = np.where(expr == 0, "expr0隐式", np.where(expr > 0, "expr显式", "expr缺失"))
    len_bin = np.digitize(length, LEN_BINS[1:-1], right=False)
    len_grp = np.array([LEN_LABELS[i] for i in len_bin])

    def cross_tab(mask, label):
        rows = []
        for g in ["expr0隐式", "expr显式", "expr缺失"]:
            for lg in LEN_LABELS:
                sel = (grp == g) & (len_grp == lg)
                if sel.sum() == 0:
                    continue
                rows.append({"错误类型": label, "expression组": g, "长度": lg,
                             "样本数": int(sel.sum()), "错误数": int(mask[sel].sum()),
                             "错误率": float(mask[sel].mean())})
        return pd.DataFrame(rows)

    tab = pd.concat([cross_tab(fn_mask, "FN漏判有毒"), cross_tab(fp_mask, "FP误报无毒")])

    # ---- 概念轮廓：FN 有毒 vs 检出有毒 ----
    prof_rows = []
    hit_mask = (ytr == 1) & (~fn_mask)
    for f in PROFILE_FEATURES:
        if f not in Ftr.columns:
            continue
        v = Ftr[f].values
        prof_rows.append({"特征": f, "FN有毒均值": float(v[fn_mask].mean()),
                          "检出有毒均值": float(v[hit_mask].mean()),
                          "无毒均值": float(v[ytr == 0].mean())})
    prof = pd.DataFrame(prof_rows)

    # ---- 落盘 ----
    err_df = pd.DataFrame({
        "id": tr["id"].values, "toxic": ytr, "prob_mean": np.round(prob_mean, 4),
        "n_fn": n_fn, "n_fp": n_fp, "expression": expr, "length": length,
        "fn": fn_mask.astype(int), "fp": fp_mask.astype(int), "content": content})
    err_df.to_csv(out / "oof_errors.csv", index=False, encoding="utf-8-sig")

    lines = ["=" * 72, "v6.10 诊断：H3 胜者配置 train OOF 误差分析（test 零参与）", "=" * 72,
             f"H3 特征 {X.shape[1]} 维；OOF τ*（宏F1）: "
             + ", ".join(f"seed{s}={t:.2f}" for s, t in zip(CV_SEEDS, taus)),
             "",
             f"稳定错误（≥2/3 种子）: FN 漏判有毒 {int(fn_mask.sum())}/{int((ytr == 1).sum())}，"
             f"FP 误报无毒 {int(fp_mask.sum())}/{int((ytr == 0).sum())}",
             "",
             "错误分布（expression 组 × 长度桶）:",
             tab.to_string(index=False),
             "",
             "概念轮廓（头部 v3 特征均值，筛选用原始未对齐值）:",
             prof.to_string(index=False)]

    fn_idx = np.where(fn_mask)[0]
    fn_idx = fn_idx[np.argsort(prob_mean[fn_idx])][:15]
    lines += ["", "典型漏判案例（prob 最低 15 例）:"]
    for i in fn_idx:
        lines.append(f"  {tr['id'].iloc[i]}  expr={expr[i]}  prob={prob_mean[i]:.3f}  "
                     f"len={length[i]}  | {content[i][:50]}")

    fp_idx = np.where(fp_mask)[0]
    fp_idx = fp_idx[np.argsort(-prob_mean[fp_idx])][:10]
    lines += ["", "典型误报案例（prob 最高 10 例）:"]
    for i in fp_idx:
        lines.append(f"  {tr['id'].iloc[i]}  expr={expr[i]}  prob={prob_mean[i]:.3f}  "
                     f"len={length[i]}  | {content[i][:50]}")

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    tab.to_csv(out / "error_breakdown.csv", index=False, encoding="utf-8-sig")
    prof.to_csv(out / "concept_profile.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
