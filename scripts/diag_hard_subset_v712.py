# -*- coding: utf-8 -*-
"""v7.12 诊断：靶向条目为何无边际增益——难样本子集上的条件判别力分析。test 零参与。

问题：24 条 v7 条目全 train 单条 AUC 0.64-0.85，并入 K1 配置后 CV 增益仅 +0.0003。
两个竞争假设（预登记判读）：
  H-A 测量盲区：新条目在难样本区域（稳定 FN vs 正确检出有毒、稳定 FP vs 正确无毒）
      的 AUC 塌到 ~0.5 → 同一教师对所有措辞的相关性感知失败，瓶颈在测量效度
  H-B 模型侧问题：难样本区域新条目仍有判别力（AUC >= 0.6）但线性读出没用上
      → 杠杆在分箱特征/正则协议

方法：
  1) 复刻 K1（基座162 + v3_soft）→ 3 种子 OOF 概率 → 稳定错误掩码（同 v6.10）
  2) 每个 v7 条目（对齐后）：全 train AUC / FN-vs-TP AUC / FP-vs-TN AUC
  3) 与现有 253 特征的最大 |corr|（冗余度）
  4) 条件增量：难样本子集内 [prob_oof + item] vs [prob_oof] 的 AUC 差（in-sample，
     仅诊断用）

产物：results/v7/diag_hard_subset/{metrics.txt, hard_subset_audit.csv}
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
            if not all(t in A.columns for t in D2_TIERS):
                continue
            hits = A[D2_TIERS].values >= 0.5
            score = np.zeros(len(df))
            for i in range(len(df)):
                h = hits[i]
                score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
            means[cid] = score
            continue
        if not kept:
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
    ap = argparse.ArgumentParser(description="v7.12 难样本条件判别力诊断")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v7_train", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v7/diag_hard_subset")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    v7tr = pd.read_csv(args.v7_train)
    v3tr = pd.read_csv(args.v3_train)
    assert (tr["id"].values == v7tr["id"].values).all()
    assert (tr["id"].values == v3tr["id"].values).all()
    y = tr["toxic"].astype(int).values

    # ---- K1 特征（基座162 + v3_soft）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xr = tr[item_cols]
    A = Xr.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            A[c] = 1.0 - Xr[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in A.columns]
    mtr = construct_features(tr, freeze, A, Xr)
    mcols = [c for c in mtr.columns if c != "id"]
    base_tr = np.hstack([mtr[mcols].values, A[kept_items].values])

    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok_c = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok_c:
        a = roc_auc_score(y, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok_c if auc[c] >= V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, A[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    Ftr = pd.concat([Ftr_c[kept_c], v3tr[kept_s]], axis=1)
    X = np.hstack([base_tr, Ftr.values])
    print(f"K1 特征: {X.shape[1]} 维")

    # ---- OOF × 3 种子 → 稳定错误掩码 ----
    taus, oofs = [], []
    for seed in tqdm(CV_SEEDS, desc="OOF CV", dynamic_ncols=True):
        t, oof = cv_select(X, y, seed, "K1")
        taus.append(t)
        oofs.append(oof)
    preds = np.stack([(o >= t).astype(int) for o, t in zip(oofs, taus)])
    prob_mean = np.mean(oofs, axis=0)
    fn = ((preds == 0) & (y == 1)[None, :]).sum(axis=0) >= 2
    fp = ((preds == 1) & (y == 0)[None, :]).sum(axis=0) >= 2
    print(f"稳定错误: FN {fn.sum()}, FP {fp.sum()}")

    # ---- v7 条目对齐 ----
    v7_cols = [c for c in v7tr.columns if c not in ("id", "toxic", "content")]
    V = v7tr[v7_cols].copy()
    v7_pol = {}
    for c in v7_cols:
        a = roc_auc_score(y, V[c].values)
        v7_pol[c] = 1 if a >= 0.5 else -1
        if v7_pol[c] == -1:
            V[c] = 1.0 - v7tr[c]

    # ---- 逐条目诊断 ----
    toxic_idx = np.where(y == 1)[0]
    ben_idx = np.where(y == 0)[0]
    rows = []
    exist_ref = X  # 与现有 253 特征的相关
    for c in v7_cols:
        v = V[c].values
        full_auc = roc_auc_score(y, v)
        # FN vs TP: 1=TP(检出), 0=FN(漏判) → AUC of v ranking TP above FN
        tp_mask, fn_mask_tox = ~fn[toxic_idx], fn[toxic_idx]
        yy = np.r_[np.ones(tp_mask.sum()), np.zeros(fn_mask_tox.sum())]
        vv = np.r_[v[toxic_idx][tp_mask], v[toxic_idx][fn_mask_tox]]
        fn_auc = roc_auc_score(yy, vv)
        tn_mask = ~fp[ben_idx]
        yy = np.r_[np.ones(tn_mask.sum()), np.zeros(fp[ben_idx].sum())]
        vv = np.r_[v[ben_idx][tn_mask], v[ben_idx][fp[ben_idx]]]
        fp_auc = roc_auc_score(yy, vv)  # 1=TN(正确无毒), 0=FP(误报) → 高值=把TN排在FP前
        max_corr = np.abs(np.corrcoef(v, exist_ref.T)[0, 1:]).max()
        # 条件增量：FN 子集内 [prob + v] vs [prob]
        inc_fn = inc_fp = np.nan
        if fn_mask_tox.sum() >= 20:
            p_sub = prob_mean[toxic_idx]
            y_bin = tp_mask.astype(int)
            base_auc = roc_auc_score(y_bin, p_sub)
            m2 = LogisticRegression(max_iter=1000).fit(
                StandardScaler().fit_transform(np.c_[p_sub, v[toxic_idx]]), y_bin)
            aug_auc = roc_auc_score(y_bin, m2.predict_proba(
                StandardScaler().fit_transform(np.c_[p_sub, v[toxic_idx]]))[:, 1])
            inc_fn = aug_auc - base_auc
        if fp[ben_idx].sum() >= 20:
            p_sub = prob_mean[ben_idx]
            y_bin = tn_mask.astype(int)
            base_auc = roc_auc_score(y_bin, p_sub)
            sc = StandardScaler().fit(np.c_[p_sub, v[ben_idx]])
            m2 = LogisticRegression(max_iter=1000).fit(sc.transform(np.c_[p_sub, v[ben_idx]]), y_bin)
            aug_auc = roc_auc_score(y_bin, m2.predict_proba(sc.transform(np.c_[p_sub, v[ben_idx]]))[:, 1])
            inc_fp = aug_auc - base_auc
        rows.append({"item": c, "full_auc": full_auc, "fn_vs_tp_auc": fn_auc,
                     "fp_vs_tn_auc": fp_auc, "max_corr_existing": max_corr,
                     "inc_fn": inc_fn, "inc_fp": inc_fp})
    aud = pd.DataFrame(rows)

    lines = ["=" * 76, "v7.12 诊断：靶向条目难样本条件判别力（test 零参与）", "=" * 76,
             f"K1 特征 {X.shape[1]} 维；稳定 FN {int(fn.sum())} / FP {int(fp.sum())}",
             "",
             f"{'条目':<8}{'全AUC':<8}{'FNvsTP':<9}{'FPvsTN':<9}{'与现有max|corr|':<16}{'FN增量':<9}{'FP增量':<9}",
             "-" * 68]
    for _, r in aud.iterrows():
        lines.append(f"{r['item']:<8}{r['full_auc']:<8.3f}{r['fn_vs_tp_auc']:<9.3f}"
                     f"{r['fp_vs_tn_auc']:<9.3f}{r['max_corr_existing']:<16.3f}"
                     f"{r['inc_fn']:<9.3f}{r['inc_fp']:<9.3f}")
    med_fn = aud["fn_vs_tp_auc"].median()
    med_fp = aud["fp_vs_tn_auc"].median()
    med_corr = aud["max_corr_existing"].median()
    lines += ["-" * 68,
              f"中位数: FNvsTP={med_fn:.3f}  FPvsTN={med_fp:.3f}  max|corr|={med_corr:.3f}",
              "",
              "预登记判读:",
              "  若 FNvsTP / FPvsTN 中位数 ≈ 0.5 → H-A 测量盲区：同一教师对所有措辞的",
              "    难样本感知失败相关，杠杆在测量侧（改写去噪/换测量方式），加特征无用",
              "  若中位数 >= 0.6 且 max|corr| < 0.7 → H-B 模型侧问题：信息在但读出没利用，",
              "    杠杆在分箱特征/正则协议",
              "  inc_fn / inc_fp = 子集内加单条目后 in-sample AUC 提升（>0.02 视为实质增量）"]

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    aud.to_csv(out / "hard_subset_audit.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
