# -*- coding: utf-8 -*-
"""v6.8 读出：合并池条目级筛选 + PRUNE6 基座，三配置 CV 竞争。

背景（v6.7 结果）：
  F2 基座+v3p 胜出，test 宏F1 0.7960 / 有毒F1 0.8133 / AUC 0.8821；
  F3 三通道（+v3d/v3v 57 维）CV 反而降（0.8063 < 0.8089）——未筛选通道是噪声。
下一步：对 v3 特征做与 v6.4 同纪律的条目级筛选，基座换 v6.6 已验证的 PRUNE6。

v3 特征筛选（全部 train 内裁决，冻结后投影 test；预登记标准）：
  1) 方差过滤：train std > 1e-9
  2) 经验极性对齐：单特征 AUC < 0.5 翻转（与 v6.4 同规则）
  3) 条目级效度：对齐后单特征 AUC >= 0.55 才保留（效标无关即删）
  4) 冗余剔除：与任一 v6 对齐保留条目 |Pearson r| >= 0.9 删（重复信号）

配置：
  G1  PRUNE6 基座              —— v6.6 胜者复刻（数据驱动负Δ构念剔除）
  G2  PRUNE6 基座 + 筛选v3p    —— stage1 通道
  G3  PRUNE6 基座 + 筛选全通道 —— p/d/v 三通道（检验筛选能否救活 F3）
5 折 CV × 3 种子（τ@宏F1）竞争，胜者 5 种子 test 各评一次。

产物：results/v6/readout_v68/{metrics.txt, cv_competition.csv,
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

ALPHA_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
TAU_GRID = np.round(np.arange(0.30, 0.701, 0.01), 3)
CV_SEEDS = [42, 43, 44]
FINAL_SEEDS = [0, 1, 2, 3, 4]
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
V3_MIN_AUC = 0.55      # 预登记：对齐后单特征 AUC 门槛
V3_MAX_CORR = 0.90     # 预登记：与 v6 条目最大相关冗余线


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
    """宏层（与 readout_v66 同口径）：dropped 构念 T 分置 0 + V1 + V2。"""
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


def main():
    ap = argparse.ArgumentParser(description="v6.8 读出：合并池条目级筛选 + PRUNE6 基座")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--block_validity_csv", default="results/v6/diag_block_validity/block_validity.csv")
    ap.add_argument("--output_dir", default="results/v6/readout_v68")
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

    # ---- PRUNE6 基座（与 v6.6 数据驱动一致）----
    bv = pd.read_csv(args.block_validity_csv)
    alive = set(freeze["constructs_alive"])
    bv = bv[bv["construct"].isin(alive)]
    dropped = set(bv.loc[bv["delta"] < 0, "construct"])

    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    item_of = {}
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            item_of[c] = cid
    mtr = construct_features(tr, freeze, Atr, Xtr_raw, dropped)
    mte = construct_features(te, freeze, Ate, Xte_raw, dropped)
    mcols = [c for c in mtr.columns if c != "id"]
    micro = [c for c, cc in item_of.items() if c in Atr.columns and cc not in dropped]
    base_tr = np.hstack([mtr[mcols].values, Atr[micro].values])
    base_te = np.hstack([mte[mcols].values, Ate[micro].values])
    base_names = mcols + micro
    print(f"PRUNE6 基座: 剔除 {sorted(dropped)}，维度 {base_tr.shape[1]}")

    # ---- v3 条目级筛选（train 冻结）----
    v3_cols = [c for c in v3tr.columns if c.startswith(("v3p_", "v3d_", "v3v_"))]
    var_ok = [c for c in v3_cols if v3tr[c].std() > 1e-9]
    pol3, auc3 = {}, {}
    for c in var_ok:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc3[c] = max(a, 1.0 - a)
        pol3[c] = 1 if a >= 0.5 else -1
    valid = [c for c in var_ok if auc3[c] >= V3_MIN_AUC]
    # 极性对齐后的 v3 矩阵
    F3tr = v3tr[valid].copy()
    F3te = v3te[valid].copy()
    for c in valid:
        if pol3[c] == -1:
            F3tr[c] = 1.0 - v3tr[c]
            F3te[c] = 1.0 - v3te[c]
    # 冗余剔除：与任一 v6 对齐保留条目 |r| >= 0.9
    v6_ref = Atr[micro]
    kept3, redundant3 = [], []
    for c in valid:
        r = np.abs(np.corrcoef(F3tr[c].values, v6_ref.values.T)[0, 1:])
        if np.nanmax(r) >= V3_MAX_CORR:
            redundant3.append(c)
        else:
            kept3.append(c)
    print(f"v3 筛选: 总 {len(v3_cols)} → 方差{len(var_ok)} → AUC>={V3_MIN_AUC} "
          f"{len(valid)} → 去冗余 {len(kept3)}（剔除 {len(redundant3)} 个冗余）")
    kept3p = [c for c in kept3 if c.startswith("v3p_")]
    kept3d = [c for c in kept3 if c.startswith("v3d_")]
    kept3v = [c for c in kept3 if c.startswith("v3v_")]
    print(f"  通道分布: v3p={len(kept3p)}  v3d={len(kept3d)}  v3v={len(kept3v)}")

    audit = pd.DataFrame({
        "feature": var_ok,
        "channel": [c.split("_")[0] for c in var_ok],
        "aligned_auc": [auc3[c] for c in var_ok],
        "polarity": [pol3[c] for c in var_ok],
        "kept": [c in kept3 for c in var_ok],
        "redundant": [c in redundant3 for c in var_ok],
    }).sort_values("aligned_auc", ascending=False)
    audit.to_csv(out / "v3_feature_audit.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "G1_PRUNE6基座": (base_tr, base_te, base_names),
        "G2_基座+筛选v3p": (np.hstack([base_tr, F3tr[kept3p].values]),
                            np.hstack([base_te, F3te[kept3p].values]),
                            base_names + kept3p),
        "G3_基座+筛选全通道": (np.hstack([base_tr, F3tr[kept3].values]),
                              np.hstack([base_te, F3te[kept3].values]),
                              base_names + kept3),
    }

    lines = ["=" * 72, "v6.8 读出（合并池条目级筛选 + PRUNE6 基座，τ@宏F1）", "=" * 72,
             f"基座剔除构念: {sorted(dropped)}",
             f"v3 筛选: 总 {len(v3_cols)} → 方差{len(var_ok)} → AUC>={V3_MIN_AUC} "
             f"{len(valid)} → 去冗余 {len(kept3)}（冗余剔除 {len(redundant3)}）",
             f"  通道分布: v3p={len(kept3p)}  v3d={len(kept3d)}  v3v={len(kept3v)}",
             "维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, (X, _, _) in cfg.items()),
             ""]

    rows = []
    for tag, (X, _, _) in tqdm(cfg.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, f1 = cv_select(X, ytr, seed, tag)
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "oof_macro_f1": f1, "oof_tau": t})
            lines.append(f"  {tag:<20} seed{seed}  alpha={a:<8} OOF宏F1={f1:.4f}  τ*={t:.2f}")
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
              "历史: v6.5 混合 0.7889±0.0055 / v6.6 剪枝 0.7859±0.0032 / v6.7 合并 0.7960±0.0028",
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
