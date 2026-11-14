# -*- coding: utf-8 -*-
"""v7.13 读出：双教师合并（GLM 基座162 + v3_soft + Qwen 宽表），三配置 CV 竞争。

背景：v7.12 诊断裁决单教师测量到顶（跨措辞相关 0.85、引用/实施混淆盲区），
用户解除方法限制。Qwen3.5-9B 的独立测量是盲区不相关证据票的来源。

筛选（train 冻结，预登记）：
  Qwen 概念通道（qw_p_/qw_b_/qw_pd_/qw_v_）：方差>0 → 经验极性对齐 → 单特征
    AUC>=0.55 → 与现有特征（v6条目 + v3 kept）max|corr|>=0.9 去冗余
  Qwen 样本通道（qw_s1a/qw_s1b/qw_cat*）：仅方差>0

配置：
  L1  K1 锚点（基座162 + v3_soft）
  L2  L1 + Qwen 全通道        —— 主候选
  L3  Qwen 单独               —— 第二教师独立上限诊断
5 折 CV × 3 种子（τ@宏F1），门槛：L2 须 ≥ L1 + 0.002 才碰 test。
胜者 5 种子 test 双口径（τ@宏F1 / τ@有毒F1）。

产物：results/v7/readout_v713/{metrics.txt, cv_competition.csv, qwen_audit.csv,
      test_predictions.csv}
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
FINAL_SEEDS = [0, 1, 2, 3, 4]
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
    return best_a, float(tau), oof


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
    ap = argparse.ArgumentParser(description="v7.13 读出：双教师合并")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--qwen_train", required=True)
    ap.add_argument("--qwen_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v7/readout_v713")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    qwtr = pd.read_csv(args.qwen_train)
    qwte = pd.read_csv(args.qwen_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (te["id"].values == v3te["id"].values).all()
    assert (tr["id"].values == qwtr["id"].values).all(), "train id 未对齐（qwen）"
    assert (te["id"].values == qwte["id"].values).all(), "test id 未对齐（qwen）"
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    # ---- 基座 162 + v3_soft（v6.9 K1 复刻）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    mtr = construct_features(tr, freeze, Atr, Xtr_raw)
    mte = construct_features(te, freeze, Ate, Xte_raw)
    mcols = [c for c in mtr.columns if c != "id"]
    base_tr = np.hstack([mtr[mcols].values, Atr[kept_items].values])
    base_te = np.hstack([mte[mcols].values, Ate[kept_items].values])

    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok_c = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok_c:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok_c if auc[c] >= V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    Fte_c = v3te[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
            Fte_c[c] = 1.0 - v3te[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    Ftr = pd.concat([Ftr_c[kept_c], v3tr[kept_s]], axis=1)
    Fte = pd.concat([Fte_c[kept_c], v3te[kept_s]], axis=1)
    v3_tr, v3_te = Ftr.values, Fte.values
    existing_tr = np.hstack([base_tr, v3_tr])          # L1 全量
    existing_names = mcols + kept_items + kept_c + kept_s

    # ---- Qwen 特征筛选（train 冻结）----
    qcols = [c for c in qwtr.columns if c not in ("id", "toxic")]
    q_concept = [c for c in qcols if c.startswith(("qw_p_", "qw_b_", "qw_pd_", "qw_v_"))]
    q_sample = [c for c in qcols if c not in q_concept]
    var_qc = [c for c in q_concept if qwtr[c].std() > 1e-9]
    qpol, qauc = {}, {}
    for c in var_qc:
        a = roc_auc_score(ytr, qwtr[c].values)
        qauc[c] = max(a, 1.0 - a)
        qpol[c] = 1 if a >= 0.5 else -1
    valid_qc = [c for c in var_qc if qauc[c] >= V3_MIN_AUC]
    Qtr_c = qwtr[valid_qc].copy()
    Qte_c = qwte[valid_qc].copy()
    for c in valid_qc:
        if qpol[c] == -1:
            Qtr_c[c] = 1.0 - qwtr[c]
            Qte_c[c] = 1.0 - qwte[c]
    kept_qc, redund_qc = [], []
    for c in valid_qc:
        r = np.abs(np.corrcoef(Qtr_c[c].values, existing_tr.T)[0, 1:])
        if np.nanmax(r) >= V3_MAX_CORR:
            redund_qc.append(c)
        else:
            kept_qc.append(c)
    kept_qs = [c for c in q_sample if qwtr[c].std() > 1e-9]
    Qtr = pd.concat([Qtr_c[kept_qc], qwtr[kept_qs]], axis=1)
    Qte = pd.concat([Qte_c[kept_qc], qwte[kept_qs]], axis=1)
    nq = len(kept_qc) + len(kept_qs)
    print(f"Qwen: 总 {len(qcols)} → 概念通道 {len(kept_qc)}（AUC门槛剔除 "
          f"{len(var_qc) - len(valid_qc)}，冗余剔除 {len(redund_qc)}）+ 样本通道 {len(kept_qs)}")

    pd.DataFrame({"feature": var_qc, "aligned_auc": [qauc[c] for c in var_qc],
                  "polarity": [qpol[c] for c in var_qc],
                  "kept": [c in kept_qc for c in var_qc]}).sort_values(
        "aligned_auc", ascending=False).to_csv(out / "qwen_audit.csv",
                                               index=False, encoding="utf-8-sig")

    q_tr, q_te = Qtr.values, Qte.values
    cfg = {
        "L1_K1锚点": (existing_tr, np.hstack([base_te, Fte.values]), existing_names),
        "L2_双教师合并": (np.hstack([existing_tr, q_tr]),
                          np.hstack([base_te, Fte.values, q_te]),
                          existing_names + kept_qc + kept_qs),
        "L3_Qwen单独": (q_tr, q_te, kept_qc + kept_qs),
    }

    lines = ["=" * 72, "v7.13 读出（双教师合并，三配置 CV 竞争，τ@宏F1）", "=" * 72,
             f"Qwen: 概念 {len(kept_qc)} + 样本 {len(kept_qs)} = {nq} 维进入 L2/L3",
             "维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, (X, _, _) in cfg.items()),
             ""]

    rows = []
    for tag, (X, _, _) in tqdm(cfg.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, oof = cv_select(X, ytr, seed, tag)
            oof_auc = roc_auc_score(ytr, oof)
            oof_f1 = f1_score(ytr, (oof >= t).astype(int), average="macro")
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "oof_macro_f1": oof_f1, "oof_auc": oof_auc, "oof_tau": t})
            lines.append(f"  {tag:<14} seed{seed}  alpha={a:<8} OOF宏F1={oof_f1:.4f}  "
                         f"OOF AUC={oof_auc:.4f}  τ*={t:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    best = mean_f1.idxmax()
    gain = mean_f1["L2_双教师合并"] - mean_f1["L1_K1锚点"]
    gate_pass = best == "L2_双教师合并" and gain >= 0.002
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best}；L2 相对 L1 增益 {gain:+.4f}",
              f"预登记门槛（L2 ≥ L1+0.002）: {'通过' if gate_pass else '未通过'}", ""]

    # Qwen 块整体增量（L2 上 leave-block-out）
    if gate_pass or best == "L2_双教师合并":
        Xb_tr, Xb_te, b_names = cfg["L2_双教师合并"]
        q_idx = [i for i, nm in enumerate(b_names) if nm in kept_qc + kept_qs]
        keep_idx = [i for i in range(len(b_names)) if i not in q_idx]
        drow = []
        for seed in CV_SEEDS:
            a, t, oof = cv_select(Xb_tr[:, keep_idx], ytr, seed, "去Qwen")
            drow.append(f1_score(ytr, (oof >= t).astype(int), average="macro"))
        d_qwen = mean_f1["L2_双教师合并"] - float(np.mean(drow))
        lines += [f"Qwen 块整体增量（leave-block-out）: 去块 OOF宏F1={np.mean(drow):.4f}  "
                  f"Δ={d_qwen:+.4f}", ""]

    if not gate_pass:
        report = "\n".join(lines)
        print(report)
        (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
        return

    # ---- 胜者 test（5 种子双口径）----
    Xw_tr, Xw_te, w_names = cfg["L2_双教师合并"]
    per_seed = []
    prob = pred_m = None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau_m, oof = cv_select(Xw_tr, ytr, seed, "L2")
        tau_t = max(TAU_GRID, key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
        prob = model.predict_proba(Xw_te)[:, 1]
        pred_m = (prob >= tau_m).astype(int)
        pred_t = (prob >= tau_t).astype(int)
        per_seed.append({
            "seed": seed, "alpha": a, "tau_macro": tau_m, "tau_toxic": tau_t,
            "macro_f1": f1_score(yte, pred_m, average="macro"),
            "toxic_f1": f1_score(yte, pred_m),
            "benign_f1": f1_score(yte, pred_m, pos_label=0),
            "auc": roc_auc_score(yte, prob),
            "toxic_f1_t": f1_score(yte, pred_t),
            "macro_f1_t": f1_score(yte, pred_t, average="macro")})
    per = pd.DataFrame(per_seed)
    lines += ["test 评估（L2，5 种子，各评一次）:",
              f"  {'seed':<6}{'alpha':<10}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}{'无毒F1':<9}{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}",
              "-" * 72]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['alpha']:<10}{r['tau_macro']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}"
                     f"{r['benign_f1']:<9.4f}{r['auc']:<8.4f}{r['tau_toxic']:<7.2f}"
                     f"{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 72,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v6.9 H3 宏 0.8082 / 有毒 0.8221 (@τ毒 0.8231) / AUC 0.8878",
              "      判死线（有毒F1口径）0.8331"]

    model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / per.iloc[0]["alpha"],
                                        max_iter=2000)).fit(Xw_tr, ytr)
    coef = pd.Series(model.model.coef_[0], index=w_names)
    nz = coef[coef.abs() > 1e-6]
    q_nz = nz[nz.index.isin(kept_qc + kept_qs)]
    lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0），其中 Qwen 特征非零 {len(q_nz)}/{nq}",
              "", "|coef| 前 20："]
    for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(20).items():
        lines.append(f"    {name:<14} {w:+.3f}")

    te_out = te[["id", "toxic"]].copy()
    te_out["prob"] = prob
    te_out["pred"] = pred_m
    te_out.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
