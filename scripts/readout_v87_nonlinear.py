# -*- coding: utf-8 -*-
"""v8.7 读出：下游分类器非线性化——v8.1 M4 特征基座不变，换分类器族。

动机（v8.6 之后的理论分歧点）：线性读出是否已达概念向量 C 的 Bayes 上限？
用户判断"稀疏线性分类器肯定不能平铺利用信息"。本脚本做两件事：

【A】全量非线性读出竞争（M4 的 264 维特征逐字节复刻，仅换分类器）：
  N0  L1-LR 锚点（线性参照，alpha 网格）
  N1  GBDT        HistGradientBoosting（叶15/300轮/lr0.05/叶样本60/l2=1）
  N2  GBDT浅      强正则版（叶7/200轮/lr0.03/叶样本100）
  N3  RF          RandomForest 400树/叶10/max_features=0.3
  N4  ET          ExtraTrees 400树/叶10/max_features=0.3
  N5  MLP         sklearn MLP (64,32)，alpha=1e-3，早停，StandardScaler
  N6  NystroemSVM RBF 核近似(200分量)+L2-LR，StandardScaler
  N7  GBDT温标    N1 + 温度缩放（T 在 train OOF 上拟合——修 v8.6 S7 的
                  τ 转移失败：GBDT 概率未校准，train OOF τ 不迁移）
  超参均为先验固定（不做 CV 调参，避免选择过拟——v8.6 S7 的教训）。
  协议：CV 5 折×3 种子（42/43/44）竞争，τ@宏F1，门槛 ≥+0.002 才碰 test，
  胜者 5 种子（0-4）test 双口径。

【B】边界非线性探针（决定性诊断）：v8.3 审计的组间 AUC（FP vs TP 0.7335 等）
  是 CV-LR（线性）测的。若该边界存在非线性可分结构，0.7335 不是 C 的真实
  上界。此处用 GBDT / NystroemSVM 两种非线性探针重测同一批错误组对，
  与线性参照逐项对比。
  预登记判读：非线性探针比线性 +0.05 以上 → C 存在非线性边界结构，
  读出侧复活；+0.02 以内 → 现有矩阵信息耗尽成立（数据处理不等式闭环），
  唯一出路是测量模式革新。

产物：results/v87/readout_nonlinear/{metrics.txt, cv_competition.csv}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import readout_v83 as r83

GATE = r83.GATE
NONLIN = [("N1_GBDT", "gbdt1"), ("N2_GBDT浅", "gbdt2"), ("N3_RF", "rf"),
          ("N4_ET", "et"), ("N5_MLP", "mlp"), ("N6_NystroemSVM", "nys"),
          ("N7_GBDT温标", "temp")]
AUDIT_REF = {"FP_vs_TP": 0.7335, "FP_vs_TN": 0.9862, "E2FN_vs_E2TP": 0.968,
             "E2FN_vs_BEN": 0.6825}


def make_model(kind, seed):
    if kind in ("gbdt1", "temp"):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=60, l2_regularization=1.0, random_state=seed)
    if kind == "gbdt2":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=7,
            min_samples_leaf=100, l2_regularization=1.0, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                      max_features=0.3, n_jobs=-1, random_state=seed)
    if kind == "et":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10,
                                    max_features=0.3, n_jobs=-1, random_state=seed)
    if kind == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                             batch_size=256, max_iter=200,
                                             early_stopping=True, n_iter_no_change=8,
                                             validation_fraction=0.1,
                                             random_state=seed))])
    if kind == "nys":
        return Pipeline([("sc", StandardScaler()),
                         ("ny", Nystroem(kernel="rbf", n_components=200,
                                         random_state=seed)),
                         ("m", LogisticRegression(max_iter=2000))])
    if kind == "lr_l2":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000))])
    raise ValueError(kind)


def fit_temperature(p, y):
    """温度缩放：p_scaled = expit(logit(p)/T)，T 在 OOF 上最小化 NLL。"""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def nll(log_t):
        q = expit(z / np.exp(log_t))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def eval_config(kind, X, y, seed):
    """返回 (tau_macro, oof_prob[用于最终指标], tau_toxic, T)。"""
    base = "gbdt1" if kind == "temp" else kind
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in tqdm(cv.split(X, y), desc=f"    {kind} s{seed}", leave=False,
                       dynamic_ncols=True):
        m = make_model(base, seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    T = None
    if kind == "temp":
        T = fit_temperature(oof, y)
        oof = expit(logit(np.clip(oof, 1e-6, 1 - 1e-6)) / T)
    tau = max(r83.TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int),
                                                   average="macro"))
    tau_t = max(r83.TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int)))
    return float(tau), oof, float(tau_t), T


def eval_anchor(X, y, seed):
    a, tau, oof = r83.cv_select(X, y, seed, "N0")
    tau_t = max(r83.TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int)))
    return a, float(tau), oof, float(tau_t)


def cv_auc_probe(kind, X, pos_idx, neg_idx, seed=42):
    """子集内 5 折 CV AUC（与 v8.3 审计同法，仅换探针模型）。"""
    idx = np.concatenate([pos_idx, neg_idx])
    ysub = np.where(np.arange(len(idx)) < len(pos_idx), 1, 0)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(idx))
    for tr, va in cv.split(X[idx], ysub):
        m = make_model(kind, seed)
        m.fit(X[idx[tr]], ysub[tr])
        oof[va] = m.predict_proba(X[idx[va]])[:, 1]
    a = roc_auc_score(ysub, oof)
    return max(a, 1 - a)


def main():
    ap = argparse.ArgumentParser(description="v8.7 非线性读出 + 边界探针")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--output_dir", default="results/v87/readout_nonlinear")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (te["id"].values == v3te["id"].values).all()
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    # ---- M4 特征装配（与 v8.3/v8.5/v8.6 S1 逐行一致）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok if auc[c] >= r83.V3_MIN_AUC]
    Ftr_c, Fte_c = v3tr[valid_c].copy(), v3te[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
            Fte_c[c] = 1.0 - v3te[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < r83.V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    prob_names = kept_items + kept_c + kept_s
    v3_tr = np.hstack([Ftr_c[kept_c].values, v3tr[kept_s].values])
    v3_te = np.hstack([Fte_c[kept_c].values, v3te[kept_s].values])
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    raw_te = np.hstack([Ate[kept_items].values, v3_te])
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rall_tr = np.zeros_like(raw_tr)
    Rall_te = np.zeros_like(raw_te)
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)
        Rall_te[:, j] = np.interp(raw_te[:, j], edges, qs)
    n_items = len(kept_items)
    Atr_items, Ate_items = Atr[kept_items].values, Ate[kept_items].values
    Mtr, Mte = r83.meta_base(tr["content"].tolist(), te["content"].tolist())
    mnames = ["meta_len", "meta_emoji", "meta_exclaim", "meta_question",
              "meta_repeatpunct", "meta_evasion"]
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1]) for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    top_names = [kept_items[i] for i in order_i]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate_items[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    Xtr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    Xte = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    feat_names = mcols + prob_names + mnames + [f"lenx_{f}" for f in top_names]
    assert Xtr.shape[1] == len(feat_names)
    print(f"M4 特征装配完成：{Xtr.shape[1]} 维（train {Xtr.shape[0]} / test {Xte.shape[0]}）")

    # ---- 【A】CV 竞争 ----
    lines = ["=" * 72, "v8.7 非线性读出（M4 特征不变，换分类器）", "=" * 72,
             "N0 L1-LR锚点 | N1/N2 GBDT | N3 RF | N4 ET | N5 MLP | "
             "N6 NystroemSVM | N7 GBDT+温标（超参先验固定，无CV调参）", ""]
    rows = []
    for seed in r83.CV_SEEDS:
        a, tau, oof, _ = eval_anchor(Xtr, ytr, seed)
        rows.append({"config": "N0_LR锚点", "seed": seed, "alpha": a,
                     "oof_macro_f1": f1_score(ytr, (oof >= tau).astype(int), average="macro"),
                     "oof_auc": roc_auc_score(ytr, oof), "oof_tau": tau})
        lines.append(f"  N0_LR锚点   seed{seed}  alpha={a:<8} OOF宏F1="
                     f"{f1_score(ytr, (oof >= tau).astype(int), average='macro'):.4f}"
                     f"  OOF AUC={roc_auc_score(ytr, oof):.4f}  τ*={tau:.2f}")
        for name, kind in NONLIN:
            tau_k, oof_k, _, _ = eval_config(kind, Xtr, ytr, seed)
            f1k = f1_score(ytr, (oof_k >= tau_k).astype(int), average="macro")
            rows.append({"config": name, "seed": seed, "alpha": "",
                         "oof_macro_f1": f1k, "oof_auc": roc_auc_score(ytr, oof_k),
                         "oof_tau": tau_k})
            lines.append(f"  {name:<12} seed{seed}  OOF宏F1={f1k:.4f}"
                         f"  OOF AUC={roc_auc_score(ytr, oof_k):.4f}  τ*={tau_k:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    nonlin_best = mean_f1[[n for n, _ in NONLIN]].idxmax()
    gain = mean_f1[nonlin_best] - mean_f1["N0_LR锚点"]
    gate_pass = gain >= GATE
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳非线性: {nonlin_best}（相对 N0 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过'}", ""]

    # ---- 【B】边界非线性探针 ----
    oofs_a = np.column_stack([eval_anchor(Xtr, ytr, s)[2] for s in r83.CV_SEEDS])
    oof_a = oofs_a.mean(axis=1)
    tau_a = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof_a >= t).astype(int),
                                                     average="macro"))
    err = (oof_a >= tau_a).astype(int) != ytr
    fp_m = err & (ytr == 0)
    tp_m = (ytr == 1) & (~err)
    tn_m = (ytr == 0) & (~err)
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    e2idx = np.where(expr == 2)[0]
    e2fn = e2idx[err[e2idx] & (ytr[e2idx] == 1)]
    e2tp = e2idx[(~err[e2idx]) & (ytr[e2idx] == 1)]
    fp_i, tp_i, tn_i = np.where(fp_m)[0], np.where(tp_m)[0], np.where(tn_m)[0]
    ben_i = np.where(ytr == 0)[0]
    pairs = [("FP_vs_TP", fp_i, tp_i), ("FP_vs_TN", fp_i, tn_i),
             ("E2FN_vs_E2TP", e2fn, e2tp), ("E2FN_vs_BEN", e2fn, ben_i)]
    lines += ["=" * 72, "【B】边界非线性探针（子集内 5 折 CV AUC，seed42）", "=" * 72,
              f"{'组对':<16}{'线性参照(v83审计)':<18}{'L2-LR复测':<12}{'GBDT':<12}{'NystroemSVM':<12}",
              "-" * 72]
    for pname, pi, ni in pairs:
        if len(pi) < 20 or len(ni) < 20:
            lines.append(f"  {pname:<16}样本不足（{len(pi)}/{len(ni)}），跳过")
            continue
        vals = [cv_auc_probe(k, Xtr, pi, ni) for k in ("lr_l2", "gbdt1", "nys")]
        lines.append(f"  {pname:<16}{AUDIT_REF.get(pname, float('nan')):<18.4f}"
                     f"{vals[0]:<12.4f}{vals[1]:<12.4f}{vals[2]:<12.4f}")
    lines += ["判读（预登记）：非线性探针比线性 +0.05 以上 → C 存在非线性边界结构，"
              "读出侧复活；+0.02 以内 → 现有矩阵信息耗尽成立，出路在测量模式革新。", ""]

    # ---- test（胜者或锚点）----
    test_name = nonlin_best if gate_pass else "N0_LR锚点"
    test_kind = dict(NONLIN)[test_name] if gate_pass else "lr"
    if not gate_pass:
        lines.append("门槛未通过，test 仅复评锚点 N0。")
    per_seed = []
    for seed in tqdm(r83.FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        if test_kind == "lr":
            a, tau, oof, tau_t = eval_anchor(Xtr, ytr, seed)
            m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                C=1.0 / a, max_iter=2000)).fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]
        else:
            tau, oof, tau_t, T = eval_config(test_kind, Xtr, ytr, seed)
            m = make_model("gbdt1" if test_kind == "temp" else test_kind, seed).fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]
            if test_kind == "temp":
                prob = expit(logit(np.clip(prob, 1e-6, 1 - 1e-6)) / T)
        pred = (prob >= tau).astype(int)
        per_seed.append({"seed": seed, "tau_macro": tau, "tau_toxic": tau_t,
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "toxic_f1": f1_score(yte, pred),
                         "benign_f1": f1_score(yte, pred, pos_label=0),
                         "auc": roc_auc_score(yte, prob),
                         "toxic_f1_t": f1_score(yte, (prob >= tau_t).astype(int))})
    per = pd.DataFrame(per_seed)
    lines += [f"test 评估（{test_name}，5 种子）:",
              f"  {'seed':<6}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}{'无毒F1':<9}"
              f"{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}", "-" * 72]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['tau_macro']:<7.2f}{r['macro_f1']:<9.4f}"
                     f"{r['toxic_f1']:<9.4f}{r['benign_f1']:<9.4f}{r['auc']:<8.4f}"
                     f"{r['tau_toxic']:<7.2f}{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 72,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v8.1 M4 宏 0.8141 / 有毒 0.8249 (@τ毒 0.8310) / AUC 0.8948",
              "      v8.6 S7 GBDT 宏 0.8093 / AUC 0.8977（CV 过门 test 未兑现）"]

    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
