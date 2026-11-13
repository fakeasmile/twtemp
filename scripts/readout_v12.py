# -*- coding: utf-8 -*-
"""v12 读出：跨教师拼接——GLM M4 基座 + Qwen 句级/目标特征（v11 线收尾）。

动机（交叉教师侦察 cross_teacher_check.txt）：Qwen 句级特征单独在
GLM FP vs TP 边界上 AUC=0.7026，M4+Qwen句级=0.7526（+0.019）。该增量
是否在正式 CV 竞争中兑现为本读出的最后裁决。

四臂（基座 = GLM M4 264 维，逐字节复刻 v8.x 装配）：
  N0_M4锚点        264 维
  Q1_+Qwen句级     267 维（Qwen assert 3 特征：p_tox_max/n_toxic/soft_asserted）
  Q2_+Qwen目标     275 维（Qwen target 11 特征）
  Q3_+句级+目标    278 维

协议：CV 5 折 × 3 种子（42/43/44）竞争，τ@宏F1，门槛 ≥+0.002 才碰
test；胜者 5 种子（0-4）test 双口径；未过则 test 仅复评锚点。
历史对照：v11.2（GLM 目标特征同构实验）增益 -0.0004/-0.0010 未过门。
产物：results/v12/readout_cross/{metrics.txt, cv_competition.csv}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import readout_v83 as r83
from analyze_target_v11 import FEATS as TGT_FEATS, aggregate as tgt_aggregate
from cross_teacher_v11 import agg_assert

GATE = r83.GATE


def assemble_m4(tr, te, v3tr, v3te, freeze, ytr):
    """GLM M4 装配（train+test，与 v8.x 逐行一致；train 段与
    verify_noise_v11.assemble_m4 同源，已复现 0.8929/0.7333）。"""
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
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate_items[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    Xtr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    Xte = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    return Xtr, Xte


def test_eval(Xtr, ytr, Xte, yte, seeds, tag):
    """test 评估（胜者 5 种子双口径；与 v8.7/v11 锚点路径同构）。
    τ宏 行报告全指标（宏F1/加权F1/ACC/宏精确/宏召回/有毒F1/有毒P/R/无毒F1/AUC），
    另附 τ毒 口径的有毒 F1。"""
    rows = []
    for seed in seeds:
        a, tau, oof = r83.cv_select(Xtr, ytr, seed, f"final[{tag}]")
        m = Pipeline([("m", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / a, max_iter=2000))]).fit(Xtr, ytr)
        p_te = m.predict_proba(Xte)[:, 1]
        pred = (p_te >= tau).astype(int)
        tau_t = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        pred_t = (p_te >= tau_t).astype(int)
        rows.append({
            "seed": seed, "alpha": a, "tau": tau,
            "macro": f1_score(yte, pred, average="macro"),
            "weighted": f1_score(yte, pred, average="weighted"),
            "acc": accuracy_score(yte, pred),
            "prec_macro": precision_score(yte, pred, average="macro"),
            "rec_macro": recall_score(yte, pred, average="macro"),
            "toxic": f1_score(yte, pred),
            "prec_t": precision_score(yte, pred),
            "rec_t": recall_score(yte, pred),
            "benign": f1_score(yte, pred, pos_label=0),
            "auc": roc_auc_score(yte, p_te),
            "tau_toxic": tau_t,
            "toxic_at_tau_t": f1_score(yte, pred_t),
        })
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser(description="v12 跨教师拼接读出")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--qwen_assert_train", default="results/v11/qwen/assert/assert_flat_train.csv")
    ap.add_argument("--qwen_assert_test", default="results/v11/qwen/assert/assert_flat_test.csv")
    ap.add_argument("--qwen_target_train", default="results/v11/qwen/target/target_flat_train.csv")
    ap.add_argument("--qwen_target_test", default="results/v11/qwen/target/target_flat_test.csv")
    ap.add_argument("--output_dir", default="results/v12/readout_cross")
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

    Xtr, Xte = assemble_m4(tr, te, v3tr, v3te, freeze, ytr)
    print(f"M4 装配 train {Xtr.shape} / test {Xte.shape}")

    ids_tr, ids_te = tr["id"].values, te["id"].values
    A_tr = agg_assert(pd.read_csv(args.qwen_assert_train), ids_tr).values
    A_te = agg_assert(pd.read_csv(args.qwen_assert_test), ids_te).values
    T_tr = tgt_aggregate(pd.read_csv(args.qwen_target_train),
                         ids_tr)[TGT_FEATS].values
    T_te = tgt_aggregate(pd.read_csv(args.qwen_target_test),
                         ids_te)[TGT_FEATS].values
    assert np.isfinite(A_tr).all() and np.isfinite(T_tr).all()
    assert np.isfinite(A_te).all() and np.isfinite(T_te).all()

    ARMS = [("N0_M4锚点", Xtr, Xte),
            ("Q1_+Qwen句级", np.hstack([Xtr, A_tr]), np.hstack([Xte, A_te])),
            ("Q2_+Qwen目标", np.hstack([Xtr, T_tr]), np.hstack([Xte, T_te])),
            ("Q3_+句级+目标", np.hstack([Xtr, A_tr, T_tr]),
             np.hstack([Xte, A_te, T_te]))]

    rows = []
    for seed in r83.CV_SEEDS:
        for name, Xt, _ in ARMS:
            a, tau, oof = r83.cv_select(Xt, ytr, seed, name)
            f1m = f1_score(ytr, (oof >= tau).astype(int), average="macro")
            rows.append({"arm": name, "seed": seed, "alpha": a, "tau": tau,
                         "cv_macro": f1m, "cv_auc": roc_auc_score(ytr, oof)})
            print(f"  {name:<14} seed{seed} alpha={a:<8} "
                  f"OOF宏F1={f1m:.4f}  OOF AUC={roc_auc_score(ytr, oof):.4f}  τ*={tau}")
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")

    means = cv.groupby("arm")["cv_macro"].mean()
    aucs = cv.groupby("arm")["cv_auc"].mean()
    best = max(means.index, key=lambda k: means[k])
    gain = means[best] - means["N0_M4锚点"]
    lines = ["=" * 72,
             "v12 跨教师拼接读出：GLM M4 + Qwen 句级/目标特征",
             "=" * 72,
             "N0_M4锚点 | Q1_+Qwen句级(3) | Q2_+Qwen目标(11) | Q3_+句级+目标(14)", ""]
    lines += [f"CV 均值: " + ", ".join(f"{k}={means[k]:.4f}" for k in means.index),
              "CV AUC: " + ", ".join(f"{k}={aucs[k]:.4f}" for k in aucs.index),
              f"最佳臂: {best}（相对 N0 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: " + ("通过" if gain >= GATE else "未通过"), ""]

    if gain >= GATE:
        ev_name, ev_Xtr, ev_Xte = best, dict(ARMS)[best][1], dict(ARMS)[best][2]
    else:
        ev_name, ev_Xtr, ev_Xte = "N0_M4锚点", Xtr, Xte
        lines += ["门槛未通过，test 仅复评锚点 N0。"]
    te_df = test_eval(ev_Xtr, ytr, ev_Xte, yte, r83.FINAL_SEEDS, ev_name)
    lines += [f"test 评估（{ev_name}，5 种子，全指标）：",
              "  seed  τ宏   宏F1    加权F1   ACC     精确宏   召回宏   |  有毒F1  有毒P   有毒R   无毒F1  |  AUC     τ毒   有毒F1@τ毒",
              "  " + "-" * 124]
    for _, rrow in te_df.iterrows():
        lines.append(
            f"  {int(rrow['seed']):<5} {rrow['tau']:.2f}  "
            f"{rrow['macro']:.4f}  {rrow['weighted']:.4f}  {rrow['acc']:.4f}  "
            f"{rrow['prec_macro']:.4f}  {rrow['rec_macro']:.4f}  |  "
            f"{rrow['toxic']:.4f}  {rrow['prec_t']:.4f}  {rrow['rec_t']:.4f}  "
            f"{rrow['benign']:.4f}  |  {rrow['auc']:.4f}  "
            f"{rrow['tau_toxic']:.2f}  {rrow['toxic_at_tau_t']:.4f}")
    lines.append("  " + "-" * 124)
    for label, col in (("宏F1     ", "macro"), ("加权F1   ", "weighted"),
                       ("ACC     ", "acc"), ("精确率宏 ", "prec_macro"),
                       ("召回率宏 ", "rec_macro"), ("有毒F1   ", "toxic"),
                       ("有毒精确 ", "prec_t"), ("有毒召回 ", "rec_t"),
                       ("无毒F1   ", "benign"), ("AUC     ", "auc"),
                       ("有毒F1@τ毒", "toxic_at_tau_t")):
        lines.append(f"  {label} mean±std: {te_df[col].mean():.4f} ± {te_df[col].std():.4f}")
    lines += ["",
              "历史: v8.1 M4 宏 0.8141±0.0011 / AUC 0.8948",
              "      v11.2 锚点 test 0.8148±0.0010 / AUC 0.8944（GLM 目标特征未过门）",
              "      交叉教师侦察: M4+Qwen句级 @FP vs TP = 0.7526（+0.019）"]
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
