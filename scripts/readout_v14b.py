# -*- coding: utf-8 -*-
"""v14b 读出：非线性门缝封堵 + 四臂全指标（CPU，用户执行）。

背景：v14 Thurstone 双门 G1'/G2' 已 FAIL（配对 delta +0.0005 < +0.010；
max|corr| 0.8583 > 0.7）。但 diag_thurstone_v14【5】条件分层显示 θ 的增量
信息集中在 M4 概率两端带的非线性结构里（低带 0.6618 / 高带 0.6369，τ 决策
带 0.55-0.57）——而 v8.7 只在 M4 264 维上测过非线性（GBDT +0.0000），
从未在 M4+BT 上测过。本脚本封堵（或打开）这最后一道门缝。

四臂（2×2 析因）：
  A0_M4_LR      L1-LR @ M4 264 维（锚点）
  A1_M4BT_LR    L1-LR @ M4+BT 270 维（线性形式：预期被 L1 剪掉）
  B0_M4_GB      GBDT @ M4 264 维（v8.7 超参先验固定：叶15/300轮/lr0.05）
  B1_M4BT_GB    GBDT @ M4+BT 270 维（非线性形式：门缝所在）

协议：CV 5 折 × 3 种子（42/43/44），τ@宏F1，门槛 ≥+0.002（G1'' 主门 =
任意臂相对 A0 的 CV 宏F1 增益）。GBDT 概率未校准，τ 在 train OOF 上选
（v8.6 S7 教训），不做温度缩放（v8.7 N7 已证无额外增益）。

test 评估（全部四臂 5 种子，双口径 τ宏/τ毒，全指标）：**诊断性输出，非
预登记主张**——双门已 FAIL，test 数字仅用于回答"这套方案到底多少 F1"，
不得写为论文结果。

产物：results/v14b/readout_v14b/{metrics.txt, cv_competition.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import readout_v83 as r83
from readout_v12 import assemble_m4
from analyze_thurstone_v14 import BT_FEATS

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def gbdt_oof(X, y, seed):
    """GBDT 5 折 OOF（v8.7 N1 先验超参），返回 (tau_macro, oof, tau_toxic)。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for trn, va in tqdm(cv.split(X, y), desc=f"    GB s{seed}", leave=False,
                        dynamic_ncols=True):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=60, l2_regularization=1.0, random_state=seed)
        m.fit(X[trn], y[trn])
        oof[va] = m.predict_proba(X[va])[:, 1]
    tau = max(r83.TAU_GRID,
              key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    tau_t = max(r83.TAU_GRID,
                key=lambda t: f1_score(y, (oof >= t).astype(int)))
    return float(tau), oof, float(tau_t)


def test_eval_lr(Xtr, ytr, Xte, yte, seeds, tag):
    """L1-LR test 全指标（v12 test_eval 同构）。"""
    rows = []
    for seed in tqdm(seeds, desc=f"  test {tag}", dynamic_ncols=True):
        a, tau, oof = r83.cv_select(Xtr, ytr, seed, f"final[{tag}]")
        m = Pipeline([("m", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / a, max_iter=2000))]).fit(Xtr, ytr)
        p_te = m.predict_proba(Xte)[:, 1]
        rows.append(_metrics_row(seed, a, tau, p_te, yte, tau_toxic=None, ytr=ytr,
                                 oof=oof))
    return pd.DataFrame(rows)


def _metrics_row(seed, alpha, tau, p_te, yte, tau_toxic, ytr, oof):
    if tau_toxic is None:
        tau_toxic = max(r83.TAU_GRID,
                        key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
    pred = (p_te >= tau).astype(int)
    pred_t = (p_te >= tau_toxic).astype(int)
    return {"seed": seed, "alpha": alpha, "tau": tau,
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
            "tau_toxic": tau_toxic,
            "toxic_at_tau_t": f1_score(yte, pred_t)}


def test_eval_gb(Xtr, ytr, Xte, yte, seeds, tag):
    """GBDT test 全指标：τ 从每次 seed 的 train OOF 上选（不校准）。"""
    rows = []
    for seed in tqdm(seeds, desc=f"  test {tag}", dynamic_ncols=True):
        tau, oof, tau_t = gbdt_oof(Xtr, ytr, seed)
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=60, l2_regularization=1.0, random_state=seed)
        m.fit(Xtr, ytr)
        p_te = m.predict_proba(Xte)[:, 1]
        rows.append(_metrics_row(seed, np.nan, tau, p_te, yte,
                                 tau_toxic=tau_t, ytr=ytr, oof=oof))
    return pd.DataFrame(rows)


FMT = ("  seed  τ宏   宏F1    加权F1   ACC     精确宏   召回宏   |  "
       "有毒F1  有毒P   有毒R   无毒F1  |  AUC     τ毒   有毒F1@τ毒")


def fmt_df(df):
    lines = ["  " + FMT, "  " + "-" * 124]
    for _, r in df.iterrows():
        alpha = f"{r['alpha']:.3f}" if np.isfinite(r["alpha"]) else "  --  "
        lines.append(
            f"  {int(r['seed']):<5} {r['tau']:.2f}  "
            f"{r['macro']:.4f}  {r['weighted']:.4f}  {r['acc']:.4f}  "
            f"{r['prec_macro']:.4f}  {r['rec_macro']:.4f}  |  "
            f"{r['toxic']:.4f}  {r['prec_t']:.4f}  {r['rec_t']:.4f}  "
            f"{r['benign']:.4f}  |  {r['auc']:.4f}  "
            f"{r['tau_toxic']:.2f}  {r['toxic_at_tau_t']:.4f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description="v14b 非线性门缝读出（四臂）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--bt_dir", default="results/v14/thurstone")
    ap.add_argument("--output_dir", default="results/v14b/readout_v14b")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    stamp("[加载] 读取矩阵与 BT 特征...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (te["id"].values == v3te["id"].values).all()
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    ftr = pd.read_csv(Path(args.bt_dir) / "bt_features_train.csv")
    fte = pd.read_csv(Path(args.bt_dir) / "bt_features_test.csv")
    assert (ftr["id"].values == tr["id"].values).all(), "BT train id 不对齐"
    assert (fte["id"].values == te["id"].values).all(), "BT test id 不对齐"
    Btr = ftr[BT_FEATS].values
    Bte = fte[BT_FEATS].values
    assert np.isfinite(Btr).all() and np.isfinite(Bte).all()

    stamp("[装配] M4 264 维...")
    Xtr, Xte = assemble_m4(tr, te, v3tr, v3te, freeze, ytr)
    XBtr = np.hstack([Xtr, Btr])
    XBte = np.hstack([Xte, Bte])

    ARMS = [("A0_M4_LR", Xtr, Xte, "lr"), ("A1_M4BT_LR", XBtr, XBte, "lr"),
            ("B0_M4_GB", Xtr, Xte, "gb"), ("B1_M4BT_GB", XBtr, XBte, "gb")]

    # ---- CV 竞争（预登记：门槛 ≥+0.002）----
    rows = []
    for seed in r83.CV_SEEDS:
        for name, Xt, _, kind in ARMS:
            if kind == "lr":
                a, tau, oof = r83.cv_select(Xt, ytr, seed, name)
            else:
                tau, oof, _ = gbdt_oof(Xt, ytr, seed)
                a = np.nan
            f1m = f1_score(ytr, (oof >= tau).astype(int), average="macro")
            auc = roc_auc_score(ytr, oof)
            rows.append({"arm": name, "seed": seed, "alpha": a, "tau": tau,
                         "cv_macro": f1m, "cv_auc": auc})
            print(f"  {name:<12} seed{seed} OOF宏F1={f1m:.4f} AUC={auc:.4f} "
                  f"τ*={tau}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")

    means = cv.groupby("arm")["cv_macro"].mean()
    aucs = cv.groupby("arm")["cv_auc"].mean()
    best = max(means.index, key=lambda k: means[k])
    gain = means[best] - means["A0_M4_LR"]
    lines = ["=" * 128,
             "v14b 非线性门缝读出（2×2：LR/GBDT × M4/M4+BT）",
             "=" * 128,
             "A0_M4_LR | A1_M4BT_LR | B0_M4_GB | B1_M4BT_GB；"
             "GBDT 超参 = v8.7 N1 先验固定", "",
             "CV 宏F1 均值: " + ", ".join(f"{k}={means[k]:.4f}" for k in means.index),
             "CV AUC 均值:  " + ", ".join(f"{k}={aucs[k]:.4f}" for k in aucs.index),
             f"最佳臂: {best}（相对 A0 增益 {gain:+.4f}）",
             f"预登记门槛（≥{r83.GATE}）: " + ("通过" if gain >= r83.GATE else "未通过"),
             "", "test 全指标（**诊断性输出，非预登记主张**——G1'/G2' 已 FAIL，"
             "数字仅回答\u201c这套方案到底多少 F1\u201d）："]

    for name, Xt, Xv, kind in ARMS:
        stamp(f"[test] {name}...")
        df = test_eval_lr(Xt, ytr, Xv, yte, r83.FINAL_SEEDS, name) \
            if kind == "lr" else test_eval_gb(Xt, ytr, Xv, yte, r83.FINAL_SEEDS, name)
        lines += [f"", f"【{name}】", *fmt_df(df),
                  f"  mean: 宏F1={df['macro'].mean():.4f}±{df['macro'].std():.4f}  "
                  f"ACC={df['acc'].mean():.4f}  有毒F1={df['toxic'].mean():.4f}  "
                  f"无毒F1={df['benign'].mean():.4f}  AUC={df['auc'].mean():.4f}"]

    lines += ["", "=" * 128,
              "判读：CV 门过 → 非线性门缝打开（进入终局文档正结果）；"
              "未过 → 门缝封死（0.8159 + 教师状态上界无死角闭环）。"]
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] metrics.txt / cv_competition.csv 已写出")


if __name__ == "__main__":
    main()
