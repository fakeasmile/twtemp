# -*- coding: utf-8 -*-
"""v8.3 审计：FN/FP 条件判别定量拆分（纯 train OOF，test 零参与，无新测量）。

背景（用户要求对"教师无字面vs意图分辨能力"的结论小心求证）：
  v7.12 的 FPvsTN=0.189 方向是 P(TN>FP)，实际含义是 FP 在 E1/F1/G1 条目上
  得分高于 TN（极性纠缠而非无信号）；expr2(反讽) vs 无毒的单条目判别上限
  达 0.82，说明教师并非对反讽全盲。本脚本补上最后一块：用 v8.3 Q1 锚点的
  train OOF 错误样本，定量拆分 expr2-FN 的两个候选机制——
    兑现断层：教师特征可把 expr2-FN 与无毒分开（多元 CV AUC 高），
              但全局线性读出的单一阈值牺牲了它们
    深水区：  教师特征本身分不开 expr2-FN 与无毒（CV AUC≈0.5），测量信息不足

预登记判读（E2FN vs 全部无毒，232 维教师秩特征，5 折 CV L2-LR）：
  CV AUC < 0.65          → 深水区主导（读出无责，测量层信息不足）
  CV AUC ≥ 0.75          → 兑现断层主导（教师有信息，全局线性边界欠账）
  0.65 ~ 0.75            → 混合
  对照组：expr2-TP / expr1-TP vs 无毒的 CV AUC（应显著更高，证明对照有效）

自校验：复现 v8.3 误差复检的 FN/FP 计数（期望 FN=989 / FP=793，expr2 FN=468，
expr0 FP=793）；不匹配则审计结论降级为"近似复现"并如实标注。

产物: results/v8/diag_v83_audit/{metrics.txt, audit_groups.csv}
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

import readout_v83 as r83  # 复用常量与函数，保证与 v8.3 Q1 特征逐位一致


def cv_auc(pos_mask, neg_mask, X, seed=42):
    """子集内 5 折 CV 的 L2-LR 多元判别 AUC（诊断用，test 无涉）。"""
    idx = np.r_[np.where(pos_mask)[0], np.where(neg_mask)[0]]
    yy = np.r_[np.ones(int(pos_mask.sum())), np.zeros(int(neg_mask.sum()))]
    Xs = X[idx]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for trn, van in cv.split(Xs, yy):
        sc = StandardScaler().fit(Xs[trn])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xs[trn]), yy[trn])
        aucs.append(roc_auc_score(yy[van], m.predict_proba(sc.transform(Xs[van]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def sf_ceil(pos_mask, neg_mask, cols, src):
    """单特征判别上限（方向折叠）；返回 [(folded_auc, name, raw_auc)] 降序。"""
    yy = np.r_[np.ones(int(pos_mask.sum())), np.zeros(int(neg_mask.sum()))]
    out = []
    for c in cols:
        v = np.r_[src[c].values[pos_mask], src[c].values[neg_mask]]
        if np.std(v) < 1e-9:
            continue
        a = roc_auc_score(yy, v)
        out.append((max(a, 1.0 - a), c, a))
    out.sort(reverse=True)
    return out


def fmt_top(results, k=3):
    parts = []
    for fa, c, a in results[:k]:
        sign = "+" if a >= 0.5 else "-"  # +: 组1得分更高 / -: 组2得分更高
        parts.append(f"{c}{sign}{fa:.3f}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="v8.3 审计：FN/FP 条件判别拆分")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v7_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--v4_sa_train", default=None,
                    help="v4 SA/Q 通道矩阵（extract_v4_sa.py 产物）；提供则做意图通道增强审计")
    ap.add_argument("--output_dir", default="results/v8/diag_v83_audit")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    # ---- 数据与特征（与 readout_v83.py main() 的 train 侧逐行一致）----
    tr = pd.read_csv(args.train_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v7tr = pd.read_csv(args.v7_train)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (tr["id"].values == v7tr["id"].values).all()
    ytr = tr["toxic"].astype(int).values

    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw = tr[item_cols]
    Atr = Xtr_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
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
    Ftr_c = v3tr[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < r83.V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    v3_names = kept_c + kept_s
    v3_tr = np.hstack([Ftr_c[kept_c].values, v3tr[kept_s].values])

    prob_names = kept_items + v3_names
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rall_tr = raw_tr.copy()
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)

    Atr_items = Atr[kept_items].values
    Mtr, _ = r83.meta_base(tr["content"].tolist(), tr["content"].tolist())
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(len(kept_items))])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])

    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    q1_tr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])

    # ---- OOF 复现（与 v8.3 误差复检同法：3 种子 OOF 均值 → τ@宏F1）----
    print("计算 train OOF（3 种子，与 v8.3 误差复检同法）...")
    oofs = np.column_stack([r83.cv_select(q1_tr, ytr, s, "审计")[2] for s in r83.CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof_prob >= t).astype(int),
                                                     average="macro"))
    pred = (oof_prob >= tau_e).astype(int)
    err = pred != ytr
    fn = err & (ytr == 1)
    fp = err & (ytr == 0)

    rings = json.loads(Path(args.v3_rings_json).read_text(encoding="utf-8"))
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    lens = np.array([len(str(t).replace(" ", "")) for t in tr["content"].tolist()])

    # ---- 自校验：与 v8.3 metrics.txt 的误差复检对照 ----
    exp_fn, exp_fp = 989, 793
    match = (int(fn.sum()) == exp_fn) and (int(fp.sum()) == exp_fp)
    e2fn = (expr == 2) & fn
    e2fn_exp = 468
    lines = ["=" * 72, "v8.3 审计：FN/FP 条件判别定量拆分（纯 train OOF，test 零参与）",
             "=" * 72,
             f"OOF τ*={tau_e:.3f}  FN {int(fn.sum())}（期望 {exp_fn}）  "
             f"FP {int(fp.sum())}（期望 {exp_fp}）  expr2-FN {int(e2fn.sum())}"
             f"（期望 {e2fn_exp}）",
             f"复现校验: {'通过——OOF 错误与 v8.3 逐样本一致' if match else '未逐位匹配，以下结论按近似复现解读'}",
             f"FP 的 expression 组成: {dict(zip(*np.unique(expr[fp], return_counts=True)))}", ""]

    # ---- 分组定义 ----
    ben = ytr == 0
    tp_all = (ytr == 1) & ~fn
    g = {
        "E2FN": (expr == 2) & (ytr == 1) & fn,
        "E2TP": (expr == 2) & (ytr == 1) & ~fn,
        "E1FN": (expr == 1) & (ytr == 1) & fn,
        "E1TP": (expr == 1) & (ytr == 1) & ~fn,
        "E3FN": (expr == 3) & (ytr == 1) & fn,
        "E3TP": (expr == 3) & (ytr == 1) & ~fn,
        "E0FP": ben & fp,
        "TN": ben & ~fp,
        "BEN": ben,
        "TP": tp_all,
    }
    lines.append("组样本量: " + ", ".join(f"{k}={int(v.sum())}" for k, v in g.items()))
    lines.append("expr2 按长度桶的 FN 率: " + ", ".join(
        f"[{lo},{hi})={((expr == 2) & (ytr == 1) & fn & (lens >= lo) & (lens < hi)).sum() / max(((expr == 2) & (ytr == 1) & (lens >= lo) & (lens < hi)).sum(), 1):.3f}"
        for lo, hi in r83.LEN_BINS))
    lines.append("expr2-FN(短<20) n=" + str(int((g['E2FN'] & (lens < 20)).sum())))

    # ---- 多元条件判别（232 维教师秩特征，5 折 CV L2-LR）----
    v7_cols = [c for c in v7tr.columns if c not in ("id", "toxic", "content")]
    v3_cols = [c for c in v3tr.columns if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    pools = [("v6条目", item_cols, tr), ("v7靶向", v7_cols, v7tr), ("v3通道", v3_cols, v3tr)]

    sa_tr = None
    sa_cols = []
    if args.v4_sa_train:
        sa_tr = pd.read_csv(args.v4_sa_train)
        assert (tr["id"].astype(str).values == sa_tr["id"].astype(str).values).all(), \
            "v4 SA 矩阵 id 与 train 矩阵不对齐"
        sa_cols = [c for c in sa_tr.columns if c != "id"]
        pools.append(("v4意图通道", sa_cols, sa_tr))

    comps = [
        ("expr2-FN vs 全部无毒", g["E2FN"], g["BEN"]),
        ("expr2-FN(短<20) vs 全部无毒", g["E2FN"] & (lens < 20), g["BEN"]),
        ("expr2-FN vs expr2-TP", g["E2FN"], g["E2TP"]),
        ("expr2-TP vs 全部无毒（对照）", g["E2TP"], g["BEN"]),
        ("expr1-FN vs 全部无毒", g["E1FN"], g["BEN"]),
        ("expr1-TP vs 全部无毒（对照）", g["E1TP"], g["BEN"]),
        ("expr3-FN vs 全部无毒", g["E3FN"], g["BEN"]),
        ("expr3-TP vs 全部无毒（对照）", g["E3TP"], g["BEN"]),
        ("expr0-FP vs TN", g["E0FP"], g["TN"]),
        ("expr0-FP vs 有毒TP", g["E0FP"], g["TP"]),
        ("expr0-FP vs expr1-TP", g["E0FP"], g["E1TP"]),
        ("expr0-FP vs expr2-TP", g["E0FP"], g["E2TP"]),
    ]
    lines += ["", "多元条件判别（教师秩特征 232 维，5 折 CV L2-LR AUC；"
              "'+'=组1得分更高，'-'=组2得分更高）:",
              f"  {'比较':<28}{'n1/n2':<12}{'CV AUC':<14}单特征上限(top3×3池)"]
    rows = []
    for tag, pm, nm in comps:
        mu, sd = cv_auc(pm, nm, Rall_tr)
        lines.append(f"  {tag:<28}{int(pm.sum())}/{int(nm.sum()):<10}"
                     f"{mu:.4f}±{sd:.4f}")
        for pname, cols, src in pools:
            top = sf_ceil(pm, nm, cols, src)
            lines.append(f"      {pname:<8} {fmt_top(top)}")
        rows.append({"comparison": tag, "n_pos": int(pm.sum()), "n_neg": int(nm.sum()),
                     "cv_auc_mean": mu, "cv_auc_std": sd})

    # ---- 意图通道增强判别（232 维 + SA/Q 12 维，关键比较）----
    aug_auc = {}
    if sa_tr is not None:
        X_aug = np.hstack([Rall_tr, sa_tr[sa_cols].values])
        key = ["expr2-FN vs 全部无毒", "expr2-FN vs expr2-TP",
               "expr0-FP vs 有毒TP", "expr0-FP vs expr2-TP"]
        lines += ["", "意图通道增强判别（教师秩特征 232 + SA/Q 12 = 244 维，5 折 CV L2-LR）:"]
        for tag, pm, nm in comps:
            if tag not in key:
                continue
            mu, sd = cv_auc(pm, nm, X_aug)
            base_mu = [r["cv_auc_mean"] for r in rows if r["comparison"] == tag][0]
            aug_auc[tag] = mu
            lines.append(f"  {tag:<28}{mu:.4f}±{sd:.4f}  (基线 {base_mu:.4f}, "
                         f"增量 {mu - base_mu:+.4f})")

    # ---- OOF 概率分布（兑现断层的直接证据）----
    lines += ["", "OOF 概率分位数（p10/p25/p50/p75/p90，τ*=%.3f）:" % tau_e]
    for k in ["E2TP", "E2FN", "E1TP", "E1FN", "E3TP", "E3FN", "TN", "E0FP"]:
        v = oof_prob[g[k]]
        qs_p = np.percentile(v, [10, 25, 50, 75, 90])
        lines.append(f"  {k:<6} n={int(g[k].sum()):<5} "
                     + " ".join(f"{q:.3f}" for q in qs_p))

    # ---- 预登记判读 ----
    e2_mu = [r["cv_auc_mean"] for r in rows if r["comparison"] == "expr2-FN vs 全部无毒"][0]
    e2tp_mu = [r["cv_auc_mean"] for r in rows if r["comparison"] == "expr2-TP vs 全部无毒（对照）"][0]
    fp_mu = [r["cv_auc_mean"] for r in rows if r["comparison"] == "expr0-FP vs TN"][0]
    fptp_mu = [r["cv_auc_mean"] for r in rows if r["comparison"] == "expr0-FP vs 有毒TP"][0]
    if fptp_mu >= 0.65:
        fp_verdict = ("教师存在把 FP 从毒样池中摘出的信号（use/mention 缺口属"
                      "'测量存在但读出未利用/极性纠缠'）")
    elif fptp_mu < 0.55:
        fp_verdict = "use/mention 维度在教师测量中缺失，FP 修复必须依赖新测量"
    else:
        fp_verdict = "use/mention 信号微弱，修复 FP 主要依赖新测量"
    if e2_mu < 0.65:
        verdict = "深水区主导：教师特征分不开 expr2-FN 与无毒，测量层信息不足，读出无责"
    elif e2_mu >= 0.75:
        verdict = "兑现断层主导：教师有信息但全局线性阈值牺牲了 expr2-FN"
    else:
        verdict = "混合：部分可分被阈值牺牲，部分为测量盲区"
    lines += ["", "预登记判读:",
              f"  expr2-FN vs 无毒 CV AUC = {e2_mu:.4f} → {verdict}",
              f"  对照 expr2-TP vs 无毒 = {e2tp_mu:.4f}"
              f"（若显著更高则对照有效）",
              f"  expr0-FP vs TN CV AUC = {fp_mu:.4f}"
              f"（结合单特征方向符号判读极性纠缠）",
              f"  expr0-FP vs 有毒TP CV AUC = {fptp_mu:.4f} → {fp_verdict}"]
    if sa_tr is not None and aug_auc:
        d1 = aug_auc.get("expr2-FN vs 全部无毒", np.nan) - e2_mu
        d2 = aug_auc.get("expr0-FP vs 有毒TP", np.nan) - fptp_mu
        if d1 >= 0.05 or d2 >= 0.05:
            sa_verdict = "SA/Q 携带中带判别信息（增量≥+0.05），可直接进读出实验（v8.4）"
        elif d1 < 0.02 and d2 < 0.02:
            sa_verdict = "现有 SA/Q 测量不携带中带信息，需设计全新意图测量"
        else:
            sa_verdict = "SA/Q 信号微弱（增量 +0.02~0.05），证据不足"
        lines.append(f"  意图通道增强: expr2-FN增量 {d1:+.4f} / FPvsTP增量 {d2:+.4f} "
                     f"→ {sa_verdict}")

    pd.DataFrame(rows).to_csv(out / "audit_groups.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
