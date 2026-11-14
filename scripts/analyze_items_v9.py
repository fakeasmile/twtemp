# -*- coding: utf-8 -*-
"""v9 意图条目门槛分析（预登记裁决，纯 train，test 零参与）。

输入：run_items_v9.py 产出的 v9 意图条目矩阵（6 条是/否 p_yes）。
方法：复算 v8.3 Q1 锚点的 train OOF（与 diag_expr_audit_v83.py 同法）得到
FN/FP 掩码，对每条 v9 条目裁决预登记门槛：
  G1 expr2-FN vs expr0-FP 单条目 AUC（方向折叠）≥ 0.70 —— 对全题最难边界有效
  G2 与现有 232 维对齐特征 max|corr| < 0.70 —— 不得是表面毒性轴的镜像
  G3 train 全局毒性-无毒 AUC（方向折叠）≥ 0.55
辅助统计（不设门槛，供机制判读）：E2FN vs BEN / E0FP vs TP / E1FN vs BEN /
E2FN vs E2TP 单条目 AUC。
存活条目联合检查：232 + 存活条目增强 CV AUC（expr2-FN vs BEN、E0FP vs TP）
对比审计基线 0.6827 / 0.7340。

预登记裁决规则：
  至少 1 条过 G1+G2+G3 → 进入 v9 读出实验（232+12+存活条目）；
  全灭 → 结论升级为"GLM-4-9B 教师能力边界"，不再设计新测量。

无效测量（p_yes 缺失）填 0.5（无信息先验）。
产物: results/v9/item_gate/{metrics.txt, v9_gate.json}
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

import readout_v83 as r83  # 复用常量与函数，保证 OOF 复现与审计逐位同法


def cv_auc(pos_mask, neg_mask, X, seed=42):
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


def fold_auc(pos_vals, neg_vals):
    """方向折叠的单特征 AUC：返回 (folded_auc, raw_auc, sign)。"""
    yy = np.r_[np.ones(len(pos_vals)), np.zeros(len(neg_vals))]
    v = np.r_[pos_vals, neg_vals]
    a = roc_auc_score(yy, v)
    return max(a, 1 - a), a, (1 if a >= 0.5 else -1)


def main():
    ap = argparse.ArgumentParser(description="v9 意图条目门槛分析")
    ap.add_argument("--v9_train", required=True, help="v9 意图条目矩阵（train）")
    ap.add_argument("--train_matrix", required=True, help="v6 条目矩阵（train）")
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--output_dir", default="results/v9/item_gate")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    # ---- 数据与 232 维对齐特征装配（与 diag_expr_audit_v83.py 逐行一致）----
    tr = pd.read_csv(args.train_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v9tr = pd.read_csv(args.v9_train)
    assert (tr["id"].astype(str).values == v3tr["id"].astype(str).values).all()
    assert (tr["id"].astype(str).values == v9tr["id"].astype(str).values).all(), \
        "v9 矩阵 id 与 train 矩阵不对齐（检查 run_items_v9 的 data_path/split）"
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
    align_feat_names = kept_items + v3_names  # 232
    align_feats = np.hstack([Atr[kept_items].values, v3_tr])

    v9_cols = [c for c in v9tr.columns if c not in ("id", "toxic", "content")]
    V9 = v9tr[v9_cols].fillna(0.5).values  # 无效测量 → 无信息先验

    # ---- OOF 复现（与审计同法：3 种子 cv_select → 均值 → τ@宏F1）----
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]

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
    q1_tr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])

    print("计算 train OOF（3 种子，与 v8.3 审计同法）...")
    oofs = np.column_stack([r83.cv_select(q1_tr, ytr, s, "v9门槛")[2] for s in r83.CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof_prob >= t).astype(int),
                                                     average="macro"))
    err = (oof_prob >= tau_e).astype(int) != ytr
    fn, fp = err & (ytr == 1), err & (ytr == 0)
    fn_total, fp_total = int(fn.sum()), int(fp.sum())

    rings = json.loads(Path(args.v3_rings_json).read_text(encoding="utf-8"))
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].astype(str).values])

    ben = ytr == 0
    g = {
        "E2FN": (expr == 2) & (ytr == 1) & fn,
        "E1FN": (expr == 1) & (ytr == 1) & fn,
        "E2TP": (expr == 2) & (ytr == 1) & ~fn,
        "E0FP": ben & fp,
        "TP": (ytr == 1) & ~fn,
        "BEN": ben,
    }
    exp_fn, exp_fp = 989, 793
    match = abs(fn_total - exp_fn) <= 3 and abs(fp_total - exp_fp) <= 3

    # ---- 逐条目门槛裁决 ----
    g1_thr, g2_thr, g3_thr = 0.70, 0.70, 0.55
    lines = ["=" * 72, "v9 意图条目门槛分析（预登记裁决，纯 train，test 零参与）",
             "=" * 72,
             f"OOF τ*={tau_e:.3f}  FN={fn_total}（期望 {exp_fn}）FP={fp_total}（期望 {exp_fp}）"
             f"  复现校验: {'通过(±3)' if match else '偏离较大，先检查数据再解读'}",
             f"组样本量: " + ", ".join(f"{k}={int(v.sum())}" for k, v in g.items()),
             f"门槛: G1 expr2-FN vs expr0-FP ≥ {g1_thr}；G2 max|corr|<232维 < {g2_thr}；"
             f"G3 全局AUC ≥ {g3_thr}", ""]
    kept, stats = [], []
    for j, c in enumerate(v9_cols):
        v = V9[:, j]
        g3_a, g3_raw, g3_sign = fold_auc(v[ytr == 1], v[ytr == 0])
        g1_a, _, g1_sign = fold_auc(v[g["E2FN"]], v[g["E0FP"]])
        e2fnben_a, _, s1 = fold_auc(v[g["E2FN"]], v[g["BEN"]])
        fptp_a, _, s2 = fold_auc(v[g["E0FP"]], v[g["TP"]])
        e1fnben_a, _, _ = fold_auc(v[g["E1FN"]], v[g["BEN"]])
        e2fntp_a = None
        if g["E2TP"].sum() > 0:
            e2fntp_a, _, _ = fold_auc(v[g["E2FN"]], v[g["E2TP"]])
        # G2: 与 232 维对齐特征的最大相关
        with np.errstate(invalid="ignore"):
            cors = np.abs(np.array([np.corrcoef(v, align_feats[:, k])[0, 1]
                                    for k in range(align_feats.shape[1])]))
        g2_max = float(np.nanmax(cors))
        g2_name = align_feat_names[int(np.nanargmax(cors))]
        ok = (g1_a >= g1_thr) and (g2_max < g2_thr) and (g3_a >= g3_thr)
        if ok:
            kept.append({"item_id": c, "polarity": int(g3_sign)})
        stats.append({"item_id": c, "g1_e2fn_vs_e0fp": g1_a, "g1_sign": int(g1_sign),
                      "g2_maxcorr": g2_max, "g2_maxcorr_with": g2_name,
                      "g3_global_auc": g3_a, "g3_sign": int(g3_sign),
                      "e2fn_vs_ben": e2fnben_a, "e0fp_vs_tp": fptp_a,
                      "e1fn_vs_ben": e1fnben_a,
                      "e2fn_vs_e2tp": None if e2fntp_a is None else e2fntp_a,
                      "pass": ok})
        lines.append(f"[{c}] {'PASS' if ok else 'FAIL'}  "
                     f"G1={g1_a:.3f}({'+' if g1_sign == 1 else '-'})  "
                     f"G2={g2_max:.3f}(vs {g2_name})  "
                     f"G3={g3_a:.3f}({'+' if g3_sign == 1 else '-'})")
        lines.append(f"      辅助: E2FNvsBEN={e2fnben_a:.3f}  E0FPvsTP={fptp_a:.3f}  "
                     f"E1FNvsBEN={e1fnben_a:.3f}  "
                     + (f"E2FNvsE2TP={e2fntp_a:.3f}" if e2fntp_a is not None else "E2FNvsE2TP=n/a"))

    # ---- 存活条目联合增强检查 ----
    lines += ["", "存活条目联合增强（232 维 + 存活 v9 条目，5 折 CV L2-LR）:"]
    if kept:
        kept_names = [k["item_id"] for k in kept]
        X_aug = np.hstack([Rall_tr, v9tr[kept_names].fillna(0.5).values])
        for tag, pm, nm, base in [
            ("expr2-FN vs 全部无毒", g["E2FN"], g["BEN"], 0.6827),
            ("expr0-FP vs 有毒TP", g["E0FP"], g["TP"], 0.7340),
        ]:
            mu, sd = cv_auc(pm, nm, X_aug)
            lines.append(f"  {tag:<24}{mu:.4f}±{sd:.4f}  (审计基线 {base:.4f}, "
                         f"增量 {mu - base:+.4f})")
        verdict = (f"共 {len(kept)} 条存活（{', '.join(kept_names)}）→ 进入 v9 读出实验"
                   if kept else "全部条目未过门槛 → 结论升级为教师能力边界")
    else:
        verdict = "全部条目未过门槛 → 结论升级为'GLM-4-9B 教师能力边界'，不再设计新测量"
    lines += ["", "裁决: " + verdict]

    (out / "v9_gate.json").write_text(json.dumps(
        {"kept": kept, "stats": stats, "oof_tau": tau_e,
         "fn_total": fn_total, "fp_total": fp_total}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
