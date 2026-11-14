# -*- coding: utf-8 -*-
"""v8.8 读出：v3b_ 通道（stage2 p_B）并入 M4 装配 —— 测量方法变量清单 Tier-1。

背景（写脚本前数据探查已确证，glm v3_soft_train_matrix.csv）：
  1. M4 装配的 prefix 过滤器只收 v3p_/v3pd_/v3d_，v3b_ 是 v3 软编码中唯一
     从未进过装配的通道（35 列，parse_v3_soft.py 产出）。
  2. 触发样本上 v3b_ + v3d_ ≡ 1（|sum-1| 中位 0.0000，35 概念全体验证），
     且触发指示 = 1[v3p_ ≥ 0.5]（确定性触发规则）。因此 v3b_ 对线性读出的
     新增基 = 每概念的硬门限指示器（软 p_yes 的阈值非线性变换），
     不是独立信息面——本实验检验"硬门限基是否携带软通道外的增量"。
  3. v3b_ 单列 AUC 中位 0.757 / max 0.817，22/35 列方差存活（13 列恒定）。

双臂设计（预登记）：
  A1 同配方并入：v3b_ 走与 v3p_ 完全相同的筛选管线（方差 → AUC≥V3_MIN_AUC →
     经验极性对齐 → 与 kept_items max|corr|<V3_MAX_CORR），追加进秩变换块。
  A2 正交残差臂：在 A1 存活列上追加与已并入全部概率特征（kept_items +
     kept_c + kept_s）max|corr|<V3_MAX_CORR 过滤，只留对现有矩阵正交的门限信息。
  N0 锚点：M4 逐字节复刻（与 v8.3/v8.5/v8.6/v8.7 装配逐行一致）。

协议：CV 5 折 × 3 种子（42/43/44），τ@宏F1，门槛 ≥+0.002 才碰 test；
胜者 5 种子（0-4）test 双口径（τ@宏 / τ@毒）。
产物：results/v88/v3b_merge/{metrics.txt, cv_competition.csv, kept_columns.json}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import readout_v83 as r83

GATE = r83.GATE


def rank_new_cols(raw_tr, raw_te):
    """对新列做与 v8.1 秩变换逐字节一致的 101 分位变换。

    秩变换逐列独立（分位数按列计算），对已有列追加新列不改变其取值，
    因此"列追加进 raw 块统一变换"与"单独变换后 hstack"严格等价。
    """
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rtr = np.zeros(raw_tr.shape, dtype=float)
    Rte = np.zeros(raw_te.shape, dtype=float)
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rtr[:, j] = np.interp(raw_tr[:, j], edges, qs)
        Rte[:, j] = np.interp(raw_te[:, j], edges, qs)
    return Rtr, Rte


def main():
    ap = argparse.ArgumentParser(description="v8.8 v3b_ 通道并入读出")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v88/v3b_merge")
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

    # ---- M4 特征装配（与 v8.3/v8.5/v8.6/v8.7 逐行一致）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    n_items = len(kept_items)
    Atr_items, Ate_items = Atr[kept_items].values, Ate[kept_items].values

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
    assert Xtr.shape[1] == len(mcols) + len(prob_names) + len(mnames) + r83.N_TOP_INTERACT
    print(f"M4 特征装配完成：{Xtr.shape[1]} 维（train {Xtr.shape[0]} / test {Xte.shape[0]}）")

    # ---- v3b_ 双臂筛选 ----
    b_cols = [c for c in all_cols if c.startswith("v3b_")]
    b_var = [c for c in b_cols if v3tr[c].std() > 1e-9]
    b_auc, b_pol = {}, {}
    for c in b_var:
        a = roc_auc_score(ytr, v3tr[c].values)
        b_auc[c] = float(max(a, 1.0 - a))
        b_pol[c] = 1 if a >= 0.5 else -1
    b_valid = [c for c in b_var if b_auc[c] >= r83.V3_MIN_AUC]
    Ftr_b, Fte_b = v3tr[b_valid].copy(), v3te[b_valid].copy()
    for c in b_valid:
        if b_pol[c] == -1:
            Ftr_b[c] = 1.0 - v3tr[c]
            Fte_b[c] = 1.0 - v3te[c]
    kept_b1 = []
    for c in b_valid:
        r = np.abs(np.corrcoef(Ftr_b[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < r83.V3_MAX_CORR:
            kept_b1.append(c)
    exist_tr = np.hstack([Atr_items, Ftr_c[kept_c].values, v3tr[kept_s].values])
    kept_b2 = []
    b2_maxcorr = {}
    for c in kept_b1:
        r = np.abs(np.corrcoef(Ftr_b[c].values, exist_tr.T)[0, 1:])
        b2_maxcorr[c] = float(np.nanmax(r))
        if b2_maxcorr[c] < r83.V3_MAX_CORR:
            kept_b2.append(c)
    print(f"v3b_ 筛选漏斗: 候选{len(b_cols)} → 方差{len(b_var)} → AUC≥{r83.V3_MIN_AUC}:"
          f"{len(b_valid)} → A1条目相关:{len(kept_b1)} → A2正交:{len(kept_b2)}")

    # ---- 双臂装配 ----
    arms, arm_X = [], {}
    if kept_b1:
        B1tr, B1te = rank_new_cols(Ftr_b[kept_b1].values, Fte_b[kept_b1].values)
        arms.append("A1_v3b同配方")
        arm_X["A1_v3b同配方"] = (np.hstack([Xtr, B1tr]), np.hstack([Xte, B1te]))
    if kept_b2:
        B2tr, B2te = rank_new_cols(Ftr_b[kept_b2].values, Fte_b[kept_b2].values)
        arms.append("A2_v3b正交残差")
        arm_X["A2_v3b正交残差"] = (np.hstack([Xtr, B2tr]), np.hstack([Xte, B2te]))

    lines = ["=" * 72, "v8.8 v3b_ 通道（stage2 p_B）并入 M4 读出", "=" * 72,
             f"机制：v3b_ = 1[触发] − v3d_（触发样本上严格互补），新增基 = "
             f"硬门限指示器 1[v3p_≥0.5]",
             f"v3b_ 筛选漏斗: 候选{len(b_cols)} → 方差{len(b_var)} → "
             f"AUC≥{r83.V3_MIN_AUC}:{len(b_valid)} → A1:{len(kept_b1)} → A2:{len(kept_b2)}",
             "  A1 存活: " + (", ".join(kept_b1) if kept_b1 else "无"),
             "  A2 存活: " + (", ".join(f"{c}(maxcorr={b2_maxcorr[c]:.2f},"
                                        f"auc={b_auc[c]:.2f})" for c in kept_b2)
                              if kept_b2 else "无"), ""]
    if kept_b2:
        for c in kept_b2:
            lines.append(f"    {c}: AUC={b_auc[c]:.4f} 极性={'+' if b_pol[c]==1 else '-'} "
                         f"与现有矩阵max|corr|={b2_maxcorr[c]:.4f}")
        lines.append("")

    # ---- CV 竞争 ----
    rows = []
    for seed in r83.CV_SEEDS:
        a0, tau0, oof0 = r83.cv_select(Xtr, ytr, seed, "N0")
        f10 = f1_score(ytr, (oof0 >= tau0).astype(int), average="macro")
        rows.append({"config": "N0_M4锚点", "seed": seed, "alpha": a0,
                     "oof_macro_f1": f10, "oof_auc": roc_auc_score(ytr, oof0),
                     "oof_tau": tau0})
        lines.append(f"  N0_M4锚点   seed{seed}  alpha={a0:<8} OOF宏F1={f10:.4f}"
                     f"  OOF AUC={roc_auc_score(ytr, oof0):.4f}  τ*={tau0:.2f}")
        for name in arms:
            Xa, _ = arm_X[name]
            aa, taua, oofa = r83.cv_select(Xa, ytr, seed, name)
            f1a = f1_score(ytr, (oofa >= taua).astype(int), average="macro")
            rows.append({"config": name, "seed": seed, "alpha": aa,
                         "oof_macro_f1": f1a, "oof_auc": roc_auc_score(ytr, oofa),
                         "oof_tau": taua})
            lines.append(f"  {name:<12} seed{seed}  alpha={aa:<8} OOF宏F1={f1a:.4f}"
                         f"  OOF AUC={roc_auc_score(ytr, oofa):.4f}  τ*={taua:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    if arms:
        best_arm = mean_f1[arms].idxmax()
        gain = float(mean_f1[best_arm] - mean_f1["N0_M4锚点"])
        gate_pass = gain >= GATE
    else:
        best_arm, gain, gate_pass = None, float("nan"), False
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items())]
    if arms:
        lines.append(f"最佳臂: {best_arm}（相对 N0 增益 {gain:+.4f}）")
        lines.append(f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过'}")

    # ---- test（胜者或锚点）----
    test_name = best_arm if gate_pass else "N0_M4锚点"
    Xw_tr, Xw_te = arm_X[test_name] if gate_pass else (Xtr, Xte)
    if not gate_pass:
        lines.append("门槛未通过，test 仅复评锚点 N0。"
                     if arms else "无存活 v3b_ 列，无臂可竞争，test 仅复评锚点。")
    per_seed = []
    for seed in tqdm(r83.FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau, oof = r83.cv_select(Xw_tr, ytr, seed, "final")
        tau_t = max(r83.TAU_GRID,
                    key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
        prob = m.predict_proba(Xw_te)[:, 1]
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
              "历史: v8.7 锚点 CV 均值 0.8122 / test 0.8153±0.0010 / AUC 0.8944",
              "      v8.1 M4 test 宏 0.8141 / AUC 0.8948"]

    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    (out / "kept_columns.json").write_text(
        json.dumps({"kept_b1": kept_b1, "kept_b2": kept_b2, "b2_maxcorr": b2_maxcorr,
                    "b_auc": b_auc, "b_pol": b_pol}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
