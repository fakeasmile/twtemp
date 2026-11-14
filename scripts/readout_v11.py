# -*- coding: utf-8 -*-
"""v11.2 读出：M4 基座 + 攻击目标三因子特征（275/269 维）CV 竞争。

v11.2 双门全过（全项目 13 轮首个 G1+G2 双 PASS 通道）后的正式读出：
  N0_M4锚点    M4 264 维逐字节复刻（与 v8.3-v8.8 装配逐行一致）
  T1_全并入    M4 + 11 个目标特征（raw 追加）
  T2_soft核心  M4 + 5 个核心特征（n_attacked / grp_max_asserted /
               soft_grp_asserted / soft_quoter_asserted / gap_soft_target，
               G1 单特征最优 soft_grp_asserted 所在族）

协议（预登记，同 v8.x）：CV 5 折×3 种子（42/43/44）竞争，τ@宏F1，
门槛 ≥+0.002 才碰 test；胜者 5 种子（0-4）test 双口径。
产物：results/v11/readout_v11/{metrics.txt, cv_competition.csv}
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
from analyze_target_v11 import FEATS as TGT_FEATS, aggregate as tgt_aggregate

GATE = r83.GATE
CORE = ["n_attacked", "grp_max_asserted", "soft_grp_asserted",
        "soft_quoter_asserted", "gap_soft_target"]
ARMS = [("N0_M4锚点", None), ("T1_全并入", "all"), ("T2_soft核心", "core")]


def main():
    ap = argparse.ArgumentParser(description="v11.2 读出：M4 + 攻击目标特征")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--target_flat_train",
                    default="results/v11/target/target_flat_train.csv")
    ap.add_argument("--target_flat_test",
                    default="results/v11/target/target_flat_test.csv")
    ap.add_argument("--output_dir", default="results/v11/readout_v11")
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

    # ---- M4 特征装配（与 v8.3/v8.5-v8.8 逐行一致）----
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

    # ---- 目标特征（v11.2 Stage3 同一 aggregate 函数，train/test 各自聚合）----
    tf_tr = pd.read_csv(args.target_flat_train)
    tf_te = pd.read_csv(args.target_flat_test)
    Ftr_t = tgt_aggregate(tf_tr, tr["id"].values)[TGT_FEATS].values
    Fte_t = tgt_aggregate(tf_te, te["id"].values)[TGT_FEATS].values
    assert np.isfinite(Ftr_t).all() and np.isfinite(Fte_t).all()
    core_idx = [TGT_FEATS.index(c) for c in CORE]
    variants = {"N0_M4锚点": (Xtr, Xte),
                "T1_全并入": (np.hstack([Xtr, Ftr_t]), np.hstack([Xte, Fte_t])),
                "T2_soft核心": (np.hstack([Xtr, Ftr_t[:, core_idx]]),
                                np.hstack([Xte, Fte_t[:, core_idx]]))}
    for name, (a, _) in variants.items():
        print(f"  {name}: {a.shape[1]} 维")

    # ---- CV 竞争（5 折 × 3 种子，τ@宏F1）----
    lines = ["=" * 72, "v11.2 读出：M4 基座 + 攻击目标三因子特征", "=" * 72,
             "N0_M4锚点 | T1_全并入(11特征) | T2_soft核心(5特征)", ""]
    rows = []
    for seed in r83.CV_SEEDS:
        for name, (Xa, _) in variants.items():
            a_alpha, tau, oof = r83.cv_select(Xa, ytr, seed, name)
            f1m = f1_score(ytr, (oof >= tau).astype(int), average="macro")
            rows.append({"config": name, "seed": seed, "alpha": a_alpha,
                         "oof_macro_f1": f1m, "oof_auc": roc_auc_score(ytr, oof),
                         "oof_tau": tau})
            lines.append(f"  {name:<12} seed{seed}  alpha={a_alpha:<8} OOF宏F1="
                         f"{f1m:.4f}  OOF AUC={roc_auc_score(ytr, oof):.4f}"
                         f"  τ*={tau:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    t_best = mean_f1[["T1_全并入", "T2_soft核心"]].idxmax()
    gain = mean_f1[t_best] - mean_f1["N0_M4锚点"]
    gate_pass = gain >= GATE
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳臂: {t_best}（相对 N0 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过'}", ""]

    # ---- test（胜者或锚点，5 种子双口径）----
    test_name = t_best if gate_pass else "N0_M4锚点"
    Xtr_use, Xte_use = variants[test_name]
    if not gate_pass:
        lines.append("门槛未通过，test 仅复评锚点 N0。")
    per_seed = []
    for seed in tqdm(r83.FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a_alpha, tau, oof = r83.cv_select(Xtr_use, ytr, seed, test_name)
        tau_t = max(r83.TAU_GRID, key=lambda t: f1_score(
            ytr, (oof >= t).astype(int)))
        m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a_alpha, max_iter=2000)
                         ).fit(Xtr_use, ytr)
        prob = m.predict_proba(Xte_use)[:, 1]
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
              f"  无毒F1 mean±std: {per['benign_f1'].mean():.4f} ± {per['benign_f1'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v8.1 M4 宏 0.8141±0.0011 / 有毒 0.8249 (@τ毒 0.8310) / AUC 0.8948",
              "      v8.8 锚点 test 0.8154±0.0016 / AUC 0.8943",
              "      v8.6 S7 GBDT CV 过门 test 未兑现（0.8093）——警惕 τ 转移"]

    comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
