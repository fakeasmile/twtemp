# -*- coding: utf-8 -*-
"""v10 pilot 选样：从 train 中分层抽取言语行为 pilot 样本（CPU，test 零参与）。

流程：复刻 M4 锚点装配 → 3 种子 OOF → 稳定错误掩码（≥2/3 种子判错）→
分层抽样：
  E0FP   200   expr0 引用/谈论型，被稳定判为有毒（无毒 F1 的全部成本）
  E2FN   150   expr2 反讽，被稳定漏检（有毒 F1 的表达组缺口）
  TP     100   被稳定正确检出的有毒（随机）
  TN     100   被稳定正确判无毒（随机）
产物：pilot_ids.csv（id/content/toxic/expression/group/oof_mean）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import readout_v83 as r83


def assemble_m4(tr, te, v3tr, v3te, freeze):
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
    ytr = tr["toxic"].astype(int).values
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
    Atr_items = Atr[kept_items].values
    Mtr, Mte = r83.meta_base(tr["content"].tolist(), te["content"].tolist())
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1]) for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    top_names = [kept_items[i] for i in order_i]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate[kept_items].values[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    Xtr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    Xte = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    return Xtr, Xte


def main():
    ap = argparse.ArgumentParser(description="v10 pilot 选样")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--output_dir", default="results/v10/pilot")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    ytr = tr["toxic"].astype(int).values

    print("复刻 M4 锚点装配 ...")
    Xtr, _ = assemble_m4(tr, te, v3tr, v3te, freeze)

    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])

    print("锚点 OOF（3 种子，稳定错误掩码）...")
    errs, oofs = [], []
    for seed in r83.CV_SEEDS:
        _, tau, oof = r83.cv_select(Xtr, ytr, seed, f"s{seed}")
        errs.append((oof >= tau).astype(int) != ytr)
        oofs.append(oof)
    E = np.column_stack(errs)
    oof_mean = np.column_stack(oofs).mean(axis=1)
    stable_err = E.sum(axis=1) >= 2
    stable_ok = E.sum(axis=1) <= 1

    fp_m = stable_err & (ytr == 0)
    fn_m = stable_err & (ytr == 1)
    tp_m = stable_ok & (ytr == 1)
    tn_m = stable_ok & (ytr == 0)
    e2fn_m = fn_m & (expr == 2)
    print(f"稳定错误：FP {fp_m.sum()}（其中 expr0 {(fp_m & (expr == 0)).sum()}）"
          f"  FN {fn_m.sum()}（其中 expr2 {e2fn_m.sum()}）")

    rng = np.random.RandomState(42)

    def take(mask, n):
        idx = np.where(mask)[0]
        return rng.choice(idx, size=min(n, len(idx)), replace=False)

    picks = {g: take(m, n) for g, m, n in
             [("E0FP", fp_m, 200), ("E2FN", e2fn_m, 150),
              ("TP", tp_m, 100), ("TN", tn_m, 100)]}
    rows = []
    for g, idx in picks.items():
        for i in idx:
            rows.append({"id": tr["id"].iloc[i], "content": tr["content"].iloc[i],
                         "toxic": int(ytr[i]), "expression": int(expr[i]),
                         "group": g, "oof_mean": oof_mean[i]})
    pilot = pd.DataFrame(rows)
    pilot.to_csv(out / "pilot_ids.csv", index=False, encoding="utf-8-sig")
    print(f"pilot 样本 {len(pilot)} 条写入 {out / 'pilot_ids.csv'}")
    print(pilot.groupby("group").agg(n=("id", "size"), toxic_rate=("toxic", "mean")))


if __name__ == "__main__":
    main()
