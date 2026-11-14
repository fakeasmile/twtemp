# -*- coding: utf-8 -*-
"""v8.8 条目裁剪模拟：为 Qwen 测量裁剪 v6 问卷（CPU，test 零参与）。

动机：Qwen3.5 处理时长约 GLM 的 3 倍，拟按 GLM 经验裁剪 v6 条目省 GPU 时。
方法：列删除模拟——在 GLM 条目矩阵上按预算 k（每构念保留 GRM 区分度 top-k，
D2 保留全部 5 档）删除条目列，重跑 v8.1 M4 装配 + 3 种子 CV，输出 F1-预算曲线。
裁剪判据（预登记）：某预算的 CV 宏 F1 >= 锚点 - 0.002 即"噪声内"；
选择满足判据的最小 k 作为 Qwen 问卷预算。

注意：这是 GLM 内部的代价模拟；GLM 弱条目在 Qwen 上未必弱，本模拟给出的是
"裁剪的信息代价上界参考"。

产物：results/v88/prune/{metrics.txt, keep_sets.json}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import readout_v83 as r83


def assemble_m4(tr, te, v3tr, v3te, freeze, keep):
    """M4 装配（与 select_pilot_v10 同源），item_cols 限制在 keep 集内。"""
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")
                 and c in keep]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = list(item_cols)
    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    ytr = tr["toxic"].astype(int).values
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
    Atr_items = Atr[kept_items].values
    Mtr, Mte = r83.meta_base(tr["content"].tolist(), te["content"].tolist())
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1]) for j in range(len(kept_items))])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate[kept_items].values[:, order_i]])
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    Xtr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    Xte = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    return Xtr, Xte, kept_items


def main():
    ap = argparse.ArgumentParser(description="v8.8 条目裁剪模拟")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--grm_npz", required=True)
    ap.add_argument("--budgets", default="8,6,5,4,3")
    ap.add_argument("--output_dir", default="results/v88/prune")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    grm = np.load(args.grm_npz, allow_pickle=True)
    a_of = {str(i): float(a) for i, a in zip(grm["items"], grm["a"])}

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    ytr = tr["toxic"].astype(int).values

    budgets = [int(x) for x in args.budgets.split(",")]
    lines = ["v8.8 条目裁剪模拟（GLM 矩阵列删除，M4 装配，3 种子 CV 宏 F1）",
             "=" * 70]
    results = {}
    for k in budgets:
        keep = set()
        for cid, items in freeze["items_per_construct"].items():
            if cid not in freeze["constructs_alive"]:
                continue
            if cid == "D2":
                keep.update(t for t in ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
                            if t in a_of)
                continue
            ranked = sorted([c for c in items if c in a_of],
                            key=lambda c: -a_of[c])
            keep.update(ranked[:k])
        Xtr, _, kept = assemble_m4(tr, te, v3tr, v3te, freeze, keep)
        f1s, aucs = [], []
        for seed in r83.CV_SEEDS:
            _, _, oof = r83.cv_select(Xtr, ytr, seed, f"k{k}s{seed}")
            tau = max(np.round(np.arange(0.30, 0.7005, 0.005), 3),
                      key=lambda t: r83.f1_score(ytr, (oof >= t).astype(int),
                                                 average="macro"))
            f1s.append(r83.f1_score(ytr, (oof >= tau).astype(int), average="macro"))
            aucs.append(roc_auc_score(ytr, oof))
        results[str(k)] = {"n_items": len(kept), "cv_f1": float(np.mean(f1s)),
                           "cv_f1_std": float(np.std(f1s)), "cv_auc": float(np.mean(aucs)),
                           "keep": sorted(keep)}
        lines.append(f"k={k}  条目 {len(kept):3d}  CV宏F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}"
                     f"  OOF AUC {np.mean(aucs):.4f}")

    full = results.get(str(budgets[0]))
    lines.append("")
    for k in budgets[1:]:
        d = results[str(k)]["cv_f1"] - full["cv_f1"]
        ok = d >= -r83.GATE
        lines.append(f"k={k}: 相对全量 Δ={d:+.4f}  {'噪声内 PASS' if ok else '代价超门槛 FAIL'}")

    (out / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "keep_sets.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
