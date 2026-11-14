# -*- coding: utf-8 -*-
"""v7.11 读出：v7 靶向条目（E1反讽/F1暗语/G1针对性贬损）并入合并池。

背景：v6.10 误差分析定位两类剩余失败模式——
  显式组 FN（dogwhistle/阴阳怪气/玩梗，768/938）与 expr0 FP（谈论vs实施，874）。
v7 增补 24 条靶向条目（cbm_concept_set_v7_addendum.csv），独立测量后并入。

工序（test 零参与，预登记）：
  1) v7 新条目：train 经验极性对齐 → 方差>0 保留；构念分 = 点二列相关加权均值
     （与 v6.4 同配方，权重 train 冻结）
  2) 三配置 CV 竞争（5 折 × 3 种子，τ@宏F1，alpha 网格含 1e-5）：
     K1 = v6.9 H3 复刻（基座162 + v3_soft）锚点
     K2 = K1 + 24 条新条目（微层）
     K3 = K1 + 3 个新构念分 + 24 条新条目（宏+微）
  3) 新构念块增量裁决：胜者配置上 leave-one-block-out（E1/F1/G1），
     Δ<0 的块声明剔除候选
  4) 胜者 5 种子 test 各评一次，τ@宏F1 与 τ@有毒F1 双口径

预登记门槛：K2/K3 须 CV 宏F1 ≥ K1 + 0.002 才碰 test；否则靶向条目判无增益。

产物：results/v7/readout_v711/{metrics.txt, cv_competition.csv, item_audit_v7.csv,
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
NEW_BLOCKS = ["E1", "F1", "G1"]


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


def construct_features(df, freeze, A, Xr, extra_blocks=None):
    """v6 宏层 21 维 + 可选新构念分。extra_blocks: {cid: (kept_items, weights)}"""
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
    if extra_blocks:
        for cid, (kept, ws) in extra_blocks.items():
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
    ap = argparse.ArgumentParser(description="v7.11 读出：靶向条目并入合并池")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v7_train", required=True, help="v7 条目矩阵（train）")
    ap.add_argument("--v7_test", required=True, help="v7 条目矩阵（test）")
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v7/readout_v711")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v7tr = pd.read_csv(args.v7_train)
    v7te = pd.read_csv(args.v7_test)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v7tr["id"].values).all(), "train id 未对齐（v7）"
    assert (te["id"].values == v7te["id"].values).all(), "test id 未对齐（v7）"
    assert (tr["id"].values == v3tr["id"].values).all(), "train id 未对齐（v3）"
    assert (te["id"].values == v3te["id"].values).all(), "test id 未对齐（v3）"
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    # ---- v7 新条目处理（train 冻结：极性对齐 + 方差过滤 + 点二列权重）----
    v7_cols = [c for c in v7tr.columns if c not in ("id", "toxic", "content")]
    Ntr_raw, Nte_raw = v7tr[v7_cols], v7te[v7_cols]
    Ntr, Nte = Ntr_raw.copy(), Nte_raw.copy()
    pol7, auc7 = {}, {}
    for c in v7_cols:
        if v7tr[c].std() <= 1e-9:
            pol7[c] = 0  # 恒定，弃用
            continue
        a = roc_auc_score(ytr, v7tr[c].values)
        auc7[c] = max(a, 1.0 - a)
        pol7[c] = 1 if a >= 0.5 else -1
    alive7 = [c for c in v7_cols if pol7[c] != 0]
    for c in alive7:
        if pol7[c] == -1:
            Ntr[c] = 1.0 - v7tr[c]
            Nte[c] = 1.0 - v7te[c]
    # 构念分权重（点二列相关，v6.4 配方）
    blocks7 = {}
    for cid in NEW_BLOCKS:
        kept = [c for c in alive7 if c.startswith(cid + "-")]
        if not kept:
            continue
        ws = np.array([abs(np.corrcoef(Ntr[c].values, ytr)[0, 1]) for c in kept])
        if ws.sum() <= 1e-9:
            ws = np.ones(len(kept))
        blocks7[cid] = (kept, ws / ws.sum())
    print(f"v7 条目: {len(v7_cols)} 条 → 存活 {len(alive7)}（恒定剔除 "
          f"{len(v7_cols) - len(alive7)}），极性翻转 {sum(1 for c in alive7 if pol7[c] == -1)}")
    for cid, (kept, _) in blocks7.items():
        print(f"  {cid}: {len(kept)} 条存活, 单条 AUC "
              + ", ".join(f"{c}={auc7[c]:.3f}" for c in kept))

    pd.DataFrame({"item": v7_cols, "auc": [auc7.get(c, np.nan) for c in v7_cols],
                  "polarity": [pol7[c] for c in v7_cols],
                  "alive": [pol7[c] != 0 for c in v7_cols]}).to_csv(
        out / "item_audit_v7.csv", index=False, encoding="utf-8-sig")

    # ---- 基座 162（v6.5 复刻）----
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

    # ---- v3 soft 特征筛选（train 冻结，与 v6.9 同规则）----
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
    v6_ref = Atr[kept_items]
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, v6_ref.values.T)[0, 1:])
        if np.nanmax(r) < V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    Ftr = pd.concat([Ftr_c[kept_c], v3tr[kept_s]], axis=1)
    Fte = pd.concat([Fte_c[kept_c], v3te[kept_s]], axis=1)

    v3_tr, v3_te = Ftr.values, Fte.values
    n7_tr, n7_te = Ntr[alive7].values, Nte[alive7].values
    m7tr_full = construct_features(v7tr, freeze, Ntr, Ntr_raw, extra_blocks=blocks7)
    m7te_full = construct_features(v7te, freeze, Nte, Nte_raw, extra_blocks=blocks7)
    # 新构念分列（只取 E1/F1/G1_mean + 无 V1/V2 重复）
    m7cols = [f"{cid}_mean" for cid in blocks7]
    m7_tr = m7tr_full[m7cols].values
    m7_te = m7te_full[m7cols].values

    base_full_tr = np.hstack([base_tr, v3_tr])
    base_full_te = np.hstack([base_te, v3_te])
    cfg = {
        "K1_H3复刻锚点": (base_full_tr, base_full_te,
                          mcols + kept_items + kept_c + kept_s),
        "K2_加24条目": (np.hstack([base_full_tr, n7_tr]),
                        np.hstack([base_full_te, n7_te]),
                        mcols + kept_items + kept_c + kept_s + alive7),
        "K3_加构念分+条目": (np.hstack([base_full_tr, m7_tr, n7_tr]),
                             np.hstack([base_full_te, m7_te, n7_te]),
                             mcols + kept_items + kept_c + kept_s + m7cols + alive7),
    }

    lines = ["=" * 72, "v7.11 读出（靶向条目并入合并池，三配置 CV 竞争，τ@宏F1）", "=" * 72,
             f"v7 条目: {len(v7_cols)} → 存活 {len(alive7)}，翻转 "
             f"{sum(1 for c in alive7 if pol7[c] == -1)}；构念分 {len(m7cols)} 维",
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
            lines.append(f"  {tag:<18} seed{seed}  alpha={a:<8} OOF宏F1={oof_f1:.4f}  "
                         f"OOF AUC={oof_auc:.4f}  τ*={t:.2f}")
    comp = pd.DataFrame(rows)
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    best_k = mean_f1.idxmax()
    gain = mean_f1.max() - mean_f1["K1_H3复刻锚点"]
    gate_pass = best_k != "K1_H3复刻锚点" and gain >= 0.002
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best_k}（相对 K1 增益 {gain:+.4f}）",
              f"预登记门槛（≥0.002）: {'通过' if gate_pass else '未通过——靶向条目判无增益，不碰 test'}",
              ""]

    # ---- 新构念块增量裁决（在最佳配置上 leave-one-block-out）----
    Xb_tr, Xb_te, b_names = cfg[best_k]
    blk_rows = []
    blk_of = {}
    for i, nm in enumerate(b_names):
        if nm in alive7:
            blk_of[nm] = "微层-" + nm.split("-")[0]
        elif nm in m7cols:
            blk_of[nm] = "宏层-" + nm.replace("_mean", "")
    for target in sorted(set(blk_of.values())):
        keep_idx = [i for i, nm in enumerate(b_names) if blk_of.get(nm) != target]
        for seed in CV_SEEDS:
            a, t, oof = cv_select(Xb_tr[:, keep_idx], ytr, seed, target)
            blk_rows.append({"block": target, "seed": seed,
                             "oof_macro_f1": f1_score(ytr, (oof >= t).astype(int),
                                                      average="macro")})
    blk = pd.DataFrame(blk_rows).groupby("block")["oof_macro_f1"].mean()
    full_mean = mean_f1[best_k]
    lines += ["新构念块增量裁决（leave-one-block-out，Δ = 全量 - 去块）:"]
    for target, v in blk.items():
        d = full_mean - v
        lines.append(f"  {target:<10} 去块 OOF宏F1={v:.4f}  Δ={d:+.4f}  "
                     f"{'保留' if d >= 0 else '负增量（剔除候选）'}")
    lines.append("")

    if not gate_pass:
        report = "\n".join(lines)
        print(report)
        (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
        return

    # ---- 胜者 test 评估（5 种子，双阈值口径）----
    per_seed = []
    prob = pred_m = None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau_m, oof = cv_select(Xb_tr, ytr, seed, best_k)
        tau_t = max(TAU_GRID, key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xb_tr, ytr)
        prob = model.predict_proba(Xb_te)[:, 1]
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
    lines += ["test 评估（胜者配置，5 种子，各评一次）:",
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
              "历史: v6.7 合并 0.7960 (有毒 0.8133) / v6.9 H3 0.8082 (有毒 0.8221, @τ毒 0.8231)",
              "      判死线（有毒F1口径）0.8331"]

    model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / per.iloc[0]["alpha"],
                                        max_iter=2000)).fit(Xb_tr, ytr)
    coef = pd.Series(model.model.coef_[0], index=b_names)
    nz = coef[coef.abs() > 1e-6]
    n7_nz = nz[nz.index.isin(alive7 + m7cols)]
    lines += ["", f"非零权重 {len(nz)}/{len(b_names)}（种子 0），其中 v7 特征非零 {len(n7_nz)}/{len(alive7) + len(m7cols)}",
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
