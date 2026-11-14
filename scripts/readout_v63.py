# -*- coding: utf-8 -*-
"""v6.3 读出：A 口径修正 + B 构念交互特征（新建脚本，不改 v6.1/v6.2 脚本）。

两项改动（用户确认）：
  A. 目标口径 = Macro F1（全 test 两类平均）。τ* 与 CV 内 alpha 选择均改按
     Macro F1 选（v6.2 按 toxic F1 选 τ 把召回推到 0.865、无毒 F1 0.730，
     宏 F1 被拖累）。
  B. 构念交互特征：19 构念均值 T 分（z 化）的两两乘积项 C(19,2)=171 维，
     与 59 维基础特征拼接 = 230 维；读出仍是 L1 线性 LR——严格线性 +
     构念层输入两条架构约束都保持，每个存活交互项仍是构念级可解释
     （如"非人化×道德辩护"联合表现）。

特征：
  基础 59 维（同 v6.2）：19 构念 × 3 泛函（mean/top2/std）+ V1 + V2
  交互 171 维：z(mean_i) × z(mean_j)
  D2 特例不变（mean 槽位 = Guttman 取档）
  T 分常模：train 统计量；test 只投影

协议：
  配置竞争只在 train 5 折 CV × 3 种子上做（base59 vs base59+交互，二者 τ*
  均按宏 F1 选，隔离特征效应）；CV 胜者冻结 → 5 种子 test 评估一次，
  Macro F1 为主口径、toxic F1 并报。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:  # 无 tqdm 时降级为无进度条
    def tqdm(x, **kwargs):
        return x

ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
TAU_GRID = np.arange(0.30, 0.70, 0.01)
GUTTMAN_MONO_THRESHOLD = 0.80
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]


def make_lr(alpha):
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / alpha, max_iter=2000, random_state=0))])


def guttman_score(tier_M):
    M = tier_M.values
    hits = M >= 0.5
    mono_ok = 0
    for i in range(M.shape[0]):
        h = hits[i]
        if h.any():
            top = int(np.where(h)[0][0])
            if h[top:].all():
                mono_ok += 1
        else:
            mono_ok += 1
    if mono_ok / M.shape[0] < GUTTMAN_MONO_THRESHOLD:
        return M.mean(axis=1), False
    score = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        h = hits[i]
        score[i] = (5 - int(np.where(h)[0][0])) if h.any() else 0.0
    return score, True


def build_construct_stats(df, items_meta, freeze):
    """19 构念 × (mean/top2/std) 泛函，D2 特例。返回 (stats_df, d2_used_guttman)。"""
    out = df[["id", "toxic"]].copy()
    d2_used_guttman = False
    for cid, kept in freeze["items_per_construct"].items():
        if not kept:
            continue
        g = items_meta[items_meta["construct"] == cid]
        item_dir = dict(zip(g["item"], g["direction"]))
        if cid == "D2":
            tier_cols = [c for c in D2_TIERS if c in df.columns]
            if len(tier_cols) == 5:
                mean_v, d2_used_guttman = guttman_score(df[tier_cols])
            else:
                mean_v = df[tier_cols].mean(axis=1).values
            Mt = df[tier_cols].values
            k = Mt.shape[1]
            out[f"{cid}_mean"] = mean_v
            out[f"{cid}_top2"] = np.sort(Mt, axis=1)[:, -min(2, k):].mean(axis=1)
            out[f"{cid}_std"] = Mt.std(axis=1, ddof=1)
            continue
        cols = [c for c in kept if c in df.columns]
        M = df[cols].copy()
        for c in cols:
            if item_dir.get(c) == "反向":
                M[c] = 1.0 - M[c]
        Mv = M.values
        k = Mv.shape[1]
        out[f"{cid}_mean"] = Mv.mean(axis=1)
        out[f"{cid}_top2"] = np.sort(Mv, axis=1)[:, -min(2, k):].mean(axis=1)
        out[f"{cid}_std"] = Mv.std(axis=1, ddof=1)
    return out, d2_used_guttman


def compute_validity(df, items_meta, df_teacher2=None):
    item_ids = items_meta["item"].tolist()
    X = df[item_ids]
    v1 = np.zeros(len(df))
    n_rev_c = 0
    for _, g in items_meta.groupby("construct"):
        rev = g[g["direction"] == "反向"]["item"].tolist()
        pos = g[g["direction"] == "正向"]["item"].tolist()
        if not rev:
            continue
        pos_mean = X[pos].mean(axis=1).values if pos else np.zeros(len(df))
        rev_mean = X[rev].mean(axis=1).values
        v1 += pos_mean * rev_mean
        n_rev_c += 1
    if n_rev_c:
        v1 /= n_rev_c
    row_mean = X.mean(axis=1).values
    row_std = X.std(axis=1, ddof=1).values
    v2 = (1.0 - np.clip(row_std, 0, 1)) * np.abs(row_mean - 0.5) * 2.0
    if df_teacher2 is not None:
        v3 = (X - df_teacher2[item_ids]).abs().mean(axis=1).values
    else:
        v3 = np.zeros(len(df))
    return pd.DataFrame({"id": df["id"], "V1": v1, "V2": v2, "V3": v3})


def t_transform(train_scores, test_scores):
    cids = [c for c in train_scores.columns if c not in ("id", "toxic")]
    tr = train_scores[["id"]].copy()
    te = test_scores[["id"]].copy()
    for c in cids:
        mu = train_scores[c].mean()
        sd = train_scores[c].std(ddof=1)
        if sd < 1e-8:
            sd = 1.0
        tr[c] = 50.0 + 10.0 * (train_scores[c] - mu) / sd
        te[c] = 50.0 + 10.0 * (test_scores[c] - mu) / sd
    return tr, te


def add_interactions(scores_T):
    """19 构念 mean T 分 z 化后两两乘积。返回 (inter_matrix, names)。"""
    mean_cols = [c for c in scores_T.columns if c.endswith("_mean")]
    Z = ((scores_T[mean_cols].values - 50.0) / 10.0)
    names, cols = [], []
    for i in range(len(mean_cols)):
        for j in range(i + 1, len(mean_cols)):
            names.append(f"{mean_cols[i][:-5]}x{mean_cols[j][:-5]}")
            cols.append(Z[:, i] * Z[:, j])
    return np.column_stack(cols), names


def assemble_base(scores_T, val_df):
    cols = [c for c in scores_T.columns if c != "id"]
    X = np.hstack([scores_T[cols].values,
                   val_df[["V1", "V2"]].values]).astype(np.float64)
    return X, list(cols) + ["V1", "V2"]


def cv_select(X, y, seed):
    """train 内 5 折 CV：alpha 按 Macro F1(0.5) 选 → OOF τ* 按 Macro F1 选。
    返回 (alpha, tau, cv_macro_f1, oof_macro_f1)。test 零参与。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in tqdm(ALPHA_GRID, desc="    alpha", leave=False, dynamic_ncols=True):
        f = []
        for i_tr, i_va in cv.split(X, y):
            m = make_lr(a).fit(X[i_tr], y[i_tr])
            f.append(f1_score(y[i_va], m.predict_proba(X[i_va])[:, 1] >= 0.5,
                              average="macro"))
        mf = float(np.mean(f))
        if mf > best_f:
            best_f, best_a = mf, a
    oof = np.zeros(len(y))
    for i_tr, i_va in cv.split(X, y):
        m = make_lr(best_a).fit(X[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(X[i_va])[:, 1]
    tau = float(TAU_GRID[int(np.argmax([f1_score(y, oof >= t, average="macro")
                                         for t in TAU_GRID]))])
    return best_a, tau, best_f, float(f1_score(y, oof >= tau, average="macro"))


def metric_block(name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return [f"[{name}]  n={len(y)}  阈值 {threshold:.2f}",
            f"  Acc          {accuracy_score(y, pred):.4f}",
            f"  Macro F1     {f1_score(y, pred, average='macro'):.4f}   ← 主口径",
            f"  Weighted F1  {f1_score(y, pred, average='weighted'):.4f}",
            f"  有毒(=1)     P {pr[0]:.4f}  R {rc[0]:.4f}  F1 {f1c[0]:.4f}",
            f"  无毒(=0)     P {pr[1]:.4f}  R {rc[1]:.4f}  F1 {f1c[1]:.4f}",
            f"  AUC          {roc_auc_score(y, p):.4f}"]


def main():
    ap = argparse.ArgumentParser(description="v6.3 读出：宏F1口径 + 构念交互 + 线性")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--output_dir", default="results/v6/readout_v63")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.freeze, encoding="utf-8") as f:
        freeze = json.load(f)
    items_df = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    items_meta = items_df[["条目id", "构念id", "方向"]].copy()
    items_meta.columns = ["item", "construct", "direction"]

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)

    tr_stats, d2_g = build_construct_stats(tr, items_meta, freeze)
    te_stats, _ = build_construct_stats(te, items_meta, freeze)
    tr_val = compute_validity(tr, items_meta)
    te_val = compute_validity(te, items_meta)
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    tr_T, te_T = t_transform(tr_stats, te_stats)
    Xtr_b, names_b = assemble_base(tr_T, tr_val)
    Xte_b, _ = assemble_base(te_T, te_val)
    itr_tr, inames = add_interactions(tr_T)
    itr_te, _ = add_interactions(te_T)
    Xtr_i = np.hstack([Xtr_b, itr_tr])
    Xte_i = np.hstack([Xte_b, itr_te])
    names_i = names_b + inames

    lines = ["=" * 64,
             "v6.3 读出：宏F1 口径（τ*/alpha 均按宏 F1 选）+ 构念交互 + 线性 L1 LR",
             f"base59 = {Xtr_b.shape[1]} 维；base59+交互 = {Xtr_i.shape[1]} 维",
             f"D2 聚合: {'Guttman 取档' if d2_g else '五档均值回退'}  "
             f"train n={len(ytr)}  test n={len(yte)}",
             "参考（历史 test 值）：v6.1 22维 τ@toxicF1  Macro 0.7696 / 有毒 0.7983",
             "                    v6.2 59维 τ@toxicF1  Macro 0.7647 / 有毒 0.7987",
             "=" * 64, "",
             "--- 配置竞争（train 5 折 CV × 3 种子，test 零参与）---",
             "配置            seed  alpha     tau*   CV宏F1(alpha@0.5)  OOF宏F1@tau*"]
    results = {}
    for tag, X in tqdm([("base59", Xtr_b), ("base59+交互", Xtr_i)],
                       desc="配置竞争", dynamic_ncols=True):
        rows = []
        for seed in CV_SEEDS:
            a, tau, fc, fo = cv_select(X, ytr, seed)
            rows.append((a, tau, fc, fo))
            lines.append(f"{tag:<14} {seed:<5} {a:<9.4g} {tau:<6.2f} "
                         f"{fc:.4f}             {fo:.4f}")
        results[tag] = rows

    winner = max(results, key=lambda k: np.mean([r[3] for r in results[k]]))
    Xtr_w = Xtr_i if winner == "base59+交互" else Xtr_b
    Xte_w = Xte_i if winner == "base59+交互" else Xte_b
    names_w = names_i if winner == "base59+交互" else names_b
    lines += ["", f"CV 胜者: {winner} → 冻结，test 评估一次", "",
              "--- 胜者配置 test 评估（5 种子，主口径 Macro F1）---"]
    per_seed = []
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau, _, _ = cv_select(Xtr_w, ytr, seed)
        m = make_lr(a).fit(Xtr_w, ytr)
        proba = m.predict_proba(Xte_w)[:, 1]
        per_seed.append({"seed": seed, "alpha": a, "tau": tau, "proba": proba, "model": m})
    f1ms, f1ts = [], []
    for r in per_seed:
        pred = r["proba"] >= r["tau"]
        f1ms.append(f1_score(yte, pred, average="macro"))
        f1ts.append(f1_score(yte, pred))
    lines.append("seed  alpha     tau*   MacroF1@tau*  有毒F1@tau*")
    for r, fm, ft in zip(per_seed, f1ms, f1ts):
        lines.append(f"{r['seed']:<5} {r['alpha']:<9.4g} {r['tau']:<6.2f} {fm:.4f}        {ft:.4f}")
    lines.append(f"mean   {np.mean([r['alpha'] for r in per_seed]):<9.4g} "
                 f"{np.mean([r['tau'] for r in per_seed]):<6.2f} "
                 f"{np.mean(f1ms):.4f}±{np.std(f1ms):.4f}  "
                 f"{np.mean(f1ts):.4f}±{np.std(f1ts):.4f}")

    r0 = per_seed[0]
    lines += ["", *metric_block("全部 test（seed 0）", yte, r0["proba"], r0["tau"]), ""]

    coef = r0["model"].named_steps["lr"].coef_[0]
    nz = [(names_w[i], coef[i]) for i in range(len(coef)) if abs(coef[i]) > 1e-6]
    nz.sort(key=lambda t: -abs(t[1]))
    lines.append(f"seed 0 非零特征 {len(nz)}/{len(coef)}，按 |权重| 前 25:")
    for nm, w in nz[:25]:
        lines.append(f"  {nm:<14} {w:+.3f}")

    report = "\n".join(lines)
    print(report)
    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    pd.DataFrame(Xtr_w, columns=names_w).to_csv(out / "features_train_winner.csv",
                                                index=False, encoding="utf-8-sig")
    pd.DataFrame(Xte_w, columns=names_w).to_csv(out / "features_test_winner.csv",
                                                index=False, encoding="utf-8-sig")
    with open(out / "v63_predictions_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "toxic", "proba", "pred"])
        for i in range(len(te)):
            w.writerow([te.iloc[i]["id"], int(yte[i]), f"{r0['proba'][i]:.4f}",
                        int(r0["proba"][i] >= r0["tau"])])
    print(f"\n产物落盘 {out}")


if __name__ == "__main__":
    main()
