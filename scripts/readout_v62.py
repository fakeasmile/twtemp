# -*- coding: utf-8 -*-
"""v6.2 读出：升级轴 1 —— 构念分多泛函聚合，线性读出（新建脚本，不改 v6.1 脚本）。

架构约束（用户确认）：
  - 下游只见构念层特征，条目分不进分类器（守 CBM 概念瓶颈：判定←构念分←条目 span）
  - 读出保持线性（L1 LogisticRegression），保证绝对可解释性

特征（单教师 59 维）：
  19 构念 × 3 泛函：
    mean  = 保留条目均值（典型水平，v6.1 原聚合）
    top2  = 条目最高两值均值（最强可观测表现，修复均值稀释）
    std   = 构念内条目 std（作答分歧度）
  + V1 一致性 + V2 极端作答（效度层）
  D2 特例：mean 槽位 = Guttman 取档（单调性 <0.80 回退五档均值），
           top2/std 在五档原始 p 上计算；D2-06/07/08 不进任何槽位
  T 分常模：构念泛函特征用 train mean/std；V1/V2 原值交给 StandardScaler
  可选 --teacher2_train/--teacher2_test → +V3（60 维）

协议：
  配置竞争只在 train 5 折 CV 上做（v6.1 22 维基线 vs v6.2 59 维，3 种子，含 CV 内
  alpha 选择与 OOF 阈值 τ*）；v6.2 配置冻结后 5 种子在 test 上评估一次，
  口径与 v6.1 metrics.txt 一致（有毒 F1 与 Macro F1 并报）。
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
    """tier_M: (N, k≤5) 五档原始 p（列序 = D2-01 最严重 → D2-05 最轻）。返回 0-5 档分。"""
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
    """每构念产出 (mean, top2, std) 三泛函 + 构念内条目处理后矩阵。
    返回 (stats_df, d2_used_guttman)。stats_df 含 id/toxic + 每构念 3 列。"""
    n = len(df)
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
            top2_v = np.sort(Mt, axis=1)[:, -min(2, k):].mean(axis=1)
            std_v = Mt.std(axis=1, ddof=1)
            out[f"{cid}_mean"] = mean_v
            out[f"{cid}_top2"] = top2_v
            out[f"{cid}_std"] = std_v
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


def build_baseline_scores(df, items_meta, freeze):
    """v6.1 基线：每构念单均值（D2 = Guttman），返回含 id/toxic + 19 构念列。"""
    out = df[["id", "toxic"]].copy()
    for cid, kept in freeze["items_per_construct"].items():
        if not kept:
            continue
        g = items_meta[items_meta["construct"] == cid]
        item_dir = dict(zip(g["item"], g["direction"]))
        if cid == "D2":
            tier_cols = [c for c in D2_TIERS if c in df.columns]
            if len(tier_cols) == 5:
                out[cid] = guttman_score(df[tier_cols])[0]
            else:
                out[cid] = df[tier_cols].mean(axis=1).values
            continue
        cols = [c for c in kept if c in df.columns]
        M = df[cols].copy()
        for c in cols:
            if item_dir.get(c) == "反向":
                M[c] = 1.0 - M[c]
        out[cid] = M.mean(axis=1).values
    return out


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
        X2 = df_teacher2[item_ids]
        v3 = (X - X2).abs().mean(axis=1).values
    else:
        v3 = np.zeros(len(df))
    return pd.DataFrame({"id": df["id"], "V1": v1, "V2": v2, "V3": v3})


def t_transform(train_scores, test_scores, exclude=("id", "toxic", "V1", "V2", "V3")):
    cids = [c for c in train_scores.columns if c not in exclude]
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


def assemble(scores_T, val_df, cols):
    X = np.hstack([scores_T[cols].values, val_df[["V1", "V2", "V3"]].values]).astype(np.float64)
    names = list(cols) + ["V1", "V2", "V3"]
    return X, names


def cv_select(X, y, seed):
    """train 内 5 折 CV：选 alpha（二类 F1）→ OOF τ*（二类 F1）。
    返回 (alpha, tau, oof_binary_f1, oof_macro_f1)。test 零参与。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in ALPHA_GRID:
        f = []
        for i_tr, i_va in cv.split(X, y):
            m = make_lr(a).fit(X[i_tr], y[i_tr])
            f.append(f1_score(y[i_va], m.predict_proba(X[i_va])[:, 1] >= 0.5))
        mf = float(np.mean(f))
        if mf > best_f:
            best_f, best_a = mf, a
    oof = np.zeros(len(y))
    for i_tr, i_va in cv.split(X, y):
        m = make_lr(best_a).fit(X[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(X[i_va])[:, 1]
    tau = float(TAU_GRID[int(np.argmax([f1_score(y, oof >= t) for t in TAU_GRID]))])
    return best_a, tau, best_f, float(f1_score(y, oof >= tau, average="macro"))


def metric_block(name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return [f"[{name}]  n={len(y)}  阈值 {threshold:.2f}",
            f"  Acc          {accuracy_score(y, pred):.4f}",
            f"  Macro F1     {f1_score(y, pred, average='macro'):.4f}",
            f"  Weighted F1  {f1_score(y, pred, average='weighted'):.4f}",
            f"  有毒(=1)     P {pr[0]:.4f}  R {rc[0]:.4f}  F1 {f1c[0]:.4f}",
            f"  无毒(=0)     P {pr[1]:.4f}  R {rc[1]:.4f}  F1 {f1c[1]:.4f}",
            f"  AUC          {roc_auc_score(y, p):.4f}"]


def main():
    ap = argparse.ArgumentParser(description="v6.2 读出：构念多泛函 + 线性")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--teacher2_train", default=None)
    ap.add_argument("--teacher2_test", default=None)
    ap.add_argument("--output_dir", default="results/v6/readout_v62")
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
    t2tr = pd.read_csv(args.teacher2_train) if args.teacher2_train else None
    t2te = pd.read_csv(args.teacher2_test) if args.teacher2_test else None

    # 构念泛函
    tr_stats, d2_g = build_construct_stats(tr, items_meta, freeze)
    te_stats, _ = build_construct_stats(te, items_meta, freeze)
    tr_base = build_baseline_scores(tr, items_meta, freeze)
    te_base = build_baseline_scores(te, items_meta, freeze)

    # 效度层
    tr_val = compute_validity(tr, items_meta, df_teacher2=t2tr)
    te_val = compute_validity(te, items_meta, df_teacher2=t2te)

    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    # T 分（构念泛函特征）
    tr62_T, te62_T = t_transform(tr_stats, te_stats)
    trb_T, teb_T = t_transform(tr_base, te_base)

    cols62 = [c for c in tr62_T.columns if c != "id"]
    colsb = [c for c in trb_T.columns if c != "id"]
    # 单教师时 V3 恒 0，从特征里剔除
    if t2tr is None:
        cols62 = [c for c in cols62]  # 构念列
        Xtr62, names62 = assemble(tr62_T, tr_val, cols62)
        Xte62, _ = assemble(te62_T, te_val, cols62)
        Xtrb, namesb = assemble(trb_T, tr_val, colsb)
        Xteb, _ = assemble(teb_T, te_val, colsb)
        # 去掉 V3 列
        keep62 = [i for i, nm in enumerate(names62) if nm != "V3"]
        keepb = [i for i, nm in enumerate(namesb) if nm != "V3"]
        Xtr62, names62 = Xtr62[:, keep62], [names62[i] for i in keep62]
        Xte62 = Xte62[:, keep62]
        Xtrb, namesb = Xtrb[:, keepb], [namesb[i] for i in keepb]
        Xteb = Xteb[:, keepb]
    else:
        Xtr62, names62 = assemble(tr62_T, tr_val, cols62)
        Xte62, _ = assemble(te62_T, te_val, cols62)
        Xtrb, namesb = assemble(trb_T, tr_val, colsb)
        Xteb, _ = assemble(teb_T, te_val, colsb)

    lines = ["=" * 64,
             "v6.2 读出：构念多泛函聚合（mean/top2/std）+ 线性 L1 LR",
             f"v6.2 特征 {Xtr62.shape[1]} 维（{len(cols62)} 构念泛函 T 分 + V1 + V2"
             f"{' + V3' if t2tr is not None else ''}）；基线 {Xtrb.shape[1]} 维",
             f"D2 聚合: {'Guttman 取档' if d2_g else '五档均值回退'}  "
             f"train n={len(ytr)}  test n={len(yte)}",
             "=" * 64, "",
             "--- 配置竞争（train 5 折 CV × 3 种子，test 零参与）---",
             "配置                seed  alpha     tau*   CV二类F1  CV宏F1"]
    for tag, X in [("v6.1基线", Xtrb), ("v6.2泛函", Xtr62)]:
        for seed in CV_SEEDS:
            a, tau, fb, fm = cv_select(X, ytr, seed)
            lines.append(f"{tag:<18} {seed:<5} {a:<9.4g} {tau:<6.2f} {fb:.4f}    {fm:.4f}")

    # v6.2 配置冻结 → test 评估一次（5 种子，口径同 v6.1）
    lines += ["", "--- v6.2 配置 test 评估（冻结后一次性）---"]
    per_seed = []
    for seed in FINAL_SEEDS:
        a, tau, _, _ = cv_select(Xtr62, ytr, seed)
        m = make_lr(a).fit(Xtr62, ytr)
        proba = m.predict_proba(Xte62)[:, 1]
        per_seed.append({"seed": seed, "alpha": a, "tau": tau, "proba": proba,
                         "model": m})
    f1ts, f1ms = [], []
    for r in per_seed:
        pred = r["proba"] >= r["tau"]
        f1ts.append(f1_score(yte, pred))
        f1ms.append(f1_score(yte, pred, average="macro"))
    lines.append("seed  alpha     tau*   有毒F1@tau*  MacroF1@tau*")
    for r, f1t, f1m in zip(per_seed, f1ts, f1ms):
        lines.append(f"{r['seed']:<5} {r['alpha']:<9.4g} {r['tau']:<6.2f} {f1t:.4f}       {f1m:.4f}")
    lines.append(f"mean   {np.mean([r['alpha'] for r in per_seed]):<9.4g} "
                 f"{np.mean([r['tau'] for r in per_seed]):<6.2f} "
                 f"{np.mean(f1ts):.4f}±{np.std(f1ts):.4f}  "
                 f"{np.mean(f1ms):.4f}±{np.std(f1ms):.4f}")

    r0 = per_seed[0]
    lines += ["", *metric_block("全部 test（seed 0）", yte, r0["proba"], r0["tau"]), ""]

    coef = r0["model"].named_steps["lr"].coef_[0]
    nz = [(names62[i], coef[i]) for i in range(len(coef)) if abs(coef[i]) > 1e-6]
    nz.sort(key=lambda t: -abs(t[1]))
    lines.append(f"seed 0 非零权重 {len(nz)}/{len(coef)}（L1 稀疏）:")
    for nm, w in nz:
        lines.append(f"  {nm:<12} {w:+.3f}")

    report = "\n".join(lines)
    print(report)
    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    pd.DataFrame(Xtr62, columns=names62).to_csv(out / "features_train_v62.csv",
                                                index=False, encoding="utf-8-sig")
    pd.DataFrame(Xte62, columns=names62).to_csv(out / "features_test_v62.csv",
                                                index=False, encoding="utf-8-sig")
    with open(out / "v62_predictions_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "toxic", "proba", "pred"])
        for i in range(len(te)):
            w.writerow([te.iloc[i]["id"], int(yte[i]), f"{r0['proba'][i]:.4f}",
                        int(r0["proba"][i] >= r0["tau"])])
    print(f"\n产物落盘 {out}")


if __name__ == "__main__":
    main()
