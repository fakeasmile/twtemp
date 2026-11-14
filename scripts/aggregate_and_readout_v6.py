# -*- coding: utf-8 -*-
"""v6 聚合 + 稀疏线性读出（新建脚本，不修改 train_concept_lr.py）。

流程（设计文档 §4/§5/§8）：
  1. 读 v6.1 冻结文件（存活构念、保留条目、反向降级清单）
  2. 读 train/test 条目矩阵 CSV
  3. 聚合构念分：
       - 普通构念：保留条目均值（反向条目按 freeze 决定是否进分；
         未降级的反向条目取 1−p 后与正向一起求均值）
       - D2 特例：D2-01..05 五档走 Guttman（命中最高档，0-5 ordinal），
         命中单调性不达标则回退五档均值；D2-06/07/08 只作效度辅助不进主分
  4. T 分常模：每个构念用 train 的 mean/std，T=50+10·z；test 用 train 统计量投影
  5. 效度层（从条目矩阵算，不增测）：
       - V1 一致性：构念内正/反向条目（原始 p）矛盾度，VRIN 思想
       - V2 极端作答：全条目 p 方差异常低且均值极端，F 量表思想
       - V3 跨教师分歧：若提供第二教师矩阵则算平均绝对差，否则置 0
  6. 稀疏线性读出（参照 train_concept_lr.py，不修改其代码）：
       - StandardScaler + L1 LogisticRegression(liblinear, C=1/alpha)
       - alpha 由 train 5 折 CV 按 F1 选出；τ* 由同 CV OOF 概率按 F1 选出
       - 5 随机种子重复，test 仅在各种子超参选定后各评一次，报 mean±std
       - 全程不碰 test 选超参；metrics.txt 只报 test

使用：
  python scripts/aggregate_and_readout_v6.py \
    --train_matrix data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv \
    --test_matrix data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_test_matrix.csv \
    --freeze results/v6/item_analysis_glm_train/v6_1_freeze.json \
    --items_csv data/concepts/cbm_concept_set_v6.csv \
    --output_dir results/v6/readout_glm
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
TAU_GRID = np.arange(0.30, 0.70, 0.01)
N_SEEDS = 5
GUTTMAN_MONO_THRESHOLD = 0.80  # 命中单调性达此比例才用 Guttman，否则回退均值


# =============================================================================
# 构念分聚合
# =============================================================================
def load_matrix(path):
    df = pd.read_csv(path)
    return df


def aggregate_construct_scores(df, items_meta, freeze):
    """返回 (construct_df, d2_used_guttman)。construct_df 含 id/toxic 列 + 各构念分。"""
    item_ids = items_meta["item"].tolist()
    construct_scores = {}

    # D2 五档条目 id（按严重度从高到低：D2-01 最严重 → D2-05 最轻）
    d2_tiers = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
    d2_scores = None
    d2_used_guttman = False

    for cid, kept in freeze["items_per_construct"].items():
        if not kept:
            continue
        g = items_meta[items_meta["construct"] == cid]
        # 方向映射
        item_dir = dict(zip(g["item"], g["direction"]))
        # 该构念的反向是否降级
        rev_downgraded = freeze["reverse_items_role"].get(cid, False)

        if cid == "D2":
            # D2 主分 = Guttman(D2-01..05)，不使用 D2-06/07/08
            tier_cols = [c for c in d2_tiers if c in df.columns]
            if len(tier_cols) != 5:
                # 缺档，回退均值（仍只用现有档次 D2-01..05，不含 D2-06/07/08）
                M = df[tier_cols].copy()
                d2_scores = M.mean(axis=1).values
            else:
                M = df[tier_cols].values  # (N, 5)
                # 命中单调性检查：对每个样本，低档命中则高档不应命中；
                # 等价：命中的最高档以下各档应全部命中
                hits = M >= 0.5
                mono_ok = 0
                for i in range(M.shape[0]):
                    h = hits[i]
                    if h.any():
                        top = np.where(h)[0][0]  # 最严重档 index（0=最严重）
                        # top 以下（更轻）应全命中
                        if h[top:].all():
                            mono_ok += 1
                    else:
                        mono_ok += 1  # 全未命中也算满足单调
                mono_ratio = mono_ok / M.shape[0]
                if mono_ratio >= GUTTMAN_MONO_THRESHOLD:
                    # Guttman 取最高档：命中的最严重档 index（0-4），无命中为 0
                    tier_score = np.zeros(M.shape[0])
                    for i in range(M.shape[0]):
                        h = hits[i]
                        if h.any():
                            # 最严重档 = 最小 index；转成 1-5 分，无命中=0
                            tier_score[i] = 5 - np.where(h)[0][0]
                        else:
                            tier_score[i] = 0.0
                    d2_scores = tier_score
                    d2_used_guttman = True
                else:
                    # 回退五档均值
                    d2_scores = M.mean(axis=1)
            construct_scores[cid] = d2_scores
            continue

        # 普通构念：保留条目均值（反向按降级决定）
        kept_cols = [c for c in kept if c in df.columns]
        M = df[kept_cols].copy()
        for c in kept_cols:
            if item_dir[c] == "反向" and not rev_downgraded:
                M[c] = 1.0 - M[c]
        construct_scores[cid] = M.mean(axis=1).values

    out = df[["id", "toxic"]].copy()
    for cid, vals in construct_scores.items():
        out[cid] = vals
    return out, d2_used_guttman


# =============================================================================
# T 分常模
# =============================================================================
def t_score_transform(train_scores, test_scores):
    """train 统计量 → T=50+10z；test 用 train 的 mean/std 投影。"""
    cids = [c for c in train_scores.columns if c not in ("id", "toxic")]
    train_T = train_scores[["id", "toxic"]].copy()
    test_T = test_scores[["id", "toxic"]].copy()
    stats = {}
    for c in cids:
        mu = train_scores[c].mean()
        sd = train_scores[c].std(ddof=1)
        if sd < 1e-8:
            sd = 1.0
        stats[c] = (float(mu), float(sd))
        train_T[c] = 50.0 + 10.0 * (train_scores[c] - mu) / sd
        test_T[c] = 50.0 + 10.0 * (test_scores[c] - mu) / sd
    return train_T, test_T, stats


# =============================================================================
# 效度层 V1 / V2 / V3
# =============================================================================
def compute_validity(df, items_meta, df_teacher2=None):
    """返回 DataFrame: id, V1, V2, V3。"""
    item_ids = items_meta["item"].tolist()
    X = df[item_ids]

    # V1 一致性：构念内正向 p 与反向 p（原始，不反转）矛盾度
    # 矛盾 = 正向均值高 且 反向均值高（未反转也高）→ 乘积大
    v1 = np.zeros(len(df))
    n_constructs_with_rev = 0
    for cid, g in items_meta.groupby("construct"):
        rev = g[g["direction"] == "反向"]["item"].tolist()
        pos = g[g["direction"] == "正向"]["item"].tolist()
        if not rev:
            continue
        pos_mean = X[pos].mean(axis=1).values if pos else np.zeros(len(df))
        rev_mean = X[rev].mean(axis=1).values
        # 正向高 × 反向高 = 矛盾
        v1 += pos_mean * rev_mean
        n_constructs_with_rev += 1
    if n_constructs_with_rev > 0:
        v1 = v1 / n_constructs_with_rev

    # V2 极端作答：全条目 p 的 std 低 + mean 极端
    row_mean = X.mean(axis=1).values
    row_std = X.std(axis=1, ddof=1).values
    v2 = (1.0 - np.clip(row_std, 0, 1)) * np.abs(row_mean - 0.5) * 2.0

    # V3 跨教师分歧
    if df_teacher2 is not None:
        X2 = df_teacher2[item_ids]
        v3 = (X - X2).abs().mean(axis=1).values
    else:
        v3 = np.zeros(len(df))

    out = pd.DataFrame({"id": df["id"], "V1": v1, "V2": v2, "V3": v3})
    return out


# =============================================================================
# 稀疏线性读出（参照 train_concept_lr.py 的 fit_sparse_lr，独立实现）
# =============================================================================
def make_lr(alpha):
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / alpha, max_iter=2000, random_state=0))])


def fit_sparse_lr(X, y, random_state=0):
    """alpha 5 折 CV 按 F1 选出 → 全量 refit；τ* 由同 CV OOF 概率按 F1 选出。
    返回 (model, tau_star, alpha_best)。test 不参与任何一步。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    scores = []
    for a in ALPHA_GRID:
        fold_f1 = []
        for i_tr, i_va in cv.split(X, y):
            m = make_lr(a).fit(X[i_tr], y[i_tr])
            p = m.predict_proba(X[i_va])[:, 1]
            fold_f1.append(f1_score(y[i_va], p >= 0.5))
        scores.append(np.mean(fold_f1))
    alpha = ALPHA_GRID[int(np.argmax(scores))]
    model = make_lr(alpha).fit(X, y)
    oof = np.zeros(len(y))
    for i_tr, i_va in cv.split(X, y):
        m = make_lr(alpha).fit(X[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(X[i_va])[:, 1]
    tau = float(TAU_GRID[int(np.argmax([f1_score(y, oof >= t) for t in TAU_GRID]))])
    return model, tau, alpha


def metric_block(name, y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return [f"[{name}]  n={len(y)}  阈值 {threshold:.2f}",
            f"  Acc          {accuracy_score(y, pred):.4f}",
            f"  Macro F1     {f1_score(y, pred, average='macro'):.4f}",
            f"  Weighted F1  {f1_score(y, pred, average='weighted'):.4f}",
            f"  有毒(=1)     P {pr[0]:.4f}  R {rc[0]:.4f}  F1 {f1c[0]:.4f}",
            f"  无毒(=0)     P {pr[1]:.4f}  R {rc[1]:.4f}  F1 {f1c[1]:.4f}",
            f"  AUC          {roc_auc_score(y, p):.4f}"]


def parse_args():
    ap = argparse.ArgumentParser(description="v6 聚合 + 稀疏线性读出")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", default=None,
                    help="test 矩阵；不提供则只做 train 聚合落盘，不评估")
    ap.add_argument("--freeze", required=True, help="v6_1_freeze.json")
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--teacher2_train", default=None,
                    help="第二教师 train 矩阵（计算 V3）；不提供 V3 置 0")
    ap.add_argument("--teacher2_test", default=None, help="第二教师 test 矩阵")
    ap.add_argument("--output_dir", default="results/v6/readout")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.freeze, encoding="utf-8") as f:
        freeze = json.load(f)
    items_df = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    items_meta = items_df[["条目id", "构念id", "方向"]].copy()
    items_meta.columns = ["item", "construct", "direction"]

    tr = load_matrix(args.train_matrix)
    te = load_matrix(args.test_matrix) if args.test_matrix else None

    # 构念分聚合
    tr_scores, d2_guttman = aggregate_construct_scores(tr, items_meta, freeze)
    te_scores, _ = aggregate_construct_scores(te, items_meta, freeze) if te is not None else (None, None)

    cids = [c for c in tr_scores.columns if c not in ("id", "toxic")]
    print(f"构念分: {len(cids)} 构念  D2 聚合方式: {'Guttman' if d2_guttman else '均值回退'}")

    # T 分常模
    tr_T, te_T, t_stats = t_score_transform(tr_scores, te_scores) if te is not None else (
        t_score_transform(tr_scores, tr_scores)[:2] + (None,))

    # 效度层
    tr_v = compute_validity(tr, items_meta,
                            df_teacher2=load_matrix(args.teacher2_train) if args.teacher2_train else None)
    te_v = compute_validity(te, items_meta,
                            df_teacher2=load_matrix(args.teacher2_test) if args.teacher2_test else None) \
        if te is not None else None

    # 特征 = T 分 + V1/V2/V3
    feat_names = cids + ["V1", "V2", "V3"]
    Xtr = np.hstack([tr_T[cids].values, tr_v[["V1", "V2", "V3"]].values]).astype(np.float32)
    ytr = tr["toxic"].astype(int).values

    # 落盘构念分与特征
    tr_scores.to_csv(out / "construct_scores_train.csv", index=False, encoding="utf-8-sig")
    if te_scores is not None:
        te_scores.to_csv(out / "construct_scores_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(Xtr, columns=feat_names).to_csv(out / "features_train.csv", index=False, encoding="utf-8-sig")

    if te is None:
        print(f"未提供 test_matrix，仅完成 train 聚合。构念分 → {out / 'construct_scores_train.csv'}")
        return

    Xte = np.hstack([te_T[cids].values, te_v[["V1", "V2", "V3"]].values]).astype(np.float32)
    yte = te["toxic"].astype(int).values
    pd.DataFrame(Xte, columns=feat_names).to_csv(out / "features_test.csv", index=False, encoding="utf-8-sig")

    # 5 seed 读出
    per_seed = []
    for seed in range(N_SEEDS):
        model, tau, alpha = fit_sparse_lr(Xtr, ytr, random_state=seed)
        proba = model.predict_proba(Xte)[:, 1]
        per_seed.append({"seed": seed, "alpha": alpha, "tau": tau, "proba": proba, "model": model})
    proba0 = per_seed[0]["proba"]

    def seed_table():
        rows = ["seed  alpha     tau*   有毒F1@tau*"]
        f1ts = []
        for r in per_seed:
            f1t = f1_score(yte, r["proba"] >= r["tau"])
            f1ts.append(f1t)
            rows.append(f"{r['seed']:<5} {r['alpha']:<9.4g} {r['tau']:<6.2f} {f1t:.4f}")
        taus = [r["tau"] for r in per_seed]
        alphas = [r["alpha"] for r in per_seed]
        rows.append(f"mean   {np.mean(alphas):<9.4g} {np.mean(taus):<6.2f} "
                    f"{np.mean(f1ts):.4f} ± {np.std(f1ts):.4f}")
        return rows

    lines = ["=" * 60,
             "v6 聚合 + 稀疏线性读出（L1 LogisticRegression）",
             f"特征: {len(feat_names)} 维（{len(cids)} 构念 T 分 + V1 一致性 + V2 极端作答 + V3 跨教师分歧）",
             f"D2 聚合: {'Guttman 取最高档' if d2_guttman else '五档均值（单调性不达标回退）'}",
             f"V3: {'双教师平均绝对差' if args.teacher2_train else '单教师置 0'}",
             f"train: n={len(ytr)}  test: n={len(yte)}  toxic=1: {int(yte.sum())}",
             "=" * 60, "",
             *metric_block("全部 test（seed 0）", yte, proba0, per_seed[0]["tau"]), "",
             "各 seed 超参（全量 refit 确定性一致，种子只影响 alpha/τ* 选择）:",
             *seed_table(), ""]

    # seed 0 top 权重（可解释性）
    coef = per_seed[0]["model"].named_steps["lr"].coef_[0]
    top = sorted(range(len(coef)), key=lambda i: -abs(coef[i]))[:10]
    lines.append("seed 0 top10 |权重|:")
    for i in top:
        lines.append(f"  {feat_names[i]:<12} {coef[i]:+.3f}")

    report = "\n".join(lines)
    print(report)
    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n指标已落盘 {out / 'metrics.txt'}")

    with open(out / "v6_predictions_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "toxic", "proba", "pred", "content"])
        for i in range(len(te)):
            w.writerow([te.iloc[i]["id"], int(yte[i]), f"{proba0[i]:.4f}",
                        int(proba0[i] >= per_seed[0]["tau"]), te.iloc[i]["content"][:80]])
    print(f"预测已落盘（seed 0） {out / 'v6_predictions_test.csv'}")


if __name__ == "__main__":
    main()
