"""概念向量 → 稀疏线性分类器直接预测（教师通道直读，soft 编码定稿版）。

特征编码（encode_features，本脚本唯一实现，train_distill.py 的冻结读出也 import
它，保证两处特征逐位同构）：
  全部概念 p_yes（v4 词典 47 维）+ 0-2 值域概念 P(直接)=p_yes·p_B + 0-2 概念 p_A
  + S1A 六档概率 + S1B 六档概率 + CAT 六通道 p_yes ≈ 99 维 soft
0-2 概念集合由 train 中任一样本触发过 stage2 动态确定（勿硬编码）。

分类器：StandardScaler + L1 LogisticRegression（liblinear，C=1/alpha），
alpha 由 train 5 折 CV 在 ALPHA_GRID 上按 F1 选出；阈值 τ* 由同 CV 的 train
OOF 概率在 TAU_GRID 上按 F1 选出；5 个随机种子重复，test 仅在各种子全部选定
后各评一次，报告 mean±std。全程不碰 test 选超参。

用法：
  python scripts/train_concept_lr.py --rings_dir data/processed/TOXICN/glm-4-9b-chat_v3
目录下需恰有 rings_{LLM模型名}_train.json 与 rings_{LLM模型名}_test.json 各一个，
脚本自动发现；全部产物（metrics.txt + 预测 CSV）落 --output_dir（默认 results/concept_lr）。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CAT_ORDER = ("CAT1A", "CAT1B", "CAT2A", "CAT2B", "CAT3", "CAT4")
ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
TAU_GRID = np.arange(0.30, 0.70, 0.01)
N_SEEDS = 5


def discover_rings(rings_dir):
    """目录下自动发现 rings_{模型名}_{split}.json：train/test 各恰一个。"""
    d = Path(rings_dir)
    if not d.is_dir():
        raise SystemExit(f"rings 目录不存在: {d}")
    tr = sorted(d.glob("rings_*_train.json"))
    te = sorted(d.glob("rings_*_test.json"))
    if len(tr) != 1 or len(te) != 1:
        raise SystemExit(f"{d} 下应恰有一个 rings_*_train.json 与一个 rings_*_test.json，"
                         f"找到 train {len(tr)} 个 / test {len(te)} 个")
    return tr[0], te[0]


def load_rings(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"rings 文件不存在: {p}")
    d = json.load(open(p, encoding="utf-8"))
    return d["results"], d["meta"]


def metric_block(name, y, p, threshold=0.5):
    """单组指标（metrics.txt 的标准块）：Acc / Macro F1 / Weighted F1 / 分类别 P R F1 / AUC。"""
    pred = (p >= threshold).astype(int)
    pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return [f"[{name}]  n={len(y)}  阈值 {threshold:.2f}",
            f"  Acc          {accuracy_score(y, pred):.4f}",
            f"  Macro F1     {f1_score(y, pred, average='macro'):.4f}",
            f"  Weighted F1  {f1_score(y, pred, average='weighted'):.4f}",
            f"  有毒(=1)     P {pr[0]:.4f}  R {rc[0]:.4f}  F1 {f1c[0]:.4f}",
            f"  无毒(=0)     P {pr[1]:.4f}  R {rc[1]:.4f}  F1 {f1c[1]:.4f}",
            f"  AUC          {roc_auc_score(y, p):.4f}"]


class ConceptArrays:
    """从教师 results 提取的概念通道数组（编码的唯一数据源）。"""

    def __init__(self, results, meta):
        self.cids = [c["id"] for c in meta["concepts"]]
        # 0-2 值域概念：train 中任一样本触发过 stage2 的（勿硬编码）
        self.s2_cids = [cid for cid in self.cids
                        if any(x["concepts"][cid]["stage2"] for x in results)]
        self.cat_names = [k for k in CAT_ORDER if k in results[0]]
        self.texts = [x["content"] for x in results]
        self.toxic = np.asarray([int(x["toxic"]) for x in results], dtype=np.int64)
        self.expr = np.asarray(
            [-1 if x.get("expression") is None else int(x["expression"]) for x in results],
            dtype=np.int64)
        self.p_yes = np.asarray([[x["concepts"][cid]["stage1"]["p_yes"] for cid in self.cids]
                                 for x in results], dtype=np.float32)
        # P(直接) = p_yes · p_B；stage2 未触发（p_yes 低于存在性阈值）按 0 计
        self.p_direct = np.asarray(
            [[x["concepts"][cid]["stage1"]["p_yes"] * x["concepts"][cid]["stage2"]["p_B"]
              if x["concepts"][cid]["stage2"] else 0.0 for cid in self.s2_cids]
             for x in results], dtype=np.float32)
        self.p_a = np.asarray([[x["concepts"][cid]["stage2"]["p_A"]
                                if x["concepts"][cid]["stage2"] else 0.0
                                for cid in self.s2_cids] for x in results], dtype=np.float32)
        self.s1a = np.asarray([[float((x.get("S1A") or {}).get("probs", {}).get(str(k), 0.0))
                                for k in range(6)] for x in results], dtype=np.float32)
        self.s1b = np.asarray([[float((x.get("S1B") or {}).get("probs", {}).get(str(k), 0.0))
                                for k in range(6)] for x in results], dtype=np.float32)
        self.cat = np.asarray([[x[k]["p_yes"] for k in self.cat_names] for x in results],
                              dtype=np.float32)

    def feature_names(self):
        names = [f"{cid}:p_yes" for cid in self.cids]
        names += [f"{cid}:P(直接)" for cid in self.s2_cids]
        names += [f"{cid}:p_A" for cid in self.s2_cids]
        names += [f"S1A:p{k}" for k in range(6)]
        names += [f"S1B:p{k}" for k in range(6)]
        names += [f"{k}:p_yes" for k in self.cat_names]
        return names


def encode_features(p_yes, p_direct, p_a, s1a, s1b, cat):
    """唯一 soft 编码实现：p_yes + P(直接) + p_A + S1A6 + S1B6 + CAT6 → (N, d)。

    输入均为 (N, ·) float 数组，列序与 ConceptArrays 各字段一致；
    train_distill.py 的学生预测通道按同一列序传入即逐位同构。
    """
    return np.hstack([p_yes, p_direct, p_a, s1a, s1b, cat]).astype(np.float32)


def encode(ca):
    """ConceptArrays → 特征矩阵（train_concept_lr 内部便捷封装）。"""
    return encode_features(ca.p_yes, ca.p_direct, ca.p_a, ca.s1a, ca.s1b, ca.cat)


def make_lr(alpha):
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / alpha, max_iter=2000, random_state=0))])


def fit_sparse_lr(X, y, random_state=0):
    """alpha 5 折 CV 选出 → 全量 refit；τ* 由同 CV 的 OOF 概率按 F1 选出。

    返回 (model, tau_star, alpha_best)。test 不参与任何一步。
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    scores = [cross_val_score(make_lr(a), X, y, cv=cv, scoring="f1").mean()
              for a in ALPHA_GRID]
    alpha = ALPHA_GRID[int(np.argmax(scores))]
    model = make_lr(alpha).fit(X, y)
    oof = np.zeros(len(y))
    for i_tr, i_va in cv.split(X, y):
        m = make_lr(alpha).fit(X[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(X[i_va])[:, 1]
    tau = float(TAU_GRID[int(np.argmax([f1_score(y, oof >= t) for t in TAU_GRID]))])
    return model, tau, alpha


def main():
    ap = argparse.ArgumentParser(description="概念向量 → L1 稀疏线性分类器")
    ap.add_argument("--rings_dir", required=True,
                    help="含 rings_{模型名}_train.json / rings_{模型名}_test.json 的目录")
    ap.add_argument("--output_dir", default="results/concept_lr",
                    help="metrics.txt 与预测 CSV 的落盘目录")
    args = ap.parse_args()

    tr_path, te_path = discover_rings(args.rings_dir)
    tr_results, meta = load_rings(tr_path)
    te_results, meta_te = load_rings(te_path)
    assert [c["id"] for c in meta["concepts"]] == [c["id"] for c in meta_te["concepts"]], \
        "train/test 概念集 id 不一致（概念顺序必须一致）"
    assert meta["model"] == meta_te["model"], "违反严格单模型原则：train/test 来自不同教师"

    ca_tr, ca_te = ConceptArrays(tr_results, meta), ConceptArrays(te_results, meta)
    assert ca_tr.s2_cids == ca_te.s2_cids, "train/test 0-2 概念集合不一致"
    Xtr, Xte = encode(ca_tr), encode(ca_te)
    ytr, yte = ca_tr.toxic, ca_te.toxic

    # 5 种子：各 seed 全量 refit 确定性一致，差异仅在 alpha / τ* 选择
    per_seed = []
    for seed in range(N_SEEDS):
        model, tau, alpha = fit_sparse_lr(Xtr, ytr, random_state=seed)
        proba = model.predict_proba(Xte)[:, 1]
        per_seed.append({"seed": seed, "alpha": alpha, "tau": tau, "proba": proba,
                         "model": model})
    proba0 = per_seed[0]["proba"]

    def seed_table():
        rows = ["seed  alpha     tau*   MacroF1@tau*"]
        for r in per_seed:
            f1t = f1_score(yte, r["proba"] >= r["tau"])
            rows.append(f"{r['seed']:<5} {r['alpha']:<9.4g} {r['tau']:<6.2f} "
                        f"{f1t:.4f}")
        taus = [r["tau"] for r in per_seed]
        alphas = [r["alpha"] for r in per_seed]
        f1ts = [f1_score(yte, r["proba"] >= r["tau"]) for r in per_seed]
        rows.append(f"mean   {np.mean(alphas):<9.4g} {np.mean(taus):<6.2f} "
                    f"{np.mean(f1ts):.4f} ± {np.std(f1ts):.4f}")
        return rows

    lines = ["=" * 60,
             "train_concept_lr 概念向量直读（L1 稀疏线性分类器）",
             f"教师: {meta['model']}   特征: {Xtr.shape[1]} 维",
             f"test: n={len(yte)}  toxic=1: {int(yte.sum())}",
             "=" * 60, "",
             *metric_block("全部", yte, proba0), "",
             "各 seed 超参（全量 refit 确定性一致，种子只影响 alpha/τ* 选择）:",
             *seed_table(), ""]

    # seed 0 的 top 权重（概念瓶颈解释性）
    names = ca_tr.feature_names()
    coef = per_seed[0]["model"].named_steps["lr"].coef_[0]
    top = sorted(range(len(coef)), key=lambda i: -abs(coef[i]))[:10]
    lines.append("seed 0 top10 |权重|:")
    for i in top:
        lines.append(f"  {names[i]:<24} {coef[i]:+.3f}")
    report = "\n".join(lines)

    print(report)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n指标已落盘 {out / 'metrics.txt'}")

    with open(out / "concept_lr_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "expression", "toxic", "proba", "content"])
        for i, r in enumerate(te_results):
            w.writerow([r.get("id", i), int(ca_te.expr[i]), int(yte[i]),
                        f"{proba0[i]:.4f}", r["content"][:50]])
    print(f"预测已落盘（seed 0） {out / 'concept_lr_predictions.csv'}")


if __name__ == "__main__":
    main()
