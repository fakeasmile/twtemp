# -*- coding: utf-8 -*-
"""v8.6 前置：LCA（潜类分析）/ 混合 IRT——把 use/mention 作为潜类而非连续维。

理论动机：v9 证伪了"教师能回答意图问题"，但未检验"教师 141 条回答联合聚类
能否恢复 use/mention 类型学"。v8.3 审计已证错误群体在特征空间中是可分簇
（FP vs TN 0.986、expr2-FN vs expr2-TP 0.968）——它们不在单一毒性轴上，
但可能构成**离散类型**。LCA 把 141 条目×5 档响应矩阵建模为 K 个潜类的
混合（每类内条目局部独立），EM 闭式解（比 GRM 更快）：
  P(y_ni = j) = Σ_k pi_k · P_k(y_ni = j)
K 由 BIC 在 K=2..6 中选择（train 内，无标签参与——纯无监督）。

产物（results/v86/lca/）：
  lca_params.npz     pi / P(K,I,5) / 类后验矩阵
  post_{train,test}.csv   id, post_1..post_K（下游 readout_v86 特征）
  metrics.txt        BIC 表、类画像（各类 top 区分条目）、类×expression×
                     y 交叉表（诊断潜类是否抓到 use/mention 结构）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from irt_grm import bin5

K_GRID = [2, 3, 4, 5, 6]
MAX_ITER = 300
TOL = 1e-6
ALPHA = 0.5          # Dirichlet 先验防零概率
N_INIT = 3
SEED = 0


def lca_em(Y, K, rng):
    N, I = Y.shape
    r = rng.random((N, K))
    r /= r.sum(axis=1, keepdims=True)
    Yoh = np.zeros((N, I, 5), dtype=np.float64)
    for j in range(5):
        Yoh[:, :, j] = (Y == j)
    pi = None
    P = None
    ll = -np.inf
    for it in range(1, MAX_ITER + 1):
        # M-step（上一轮后验）
        Nk = r.sum(axis=0)                                    # (K,)
        pi_new = r.mean(axis=0)
        cnt = np.einsum("nk,nij->kij", r, Yoh)                # (K,I,5)
        P_new = (cnt + ALPHA) / (Nk[:, None, None] + ALPHA * 5.0)
        if pi is not None:
            delta = max(np.abs(P_new - P).max(), np.abs(pi_new - pi).max())
        else:
            delta = np.inf
        pi, P = pi_new, P_new
        if delta < TOL:
            break
        # E-step
        ll_k = np.log(np.clip(pi, 1e-12, None))[None, :].repeat(N, axis=0)  # (N,K)
        for i in range(I):
            ll_k += np.log(P[:, i, :])[:, Y[:, i]].T          # (K,N) → (N,K)
        m = ll_k.max(axis=1, keepdims=True)
        logZ = m + np.log(np.exp(ll_k - m).sum(axis=1, keepdims=True))
        r = np.exp(ll_k - logZ)
        ll = float(logZ.sum())
    return pi, P, ll, r, it


def posterior(P, pi, Y):
    N, I = Y.shape
    K = len(pi)
    ll_k = np.log(np.clip(pi, 1e-12, None))[None, :].repeat(N, axis=0)
    for i in range(I):
        ll_k += np.log(P[:, i, :])[:, Y[:, i]].T
    m = ll_k.max(axis=1, keepdims=True)
    logZ = m + np.log(np.exp(ll_k - m).sum(axis=1, keepdims=True))
    r = np.exp(ll_k - logZ)
    return r, float(logZ.sum())


def main():
    ap = argparse.ArgumentParser(description="v8.6 LCA 潜类分析")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--output_dir", default="results/v86/lca")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr, Ate = tr[item_cols].copy(), te[item_cols].copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
            Ate[c] = 1.0 - te[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    Ytr, Yte = bin5(Atr[kept_items].values), bin5(Ate[kept_items].values)
    N, I = Ytr.shape

    print(f"LCA：{I} 条目 × {N} train 样本，K 网格 {K_GRID}，{N_INIT} 次随机重启")
    rows_bic, fits = [], {}
    for K in K_GRID:
        best = None
        for s in range(N_INIT):
            rng = np.random.default_rng(SEED + 100 * K + s)
            pi, P, ll, r, it = lca_em(Ytr, K, rng)
            if best is None or ll > best[2]:
                best = (pi, P, ll, r, it)
        pi, P, ll, r, it = best
        n_par = (K - 1) + K * I * 4
        bic = -2 * ll + n_par * np.log(N)
        rows_bic.append({"K": K, "loglik": round(ll, 1), "n_par": n_par,
                         "BIC": round(bic, 1),
                         "iters": it, "sizes": [int(x) for x in np.round(r.sum(0))]})
        fits[K] = (pi, P, ll, r)
        print(f"  K={K}  loglik={ll:.1f}  BIC={bic:.1f}  iters={it}")
    bic_df = pd.DataFrame(rows_bic)
    K_best = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])
    print(f"BIC 最优 K = {K_best}")
    pi, P, ll, r = fits[K_best]

    post_tr, ll_tr = posterior(P, pi, Ytr)
    post_te, ll_te = posterior(P, pi, Yte)
    pd.DataFrame({"id": tr["id"].values,
                  **{f"post_{k+1}": post_tr[:, k] for k in range(K_best)}}).to_csv(
        out / "post_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": te["id"].values,
                  **{f"post_{k+1}": post_te[:, k] for k in range(K_best)}}).to_csv(
        out / "post_test.csv", index=False, encoding="utf-8-sig")
    np.savez(out / "lca_params.npz", pi=pi, P=P, K=K_best,
             items=np.array(kept_items, dtype="<U32"))

    # ---- 摘要与诊断 ----
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    ytr = tr["toxic"].astype(int).values
    cls = post_tr.argmax(axis=1)
    # 类画像：类平均响应档位 vs 全局（对齐条目 0-4 档均值差）
    prof = np.stack([(Ytr[cls == k].mean(axis=0)) for k in range(K_best)])
    prof_all = Ytr.mean(axis=0)
    lines = ["=" * 72, "v8.6 LCA 潜类摘要", "=" * 72,
             f"K 选择（BIC，train 内无监督）：",
             bic_df.to_string(index=False), "",
             f"选中 K={K_best}；train/test loglik {ll_tr:.1f}/{ll_te:.1f}",
             f"类占比（train）：", ""]
    lines.append(f"  {'类':<6}{'占比':<8}{'毒性率':<10}expression 分布 (0/1/2/3/-1)")
    for k in range(K_best):
        m = cls == k
        ed = [float((expr[m] == g).mean()) for g in (0, 1, 2, 3, -1)]
        lines.append(f"  C{k+1:<5}{m.mean():<8.3f}{ytr[m].mean():<10.3f}"
                     + "/".join(f"{v:.2f}" for v in ed))
    lines += ["", f"{'类':<6}{'post对y的AUC':<14}（诊断，train 内）"]
    from sklearn.metrics import roc_auc_score
    for k in range(K_best):
        a = roc_auc_score(ytr, post_tr[:, k])
        lines.append(f"  C{k+1:<5}{max(a, 1-a):<14.4f} 方向'{'+' if a >= 0.5 else '-'}'")
    lines += ["", "类画像（各类响应档位均值 − 全局均值，|偏差| 前 10 条目）："]
    for k in range(K_best):
        d = prof[k] - prof_all
        top = np.argsort(-np.abs(d))[:10]
        seg = ", ".join(f"{kept_items[i]}({d[i]:+.2f})" for i in top)
        lines.append(f"  C{k+1}: {seg}")

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
