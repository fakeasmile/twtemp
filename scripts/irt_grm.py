# -*- coding: utf-8 -*-
"""v8.5 R3 前置：GRM（等级反应模型，Samejima）EM 拟合与 θ 估计。

理论动机（v8.5 三层回顾·测量理论侧）：
  v6-v8 的构念分一律是"条目 p_yes 加权均值"——经典测验理论（CTT）聚合。
  CTT 对所有条目用冻结权重，忽略条目区分度差异。GRM 把每个 5 档条目建模为
    P(y>=k | θ) = sigmoid(a_i * (θ - b_ik)),  k = 1..4
  EM 估计出：
    a_i   条目区分度（信息权重——R3 用它替换冻结 item_weights 做构念聚合）；
    b_ik  条目阈值；
    θ     每个样本的潜在特质 EAP 估计（R3 用 θ_eap/θ_se 替换均值聚合的
          全局构念分列）。
  识别性：θ 先验 N(0,1) 固定尺度与位置（41 个格点，-4..4）。

分箱：对齐翻转（item_polarity==-1 取 1-p）后的条目 p_yes 按固定边界
  [0.2,0.4,0.6,0.8) digitize 成 0-4 五档（保留原始水平信息；退化条目由
  M-step 有界优化自然处理，不做特殊剔除）。

M-step：E-step 后验下，每条目的期望完全数据对数似然只依赖充分统计量
  Nqj[q,j] = Σ_n post_nq·1[y_ni=j]（41×5 表），目标函数与解析梯度在
  quadrature 格点上计算，L-BFGS-B 有界优化（a∈[0.05,5]，b∈[-5,5]）。
  阈值次序违反时排序并计数（罕见）。内置解析梯度 vs 有限差分自检。

产物（results/v85/grm/）：
  grm_params.npz    a / b / θ格点 / 权重 / 条目信息网格 / 条目名（下游复用）
  grm_params.json   条目清单 + a + b1..b4 + 收敛信息（人读）
  theta_train.csv / theta_test.csv   id, theta_eap, theta_se
                    （GRM 只在 train 拟合，test 用拟合法参数投影——无泄漏）
  metrics.txt       收敛轨迹、a 分布、弱条目、阈值违反、分档占比、θ 分带 SE
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

BIN_EDGES = np.array([0.2, 0.4, 0.6, 0.8])
N_QUAD = 41
TH_LO, TH_HI = -4.0, 4.0
A_BOUNDS = (0.05, 5.0)
B_BOUNDS = (-5.0, 5.0)


def bin5(P):
    """p_yes 矩阵 → 0-4 五档类别。"""
    return np.digitize(P, BIN_EDGES).astype(int)


def make_quad():
    th = np.linspace(TH_LO, TH_HI, N_QUAD)
    w = norm.pdf(th)
    return th, w / w.sum()


def grm_p(a, b, th):
    """类别概率 p_j = P(y=j), j=0..4。返回 (p, C)，C[:,k]=P(y>=k)，k=0..5。"""
    z = a * (th[:, None] - b[None, :])          # (Q,4)
    C = np.zeros((len(th), 6))
    C[:, 0] = 1.0
    C[:, 1:5] = expit(z)
    p = C[:, :5] - C[:, 1:6]
    return np.clip(p, 1e-12, 1.0), C


def grm_derivs(a, b, th, C):
    """dC/dθ, dC/da, dC/db 的解析导数（C 为 (Q,6) 累积概率）。"""
    sig = C[:, 1:5] * (1.0 - C[:, 1:5])          # (Q,4)
    dC_th = np.zeros_like(C)
    dC_th[:, 1:5] = a * sig
    dC_a = np.zeros_like(C)
    dC_a[:, 1:5] = (th[:, None] - b[None, :]) * sig
    dC_b = np.zeros((len(th), 6, 4))
    for m in range(4):
        dC_b[:, m + 1, m] = -a * sig[:, m]
    return dC_th, dC_a, dC_b


def item_info_grid(a, b, th):
    """单条目信息函数 I_i(θ) = Σ_j (dp_j/dθ)²/p_j，返回 (Q,)。"""
    p, C = grm_p(a, b, th)
    dC_th, _, _ = grm_derivs(a, b, th, C)
    dp = dC_th[:, :5] - dC_th[:, 1:6]
    return np.sum(dp * dp / p, axis=1)


def _mstep_obj(Nqj, th):
    """期望完全数据对数似然的负值与解析梯度（充分统计量 Nqj: (Q,5)）。"""
    def f_grad(u):
        a, b = u[0], u[1:]
        p, C = grm_p(a, b, th)
        nll = -np.sum(Nqj * np.log(p))
        dC_th, dC_a, dC_b = grm_derivs(a, b, th, C)
        dp_a = dC_a[:, :5] - dC_a[:, 1:6]
        dp_b = dC_b[:, :5, :] - dC_b[:, 1:6, :]
        g = np.empty(5)
        g[0] = -np.sum(Nqj * dp_a / p)
        g[1:] = -np.sum(Nqj[:, :, None] * dp_b / p[:, :, None], axis=(0, 1))
        return nll, g
    return f_grad


def mstep_item(Nqj, th, a0, b0):
    f = _mstep_obj(Nqj, th)
    res = minimize(f, np.concatenate([[a0], b0]), jac=True, method="L-BFGS-B",
                   bounds=[A_BOUNDS] + [B_BOUNDS] * 4)
    return res.x[0], res.x[1:]


def grad_check(Nqj, th, a0, b0, eps=1e-6):
    """解析梯度 vs 中心差分，返回最大相对误差。"""
    f = _mstep_obj(Nqj, th)
    u0 = np.concatenate([[a0], b0])
    _, g = f(u0)
    g_num = np.empty(5)
    for k in range(5):
        up, um = u0.copy(), u0.copy()
        up[k] += eps
        um[k] -= eps
        g_num[k] = (f(up)[0] - f(um)[0]) / (2.0 * eps)
    rel = np.abs(g - g_num) / (np.abs(g_num) + 1e-8)
    return float(rel.max())


def estep(Y, a, b, th, lw):
    """E-step：返回 (post (N,Q), 边际对数似然标量)。"""
    logL = np.zeros((Y.shape[0], len(th)))
    for i in range(Y.shape[1]):
        p, _ = grm_p(a[i], b[i], th)
        logL += np.log(p)[:, Y[:, i]].T          # 每人取自己类别列
    lp = logL + lw[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    post = ep / ep.sum(axis=1, keepdims=True)
    marg = float(np.log(ep.sum(axis=1)).sum())
    return post, marg


def eap_se(post, th):
    eap = post @ th
    var = np.maximum((post * (th[None, :] - eap[:, None]) ** 2).sum(axis=1), 1e-12)
    return eap, np.sqrt(var)


def init_params(Y):
    """a=1；b_k 经验初始化：边际 P(y>=k)=Φ(-b_k/√2)（a=1, θ~N(0,1) 下）。"""
    I = Y.shape[1]
    a = np.ones(I)
    b = np.zeros((I, 4))
    for i in range(I):
        prop = np.array([(Y[:, i] >= k).mean() for k in (1, 2, 3, 4)])
        b[i] = -np.sqrt(2.0) * norm.ppf(np.clip(prop, 0.01, 0.99))
    return a, b


def fit_grm(Y, max_iter=100, tol=1e-4, verbose=True):
    th, w = make_quad()
    lw = np.log(w)
    a, b = init_params(Y)
    n_viol, hist, converged = 0, [], False
    for it in range(1, max_iter + 1):
        post, marg = estep(Y, a, b, th, lw)
        hist.append(marg)
        a_new, b_new = np.empty_like(a), np.empty_like(b)
        it_items = tqdm(range(Y.shape[1]), desc=f"    EM {it}", leave=False,
                        dynamic_ncols=True) if verbose else range(Y.shape[1])
        for i in it_items:
            Oh = (Y[:, i][:, None] == np.arange(5)).astype(float)
            Nqj = post.T @ Oh                     # (Q,5) 充分统计量
            ai, bi = mstep_item(Nqj, th, max(a[i], 0.1), b[i])
            if np.any(np.diff(bi) < -1e-8):
                n_viol += 1
                bi = np.sort(bi)
            a_new[i], b_new[i] = ai, bi
        delta = max(np.abs(a_new - a).max(), np.abs(b_new - b).max())
        a, b = a_new, b_new
        if verbose and (it == 1 or it % 10 == 0):
            print(f"  EM iter {it:>3}  marginal loglik={marg:.1f}  "
                  f"max|Δparam|={delta:.2e}")
        if delta < tol:
            converged = True
            break
    post, marg = estep(Y, a, b, th, lw)
    eap, se = eap_se(post, th)
    info = np.column_stack([item_info_grid(a[i], b[i], th)
                            for i in range(Y.shape[1])]).T     # (I, Q)
    return {"a": a, "b": b, "th": th, "w": w, "eap": eap, "se": se,
            "info": info, "hist": hist, "converged": converged,
            "n_iter": len(hist), "n_viol": n_viol, "marg": marg}


def apply_grm(fit, Y_new):
    """用拟合法参数对新数据做 E-step → EAP θ 与 SE。"""
    lw = np.log(fit["w"])
    post, _ = estep(Y_new, fit["a"], fit["b"], fit["th"], lw)
    return eap_se(post, fit["th"])


def main():
    ap = argparse.ArgumentParser(description="v8.5 GRM 拟合与 θ 估计")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v85/grm")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    # 对齐翻转（与 readout_v83 同法：反向条目取 1-p）
    Atr, Ate = tr[item_cols].copy(), te[item_cols].copy()
    n_flip = 0
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
            Ate[c] = 1.0 - te[c]
            n_flip += 1
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]

    Ytr = bin5(Atr[kept_items].values)
    Yte = bin5(Ate[kept_items].values)
    print(f"GRM 拟合：{len(kept_items)} 条目 × {len(Ytr)} train 样本，"
          f"5 档（边界 {BIN_EDGES.tolist()}），翻转 {n_flip} 条，{N_QUAD} 格点")

    fit = fit_grm(Ytr)

    # 梯度自检（用拟合收敛后的 post 的充分统计量）
    post_f, _ = estep(Ytr, fit["a"], fit["b"], fit["th"], np.log(fit["w"]))
    rng = np.random.default_rng(0)
    chk_idx = rng.choice(len(kept_items), size=min(3, len(kept_items)), replace=False)
    errs = []
    for i in chk_idx:
        Oh = (Ytr[:, i][:, None] == np.arange(5)).astype(float)
        errs.append(grad_check(post_f.T @ Oh, fit["th"], fit["a"][i], fit["b"][i]))
    print(f"梯度自检（3 条目，解析 vs 有限差分）最大相对误差 = {max(errs):.2e}")

    # test 投影（train 拟合参数）
    eap_te, se_te = apply_grm(fit, Yte)

    # ---- 保存 ----
    np.savez(out / "grm_params.npz", a=fit["a"], b=fit["b"], th=fit["th"],
             w=fit["w"], info=fit["info"],
             items=np.array(kept_items, dtype="<U32"))
    pd.DataFrame({"id": tr["id"].values, "theta_eap": fit["eap"],
                  "theta_se": fit["se"]}).to_csv(
        out / "theta_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": te["id"].values, "theta_eap": eap_te,
                  "theta_se": se_te}).to_csv(
        out / "theta_test.csv", index=False, encoding="utf-8-sig")
    json.dump({"items": kept_items,
               "a": fit["a"].tolist(), "b": fit["b"].tolist(),
               "n_iter": fit["n_iter"], "converged": fit["converged"],
               "n_threshold_violations": fit["n_viol"]},
              open(out / "grm_params.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- 摘要 ----
    con_of = {}
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            con_of[c] = cid
    a = fit["a"]
    pct = np.percentile(a, [10, 25, 50, 75, 90])
    lines = ["=" * 72, "v8.5 GRM 拟合摘要", "=" * 72,
             f"条目 {len(kept_items)}（翻转 {n_flip}）/ train n={len(Ytr)} / test n={len(Yte)}",
             f"EM：{fit['n_iter']} 轮，converged={fit['converged']}，"
             f"阈值次序违反 {fit['n_viol']} 次（已排序处理）",
             f"边际 loglik：首={fit['hist'][0]:.1f} 末={fit['hist'][-1]:.1f}",
             f"梯度自检最大相对误差：{max(errs):.2e}",
             "",
             f"a（区分度）分位数 P10/25/50/75/90: "
             + "/".join(f"{v:.3f}" for v in pct),
             f"a<0.4（弱区分）: {int((a < 0.4).sum())} 条；a>1.2（高区分）: {int((a > 1.2).sum())} 条",
             "", "区分度最高 10 条："]
    order = np.argsort(-a)
    for i in order[:10]:
        lines.append(f"    {kept_items[i]:<8} ({con_of.get(kept_items[i], '?'):<4}) a={a[i]:.3f}")
    lines.append("区分度最低 10 条：")
    for i in order[-10:]:
        lines.append(f"    {kept_items[i]:<8} ({con_of.get(kept_items[i], '?'):<4}) a={a[i]:.3f}")

    cat_prop = np.stack([(Ytr == k).mean(axis=0) for k in range(5)])   # (5, I)
    degen = [(kept_items[i], float(cat_prop[:, i].max()))
             for i in range(len(kept_items)) if cat_prop[:, i].max() > 0.98]
    lines += ["", f"退化条目（某档占比>98%）：{len(degen)} 条"]
    for name, p in degen[:10]:
        lines.append(f"    {name:<8} 最大档占比 {p:.3f}")

    bands = [(-4, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 4)]
    tot_info = fit["info"].sum(axis=0)                     # (Q,) 测验信息
    lines += ["", f"{'θ带':<12}{'train N':<10}{'θ带内EAP均值':<14}{'平均SE':<10}{'平均测验信息':<12}"]
    for lo, hi in bands:
        sel = (fit["eap"] >= lo) & (fit["eap"] < hi)
        gsel = (fit["th"] >= lo) & (fit["th"] < hi)
        mi = tot_info[gsel].mean() if gsel.any() else float("nan")
        lines.append(f"  [{lo:+.0f},{hi:+.0f})   {int(sel.sum()):<10}"
                     f"{fit['eap'][sel].mean() if sel.any() else float('nan'):<14.3f}"
                     f"{fit['se'][sel].mean() if sel.any() else float('nan'):<10.3f}{mi:<12.2f}")

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
