# -*- coding: utf-8 -*-
"""v8.6 读出：测量数据上的剩余理论杠杆，七配置 CV 竞争。

v8.5 闭合了损失侧（样本加权）与 CTT→IRT 重加权；本版在同一份测量矩阵上
检验四个未试过的变换/结构理论 + 一个自动方法对照臂：

  S1  M4 锚点复刻（与 v8.3 Q1 / v8.5 R1 逐特征一致）。
  S2  van der Waerden 正态得分——v8.1 秩变换的直系升级：rank→Φ⁻¹(rank)
      把每个边缘分布精确拉成高斯（线性读出下理论上最优的单调变换，
      Gaussian marginals + linear link 匹配）。仅替换 232 维秩块。
  S3  S1 + LCA 潜类后验（lca_typology.py 产出，K 列，无监督无标签泄漏）
      ——use/mention 作为潜类而非连续维。
  S4  S1 + θ_eap 线性样条基（6 个 train 分位结点的 hinge）——中带非线性
      的最小实现：决策边界在 θ 轴上分段可调（v8.2 的 2 维交互是手工的、
      死了；这是审计指认的一维带状现象的直接解）。
  S5  S3 + S4（潜类 + 样条联合）。
  S6  两阶段级联（潜类门控的判别版）：全局 LR OOF 分数三分带 → 带内局部
      L1-LR（用全局最优 alpha，带内 5 折 OOF 选 τ）→ 带内概率合成全局
      判定。审计已证局部可分（FP vs TN 0.986 / E2FN vs E2TP 0.968），
      全局单阈值可证无法利用。带 <50 样本回退全局模型。
  S7  GBDT 兜底（HistGradientBoosting，CPU sklearn，自动非线性对照臂）。

预登记协议同 v8.x：CV 5 折 × 3 种子（42/43/44）竞争，τ@宏F1@0.005，
门槛 ≥+0.002 才碰 test；胜者 5 种子双口径 test + OOF 误差复检。
依赖：results/v86/lca/（lca_typology.py）、results/v85/grm/（θ_eap）。
产物：results/v86/readout_v86/{metrics.txt, cv_competition.csv, test_predictions.csv}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import readout_v83 as r83

GATE = r83.GATE
N_HINGE = 6
MIN_BAND = 50
KNOT_QS = np.linspace(0.125, 0.875, N_HINGE)
GBDT_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                   min_samples_leaf=40, l2_regularization=1.0)

TAGS = ["S1_M4锚点", "S2正态得分", "S3潜类", "S4θ样条", "S5组合", "S6级联", "S7GBDT"]


def rank_transform(raw_tr, raw_te, mode):
    """秩块变换：uniform = v8.1 原法（分位→[0,1]）；normal = 分位→Φ⁻¹。"""
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    out_tr, out_te = raw_tr.copy(), raw_te.copy()
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        q_tr = np.interp(raw_tr[:, j], edges, qs)
        q_te = np.interp(raw_te[:, j], edges, qs)
        if mode == "normal":
            q_tr = norm.ppf(np.clip(q_tr, 1e-4, 1 - 1e-4))
            q_te = norm.ppf(np.clip(q_te, 1e-4, 1 - 1e-4))
        out_tr[:, j], out_te[:, j] = q_tr, q_te
    return out_tr, out_te


def hinge_basis(theta_tr, theta_te):
    """θ_eap 的线性样条基：θ + 6 个 train 分位结点的 hinge max(0, θ-k)。"""
    knots = np.quantile(theta_tr, KNOT_QS)
    Htr = np.column_stack([theta_tr] + [np.maximum(0.0, theta_tr - k) for k in knots])
    Hte = np.column_stack([theta_te] + [np.maximum(0.0, theta_te - k) for k in knots])
    return Htr, Hte, ["TH"] + [f"hinge_{q:.3f}" for q in KNOT_QS]


def cv_select_gbdt(X, y, seed, tag=""):
    """GBDT 的 OOF 与 τ 选择（无 alpha 网格，结构同 r83.cv_select）。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in tqdm(cv.split(X, y), desc=f"    gbdt[{tag} s{seed}]", leave=False,
                       dynamic_ncols=True):
        m = HistGradientBoostingClassifier(random_state=seed, **GBDT_PARAMS)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    tau = max(r83.TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int),
                                                   average="macro"))
    return float(tau), oof


def cascade(X, y, seed):
    """S6 级联：返回 (oof_prob, oof_global, edges, taus, alpha_g)。train 内全 OOF。"""
    a_g, tau_g, oof_g = r83.cv_select(X, y, seed, "S6global")
    edges = np.quantile(oof_g, [1 / 3, 2 / 3])
    oof_f = oof_g.copy()
    taus = []
    for lo, hi, tag in ((-np.inf, edges[0], "低带"), (edges[0], edges[1], "中带"),
                        (edges[1], np.inf, "高带")):
        idx = np.where((oof_g > lo) & (oof_g <= hi))[0]
        if len(idx) < MIN_BAND:
            taus.append(tau_g)
            continue
        cvb = StratifiedKFold(5, shuffle=True, random_state=seed)
        ob = np.zeros(len(idx))
        for tr, va in cvb.split(X[idx], y[idx]):
            m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                C=1.0 / a_g, max_iter=2000))
            m.fit(X[idx[tr]], y[idx[tr]])
            ob[va] = m.predict_proba(X[idx[va]])[:, 1]
        tau_b = max(r83.TAU_GRID, key=lambda t: f1_score(y[idx], (ob >= t).astype(int),
                                                         average="macro"))
        oof_f[idx] = ob
        taus.append(tau_b)
    return oof_f, oof_g, edges, taus, a_g


def band_assign(scores, edges):
    """带划分（与 cascade 的掩码语义一致：低带 ≤ e0，中带 (e0,e1]，高带 > e1）。"""
    return np.where(scores <= edges[0], 0, np.where(scores <= edges[1], 1, 2))


def cascade_test(X, y, Xte, seed):
    """test 侧：train 全量 OOF 定带界与 τ（train 带成员按**全局** OOF 划分）
    → test 投影。"""
    oof_f, oof_g, edges, taus, a_g = cascade(X, y, seed)
    g = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / a_g, max_iter=2000)).fit(X, y)
    sc_te = g.predict_proba(Xte)[:, 1]
    band_of = band_assign(sc_te, edges)
    prob_te = sc_te.copy()
    sc_tr_band = band_assign(oof_g, edges)
    for b in range(3):
        idx = np.where(sc_tr_band == b)[0]
        tidx = np.where(band_of == b)[0]
        if len(idx) < MIN_BAND or len(tidx) == 0:
            continue
        m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a_g, max_iter=2000)).fit(X[idx], y[idx])
        prob_te[tidx] = m.predict_proba(Xte[tidx])[:, 1]
    return prob_te, edges, taus, a_g, sc_tr_band


def main():
    ap = argparse.ArgumentParser(description="v8.6 读出：七配置竞争")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True)
    ap.add_argument("--grm_dir", default="results/v85/grm")
    ap.add_argument("--lca_dir", default="results/v86/lca")
    ap.add_argument("--output_dir", default="results/v86/readout_v86")
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

    # ---- 条目对齐翻转 / v3 筛选 / 元特征（与 v8.3/v8.5 逐行一致）----
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
    v3_names = kept_c + kept_s
    v3_tr = np.hstack([Ftr_c[kept_c].values, v3tr[kept_s].values])
    v3_te = np.hstack([Fte_c[kept_c].values, v3te[kept_s].values])

    prob_names = kept_items + v3_names
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    raw_te = np.hstack([Ate[kept_items].values, v3_te])
    Rall_tr, Rall_te = rank_transform(raw_tr, raw_te, "uniform")
    Rns_tr, Rns_te = rank_transform(raw_tr, raw_te, "normal")

    n_items = len(kept_items)
    Atr_items, Ate_items = Atr[kept_items].values, Ate[kept_items].values
    Mtr, Mte = r83.meta_base(tr["content"].tolist(), te["content"].tolist())
    mnames = ["meta_len", "meta_emoji", "meta_exclaim", "meta_question",
              "meta_repeatpunct", "meta_evasion"]
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    top_names = [kept_items[i] for i in order_i]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate_items[:, order_i]])
    meta_all_names = mnames + [f"lenx_{f}" for f in top_names]

    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    base_tr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    base_te = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    base_ns_tr = np.hstack([macro_raw_tr[mcols].values, Rns_tr, meta_tr])
    base_ns_te = np.hstack([macro_raw_te[mcols].values, Rns_te, meta_te])
    r1_names = mcols + prob_names + meta_all_names

    # ---- LCA 后验 ----
    ldir = Path(args.lca_dir)
    ptr = pd.read_csv(ldir / "post_train.csv")
    pte = pd.read_csv(ldir / "post_test.csv")
    assert (ptr["id"].values == tr["id"].values).all()
    assert (pte["id"].values == te["id"].values).all()
    lca_names = [c for c in ptr.columns if c != "id"]
    Ltr, Lte = ptr[lca_names].values, pte[lca_names].values

    # ---- θ 样条 ----
    th_tr = pd.read_csv(Path(args.grm_dir) / "theta_train.csv")
    th_te = pd.read_csv(Path(args.grm_dir) / "theta_test.csv")
    assert (th_tr["id"].values == tr["id"].values).all()
    assert (th_te["id"].values == te["id"].values).all()
    Htr, Hte, h_names = hinge_basis(th_tr["theta_eap"].values,
                                    th_te["theta_eap"].values)

    cfg = {
        "S1_M4锚点": (base_tr, base_te, r1_names, "lr"),
        "S2正态得分": (base_ns_tr, base_ns_te, r1_names, "lr"),
        "S3潜类": (np.hstack([base_tr, Ltr]), np.hstack([base_te, Lte]),
                 r1_names + lca_names, "lr"),
        "S4θ样条": (np.hstack([base_tr, Htr]), np.hstack([base_te, Hte]),
                 r1_names + h_names, "lr"),
        "S5组合": (np.hstack([base_tr, Ltr, Htr]), np.hstack([base_te, Lte, Hte]),
                 r1_names + lca_names + h_names, "lr"),
        "S6级联": (base_tr, base_te, r1_names, "cascade"),
        "S7GBDT": (base_tr, base_te, r1_names, "gbdt"),
    }
    lines = ["=" * 72, "v8.6 读出（七配置 CV 竞争）", "=" * 72,
             "S2 正态得分 / S3 LCA潜类 / S4 θ样条 / S5 组合 / S6 级联 / S7 GBDT",
             f"LCA K={len(lca_names)} 类，样条 {N_HINGE} 结点（θ 分位）",
             "维度: " + ", ".join(f"{k}={v[0].shape[1]}" for k, v in cfg.items()), ""]

    def eval_oof(tag, seed):
        X, _, _, kind = cfg[tag]
        if kind == "lr":
            a, t, oof = r83.cv_select(X, ytr, seed, tag)
            return {"alpha": a, "tau": t, "oof": oof, "W": None}
        if kind == "gbdt":
            t, oof = cv_select_gbdt(X, ytr, seed, tag)
            return {"alpha": None, "tau": t, "oof": oof, "W": None}
        oof, _, edges, taus, a_g = cascade(X, ytr, seed)
        t = max(r83.TAU_GRID, key=lambda tt: f1_score(ytr, (oof >= tt).astype(int),
                                                      average="macro"))
        return {"alpha": a_g, "tau": t, "oof": oof, "W": (tuple(np.round(edges, 3)),
                                                          taus)}

    rows = []
    for seed in r83.CV_SEEDS:
        for tag in TAGS:
            res = eval_oof(tag, seed)
            rows.append({"config": tag, "seed": seed,
                         "alpha": res["alpha"] if res["alpha"] is not None else "",
                         "oof_macro_f1": f1_score(ytr, (res["oof"] >= res["tau"]).astype(int),
                                                  average="macro"),
                         "oof_auc": roc_auc_score(ytr, res["oof"]),
                         "oof_tau": res["tau"]})
    comp = pd.DataFrame(rows)
    for r in comp.itertuples():
        lines.append(f"  {r.config:<10} seed{r.seed}  alpha={str(r.alpha):<8} "
                     f"OOF宏F1={r.oof_macro_f1:.4f}  OOF AUC={r.oof_auc:.4f}  τ*={r.oof_tau:.2f}")
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    best = mean_f1.idxmax()
    gain = mean_f1[best] - mean_f1["S1_M4锚点"]
    gate_pass = best != "S1_M4锚点" and gain >= GATE
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best}（相对 S1 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过'}", ""]

    test_tag = best if gate_pass else "S1_M4锚点"
    if not gate_pass:
        lines.append("按预登记协议，门槛未通过时候选配置不碰 test；test 仅复评锚点 S1。")
    Xw_tr, Xw_te, w_names, kind = cfg[test_tag]
    per_seed, prob, pred_m = [], None, None
    for seed in tqdm(r83.FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        res = eval_oof(test_tag, seed)
        if kind == "cascade":
            prob_te, edges, taus, a_g, band_tr = cascade_test(Xw_tr, ytr, Xw_te, seed)
            prob, pred_m = prob_te, (prob_te >= res["tau"]).astype(int)
            info = {"alpha": a_g, "W": f"edges={np.round(edges,3).tolist()}"}
        elif kind == "gbdt":
            m = HistGradientBoostingClassifier(random_state=seed, **GBDT_PARAMS)
            m.fit(Xw_tr, ytr)
            prob = m.predict_proba(Xw_te)[:, 1]
            pred_m = (prob >= res["tau"]).astype(int)
            info = {"alpha": None, "W": None}
        else:
            a = res["alpha"]
            m = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
            prob = m.predict_proba(Xw_te)[:, 1]
            pred_m = (prob >= res["tau"]).astype(int)
            info = {"alpha": a, "W": None}
        tau_m, tau_t = res["tau"], max(r83.TAU_GRID, key=lambda tt: f1_score(
            ytr, (res["oof"] >= tt).astype(int)))
        per_seed.append({"seed": seed, "alpha": info["alpha"], "tau_macro": tau_m,
                         "tau_toxic": tau_t,
                         "macro_f1": f1_score(yte, pred_m, average="macro"),
                         "toxic_f1": f1_score(yte, pred_m),
                         "benign_f1": f1_score(yte, pred_m, pos_label=0),
                         "auc": roc_auc_score(yte, prob),
                         "toxic_f1_t": f1_score(yte, (prob >= tau_t).astype(int)),
                         "macro_f1_t": f1_score(yte, (prob >= tau_t).astype(int),
                                                average="macro")})
    per = pd.DataFrame(per_seed)
    lines += [f"test 评估（{test_tag}，5 种子）:",
              f"  {'seed':<6}{'alpha':<10}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}"
              f"{'无毒F1':<9}{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}",
              "-" * 72]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{str(r['alpha']):<10}{r['tau_macro']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}{r['benign_f1']:<9.4f}"
                     f"{r['auc']:<8.4f}{r['tau_toxic']:<7.2f}{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 72,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v8.1/v8.3 M4 宏 0.8141 / 有毒 0.8249 (@τ毒 0.8310) / AUC 0.8948",
              "      v8.5 R1 复现 宏 0.8153 / 有毒 0.8267 / AUC 0.8942"]

    # ---- OOF 稳定误差复检 ----
    oofs = np.column_stack([eval_oof(test_tag, s)["oof"] for s in r83.CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof_prob >= t).astype(int),
                                                     average="macro"))
    err = (oof_prob >= tau_e).astype(int) != ytr
    fn, fp = err & (ytr == 1), err & (ytr == 0)
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    lens = np.array([len(str(t).replace(" ", "")) for t in tr["content"].tolist()])
    lines += ["", "OOF 稳定误差复检（胜者配置，τ@宏F1）:",
              f"  FN {int(fn.sum())}/{int((ytr == 1).sum())}   FP {int(fp.sum())}/{int((ytr == 0).sum())}",
              f"  {'类型':<4}{'expression组':<10}{'长度桶':<8}{'样本数':<8}{'错误数':<8}{'错误率':<8}"]
    for label, mask in (("FN", fn), ("FP", fp)):
        for g in sorted(set(expr.tolist())):
            for lo, hi in r83.LEN_BINS:
                sel = (expr == g) & (lens >= lo) & (lens < hi)
                if sel.sum() == 0:
                    continue
                lines.append(f"  {label:<4}{g:<12}[{lo},{hi}){'':<2}"
                             f"{int(sel.sum()):<10}{int((mask & sel).sum()):<10}"
                             f"{(mask & sel).sum() / sel.sum():.4f}")

    # ---- 非零权重（种子 0，仅 LR 类配置）----
    if kind == "lr":
        a0 = per.iloc[0]["alpha"]
        m0 = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                             C=1.0 / a0, max_iter=2000)).fit(Xw_tr, ytr)
        coef = pd.Series(m0.model.coef_[0], index=w_names)
        nz = coef[coef.abs() > 1e-6]
        lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0）", "|coef| 前 20："]
        for name, w in nz.reindex(nz.abs().sort_values(ascending=False).index).head(20).items():
            lines.append(f"    {name:<18} {w:+.3f}")

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
