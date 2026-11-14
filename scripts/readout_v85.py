# -*- coding: utf-8 -*-
"""v8.5 读出：训练侧（损失结构）+ 测量理论侧（GRM θ）三配置竞争。

背景：v8.1 M4 之后五轮迭代全部在"特征工程"侧，损失结构从未动过。
v8.5 从两条理论线各取一个最小可检验增量：

  R1  M4 锚点复刻（与 v8.3 Q1 逐特征一致，保证可比）。
  R2  类条件样本加权——macro-F1 的一阶 surrogate：宏 F1 对类内错误一视
      同仁，而 OOF 误差表显示 expr2/expr3（反讽/语境类）错误率远高于
      expr1；对这两组样本上权 = 让 loss 更接近宏 F1 的代理目标。
      实现：liblinear LR fit 传 sample_weight（不装任何新依赖），
      权重网格 W∈{1,2,3}（expr∈{2,3} 上权 W 倍），与 alpha 联合选择。
  R3  GRM 宏层替换——v6-v8 的构念分是 CTT 均值聚合；R3 用 GRM（EM，
      见 irt_grm.py）的 a_i 区分度做构念内信息加权，全局列替换为
      [θ_eap, θ_se]（潜在特质 EAP 与测量误差）。其余（秩条目+v3、
      meta/交互）与 R1 完全一致——纯宏层 A/B。

预登记协议：CV 5 折 × 3 种子（42/43/44）竞争，τ@宏F1（网格 0.30-0.70
@0.005），alpha 网格同 v8.x；门槛：胜者相对 R1 增益 ≥+0.002 才碰 test；
胜者 5 种子（0-4）双口径（τ@宏F1 / τ@有毒F1）test 评估 + OOF 误差复检
（expression×长度桶）。

依赖：先跑 irt_grm.py 生成 results/v85/grm/（R3 需要）。
产物：results/v85/readout_v85/{metrics.txt, cv_competition.csv, test_predictions.csv}
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

import readout_v83 as r83

GATE = r83.GATE
W_GRID = [1.0, 2.0, 3.0]
D2_TIERS = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]

TAG_R1 = "R1_M4锚点"
TAG_R2 = "R2_表达上权"
TAG_R3 = "R3_GRM宏层"


class PipelineW:
    """带 sample_weight 的标准化 LR 管线（liblinear，无新依赖）。"""

    def __init__(self, model):
        self.scaler = StandardScaler()
        self.model = model

    def fit(self, X, y, sw=None):
        self.model.fit(self.scaler.fit_transform(X), y, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def cv_select_w(X, y, seed, sw=None, tag=""):
    """r83.cv_select 的样本加权版：唯一差异是 fit 传 sample_weight
    （评分仍用无权宏 F1，与目标指标一致）。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best_a, best_f = None, -1.0
    for a in tqdm(r83.ALPHA_GRID, desc=f"    alpha[{tag} s{seed}]",
                  leave=False, dynamic_ncols=True):
        f1s = []
        for tr, va in cv.split(X, y):
            m = PipelineW(LogisticRegression(penalty="l1", solver="liblinear",
                                             C=1.0 / a, max_iter=2000))
            m.fit(X[tr], y[tr], sw[tr] if sw is not None else None)
            prob = m.predict_proba(X[va])[:, 1]
            f1s.append(max(f1_score(y[va], (prob >= t).astype(int), average="macro")
                           for t in r83.TAU_GRID))
        f = float(np.mean(f1s))
        if f > best_f:
            best_a, best_f = a, f
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = PipelineW(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / best_a, max_iter=2000))
        m.fit(X[tr], y[tr], sw[tr] if sw is not None else None)
        oof[va] = m.predict_proba(X[va])[:, 1]
    tau = max(r83.TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int),
                                                   average="macro"))
    return best_a, float(tau), oof


def main():
    ap = argparse.ArgumentParser(description="v8.5 读出：R1/R2/R3 竞争")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True, help="train rings JSON（expression）")
    ap.add_argument("--grm_dir", default="results/v85/grm")
    ap.add_argument("--output_dir", default="results/v85/readout_v85")
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

    # ---- 条目对齐翻转（与 v8.3 逐行一致）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]

    # ---- v3 筛选（同 v8.1/v8.2/v8.3）----
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

    # ---- 秩变换（条目 + v3）----
    prob_names = kept_items + v3_names
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    raw_te = np.hstack([Ate[kept_items].values, v3_te])
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rall_tr, Rall_te = raw_tr.copy(), raw_te.copy()
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)
        Rall_te[:, j] = np.interp(raw_te[:, j], edges, qs)
    n_items = len(kept_items)
    Rtr_items, Rte_items = Rall_tr[:, :n_items], Rall_te[:, :n_items]

    # ---- V1r/V2r 用未翻转极性的原始条目秩 ----
    Rraw_tr = Xtr_raw[kept_items].values.copy()
    Rraw_te = Xte_raw[kept_items].values.copy()
    for j in range(Rraw_tr.shape[1]):
        edges = np.quantile(Rraw_tr[:, j], qs)
        Rraw_tr[:, j] = np.interp(Rraw_tr[:, j], edges, qs)
        Rraw_te[:, j] = np.interp(Rraw_te[:, j], edges, qs)

    # ---- 元特征 + len×top5（v8.1 M4 同法）----
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

    # ---- R1 宏层（v8.1 旧 T 逻辑）----
    macro_raw_tr = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = r83.construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]
    q1_tr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    q1_te = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    r1_names = mcols + prob_names + meta_all_names

    # ---- R3 宏层（GRM：θ_eap/θ_se + a_i 信息加权构念分 + D2 + V1r/V2r）----
    gdir = Path(args.grm_dir)
    gm = np.load(gdir / "grm_params.npz")
    gm_items = [str(x) for x in gm["items"]]
    assert gm_items == kept_items, "GRM 条目顺序与 freeze 不一致，请重跑 irt_grm.py"
    a_gm = gm["a"]
    th_tr = pd.read_csv(gdir / "theta_train.csv")
    th_te = pd.read_csv(gdir / "theta_test.csv")
    assert (th_tr["id"].values == tr["id"].values).all()
    assert (th_te["id"].values == te["id"].values).all()
    col_of = {c: j for j, c in enumerate(kept_items)}
    g_tr = [th_tr["theta_eap"].values, th_tr["theta_se"].values]
    g_te = [th_te["theta_eap"].values, th_te["theta_se"].values]
    g_names = ["GRM_theta", "GRM_theta_se"]
    wmap = freeze.get("item_weights", {})
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"] or cid == "D2":
            continue
        idx = [col_of[c] for c in kept if c in col_of]
        if not idx:
            continue
        wgt = a_gm[idx]                                   # GRM 区分度做信息权重
        wgt = wgt / wgt.sum()
        vtr = Rtr_items[:, idx] @ wgt
        vte = Rte_items[:, idx] @ wgt
        mu, sd = vtr.mean(), vtr.std()
        g_tr.append(50.0 + 10.0 * (vtr - mu) / (sd if sd > 1e-9 else 1.0))
        g_te.append(50.0 + 10.0 * (vte - mu) / (sd if sd > 1e-9 else 1.0))
        g_names.append(f"{cid}_gm")
    if "D2" in freeze["constructs_alive"]:
        d2raw_tr = r83.d2_score(Atr, D2_TIERS)
        d2raw_te = r83.d2_score(Ate, D2_TIERS)
        mu, sd = d2raw_tr.mean(), d2raw_tr.std()
        sd = sd if sd > 1e-9 else 1.0
        g_tr.append(50.0 + 10.0 * (d2raw_tr - mu) / sd)
        g_te.append(50.0 + 10.0 * (d2raw_te - mu) / sd)
        g_names.append("D2_mean")
    Vr_tr, Vr_te, vr_names = r83.vv_rank(Rraw_tr, Rraw_te, kept_items, freeze)
    g_tr += [Vr_tr[:, 0], Vr_tr[:, 1]]
    g_te += [Vr_te[:, 0], Vr_te[:, 1]]
    g_names += vr_names
    q3_tr = np.hstack([np.column_stack(g_tr), Rall_tr, meta_tr])
    q3_te = np.column_stack(g_te)
    q3_te = np.hstack([q3_te, Rall_te, meta_te])
    r3_names = g_names + prob_names + meta_all_names

    # ---- R2 权重（expression 组，train rings json）----
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    mask23 = np.isin(expr, [2, 3])
    weight_of = {W: 1.0 + (W - 1.0) * mask23.astype(float) for W in W_GRID}

    cfg = {
        TAG_R1: (q1_tr, q1_te, r1_names),
        TAG_R2: (q1_tr, q1_te, r1_names),   # 同特征，损失侧加权
        TAG_R3: (q3_tr, q3_te, r3_names),
    }
    lines = ["=" * 72, "v8.5 读出（R1 锚点 / R2 样本加权 / R3 GRM 宏层，CV 竞争）",
             "=" * 72,
             "R2: expr∈{2,3} 上权 W∈{1,2,3}（macro-F1 surrogate 一阶近似），"
             "与 alpha 联合选择",
             "R3: GRM a_i 信息加权构念分 + θ_eap/θ_se 替换 CTT 均值宏层"
             "（其余与 R1 一致）",
             "维度: " + ", ".join(f"{k}={v[0].shape[1]}" for k, v in cfg.items()), ""]

    def select_seed(tag, seed, r1_res=None):
        """返回 (alpha, tau_macro, tau_toxic, oof, W or None)。"""
        if tag == TAG_R2:
            best = None
            for W in W_GRID:
                if W == 1.0 and r1_res is not None:
                    a, t, oof = r1_res          # W=1 与 R1 同损失，复用
                else:
                    a, t, oof = cv_select_w(q1_tr, ytr, seed, weight_of[W],
                                            f"R2W{W:g}")
                f = f1_score(ytr, (oof >= t).astype(int), average="macro")
                if best is None or f > best[0]:
                    best = (f, W, a, t, oof)
            _, W, a, t, oof = best
        elif tag == TAG_R3:
            a, t, oof = r83.cv_select(q3_tr, ytr, seed, tag)
            W = None
        else:
            a, t, oof = r83.cv_select(q1_tr, ytr, seed, tag)
            W = None
        tau_t = max(r83.TAU_GRID, key=lambda tt: f1_score(ytr, (oof >= tt).astype(int)))
        return a, t, tau_t, oof, W

    rows = []
    for seed in r83.CV_SEEDS:
        a1, t1, oof1 = r83.cv_select(q1_tr, ytr, seed, TAG_R1)
        r1_res = (a1, t1, oof1)
        for tag in (TAG_R1, TAG_R2, TAG_R3):
            if tag == TAG_R1:
                a, t, tt, oof, W = a1, t1, None, oof1, None
                tt = max(r83.TAU_GRID, key=lambda x: f1_score(ytr, (oof >= x).astype(int)))
            else:
                a, t, tt, oof, W = select_seed(tag, seed, r1_res)
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "weight": W if W is not None else "",
                         "oof_macro_f1": f1_score(ytr, (oof >= t).astype(int),
                                                  average="macro"),
                         "oof_auc": roc_auc_score(ytr, oof), "oof_tau": t})
    comp = pd.DataFrame(rows)
    for r in comp.itertuples():
        lines.append(f"  {r.config:<12} seed{r.seed}  alpha={r.alpha:<8}"
                     f"{'W=' + str(r.weight) if r.weight != '' else '':<6} "
                     f"OOF宏F1={r.oof_macro_f1:.4f}  OOF AUC={r.oof_auc:.4f}  τ*={r.oof_tau:.2f}")
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    best = mean_f1.idxmax()
    gain = mean_f1[best] - mean_f1[TAG_R1]
    gate_pass = best != TAG_R1 and gain >= GATE
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best}（相对 R1 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过'}", ""]

    test_tag = best if gate_pass else TAG_R1
    if not gate_pass:
        lines.append("按预登记协议，门槛未通过时候选配置不碰 test；test 仅复评锚点 R1。")
    Xw_tr, Xw_te, w_names = cfg[test_tag]
    per_seed, prob, pred_m = [], None, None
    for seed in tqdm(r83.FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau_m, tau_t, _, W = select_seed(test_tag, seed)
        if test_tag == TAG_R2:
            model = PipelineW(LogisticRegression(penalty="l1", solver="liblinear",
                                                 C=1.0 / a, max_iter=2000))
            model.fit(Xw_tr, ytr, weight_of[W])
        else:
            model = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                    C=1.0 / a, max_iter=2000))
            model.fit(Xw_tr, ytr)
        prob = model.predict_proba(Xw_te)[:, 1]
        pred_m = (prob >= tau_m).astype(int)
        per_seed.append({"seed": seed, "alpha": a, "W": W, "tau_macro": tau_m,
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
              f"  {'seed':<6}{'alpha':<10}{'W':<5}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}"
              f"{'无毒F1':<9}{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}",
              "-" * 76]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['alpha']:<10}{str(r['W']):<5}{r['tau_macro']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}{r['benign_f1']:<9.4f}"
                     f"{r['auc']:<8.4f}{r['tau_toxic']:<7.2f}{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 76,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v8.1/v8.3 M4 宏 0.8141 / 有毒 0.8249 (@τ毒 0.8310) / AUC 0.8948"]

    # ---- OOF 稳定误差复检（胜者配置，train 内）----
    oofs = np.column_stack([select_seed(test_tag, s)[3] for s in r83.CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(r83.TAU_GRID, key=lambda t: f1_score(ytr, (oof_prob >= t).astype(int),
                                                     average="macro"))
    err = (oof_prob >= tau_e).astype(int) != ytr
    fn, fp = err & (ytr == 1), err & (ytr == 0)
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

    # ---- 非零权重（种子 0）----
    if test_tag == TAG_R2:
        model0 = PipelineW(LogisticRegression(penalty="l1", solver="liblinear",
                                              C=1.0 / per.iloc[0]["alpha"],
                                              max_iter=2000))
        model0.fit(Xw_tr, ytr, weight_of[per.iloc[0]["W"]])
    else:
        model0 = r83.Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                                 C=1.0 / per.iloc[0]["alpha"],
                                                 max_iter=2000))
        model0.fit(Xw_tr, ytr)
    coef = pd.Series(model0.model.coef_[0], index=w_names)
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
