# -*- coding: utf-8 -*-
"""v8.3 读出：常模后聚合（构念层复活）+ 构念内反应模式特征。

背景（v8.1 分析）：v8.1 的增益全部来自"重新表达"而非新信息——秩变换统一了
度量衡（+0.003），元特征注入了标签的全文语境（+0.002）。剩余锁定信息在
联合结构里，两个候选：
  (1) 聚合顺序错误：历史上构念分 = 原始条目加权均值 → z（饱和污染均值，
      且 z 是线性变换不修形状）。正确顺序 = 条目先秩常模 → 加权聚合。
  (2) 构念内反应模式（MMPI 编码思想）：同构念"三高"与"一高两低"均值相同、
      模式不同。profile = 构念内秩的 max/std/top2 均值，从未存在过。
另一修正：历史版本构念 T 分对 train/test 各自计算统计量（test T 分用了
test 自身均值/方差，属轻微测试集统计使用）；本版全部改为 train 统计量冻结
投影。Q1 锚点保留旧逻辑以与 v8.1 可比，其余配置用冻结统计。
门槛：胜者 ≥ Q1 + 0.002 才碰 test。无论过否，输出胜者（或锚点）OOF 稳定
误差的 expression×长度复检表。

配置：
  Q1  M4 复刻锚点（秩变换条目+v3 + 元特征/交互 + 原始构念分[旧 T 逻辑]）
  Q2  Q1 但构念分改为"秩常模后加权聚合"（train 统计冻结），并加 D2 与 V1r/V2r
  Q3  Q2 + 构念内 profile（rmax/rstd/rtop2）
产物：results/v8/readout_v83/{metrics.txt, cv_competition.csv, test_predictions.csv}
"""

import argparse
import json
import re
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
TAU_GRID = np.round(np.arange(0.30, 0.7005, 0.005), 3)
CV_SEEDS = [42, 43, 44]
FINAL_SEEDS = [0, 1, 2, 3, 4]
V3_MIN_AUC = 0.55
V3_MAX_CORR = 0.90
N_QUANT = 101
N_TOP_INTERACT = 5
GATE = 0.002
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
EVASION_LEX = ("朲", "亽", "㐺", "亻", "怹")
LEN_BINS = [(0, 10), (10, 20), (20, 40), (40, 10 ** 9)]


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


def d2_score(A, tiers):
    hits = A[tiers].values >= 0.5
    score = np.zeros(len(A))
    for i in range(len(A)):
        h = hits[i]
        score[i] = (len(tiers) - int(np.where(h)[0][0])) if h.any() else 0.0
    return score


def construct_raw(df, freeze, A, Xr):
    """v8.1 及以前的构念分逻辑（原始值加权均值，T 分用传入 df 自身统计）。"""
    wmap = freeze.get("item_weights", {})
    out = pd.DataFrame({"id": df["id"].values})
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"]:
            continue
        kept = [c for c in kept if c in A.columns]
        if cid == "D2":
            tiers = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
            if not all(t in A.columns for t in tiers):
                continue
            v = d2_score(A, tiers)
        else:
            if not kept:
                continue
            ws = np.array([wmap.get(c, 1.0 / len(kept)) for c in kept])
            ws = ws / ws.sum()
            v = (A[kept].values * ws).sum(axis=1)
        out[f"{cid}_mean"] = 50.0 + 10.0 * (v - v.mean()) / (v.std() if v.std() > 1e-9 else 1.0)
    v1 = np.zeros(len(df))
    for cid, kept in freeze["items_per_construct"].items():
        pos = [c for c in kept if c in Xr.columns and freeze["item_polarity"].get(c, 1) == 1]
        neg = [c for c in kept if c in Xr.columns and freeze["item_polarity"].get(c, 1) == -1]
        if pos and neg:
            v1 += Xr[pos].mean(axis=1).values * Xr[neg].mean(axis=1).values
    v1 /= max(len(freeze["items_per_construct"]), 1)
    m, sd = Xr.mean(axis=1).values, Xr.std(axis=1).values
    v2 = (1.0 - np.clip(sd, 0, 1)) * np.abs(m - 0.5) * 2.0
    out["V1"], out["V2"] = v1, v2
    return out


def construct_rank(Rtr_items, Rte_items, item_names, freeze):
    """常模后聚合：构念分 = 秩条目加权均值；T 分用 train 统计冻结。"""
    wmap = freeze.get("item_weights", {})
    col_of = {c: j for j, c in enumerate(item_names)}
    means_tr, means_te, names = {}, {}, []
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"] or cid == "D2":
            continue
        kept = [c for c in kept if c in col_of]
        if not kept:
            continue
        ws = np.array([wmap.get(c, 1.0 / len(kept)) for c in kept])
        ws = ws / ws.sum()
        vtr = Rtr_items[:, [col_of[c] for c in kept]] @ ws
        vte = Rte_items[:, [col_of[c] for c in kept]] @ ws
        mu, sd = vtr.mean(), vtr.std()
        key = f"{cid}_mean"
        means_tr[key] = 50.0 + 10.0 * (vtr - mu) / (sd if sd > 1e-9 else 1.0)
        means_te[key] = 50.0 + 10.0 * (vte - mu) / (sd if sd > 1e-9 else 1.0)
        names.append(key)
    return means_tr, means_te, names


def profile_features(Rtr_items, Rte_items, item_names, freeze):
    """构念内反应模式（秩尺度）：max / std / top2 均值。"""
    col_of = {c: j for j, c in enumerate(item_names)}
    feats_tr, feats_te, names = [], [], []
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"] or cid == "D2":
            continue
        kept = [c for c in kept if c in col_of]
        if len(kept) < 3:
            continue
        idx = [col_of[c] for c in kept]
        Btr = np.sort(Rtr_items[:, idx], axis=1)
        Bte = np.sort(Rte_items[:, idx], axis=1)
        for tag, fn in (("rmax", lambda B: B[:, -1]),
                        ("rstd", lambda B: B.std(axis=1)),
                        ("rtop2", lambda B: B[:, -2:].mean(axis=1))):
            feats_tr.append(fn(Btr))
            feats_te.append(fn(Bte))
            names.append(f"{cid}_{tag}")
    return (np.column_stack(feats_tr) if feats_tr else np.zeros((Rtr_items.shape[0], 0)),
            np.column_stack(feats_te) if feats_te else np.zeros((Rte_items.shape[0], 0)),
            names)


def vv_rank(Rtr_items, Rte_items, item_names, freeze):
    """V1/V2 的秩版本：V1 = 正/反向条目秩均值乘积；V2 = (1-std)|m-0.5|*2。
    输入必须是未翻转极性的原始条目秩（效度指标的语义依赖正/反向条目分裂）。"""
    col_of = {c: j for j, c in enumerate(item_names)}
    pos, neg = [], []
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            if c not in col_of:
                continue
            (pos if freeze["item_polarity"].get(c, 1) == 1 else neg).append(col_of[c])
    if pos and neg:
        m1_tr = Rtr_items[:, pos].mean(axis=1)
        m0_tr = Rtr_items[:, neg].mean(axis=1)
        m1_te = Rte_items[:, pos].mean(axis=1)
        m0_te = Rte_items[:, neg].mean(axis=1)
    else:
        m1_tr = m0_tr = m1_te = m0_te = np.zeros(len(Rtr_items))
    s_tr = Rtr_items.std(axis=1)
    s_te = Rte_items.std(axis=1)
    mm_tr = Rtr_items.mean(axis=1)
    mm_te = Rte_items.mean(axis=1)
    v1_tr, v1_te = m1_tr * m0_tr, m1_te * m0_te
    v2_tr = (1.0 - np.clip(s_tr, 0, 1)) * np.abs(mm_tr - 0.5) * 2.0
    v2_te = (1.0 - np.clip(s_te, 0, 1)) * np.abs(mm_te - 0.5) * 2.0
    return (np.column_stack([v1_tr, v2_tr]), np.column_stack([v1_te, v2_te]),
            ["V1r", "V2r"])


def meta_base(texts_tr, texts_te):
    def base(texts):
        len_c = np.array([len(str(t).replace(" ", "")) for t in texts], dtype=float)
        emoji = np.array([len(EMOJI_RE.findall(str(t))) for t in texts], dtype=float)
        exclaim = np.array([str(t).count("！") + str(t).count("!") for t in texts], dtype=float)
        question = np.array([str(t).count("？") + str(t).count("?") for t in texts], dtype=float)
        repeat = np.array([len(re.findall(r"(？{2,}|！{2,}|\.{3,}|…{2,})", str(t)))
                           for t in texts], dtype=float)
        evade = np.array([sum(str(t).count(k) for k in EVASION_LEX) for t in texts], dtype=float)
        return np.column_stack([np.log1p(len_c), emoji, exclaim, question, repeat, evade])

    Mtr, Mte = base(texts_tr), base(texts_te)
    mu, sd = Mtr.mean(axis=0), Mtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (Mtr - mu) / sd, (Mte - mu) / sd


def main():
    ap = argparse.ArgumentParser(description="v8.3 读出：常模后聚合 + profile")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True, help="train rings JSON（expression）")
    ap.add_argument("--output_dir", default="results/v8/readout_v83")
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

    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Xtr_raw, Xte_raw = tr[item_cols], te[item_cols]
    Atr, Ate = Xtr_raw.copy(), Xte_raw.copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Xtr_raw[c]
            Ate[c] = 1.0 - Xte_raw[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]

    # v3 筛选（同 v8.1/v8.2）
    all_cols = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in all_cols if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in all_cols if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc = {}, {}
    for c in var_ok:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok if auc[c] >= V3_MIN_AUC]
    Ftr_c = v3tr[valid_c].copy()
    Fte_c = v3te[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr_c[c] = 1.0 - v3tr[c]
            Fte_c[c] = 1.0 - v3te[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr_c[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    v3_names = kept_c + kept_s
    v3_tr = np.hstack([Ftr_c[kept_c].values, v3tr[kept_s].values])
    v3_te = np.hstack([Fte_c[kept_c].values, v3te[kept_s].values])

    # 秩变换（条目 + v3）
    prob_names = kept_items + v3_names
    raw_tr = np.hstack([Atr[kept_items].values, v3_tr])
    raw_te = np.hstack([Ate[kept_items].values, v3_te])
    qs = np.linspace(0.0, 1.0, N_QUANT)
    Rall_tr, Rall_te = raw_tr.copy(), raw_te.copy()
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)
        Rall_te[:, j] = np.interp(raw_te[:, j], edges, qs)
    n_items = len(kept_items)
    Rtr_items, Rte_items = Rall_tr[:, :n_items], Rall_te[:, :n_items]

    # V1r/V2r 用未翻转极性的原始条目秩（效度指标需保留正/反向条目的原始极性分裂）
    Rraw_tr = Xtr_raw[kept_items].values.copy()
    Rraw_te = Xte_raw[kept_items].values.copy()
    for j in range(Rraw_tr.shape[1]):
        edges = np.quantile(Rraw_tr[:, j], qs)
        Rraw_tr[:, j] = np.interp(Rraw_tr[:, j], edges, qs)
        Rraw_te[:, j] = np.interp(Rraw_te[:, j], edges, qs)

    # 元特征 + len×top5（v8.1 M4：交互对象与 top5 选择均用原始对齐条目值）
    Atr_items = Atr[kept_items].values
    Ate_items = Ate[kept_items].values
    Mtr, Mte = meta_base(tr["content"].tolist(), te["content"].tolist())
    mnames = ["meta_len", "meta_emoji", "meta_exclaim", "meta_question",
              "meta_repeatpunct", "meta_evasion"]
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(n_items)])
    order_i = np.argsort(-pbr)[:N_TOP_INTERACT]
    top_names = [kept_items[i] for i in order_i]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])
    meta_te = np.hstack([Mte, Mte[:, 0:1] * Ate_items[:, order_i]])
    meta_all_names = mnames + [f"lenx_{f}" for f in top_names]

    macro_raw_tr = construct_raw(tr, freeze, Atr, Xtr_raw)
    macro_raw_te = construct_raw(te, freeze, Ate, Xte_raw)
    mcols = [c for c in macro_raw_tr.columns if c != "id"]

    c_tr, c_te, c_names = construct_rank(Rtr_items, Rte_items, kept_items, freeze)
    Ctr = np.column_stack([c_tr[k] for k in c_names])
    Cte = np.column_stack([c_te[k] for k in c_names])
    Pf_tr, Pf_te, p_names = profile_features(Rtr_items, Rte_items, kept_items, freeze)
    Vr_tr, Vr_te, vr_names = vv_rank(Rraw_tr, Rraw_te, kept_items, freeze)
    if "D2" in freeze["constructs_alive"]:
        d2raw_tr = d2_score(Atr, ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"])
        d2raw_te = d2_score(Ate, ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"])
        mu, sd = d2raw_tr.mean(), d2raw_tr.std()
        sd = sd if sd > 1e-9 else 1.0
        d2_tr = (50.0 + 10.0 * (d2raw_tr - mu) / sd)[:, None]
        d2_te = (50.0 + 10.0 * (d2raw_te - mu) / sd)[:, None]
        d2_name = ["D2_mean"]
    else:
        d2_tr = np.zeros((len(ytr), 0))
        d2_te = np.zeros((len(yte), 0))
        d2_name = []

    base_rank_tr = np.hstack([Rall_tr, meta_tr])
    base_rank_te = np.hstack([Rall_te, meta_te])
    base_rank_names = prob_names + meta_all_names

    q1_tr = np.hstack([macro_raw_tr[mcols].values, Rall_tr, meta_tr])
    q1_te = np.hstack([macro_raw_te[mcols].values, Rall_te, meta_te])
    q2_tr = np.hstack([base_rank_tr, Ctr, d2_tr, Vr_tr])
    q2_te = np.hstack([base_rank_te, Cte, d2_te, Vr_te])
    q3_tr = np.hstack([q2_tr, Pf_tr])
    q3_te = np.hstack([q2_te, Pf_te])

    cfg = {
        "Q1_M4锚点": (q1_tr, q1_te, mcols + prob_names + meta_all_names),
        "Q2_秩构念": (q2_tr, q2_te, base_rank_names + c_names + d2_name + vr_names),
        "Q3_加模式": (q3_tr, q3_te, base_rank_names + c_names + d2_name + vr_names + p_names),
    }
    lines = ["=" * 72, "v8.3 读出（常模后聚合 + 构念内模式，三配置 CV 竞争，τ@宏F1@0.005）",
             "=" * 72,
             f"构念分来源: Q1=原始值旧T逻辑 / Q2+=秩常模加权(train冻结T) + D2(Guttman) "
             f"+ V1r/V2r",
             f"profile 特征 {len(p_names)} 维（rmax/rstd/rtop2 × 构念）",
             "维度: " + ", ".join(f"{tag}={X.shape[1]}" for tag, (X, _, _) in cfg.items()), ""]

    rows = []
    for tag, (X, _, _) in tqdm(cfg.items(), desc="配置竞争", dynamic_ncols=True):
        for seed in CV_SEEDS:
            a, t, oof = cv_select(X, ytr, seed, tag)
            rows.append({"config": tag, "seed": seed, "alpha": a,
                         "oof_macro_f1": f1_score(ytr, (oof >= t).astype(int), average="macro"),
                         "oof_auc": roc_auc_score(ytr, oof), "oof_tau": t})
    comp = pd.DataFrame(rows)
    for r in comp.itertuples():
        lines.append(f"  {r.config:<10} seed{r.seed}  alpha={r.alpha:<8} "
                     f"OOF宏F1={r.oof_macro_f1:.4f}  OOF AUC={r.oof_auc:.4f}  τ*={r.oof_tau:.2f}")
    mean_f1 = comp.groupby("config")["oof_macro_f1"].mean()
    mean_auc = comp.groupby("config")["oof_auc"].mean()
    best = mean_f1.idxmax()
    gain = mean_f1[best] - mean_f1["Q1_M4锚点"]
    gate_pass = best != "Q1_M4锚点" and gain >= GATE
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best}（相对 Q1 增益 {gain:+.4f}）",
              f"预登记门槛（≥{GATE}）: {'通过' if gate_pass else '未通过——常模后聚合无增量'}", ""]

    test_tag = best if gate_pass else "Q1_M4锚点"
    if not gate_pass:
        lines.append("按预登记协议，门槛未通过时候选配置不碰 test；test 仅复评锚点 Q1_M4锚点。")
    Xw_tr, Xw_te, w_names = cfg[test_tag]
    per_seed, prob, pred_m = [], None, None
    for seed in tqdm(FINAL_SEEDS, desc="test 评估", dynamic_ncols=True):
        a, tau_m, oof = cv_select(Xw_tr, ytr, seed, best)
        tau_t = max(TAU_GRID, key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
        model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                            C=1.0 / a, max_iter=2000)).fit(Xw_tr, ytr)
        prob = model.predict_proba(Xw_te)[:, 1]
        pred_m = (prob >= tau_m).astype(int)
        per_seed.append({"seed": seed, "alpha": a, "tau_macro": tau_m, "tau_toxic": tau_t,
                         "macro_f1": f1_score(yte, pred_m, average="macro"),
                         "toxic_f1": f1_score(yte, pred_m),
                         "benign_f1": f1_score(yte, pred_m, pos_label=0),
                         "auc": roc_auc_score(yte, prob),
                         "toxic_f1_t": f1_score(yte, (prob >= tau_t).astype(int)),
                         "macro_f1_t": f1_score(yte, (prob >= tau_t).astype(int),
                                                average="macro")})
    per = pd.DataFrame(per_seed)
    lines += [f"test 评估（{test_tag}，5 种子）:",
              f"  {'seed':<6}{'alpha':<10}{'τ*宏':<7}{'宏F1':<9}{'有毒F1':<9}{'无毒F1':<9}{'AUC':<8}{'τ*毒':<7}{'有毒F1@τ毒':<10}",
              "-" * 72]
    for r in per_seed:
        lines.append(f"  {r['seed']:<6}{r['alpha']:<10}{r['tau_macro']:<7.2f}"
                     f"{r['macro_f1']:<9.4f}{r['toxic_f1']:<9.4f}{r['benign_f1']:<9.4f}"
                     f"{r['auc']:<8.4f}{r['tau_toxic']:<7.2f}{r['toxic_f1_t']:<10.4f}")
    lines += ["-" * 72,
              f"  宏F1   mean±std: {per['macro_f1'].mean():.4f} ± {per['macro_f1'].std():.4f}",
              f"  有毒F1 mean±std: {per['toxic_f1'].mean():.4f} ± {per['toxic_f1'].std():.4f}",
              f"  有毒F1@τ毒 mean±std: {per['toxic_f1_t'].mean():.4f} ± {per['toxic_f1_t'].std():.4f}",
              f"  AUC    mean±std: {per['auc'].mean():.4f} ± {per['auc'].std():.4f}",
              "",
              "历史: v8.1 M4 宏 0.8141 / 有毒 0.8249 (@τ毒 0.8310) / AUC 0.8948"]

    # ---- OOF 稳定误差复检（胜者配置，train 内）----
    oofs = np.column_stack([cv_select(Xw_tr, ytr, s, "误差复检")[2] for s in CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(TAU_GRID, key=lambda t: f1_score(ytr, (oof_prob >= t).astype(int),
                                                 average="macro"))
    err = (oof_prob >= tau_e).astype(int) != ytr
    fn = err & (ytr == 1)
    fp = err & (ytr == 0)
    print("加载 rings JSON 取 expression ...")
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    ids = tr["id"].values
    expr = np.array([expr_map.get(i, -1) for i in ids])
    lens = np.array([len(str(t).replace(" ", "")) for t in tr["content"].tolist()])
    lines += ["", "OOF 稳定误差复检（胜者配置，τ@宏F1）:",
              f"  FN {int(fn.sum())}/{int((ytr == 1).sum())}   FP {int(fp.sum())}/{int((ytr == 0).sum())}",
              f"  {'类型':<4}{'expression组':<10}{'长度桶':<8}{'样本数':<8}{'错误数':<8}{'错误率':<8}"]
    for label, mask in (("FN", fn), ("FP", fp)):
        for g in sorted(set(expr.tolist())):
            for lo, hi in LEN_BINS:
                sel = (expr == g) & (lens >= lo) & (lens < hi)
                if sel.sum() == 0:
                    continue
                lines.append(f"  {label:<4}{g:<12}[{lo},{hi}){'':<2}"
                             f"{int(sel.sum()):<10}{int((mask & sel).sum()):<10}"
                             f"{(mask & sel).sum() / sel.sum():.4f}")

    model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / per.iloc[0]["alpha"],
                                        max_iter=2000)).fit(Xw_tr, ytr)
    coef = pd.Series(model.model.coef_[0], index=w_names)
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
