# -*- coding: utf-8 -*-
"""v8.2 读出：扩展交互池（v8.1 M4 基座），四配置 CV 竞争。

背景：v8.1 证实秩变换（+0.0031 CV）与元特征交互（lenx_E2 -0.247 等，11/11 非零）
均有效，M4 test 宏 0.8141 / 有毒@τ毒 0.8310 / AUC 0.8948。本版扩展交互池：
  len×top-20、question×top-10、evasion×top-10、头部特征两两交互（top-10）
τ 网格细化到 0.005 步长。门槛：胜者 ≥ P1 锚点 + 0.002 才碰 test。

配置：
  P1  M4 复刻锚点（秩变换 + v8.1 元特征与交互）
  P2  P1 + 扩展元交互（len×20 / question×10 / evasion×10）
  P3  P1 + 特征两两交互（top-10, 45 对）
  P4  P1 + 全部扩展
产物：results/v8/readout_v82/{metrics.txt, cv_competition.csv, test_predictions.csv}
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
TOP_LEN, TOP_Q, TOP_EV, TOP_PAIR = 20, 10, 10, 10
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
EVASION_LEX = ("朲", "亽", "㐺", "亻", "怹")


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
        for tr, va in tqdm(cv.split(X, y), total=5, leave=False, dynamic_ncols=True,
                           desc=f"      s{seed} a={a}"):
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
    for tr, va in tqdm(cv.split(X, y), total=5, leave=False, dynamic_ncols=True,
                       desc=f"      s{seed} a={a} OOF"):
        m = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / best_a, max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    tau = max(TAU_GRID, key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return best_a, float(tau), oof


def construct_features(df, freeze, A, Xr):
    wmap = freeze.get("item_weights", {})
    means = {}
    for cid, kept in freeze["items_per_construct"].items():
        if cid not in freeze["constructs_alive"]:
            continue
        kept = [c for c in kept if c in A.columns]
        if cid == "D2":
            tiers = ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"]
            if not all(t in A.columns for t in tiers):
                continue
            hits = A[tiers].values >= 0.5
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
    ap = argparse.ArgumentParser(description="v8.2 读出：扩展交互池")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v8/readout_v82")
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
    mtr = construct_features(tr, freeze, Atr, Xtr_raw)
    mte = construct_features(te, freeze, Ate, Xte_raw)
    mcols = [c for c in mtr.columns if c != "id"]

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

    base_tr = np.hstack([mtr[mcols].values, Atr[kept_items].values, v3_tr])
    base_te = np.hstack([mte[mcols].values, Ate[kept_items].values, v3_te])
    prob_names = kept_items + v3_names
    all_names = mcols + kept_items + v3_names
    prob_idx = [i for i, nm in enumerate(all_names) if nm in prob_names]

    # 秩变换（v8.1 M4 基座）
    rank_tr, rank_te = base_tr.copy(), base_te.copy()
    qs = np.linspace(0.0, 1.0, N_QUANT)
    for j in prob_idx:
        edges = np.quantile(base_tr[:, j], qs)
        rank_tr[:, j] = np.interp(base_tr[:, j], edges, qs)
        rank_te[:, j] = np.interp(base_te[:, j], edges, qs)

    # 元特征
    Mtr, Mte = meta_base(tr["content"].tolist(), te["content"].tolist())
    mnames = ["meta_len", "meta_emoji", "meta_exclaim", "meta_question",
              "meta_repeatpunct", "meta_evasion"]
    pbr = np.array([abs(np.corrcoef(rank_tr[:, all_names.index(nm)], ytr)[0, 1])
                    for nm in prob_names])
    order = [prob_names[i] for i in np.argsort(-pbr)]

    def inter(base_mat, meta_col, feats, tag):
        cols, names = [], []
        for f in feats:
            cols.append(meta_col[:, None] * rank_tr[:, [all_names.index(f)]])
            names.append(f"{tag}x_{f}")
        return cols, names

    len_cols, len_names = inter(None, Mtr[:, 0], order[:TOP_LEN], "len")
    q_cols, q_names = inter(None, Mtr[:, 3], order[:TOP_Q], "que")
    ev_cols, ev_names = inter(None, Mtr[:, 5], order[:TOP_EV], "eva")
    pair_idx = [all_names.index(f) for f in order[:TOP_PAIR]]
    pair_cols, pair_names = [], []
    for i in range(TOP_PAIR):
        for j in range(i + 1, TOP_PAIR):
            pair_cols.append(rank_tr[:, [pair_idx[i]]] * rank_tr[:, [pair_idx[j]]])
            pair_names.append(f"{order[i]}*{order[j]}")

    len_cols_te, q_cols_te, ev_cols_te = [], [], []
    for f in order[:TOP_LEN]:
        len_cols_te.append(Mte[:, 0:1] * rank_te[:, [all_names.index(f)]])
    for f in order[:TOP_Q]:
        q_cols_te.append(Mte[:, 3:4] * rank_te[:, [all_names.index(f)]])
    for f in order[:TOP_EV]:
        ev_cols_te.append(Mte[:, 5:6] * rank_te[:, [all_names.index(f)]])
    pair_cols_te = []
    for i in range(TOP_PAIR):
        for j in range(i + 1, TOP_PAIR):
            pair_cols_te.append(rank_te[:, [pair_idx[i]]] * rank_te[:, [pair_idx[j]]])

    v81_extra_tr = np.hstack([Mtr, Mtr[:, 0:1] * rank_tr[:, [
        all_names.index(f) for f in order[:5]]]])
    v81_extra_te = np.hstack([Mte, Mte[:, 0:1] * rank_te[:, [
        all_names.index(f) for f in order[:5]]]])
    v81_names = mnames + [f"lenx_{f}" for f in order[:5]]

    def stack(cols):
        return np.hstack(cols) if cols else np.zeros((len(ytr), 0))

    p2_tr = np.hstack([rank_tr, v81_extra_tr, stack(len_cols + q_cols + ev_cols)])
    p2_te = np.hstack([rank_te, v81_extra_te, stack(len_cols_te + q_cols_te + ev_cols_te)])
    p3_tr = np.hstack([rank_tr, v81_extra_tr, stack(pair_cols)])
    p3_te = np.hstack([rank_te, v81_extra_te, stack(pair_cols_te)])
    p4_tr = np.hstack([rank_tr, v81_extra_tr, stack(len_cols + q_cols + ev_cols + pair_cols)])
    p4_te = np.hstack([rank_te, v81_extra_te,
                       stack(len_cols_te + q_cols_te + ev_cols_te + pair_cols_te)])
    p1_tr = np.hstack([rank_tr, v81_extra_tr])
    p1_te = np.hstack([rank_te, v81_extra_te])

    cfg = {
        "P1_M4锚点": (p1_tr, p1_te, all_names + v81_names),
        "P2_扩元交互": (p2_tr, p2_te, all_names + v81_names + len_names + q_names + ev_names),
        "P3_两两交互": (p3_tr, p3_te, all_names + v81_names + pair_names),
        "P4_全部": (p4_tr, p4_te, all_names + v81_names + len_names + q_names + ev_names + pair_names),
    }
    lines = ["=" * 72, "v8.2 读出（扩展交互池，四配置 CV 竞争，τ@宏F1@0.005）", "=" * 72,
             f"交互新增: len×{TOP_LEN} + que×{TOP_Q} + eva×{TOP_EV} + 两两 {TOP_PAIR}C2="
             f"{TOP_PAIR * (TOP_PAIR - 1) // 2}",
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
    gain = mean_f1[best] - mean_f1["P1_M4锚点"]
    gate_pass = best != "P1_M4锚点" and gain >= 0.002
    lines += ["", "CV 均值: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_f1.items()),
              "CV AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_auc.items()),
              f"最佳配置: {best}（相对 P1 增益 {gain:+.4f}）",
              f"预登记门槛（≥0.002）: {'通过' if gate_pass else '未通过——交互池到顶，不碰 test'}", ""]

    if not gate_pass:
        report = "\n".join(lines)
        print(report)
        (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        comp.to_csv(out / "cv_competition.csv", index=False, encoding="utf-8-sig")
        return

    Xw_tr, Xw_te, w_names = cfg[best]
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
    lines += [f"test 评估（{best}，5 种子）:",
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

    model = Pipeline(LogisticRegression(penalty="l1", solver="liblinear",
                                        C=1.0 / per.iloc[0]["alpha"],
                                        max_iter=2000)).fit(Xw_tr, ytr)
    coef = pd.Series(model.model.coef_[0], index=w_names)
    nz = coef[coef.abs() > 1e-6]
    extra = nz[nz.index.str.contains(("lenx_|que|eva|\\*|meta_"))]
    lines += ["", f"非零权重 {len(nz)}/{len(w_names)}（种子 0）；新增交互/元特征非零 "
              f"{len(extra)}/{len(w_names) - len(all_names)}："]
    for name, w in extra.reindex(extra.abs().sort_values(ascending=False).index).head(25).items():
        lines.append(f"    {name:<28} {w:+.3f}")

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
