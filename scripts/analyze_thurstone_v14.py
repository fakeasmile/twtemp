# -*- coding: utf-8 -*-
"""v14 主线分析：Bradley-Terry 尺度拟合 + G1'/G2' 预登记双门（CPU）。

输入：results/v14/thurstone/pairs_flat_{train,test}.csv（96,088 对软胜率，
      首 token 覆盖率 1.0 已由 meta.json 确认）
      results/v14/reverify/{anchor_predictions_train.csv, fp_tp_ids.csv,
      reverify.txt 的 fresh B1=0.8929}
      原始矩阵 + freeze（fresh 重算 M4 264 维，门槛对照臂）

【BT 拟合】Bradley-Terry：P(i≻a)=σ(θ_i−θ_a)，软标签极大似然
  （scipy L-BFGS，无监督——标签零接触）。节点 = train 9600 + test 2411
  + 锚点 24；边 = 每 (text,anchor) 软胜率 p_text（已哈希换位校正）。
  拟合质量诊断：高侧锚点 θ 均值应 > 低侧锚点（方向自检）；边级残差。

【特征】BT_FEATS = bt_w（尺度值）, bt_win_high, bt_win_low, bt_gap
  (=win_high−win_low，比较模式核心签名), bt_std（8 边一致性）,
  bt_soft_margin（logit(win_high)−logit(win_low)）

【预登记双门】（引用 reverify fresh 值，历史值退役）：
  G1' 配对判据：同一 CV 划分下，(M4+BT特征) 与 M4 在 E0无毒 vs 有毒
      配对上的 OOF AUC 差值（3 种子逐种子配对）均值 ≥ +0.010
      （fresh B1 = 0.8929，等价于联合 AUC ≥ 0.9029）
  G2' 正交门：6 个 BT 特征与 M4 264 维 max|pearson| < 0.7
  附加诊断（无门）：BT 特征单独 @E0 配对、@FP vs TP（fresh 定界 770/4170）
  双门过 → readout_v14（CV ≥ +0.002 才碰 test）

产物：results/v14/thurstone/{analysis.txt, bt_features_train.csv,
      bt_features_test.csv, gate.json}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83
from readout_v12 import assemble_m4

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


BT_FEATS = ["bt_w", "bt_win_high", "bt_win_low", "bt_gap", "bt_std",
            "bt_soft_margin"]


def fit_bt(edges_text, edges_anchor, edges_p, n_nodes, damp=1e-4):
    """软标签 BT 极大似然。边 (t,a,p)：贡献 p·logσ(z)+(1−p)·logσ(−z)，
    z=θ_t−θ_a。返回 (θ, 收敛信息)。"""
    n_edges = len(edges_p)
    p = np.clip(edges_p, 1e-6, 1.0 - 1e-6)

    def nll_grad(theta):
        z = theta[edges_text] - theta[edges_anchor]
        nll = float(np.sum(p * np.logaddexp(0, -z)
                           + (1 - p) * np.logaddexp(0, z)))
        g_edge = expit(z) - p                      # d(nll)/dz = σ(z) − p
        grad = np.zeros(n_nodes)
        np.add.at(grad, edges_text, g_edge)
        np.add.at(grad, edges_anchor, -g_edge)
        return nll + damp * float(np.sum(theta ** 2)), grad + 2 * damp * theta

    res = minimize(nll_grad, np.zeros(n_nodes), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "maxcor": 20})
    return res.x, {"success": bool(res.success), "nit": int(res.nit),
                   "nll": float(res.fun)}


def edge_auc(Xsub, ysub, idx, seeds=(42, 43, 44), C=0.01):
    """子集内 5 折 CV AUC（与 verify_noise/reverify 方法学一致）。"""
    vals = []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(idx))
        for trn, va in cv.split(Xsub[idx], ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=C, max_iter=2000))])
            m.fit(Xsub[idx][trn], ysub[trn])
            oof[va] = m.predict_proba(Xsub[idx][va])[:, 1]
        a = roc_auc_score(ysub, oof)
        vals.append(max(a, 1 - a))
    return float(np.mean(vals)), float(np.std(vals))


def paired_delta(X_a, X_b, ysub, idx, seeds=(42, 43, 44), C=0.01):
    """同 CV 划分下两特征矩阵的逐种子配对 AUC 差（X_b − X_a）。"""
    deltas = []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof_a = np.zeros(len(idx))
        oof_b = np.zeros(len(idx))
        for trn, va in cv.split(X_a[idx], ysub):
            ma = Pipeline([("sc", StandardScaler()),
                           ("m", LogisticRegression(C=C, max_iter=2000))])
            mb = Pipeline([("sc", StandardScaler()),
                           ("m", LogisticRegression(C=C, max_iter=2000))])
            ma.fit(X_a[idx][trn], ysub[trn])
            mb.fit(X_b[idx][trn], ysub[trn])
            oof_a[va] = ma.predict_proba(X_a[idx][va])[:, 1]
            oof_b[va] = mb.predict_proba(X_b[idx][va])[:, 1]
        aa = max(roc_auc_score(ysub, oof_a), 1 - roc_auc_score(ysub, oof_a))
        ab = max(roc_auc_score(ysub, oof_b), 1 - roc_auc_score(ysub, oof_b))
        deltas.append(ab - aa)
    return float(np.mean(deltas)), float(np.std(deltas)), deltas


def main():
    ap = argparse.ArgumentParser(description="v14 BT 拟合 + G1'/G2' 双门")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json")
    ap.add_argument("--pairs_dir", default="results/v14/thurstone")
    ap.add_argument("--reverify_dir", default="results/v14/reverify")
    ap.add_argument("--output_dir", default="results/v14/thurstone")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.pairs_dir) / "meta.json").read_text(encoding="utf-8"))
    # fresh B1 不从文本解析——直接在门槛段用同方法学重算（edge_auc on Xtr）
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values
    ids_tr, ids_te = tr["id"].values, te["id"].values

    # ---- fail-fast ----
    assert {"id", "oof_s42b"} <= set(pd.read_csv(
        Path(args.reverify_dir) / "anchor_predictions_train.csv").columns)
    ptr = pd.read_csv(Path(args.pairs_dir) / "pairs_flat_train.csv")
    pte = pd.read_csv(Path(args.pairs_dir) / "pairs_flat_test.csv")
    assert {"text_id", "anchor_id", "side", "p_text"} <= set(ptr.columns)
    anchors = set(meta["high_anchors"]) | set(meta["low_anchors"])
    assert anchors <= set(ids_tr), "锚点不在 train 内"
    assert not ptr["p_text"].isna().any() and not pte["p_text"].isna().any()

    # ---- 节点编号：train 0..n1-1, test n1..n1+n2-1, anchors 其后 ----
    n1, n2, n_a = len(ids_tr), len(ids_te), len(anchors)
    node_of = {tid: i for i, tid in enumerate(ids_tr)}
    node_of.update({tid: n1 + i for i, tid in enumerate(ids_te)})
    anchor_list = list(meta["high_anchors"]) + list(meta["low_anchors"])
    node_of.update({aid: n1 + n2 + i for i, aid in enumerate(anchor_list)})
    N = n1 + n2 + n_a
    side_is_high = {aid: (aid in set(meta["high_anchors"]))
                    for aid in anchor_list}

    def collect(pdf):
        t = pdf["text_id"].map(node_of).values
        a = pdf["anchor_id"].map(node_of).values
        p = pdf["p_text"].values.astype(float)
        return t, a, p

    stamp(f"[BT] 边 {len(ptr) + len(pte)}，节点 {N}（train {n1} + test {n2} + "
          f"锚点 {n_a}），L-BFGS 拟合中...")
    t_ptr, a_ptr, p_ptr = collect(ptr)
    t_te, a_te, p_te = collect(pte)
    t_all = np.concatenate([t_ptr, t_te])
    a_all = np.concatenate([a_ptr, a_te])
    p_all = np.concatenate([p_ptr, p_te])
    theta, info = fit_bt(t_all, a_all, p_all, N)
    stamp(f"[BT] 收敛 success={info['success']} nit={info['nit']} "
          f"NLL/边={info['nll'] / len(p_all):.4f}")

    # ---- 方向自检：高侧锚点 θ 应 > 低侧 ----
    th_high = theta[n1 + n2:n1 + n2 + len(meta["high_anchors"])].mean()
    th_low = theta[n1 + n2 + len(meta["high_anchors"]):].mean()

    # ---- 逐文本特征 ----
    def feats(pdf):
        g = pdf.groupby("text_id")
        rows = []
        for tid, sub in g:
            hi = sub[sub["side"] == "high"]["p_text"]
            lo = sub[sub["side"] == "low"]["p_text"]
            wh = float(hi.mean()) if len(hi) else np.nan
            wl = float(lo.mean()) if len(lo) else np.nan
            eps = 1e-6
            rows.append({
                "id": tid,
                "bt_w": float(theta[node_of[tid]]),
                "bt_win_high": wh,
                "bt_win_low": wl,
                "bt_gap": wh - wl,
                "bt_std": float(sub["p_text"].std()) if len(sub) > 1 else 0.0,
                "bt_soft_margin": float(np.log((wh + eps) / (1 - wh + eps))
                                        - np.log((wl + eps) / (1 - wl + eps))),
                "n_edges": int(len(sub)),
            })
        df = pd.DataFrame(rows).set_index("id")
        return df

    ftr = feats(ptr).loc[ids_tr].reset_index()
    fte = feats(pte).loc[ids_te].reset_index()
    assert (ftr["id"].values == ids_tr).all(), "train 特征 id 顺序与矩阵不一致"
    assert (fte["id"].values == ids_te).all(), "test 特征 id 顺序与矩阵不一致"
    Ftr = ftr[BT_FEATS].values
    Fte = fte[BT_FEATS].values
    assert np.isfinite(Ftr).all() and np.isfinite(Fte).all()
    ftr.to_csv(out / "bt_features_train.csv", index=False, encoding="utf-8-sig")
    fte.to_csv(out / "bt_features_test.csv", index=False, encoding="utf-8-sig")

    # ---- M4 fresh 装配（门槛对照臂）----
    stamp("[装配] M4 264 维 fresh 重算...")
    Xtr, Xte = assemble_m4(tr, te, v3tr, v3te, freeze, ytr)

    # ---- G2' 正交门 ----
    corr = np.abs(np.corrcoef(np.hstack([Ftr, Xtr]).T))[:len(BT_FEATS),
                                                        len(BT_FEATS):]
    g2_max = float(np.nanmax(corr))
    g2_pass = g2_max < 0.7

    # ---- G1' 主门（配对 delta，E0无毒 vs 有毒）----
    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids_tr])
    e0 = np.where((ytr == 0) & (expr == 0))[0]
    tox = np.where(ytr == 1)[0]
    idx_e0 = np.concatenate([e0, tox])
    ysub = np.where(np.arange(len(idx_e0)) < len(e0), 0, 1)
    stamp(f"[门槛] G1' E0无毒 n={len(e0)} vs 有毒 n={len(tox)} 配对 CV...")
    d_mean, d_std, deltas = paired_delta(Xtr, np.hstack([Xtr, Ftr]), ysub, idx_e0)
    b1_mean, b1_std = edge_auc(Xtr, ysub, idx_e0)
    g1_pass = d_mean >= 0.010

    # ---- 附加诊断：BT 单独 @E0 与 @FP vs TP ----
    # TP/FP 定界必须与 reverify 一致：seed42 OOF 预测为正（pred_s42b==1），
    # 其中 y=0 为 FP（770）、y=1 为 TP（4170）——不是全部有毒样本
    apred = pd.read_csv(Path(args.reverify_dir) / "anchor_predictions_train.csv")
    assert {"id", "pred_s42b"} <= set(apred.columns)
    pred42 = apred.set_index("id").loc[ids_tr, "pred_s42b"].values
    fp_idx = np.where((pred42 == 1) & (ytr == 0))[0]
    tp_idx = np.where((pred42 == 1) & (ytr == 1))[0]
    assert len(fp_idx) == 770 and len(tp_idx) == 4170, \
        f"FP/TP 定界与 reverify 不符: FP={len(fp_idx)}, TP={len(tp_idx)}"
    bt_e0_mean, bt_e0_std = edge_auc(Ftr, ysub, idx_e0)
    idx_b2 = np.concatenate([fp_idx, tp_idx])
    ysub2 = np.where(np.arange(len(idx_b2)) < len(fp_idx), 0, 1)
    bt_b2_mean, bt_b2_std = edge_auc(Ftr, ysub2, idx_b2)
    m4_b2_mean, m4_b2_std = edge_auc(Xtr, ysub2, idx_b2)

    L = ["=" * 84, "v14 Thurstone 成对比较：BT 拟合 + G1'/G2' 预登记双门",
         "=" * 84,
         f"边 {len(p_all)}（全部有效），节点 {N}；BT 收敛 "
         f"success={info['success']} nit={info['nit']} NLL/边={info['nll']/len(p_all):.4f}",
         f"方向自检：高侧锚点 θ̄={th_high:.3f} > 低侧锚点 θ̄={th_low:.3f} : "
         f"{'正确' if th_high > th_low else '反向！测量可能失效'}", "",
         "BT 特征组间均值（train）",
         f"  {'组':<14}" + " ".join(f"{c:>12}" for c in BT_FEATS),
         "  " + "-" * (14 + 13 * len(BT_FEATS))]
    for name, m in [("有毒(y=1)", ytr == 1), ("无毒(y=0)", ytr == 0),
                    ("E0无毒", (ytr == 0) & (expr == 0)),
                    ("FP(seed42)", np.isin(np.arange(len(ytr)), fp_idx)),
                    ("TP(seed42)", np.isin(np.arange(len(ytr)), tp_idx))]:
        L.append(f"  {name:<14}" + " ".join(f"{Ftr[m, j].mean():>12.4f}"
                                            for j in range(len(BT_FEATS))))
    L += ["",
          "【G1' 主门】M4+BT vs M4 @E0无毒vs有毒 配对 AUC delta（3 种子）",
          f"  逐种子 delta: {[round(d, 4) for d in deltas]}",
          f"  mean delta = {d_mean:+.4f} ± {d_std:.4f}   "
          f"门槛 ≥ +0.010 : {'PASS' if g1_pass else 'FAIL'}",
          f"  （M4 基线 fresh = {b1_mean:.4f}±{b1_std:.4f}；等价联合 ≥ "
          f"{b1_mean + 0.010:.4f}）", "",
          "【G2' 正交门】BT 6 特征 vs M4 264 维 max|corr|",
          f"  max|corr| = {g2_max:.4f}   门槛 < 0.7 : {'PASS' if g2_pass else 'FAIL'}",
          "", "【附加诊断（无门）】",
          f"  BT 6 特征单独 @E0无毒vs有毒: AUC = {bt_e0_mean:.4f} ± {bt_e0_std:.4f}",
          f"  BT 单独 @FP({len(fp_idx)}) vs TP({len(tp_idx)}): AUC = {bt_b2_mean:.4f} ± {bt_b2_std:.4f}"
          f"   （M4 同配对 fresh = {m4_b2_mean:.4f} ± {m4_b2_std:.4f}）", "",
          "=" * 84,
          ("双门通过 → 下一步 readout_v14（CV ≥ +0.002 才碰 test）"
           if (g1_pass and g2_pass) else
           "未双通过 → 按预登记协议，本轮判死；若 G1' 过 G2' 败 → 特征冗余；"
           "若 G1' 败 → 比较模式无增量信息")]

    report = "\n".join(L)
    print(report)
    (out / "analysis.txt").write_text(report + "\n", encoding="utf-8")
    json.dump({"g1_pass": bool(g1_pass), "g1_delta_mean": d_mean,
                "g1_delta_std": d_std, "b1_fresh": b1_mean,
                "g2_pass": bool(g2_pass), "g2_max_corr": g2_max,
                "bt_e0_auc": bt_e0_mean, "bt_b2_auc": bt_b2_mean,
                "m4_b2_fresh": m4_b2_mean,
                "bt_info": info, "theta_high_mean": float(th_high),
                "theta_low_mean": float(th_low)},
               open(out / "gate.json", "w", encoding="utf-8"), ensure_ascii=False,
               indent=1)
    stamp("[完成] analysis.txt / gate.json / bt_features_*.csv 已写出")


if __name__ == "__main__":
    main()
