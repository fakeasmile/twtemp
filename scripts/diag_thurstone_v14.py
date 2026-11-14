# -*- coding: utf-8 -*-
"""v14 诊断：run_thurstone_v14 测量产物的深度分析（CPU，用户执行）。

目的：实验暂停后，把 96,088 对比较判断数据本身吃透——测量质量、偏置
结构、BT 尺度的判别剖面、与 M4 的条件冗余定位。全部离线重算，零新测量。

【1】甲/乙覆盖度核算（用户点名要求）：首 token 恰为甲/乙的比例（重算，
     不依赖 meta）、m_jia+m_yi 质量和分布（逸散率）、p_text 分位剖面
【2】甲位偏置定量：BT 模型预测 σ(θ_t−θ_a) 与原始 P(甲) 之差，按换位
     分组——甲位加性偏置 β 的估计（θ 固定为 bt_w，仅拟合 24 锚点 θ）
【3】边级残差：|p_text − σ(θ_t−θ_a)| 分布；残差>0.25 的边占比与
     残差 top 边的双侧特征（不一致性 = 比较判断的非传递信号）
【4】θ 判别剖面：bt_w 分 20 分位箱 × P(y=1)/P(expr0)/n——比较尺度
     携带标签信息的直接曲线
【5】条件分层 AUC（核心判读表）：M4 OOF 概率(oof_s42b) 分 8 箱，箱内
     bt_w 对 y 的 AUC。全箱≈0.5 → 完全冗余终局确认；某箱>0.6 → 存在
     条件增量信息（指定哪一段概率带）
【6】分歧样本导出：M4 概率与 bt_w 标准分之差 top±200 → csv（id/content/
     oof/bt_w/y/FP标记），供人工复核两种测量分歧落在什么文本上
【7】锚点效度：24 锚点 θ vs 参考分(oof_s42b) 排序相关、锚点 θ vs 真实
     标签 AUC——锚点设计本身的效度

产物：results/v14/thurstone/{diag_report.txt, diag_disagreement.csv,
      theta_profile.csv, theta_by_m4bin.csv}
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

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="v14 Thurstone 测量产物深度诊断")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--pairs_dir", default="results/v14/thurstone")
    ap.add_argument("--reverify_dir", default="results/v14/reverify")
    ap.add_argument("--rings_train_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json")
    ap.add_argument("--output_dir", default="results/v14/thurstone")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 加载 + fail-fast ----
    stamp("[加载] 读取测量产物与矩阵...")
    meta = json.loads((Path(args.pairs_dir) / "meta.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(Path(args.pairs_dir) / "pairs_flat_train.csv")
    feat = pd.read_csv(Path(args.pairs_dir) / "bt_features_train.csv")
    apred = pd.read_csv(Path(args.reverify_dir) / "anchor_predictions_train.csv")
    tr = pd.read_csv(args.train_matrix)
    for df, cols in ((pairs, ["text_id", "anchor_id", "side", "swapped",
                              "p_text", "m_jia", "m_yi", "tok_id", "tok"]),
                     (feat, ["id", "bt_w"]),
                     (apred, ["id", "oof_s42b", "pred_s42b"]),
                     (tr, ["id", "content", "toxic"])):
        assert set(cols) <= set(df.columns), f"缺列: {set(cols) - set(df.columns)}"
    ids_tr = tr["id"].values
    ytr = tr["toxic"].astype(int).values
    text_of = dict(zip(tr["id"], tr["content"].astype(str)))
    assert (feat["id"].values == ids_tr).all(), "bt_features id 序与矩阵不一致"
    w = feat.set_index("id").loc[ids_tr, "bt_w"].values
    oof = apred.set_index("id").loc[ids_tr, "oof_s42b"].values
    pred42 = apred.set_index("id").loc[ids_tr, "pred_s42b"].values
    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids_tr])
    node_of = {tid: i for i, tid in enumerate(ids_tr)}
    fp_idx = np.where((pred42 == 1) & (ytr == 0))[0]
    tp_idx = np.where((pred42 == 1) & (ytr == 1))[0]
    assert len(fp_idx) > 0 and len(tp_idx) > 0, "seed42 定界退化：FP/TP 为空"
    stamp(f"[定界] FP n={len(fp_idx)}  TP n={len(tp_idx)}（reverify 口径）")
    assert not pairs["p_text"].isna().any()

    # ---- 锚点 θ 拟合（文本 θ 固定为 bt_w，仅拟合 24 锚点）----
    stamp("[BT] 固定文本 θ，拟合 24 锚点 θ...")
    high_ids, low_ids = meta["high_anchors"], meta["low_anchors"]
    anchor_list = high_ids + low_ids
    n_anchor = len(anchor_list)
    a_node = {aid: i for i, aid in enumerate(anchor_list)}
    idx_text = pairs["text_id"].map(node_of).values
    th_text = w[idx_text]
    a_idx = pairs["anchor_id"].map(a_node).values
    p = np.clip(pairs["p_text"].values, 1e-6, 1 - 1e-6)
    side_high = (pairs["side"] == "high").values

    def anchor_nll(theta_a):
        z = th_text - theta_a[a_idx]
        nll = float(np.sum(p * np.logaddexp(0, -z)
                           + (1 - p) * np.logaddexp(0, z)))
        g = np.zeros(n_anchor)
        # d(nll)/dθ_a = −(σ(z)−p)（与 analyze_thurstone_v14.fit_bt 同号）
        np.add.at(g, a_idx, -(expit(z) - p))
        return nll, g

    res = minimize(anchor_nll, np.zeros(n_anchor), jac=True, method="L-BFGS-B")
    theta_a = res.x
    assert res.success, f"锚点 θ 拟合未收敛: {res.message}"
    assert np.abs(theta_a).max() > 1e-6, "锚点 θ 退化为全零（拟合失败）"
    stamp(f"[BT] 锚点 θ 拟合收敛 success={res.success} "
          f"θ范围[{theta_a.min():.2f},{theta_a.max():.2f}]")

    # ---- 【1】覆盖度核算 ----
    lab = meta["label_token_ids"]
    n_first_jia = int((pairs["tok_id"] == lab["甲"]).sum())
    n_first_yi = int((pairs["tok_id"] == lab["乙"]).sum())
    mass = pairs["m_jia"] + pairs["m_yi"]
    L = ["=" * 84, "run_thurstone_v14 测量产物深度诊断", "=" * 84, "",
         "【1】甲/乙覆盖度核算（离线重算）",
         f"  边总数 {len(pairs)}；首token=甲 {n_first_jia} ({n_first_jia/len(pairs):.4f})，"
         f"首token=乙 {n_first_yi} ({n_first_yi/len(pairs):.4f})，"
         f"其他 {len(pairs)-n_first_jia-n_first_yi}",
         f"  甲乙质量和分位: P10={mass.quantile(.1):.4f} P50={mass.quantile(.5):.4f} "
         f"P90={mass.quantile(.9):.4f}；mass<0.9 占比 {(mass<0.9).mean():.4f}（质量逸散）",
         f"  p_text 分位: P10={pairs['p_text'].quantile(.1):.3f} P25={pairs['p_text'].quantile(.25):.3f} "
         f"P50={pairs['p_text'].quantile(.5):.3f} P75={pairs['p_text'].quantile(.75):.3f} "
         f"P90={pairs['p_text'].quantile(.9):.3f}",
         f"  极端判断占比（p<0.1 或 p>0.9）: {((pairs['p_text']<0.1)|(pairs['p_text']>0.9)).mean():.4f}", ""]

    # ---- 【2】甲位偏置 ----
    z_full = th_text - theta_a[a_idx]
    p_jia_raw = pairs["m_jia"] / np.maximum(mass, 1e-12)  # 原始甲选择率
    text_in_jia = ~pairs["swapped"].values  # 未换位=文本在甲位
    pred_jia = expit(z_full)
    # 甲位加性偏置 β：P(甲)≈σ(z + β·[文本在甲位])；分别估两组均值差
    b_j = float(np.mean(p_jia_raw[text_in_jia] - pred_jia[text_in_jia]))
    b_y = float(np.mean(p_jia_raw[~text_in_jia] - (1 - pred_jia[~text_in_jia])))
    L += ["【2】甲位偏置定量（P(甲)−BT预测，正=偏向甲位）",
          f"  文本在甲位的边: 偏置均值 {b_j:+.4f}（n={int(text_in_jia.sum())}）",
          f"  文本在乙位的边: 偏置均值 {b_y:+.4f}（n={int((~text_in_jia).sum())}）",
          f"  甲位净偏置 ≈ {0.5*(b_j-b_y):+.4f}（>0 表示模型偏好回答甲）", ""]

    # ---- 【3】边级残差 ----
    resid = np.abs(pairs["p_text"].values - pred_jia)
    L += ["【3】边级残差 |p_text−σ(θ)|",
          f"  均值 {resid.mean():.4f}；P50={np.quantile(resid,.5):.4f} "
          f"P90={np.quantile(resid,.9):.4f} P99={np.quantile(resid,.99):.4f}",
          f"  残差>0.25 占比 {(resid>0.25).mean():.4f}（比较判断不一致边）",
          f"  残差>0.25 且高侧边占比 {(resid[(resid>0.25)&side_high].sum()/max((resid>0.25).sum(),1)):.4f}"
          f"（高侧边占全部边 50%，若此处显著偏离 50% 说明不一致集中于毒性边界）", ""]

    # ---- 【4】θ 判别剖面 ----
    stamp("[剖面] θ 分位箱 × 阳性率...")
    q20 = np.quantile(w, np.linspace(0, 1, 21))
    prof = []
    for k in range(20):
        m = (w >= q20[k]) & (w < q20[k + 1] if k < 19 else w <= q20[k + 1])
        if m.sum() == 0:
            continue
        prof.append({"bin": k + 1, "w_lo": q20[k], "w_hi": q20[k + 1],
                     "n": int(m.sum()),
                     "p_toxic": float(ytr[m].mean()),
                     "p_expr0": float((expr[m] == 0).mean()),
                     "p_fp_of_pred": float((np.isin(np.arange(len(ytr)), fp_idx)[m]).mean())})
    pdf = pd.DataFrame(prof)
    pdf.to_csv(out / "theta_profile.csv", index=False, encoding="utf-8-sig")
    L += ["【4】θ 判别剖面（bt_w 20 分位箱，train 全体；明细 theta_profile.csv）",
          f"  {'箱':>3} {'w区间':<16} {'n':>6} {'P(y=1)':>8} {'P(expr0)':>9} {'P(FP|箱)':>9}",
          "  " + "-" * 52]
    for _, r in pdf.iterrows():
        L.append(f"  {int(r['bin']):>3} [{r['w_lo']:>6.2f},{r['w_hi']:>6.2f}) "
                 f"{int(r['n']):>6} {r['p_toxic']:>8.3f} {r['p_expr0']:>9.3f} "
                 f"{r['p_fp_of_pred']:>9.3f}")
    L += ["", f"  θ 单特征 5折CV AUC（3种子，StandardScaler+LR C=0.01）: "
          f"{cv_auc_single(w, ytr):.4f}", ""]

    # ---- 【5】条件分层 AUC（核心判读）----
    stamp("[分层] M4 概率箱内 θ 的条件 AUC...")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    rows5 = []
    for k in range(len(bins) - 1):
        m = (oof >= bins[k]) & (oof < bins[k + 1])
        n = int(m.sum())
        if n < 30 or len(set(ytr[m])) < 2:
            rows5.append({"bin_lo": bins[k], "bin_hi": bins[k + 1], "n": n,
                          "p_toxic": float(ytr[m].mean()) if n else np.nan,
                          "auc_theta_in_bin": np.nan})
            continue
        a = cv_auc_single(w[m], ytr[m])
        rows5.append({"bin_lo": bins[k], "bin_hi": bins[k + 1], "n": n,
                      "p_toxic": float(ytr[m].mean()),
                      "auc_theta_in_bin": float(a)})
    df5 = pd.DataFrame(rows5)
    df5.to_csv(out / "theta_by_m4bin.csv", index=False, encoding="utf-8-sig")
    L += ["【5】条件分层 AUC（核心判读表）：M4 OOF 概率箱内，bt_w 对 y 的 AUC",
          "  （全箱≈0.5 → 比较测量在 M4 条件下零增量，冗余终局确认）",
          f"  {'M4概率箱':<16} {'n':>6} {'P(y=1)':>8} {'θ箱内AUC':>9}",
          "  " + "-" * 44]
    for _, r in df5.iterrows():
        a = f"{r['auc_theta_in_bin']:.4f}" if np.isfinite(r["auc_theta_in_bin"]) else "  n/a"
        L.append(f"  [{r['bin_lo']:.1f},{r['bin_hi']:.1f}) {int(r['n']):>6} "
                 f"{r['p_toxic']:>8.3f} {a:>9}")
    L += [""]

    # ---- 【6】分歧样本导出 ----
    zw = (w - w.mean()) / (w.std() + 1e-12)
    zo = (oof - oof.mean()) / (oof.std() + 1e-12)
    d = zw - zo
    order = np.argsort(-np.abs(d))[:400]
    is_fp = np.isin(np.arange(len(ytr)), fp_idx)
    dis = pd.DataFrame({
        "id": ids_tr[order], "content": [text_of[i] for i in ids_tr[order]],
        "oof_s42b": oof[order], "bt_w": w[order], "z_diff": d[order],
        "toxic": ytr[order], "expr": expr[order], "is_FP": is_fp[order],
    })
    dis.to_csv(out / "diag_disagreement.csv", index=False, encoding="utf-8-sig")
    top = dis.head(200)
    agree_dis = {
        "top200_阳性率": float(top["toxic"].mean()),
        "top200_FP占比": float(top["is_FP"].mean()),
        "top200_M4高θ低(z_diff<-1)数": int((top["z_diff"] < -1).sum()),
        "top200_M4低θ高(z_diff>1)数": int((top["z_diff"] > 1).sum()),
    }
    L += ["【6】分歧样本（|z(θ)−z(M4)| top400 导出 diag_disagreement.csv）",
          f"  top200 阳性率 {agree_dis['top200_阳性率']:.3f}（train 基线 0.54）"
          f"，FP 占比 {agree_dis['top200_FP占比']:.3f}（FP 占 train 0.08）",
          f"  M4高/θ低 n={agree_dis['top200_M4高θ低(z_diff<-1)数']}；"
          f"M4低/θ高 n={agree_dis['top200_M4低θ高(z_diff>1)数']}", ""]

    # ---- 【7】锚点效度 ----
    ref = dict(zip(apred["id"], apred["oof_s42b"]))
    a_ref = np.array([ref[aid] for aid in anchor_list])
    a_y = np.array([int(ytr[node_of[aid]]) for aid in anchor_list])
    r_theta_ref = float(np.corrcoef(theta_a, a_ref)[0, 1])
    if not np.isfinite(r_theta_ref):
        r_theta_ref = float("nan")  # 常数序列时 corrcoef 为 NaN，如实上报
    auc_anchor = roc_auc_score(a_y, theta_a)
    L += ["【7】锚点效度（24 锚点）",
          f"  θ(锚点) vs 参考分 oof_s42b 的 pearson r = {r_theta_ref:.4f}",
          f"  θ(锚点) 对真实标签 AUC = {auc_anchor:.4f}"
          f"（锚点两侧各12，理想接近1）",
          f"  高侧锚点 θ: {[round(float(x), 2) for x in theta_a[:len(high_ids)]]}",
          f"  低侧锚点 θ: {[round(float(x), 2) for x in theta_a[len(high_ids):]]}", "",
          "=" * 84]

    report = "\n".join(L)
    print(report)
    (out / "diag_report.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] diag_report.txt / diag_disagreement.csv / theta_profile.csv / "
          "theta_by_m4bin.csv 已写出")


def cv_auc_single(x, y, seeds=(42, 43, 44), C=0.01):
    """单特征 5 折 CV AUC（方向自由，3 种子均值）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    vals = []
    for seed in seeds:
        oof = np.zeros(len(y))
        for trn, va in StratifiedKFold(5, shuffle=True,
                                       random_state=seed).split(X, y):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=C, max_iter=2000))])
            m.fit(X[trn], y[trn])
            oof[va] = m.predict_proba(X[va])[:, 1]
        a = roc_auc_score(y, oof)
        vals.append(max(a, 1 - a))
    return float(np.mean(vals))


if __name__ == "__main__":
    main()
