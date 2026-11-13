# -*- coding: utf-8 -*-
"""v11-Qwen 交叉教师检验：Qwen 在 GLM 的 FP/TP 边界上看到了什么（CPU）。

背景：Qwen 版 v11.1/v11.2 按预登记判死（G2 FAIL 0.7670 / G1 FAIL 0.7116）。
但判死的是"作为并入 GLM 矩阵的增量通道"；多教师路线的真正命题是
【不同教师是否看错不同的样本】。本检验直接测：

  1. 句级一致性：两教师逐句 p_tox 的相关与分歧率
  2. FP vs TP 边界（seed42 锚点，GLM FP=766/TP=4171）：
     Qwen 的句级特征能否把 GLM 的 FP 与 TP 分开？
     对照：GLM 自己的句级特征（fp_tp_check 已测 0.61-0.63）、M4（0.7333）
  3. GLM-FP 文本上 Qwen 的绝对读数（FP 池里 Qwen 认为毒的比例）

预登记判读：Qwen 句级特征在 FP vs TP 上 AUC ≥ 0.75（显著高于 M4 的
0.7333）→ 交叉互补存在，Qwen 全量文本级测量值得上；
≤ 0.7333 → 连不同教师的句级视图都坍缩到同一边界，多教师侦察为负。

产物：results/v11/qwen/cross_teacher_check.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83
from verify_noise_v11 import assemble_m4
from analyze_target_v11 import FEATS as TGT_FEATS, aggregate as tgt_aggregate

OUT = Path("results/v11/qwen/cross_teacher_check.txt")


def agg_assert(flat, ids):
    """每文本句级特征（单一教师 assert_flat）。"""
    by = {k: v for k, v in flat.groupby("id", sort=False)}
    rows = []
    for i in ids:
        g = by.get(i)
        if g is None or len(g) == 0:
            rows.append({"as_p_tox_max": 0.0, "as_n_toxic": 0.0,
                         "as_soft_asserted": 0.0})
            continue
        p = g["p_tox"].values
        wa = g["w_断言"].values if "w_断言" in g.columns else np.ones_like(p)
        toxic = p >= 0.5
        rows.append({
            "as_p_tox_max": float(p.max()),
            "as_n_toxic": float(toxic.sum()),
            "as_soft_asserted": float(np.mean(p * wa)) if len(p) else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv")
    v3tr = pd.read_csv(
        "data/processed/TOXICN/glm-4-9b-chat_v3/v3_soft_train_matrix.csv")
    freeze = json.loads(Path(
        "results/v6/item_analysis_v64_train/v6_4_freeze.json").read_text(encoding="utf-8"))
    ytr = tr["toxic"].astype(int).values
    ids = tr["id"].values

    glm = pd.read_csv("results/v11/assert/assert_flat_train.csv")
    qwen = pd.read_csv("results/v11/qwen/assert/assert_flat_train.csv")
    m = glm.merge(qwen, on=["id", "sent_idx"], suffixes=("_glm", "_qwen"))
    lines = ["=" * 72, "交叉教师检验：GLM vs Qwen 句级测量（同句同文本）",
             "=" * 72, f"可对齐句子 {len(m)}（GLM {len(glm)} / Qwen {len(qwen)}）"]

    pg, pq = m["p_tox_glm"].values, m["p_tox_qwen"].values
    r = np.corrcoef(pg, pq)[0, 1]
    dis_hi = int(np.sum((pg >= 0.5) & (pq < 0.5)))
    dis_lo = int(np.sum((pg < 0.5) & (pq >= 0.5)))
    lines += [f"逐句 p_tox Pearson r = {r:.4f}",
              f"GLM 认为毒/Qwen 认为无毒: {dis_hi} 句",
              f"GLM 认为无毒/Qwen 认为毒: {dis_lo} 句", ""]

    # ---- FP/TP 定义（seed42 锚点，与 fp_tp_check 同法）----
    X = assemble_m4(tr, v3tr, freeze, ytr)
    a_alpha, tau, oof = r83.cv_select(X, ytr, 42, "anchor")
    err = (oof >= tau).astype(int) != ytr
    fp_i = np.where(err & (ytr == 0))[0]
    tp_i = np.where((~err) & (ytr == 1))[0]
    tn_i = np.where((~err) & (ytr == 0))[0]
    print(f"anchor alpha={a_alpha} tau={tau} FP={len(fp_i)} TP={len(tp_i)}")

    id2row = {i: k for k, i in enumerate(ids)}
    A_glm = agg_assert(glm, ids)
    A_qwen = agg_assert(qwen, ids)
    tf = pd.read_csv("results/v11/qwen/target/target_flat_train.csv")
    T_qwen = tgt_aggregate(tf, ids)[TGT_FEATS].values
    tgt_cols = TGT_FEATS

    # ---- FP vs TP 判别 ----
    idx = np.concatenate([fp_i, tp_i])
    ysub = np.where(np.arange(len(idx)) < len(fp_i), 0, 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    def sub_auc(F):
        oof = np.zeros(len(idx))
        for trn, va in cv.split(F[idx], ysub):
            mm = Pipeline([("sc", StandardScaler()),
                           ("m", LogisticRegression(C=0.01, max_iter=2000))])
            mm.fit(F[idx][trn], ysub[trn])
            oof[va] = mm.predict_proba(F[idx][va])[:, 1]
        a = roc_auc_score(ysub, oof)
        return max(a, 1 - a)

    lines += [f"FP vs TP（GLM seed42 锚点 FP={len(fp_i)}/TP={len(tp_i)}，"
              f"参照 M4=0.7333）", "-" * 72]
    for name, F, cols in (("GLM句级3特征", A_glm.values, list(A_glm.columns)),
                          ("Qwen句级3特征", A_qwen.values, list(A_qwen.columns)),
                          ("Qwen目标11特征", T_qwen, tgt_cols),
                          ("GLM句级+Qwen句级", np.hstack([A_glm.values, A_qwen.values]),
                           None),
                          ("M4+Qwen句级", np.hstack([X, A_qwen.values]), None)):
        a = sub_auc(F)
        lines.append(f"  {name:<18} AUC = {a:.4f}")

    lines += ["", "GLM-FP 池上 Qwen 的绝对读数", "-" * 72]
    for name, ii in (("FP", fp_i), ("TP", tp_i), ("TN", tn_i)):
        rows_i = [id2row[ids[k]] for k in ii]
        qm = A_qwen["as_p_tox_max"].values[rows_i]
        tg = T_qwen[rows_i, tgt_cols.index("soft_grp_asserted")]
        lines.append(f"  {name} n={len(ii):<5} Qwen p_tox_max 均值={qm.mean():.4f}"
                     f"  ≥0.5 占比={np.mean(qm >= 0.5):.3f}"
                     f"  soft_grp_asserted 均值={tg.mean():.4f}")

    q_fp = A_qwen["as_p_tox_max"].values[[id2row[ids[k]] for k in fp_i]]
    q_tp = A_qwen["as_p_tox_max"].values[[id2row[ids[k]] for k in tp_i]]
    best_gate = sub_auc(A_qwen.values)
    verdict = ("交叉互补存在（Qwen 句级视图在 GLM FP/TP 边界上显著有效）"
               if best_gate >= 0.75 else
               "多教师侦察为负（不同教师的句级视图坍缩到同一边界）"
               if best_gate <= 0.7333 else "灰色地带，需人工判读")
    lines += ["", f"[裁决] Qwen 句级特征 FP vs TP AUC = {best_gate:.4f} → {verdict}",
              "=" * 72]
    report = "\n".join(lines)
    print(report)
    OUT.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
