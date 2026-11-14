# -*- coding: utf-8 -*-
"""v14 Stage 0：M4 锚点 fresh 复算 + 门槛基准重建（CPU，全部数字从原始矩阵重算）。

用户裁决（2026-11-13）：过往数据经验只能作为参考，不能作为决定性数据；
任何需要数据下结论的场合必须重新写脚本验证。本脚本产出 v14 全部后续
门槛判断所依赖的 fresh 基准，历史值（0.8156/0.8929/0.7335 等）仅作对照。

【A】锚点复算：M4 264 维装配（readout_v12.assemble_m4 逐字节复用），
     5 种子（0-4）train OOF τ@宏F1 → test 全指标 mean±std。
     历史对照：v12 N0 = 0.8156±0.0012 / AUC 0.8948（抖动容差 ±0.002）。
【B】同配对 fresh 基准（v14 Thurstone 门槛的对照臂，方法学与
     verify_noise_v11.cv_pair_auc 逐行一致：3 种子 42/43/44 × 5 折，
     StandardScaler + L2-LR C=0.01，子集内 OOF AUC 取 max(a,1-a)）：
     B1  E0无毒(expr0,y=0) vs 全部有毒 —— G1' 主门靶点
     B2  FP vs TP（seed42 锚点 OOF 定界）—— 成对比较的原始靶点
【C】fresh 产物导出：anchor_predictions_train.csv（全 train OOF/预测/
     expr）、fp_tp_ids.csv（seed42 FP/TP id 名单）——后续门槛脚本只准
     消费本脚本产物，不得引用历史数字。

产物：results/v14/reverify/{reverify.txt, anchor_predictions_train.csv,
      fp_tp_ids.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import readout_v83 as r83
from readout_v12 import assemble_m4

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cv_pair_auc(X, ysub, idx, seeds=(42, 43, 44), C=0.01):
    """子集内 5 折 CV AUC，3 种子均值（与 verify_noise_v11.cv_pair_auc 逐行一致）。"""
    vals = []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(idx))
        for trn, va in cv.split(X[idx], ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=C, max_iter=2000))])
            m.fit(X[idx][trn], ysub[trn])
            oof[va] = m.predict_proba(X[idx][va])[:, 1]
        a = roc_auc_score(ysub, oof)
        vals.append(max(a, 1 - a))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser(description="v14 Stage 0 锚点 fresh 复算")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json")
    ap.add_argument("--output_dir", default="results/v14/reverify")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    # ---- fail-fast：本脚本对常量的全部假设在此处断言（长计算前暴露）----
    assert set(r83.FINAL_SEEDS) == {0, 1, 2, 3, 4}, r83.FINAL_SEEDS
    assert 42 in r83.CV_SEEDS, r83.CV_SEEDS
    assert callable(r83.cv_select) and callable(assemble_m4)
    assert {"item_polarity", "items_per_construct"} <= set(freeze.keys())

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (te["id"].values == v3te["id"].values).all()
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values
    ids_tr = tr["id"].values

    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids_tr])

    stamp("[装配] M4 特征装配中...")
    Xtr, Xte = assemble_m4(tr, te, v3tr, v3te, freeze, ytr)
    stamp(f"[装配] 完成 train {Xtr.shape} / test {Xte.shape}")

    # ---- 【A】锚点 5 种子复算 ----
    S = len(r83.FINAL_SEEDS)
    oof = np.zeros((len(ytr), S))
    pte = np.zeros((len(yte), S))
    taus = np.zeros(S)
    alphas = []
    for k, seed in enumerate(tqdm(r83.FINAL_SEEDS, desc="Stage0 锚点 5 种子",
                                  dynamic_ncols=True)):
        a, tau, o = r83.cv_select(Xtr, ytr, seed, f"rv[seed{seed}]")
        oof[:, k] = o
        m = Pipeline([("m", LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1.0 / a, max_iter=2000))]).fit(Xtr, ytr)
        pte[:, k] = m.predict_proba(Xte)[:, 1]
        taus[k] = tau
        alphas.append(a)
        stamp(f"[锚点] seed={seed} 完成（{k + 1}/{S}）  alpha={a}  τ={tau:.2f}")
    pred_te = (pte >= taus[None, :]).astype(int)
    rows = []
    for k in range(S):
        pred = pred_te[:, k]
        rows.append({
            "seed": r83.FINAL_SEEDS[k], "alpha": alphas[k], "tau": taus[k],
            "macro": f1_score(yte, pred, average="macro"),
            "weighted": f1_score(yte, pred, average="weighted"),
            "acc": accuracy_score(yte, pred),
            "prec_macro": precision_score(yte, pred, average="macro"),
            "rec_macro": recall_score(yte, pred, average="macro"),
            "toxic": f1_score(yte, pred),
            "prec_t": precision_score(yte, pred),
            "rec_t": recall_score(yte, pred),
            "benign": f1_score(yte, pred, pos_label=0),
            "auc": roc_auc_score(yte, pte[:, k]),
        })
    df = pd.DataFrame(rows)

    L = ["=" * 78, "v14 Stage 0：M4 锚点 fresh 复算（历史值仅作对照，不作裁决依据）",
         "=" * 78,
         "【A】test 全指标（5 种子 mean±std）",
         "  指标            mean     std     历史对照(v12 N0)",
         "  " + "-" * 64]
    HIST = {"macro": "0.8156±0.0012", "weighted": "-", "acc": "-",
            "prec_macro": "-", "rec_macro": "-", "toxic": "0.8279 附近",
            "prec_t": "-", "rec_t": "-", "benign": "0.8038 附近",
            "auc": "0.8948±0.0001"}
    for c in ("macro", "weighted", "acc", "prec_macro", "rec_macro",
              "toxic", "prec_t", "rec_t", "benign", "auc"):
        L.append(f"  {c:<15}{df[c].mean():.4f}  ±{df[c].std():.4f}   {HIST[c]}")
    L += [f"  τ/seed: {np.round(taus, 2).tolist()}",
          f"  宏F1 抖动检查（|fresh−0.8156|≤0.002）: "
          f"{'通过' if abs(df['macro'].mean() - 0.8156) <= 0.002 else '超出容差，须人工判读'}",
          ""]

    # ---- 【B】同配对 fresh 基准 ----
    e0fp = np.where((ytr == 0) & (expr == 0))[0]
    tox = np.where(ytr == 1)[0]
    idx_b1 = np.concatenate([e0fp, tox])
    ysub_b1 = np.where(np.arange(len(idx_b1)) < len(e0fp), 0, 1)
    stamp(f"[基准] B1 E0无毒 n={len(e0fp)} vs 有毒 n={len(tox)} CV 中...")
    a_b1, s_b1 = cv_pair_auc(Xtr, ysub_b1, idx_b1)

    # seed42 边界定界 OOF 独立计算（FINAL_SEEDS 为 0-4，不含 42；种子约定
    # 沿用 v8.3 审计 = seed42，但按用户新规则一律 fresh 重算）
    stamp("[基准] B2 seed42 定界 OOF 计算中...")
    a42, tau42, oof42 = r83.cv_select(Xtr, ytr, 42, "rv[seed42-boundary]")
    pred42 = (oof42 >= tau42).astype(int)
    stamp(f"[基准] B2 seed42 定界完成  alpha={a42}  τ={tau42:.2f}")
    fp = np.where((pred42 == 1) & (ytr == 0))[0]
    tp = np.where((pred42 == 1) & (ytr == 1))[0]
    assert len(fp) > 0 and len(tp) > 0, \
        f"seed42 OOF 边界退化：FP={len(fp)}, TP={len(tp)}——检查上游 cv_select"
    idx_b2 = np.concatenate([fp, tp])
    ysub_b2 = np.where(np.arange(len(idx_b2)) < len(fp), 0, 1)
    stamp(f"[基准] B2 FP n={len(fp)} vs TP n={len(tp)} CV 中...")
    a_b2, s_b2 = cv_pair_auc(Xtr, ysub_b2, idx_b2)

    L += ["【B】v14 门槛靶点的 fresh 基准（3 种子×5 折，同 verify_noise 方法学）",
          "-" * 78,
          f"  B1  M4 @ E0无毒(expr0) vs 有毒:  AUC = {a_b1:.4f} ± {s_b1:.4f}"
          f"   （G1' 主门对照臂；历史值 0.8929±0.0004 仅参考）",
          f"  B2  M4 @ FP vs TP（seed42 锚点定界）:  AUC = {a_b2:.4f} ± {s_b2:.4f}"
          f"   （成对比较原始靶点；历史值 0.7333 仅参考）",
          f"  FP/TP 定界: seed42 τ={tau42:.2f} alpha={a42}, "
          f"FP n={len(fp)}, TP n={len(tp)}", "",
          "  预登记规则（v14 Thurstone 门槛将引用本节 fresh 值，不引用历史值）：",
          "    G1' = M4+BT特征 联合 AUC ≥ B1_fresh + 0.010（配对 5 折 CV）",
          "    G2' = BT 特征与现有 264 维 max|corr| < 0.7",
          "    双门过 → readout CV 增益 ≥ +0.002 才碰 test", ""]

    # ---- 【C】fresh 产物导出（每个数组长度先断言，杜绝长度错位）----
    n_tr, n_te = len(ids_tr), len(te)
    assert len(ytr) == n_tr and len(expr) == n_tr
    assert oof.shape == (n_tr, S) and pte.shape == (n_te, S)
    assert len(oof42) == n_tr and len(pred42) == n_tr
    pd.DataFrame({
        "id": ids_tr, "toxic": ytr, "expr": expr,
        "oof_s42b": oof42, "pred_s42b": pred42,
        **{f"oof_s{s}": oof[:, j] for j, s in enumerate(r83.FINAL_SEEDS)},
    }).to_csv(out / "anchor_predictions_train.csv", index=False,
              encoding="utf-8-sig")
    pd.DataFrame({
        "id": te["id"].values, "toxic": yte,
        **{f"p_te_s{s}": pte[:, j] for j, s in enumerate(r83.FINAL_SEEDS)},
    }).to_csv(out / "anchor_predictions_test.csv", index=False,
              encoding="utf-8-sig")
    pd.concat([
        pd.DataFrame({"id": ids_tr[fp], "group": "FP"}),
        pd.DataFrame({"id": ids_tr[tp], "group": "TP"}),
    ], ignore_index=True).to_csv(out / "fp_tp_ids.csv", index=False,
                                 encoding="utf-8-sig")
    L += ["【C】fresh 产物已导出（后续门槛脚本只准消费本目录）：",
          "  anchor_predictions_train.csv（全 train OOF/预测/expr，5 种子 + seed42b）",
          "  anchor_predictions_test.csv（test 概率，5 种子）",
          f"  fp_tp_ids.csv（seed42 FP {len(fp)} + TP {len(tp)}）", "",
          "=" * 78, "Stage 0 完成。后续 v14 脚本（run_thurstone/analyze）以本报告为准。"]

    report = "\n".join(L)
    print(report)
    (out / "reverify.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
