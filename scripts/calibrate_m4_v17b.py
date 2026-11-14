# -*- coding: utf-8 -*-
"""v17 臂③ Stage2（本机）：DC-PMI 校准 M4 与原 M4 的同折竞争。

预登记（train 全量 CV 竞争，门槛 ≥ +0.002 才碰 test）：
  校准 = σ(logit(p_yes_raw) − logit(p_prior))，逐条目列，占位符两臂
      （na="N/A" / wu="无"，先验来自 Stage1 重测）
  校准后的条目矩阵替换原矩阵喂 assemble_m4（零改动，极性/分位/元特征
  全部原协议），得 M4_cal_na / M4_cal_wu
  探针 = StandardScaler + L2-LR(C=1.0)，5 折 × 3 种子（42/43/44）OOF，
      同折配对；AUC 与宏F1@τ 双指标报告（τ 取 r83.TAU_GRID 上 OOF 宏F1 最大）
  门槛 = best_cal OOF AUC ≥ M4_raw OOF AUC + 0.002
    过 → test 阶段：全 train 拟合 → test 预测（τ 用各自 OOF 最优），
         与 M4_raw test fresh 配对比较（历史 test 0.8138±0.0022 仅参考）
产物：results/v17/dcpmi/{calibrate_report.txt, calibrate_cv.csv,
      items_cal_{na,wu}_{train,test}.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83
from readout_v12 import assemble_m4

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()
SEEDS = (42, 43, 44)
GATE = 0.002
EPS = 1e-4


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def calibrate_items_matrix(df, prior, ph):
    """条目列逐列做 DC-PMI 校准；id/toxic/content 原样保留。"""
    out = df.copy()
    item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
    assert set(item_cols) == set(prior.keys()), \
        f"条目列与先验 id 不对齐：缺 {sorted(set(item_cols) - set(prior.keys()))[:3]}"
    prior_vec = np.array([prior[c] for c in item_cols])
    pv = _logit(prior_vec)
    A = df[item_cols].values.astype(float)
    assert np.isfinite(A).all() and ((A >= 0) & (A <= 1)).all()
    out[item_cols] = 1.0 / (1.0 + np.exp(-(_logit(A) - pv)))
    return out


def probe_oof(X, y, seed):
    """OOF：返回 (oof, auc, tau, macro_f1@tau, n_iter_mean)。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    nits = []
    for trn, va in cv.split(X, y):
        sc = StandardScaler().fit(X[trn])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[trn]),
                                                         y[trn])
        oof[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
        nits.append(int(np.ravel(m.n_iter_)[0]))
    tau = max(r83.TAU_GRID,
              key=lambda t: f1_score(y, (oof >= t).astype(int),
                                     average="macro"))
    f1m = f1_score(y, (oof >= tau).astype(int), average="macro")
    return oof, roc_auc_score(y, oof), float(tau), float(f1m), \
        float(np.mean(nits))


def test_eval(Xtr, ytr, Xte, yte, tau):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p), \
        f1_score(yte, (p >= tau).astype(int), average="macro")


def main():
    ap = argparse.ArgumentParser(description="v17b DC-PMI 校准 M4 竞争")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--prior_json", default="results/v17/dcpmi/prior_glm-4-9b-chat.json")
    ap.add_argument("--output_dir", default="results/v17/dcpmi")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] ...")
    pj = json.loads(Path(args.prior_json).read_text(encoding="utf-8"))
    prior_all = pj["prior"]
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values

    cal_mats = {}
    for ph, prior in prior_all.items():
        cal_mats[ph] = (calibrate_items_matrix(tr, prior, ph),
                        calibrate_items_matrix(te, prior, ph))
        stamp(f"[校准] 占位符 {ph} 完成（train/test 条目矩阵已生成）")
        cal_mats[ph][0].to_csv(out / f"items_cal_{ph}_train.csv",
                               index=False, encoding="utf-8-sig")
        cal_mats[ph][1].to_csv(out / f"items_cal_{ph}_test.csv",
                               index=False, encoding="utf-8-sig")

    stamp("[装配] M4 fresh（raw 参照 + 各校准臂）...")
    mats = {"raw": assemble_m4(tr, te, v3tr, v3te, freeze, ytr)}
    for ph in prior_all:
        mats[ph] = assemble_m4(cal_mats[ph][0], cal_mats[ph][1],
                               v3tr, v3te, freeze, ytr)
    for k, (Xa, Xb) in mats.items():
        assert Xa.shape == mats["raw"][0].shape and np.isfinite(Xa).all()

    folds_by_seed = {s: list(StratifiedKFold(5, shuffle=True,
                                             random_state=s).split(mats["raw"][0],
                                                                   ytr))
                     for s in SEEDS}

    rows, oof_store, tau_store = [], {}, {}
    for name in tqdm(["raw"] + list(prior_all), desc="OOF[5折×3种子]",
                     dynamic_ncols=True):
        Xa = mats[name][0]
        for seed in SEEDS:
            oof, auc, tau, f1m, nit = probe_oof(Xa, ytr, seed)
            rows.append({"arm": name, "seed": seed, "oof_auc": auc,
                         "tau": tau, "oof_macro_f1": f1m, "n_iter_mean": nit})
            oof_store[(name, seed)] = oof
            tau_store[(name, seed)] = tau
            print(f"  {name:<5} seed{seed} AUC={auc:.4f} 宏F1={f1m:.4f} "
                  f"τ*={tau}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "calibrate_cv.csv", index=False, encoding="utf-8-sig")

    def ms(df, col):
        g = df.groupby("arm")[col]
        return {a: (g.mean()[a], g.std()[a]) for a in g.mean().index}

    auc_s, f1_s = ms(cv, "oof_auc"), ms(cv, "oof_macro_f1")
    raw_auc = auc_s["raw"][0]
    best_ph = max(prior_all, key=lambda p: auc_s[p][0])
    best_auc = auc_s[best_ph][0]
    passed = best_auc >= raw_auc + GATE

    lines = ["=" * 72, "v17 臂③ DC-PMI 校准 M4 同折竞争（AUC / 宏F1 双指标）",
             "=" * 72,
             f"先验来源: {args.prior_json}（占位符 {list(prior_all)}）；"
             f"校准=σ(logit(p_raw)−logit(p_prior)) 逐条目列", "",
             f"{'臂':<8}{'OOF AUC':<18}{'OOF 宏F1':<18}{'ΔAUC vs raw':<14}",
             "-" * 72]
    for name in ["raw"] + list(prior_all):
        a, a_sd = auc_s[name]
        f, f_sd = f1_s[name]
        d = "" if name == "raw" else f"{a - raw_auc:+.4f}"
        lines.append(f"{name:<8}{a:.4f} ± {a_sd:.4f}   {f:.4f} ± {f_sd:.4f}   {d}")
    lines += ["-" * 72,
              f"最佳校准臂: {best_ph}（{best_auc:.4f}）",
              f"预登记门槛（≥ raw + {GATE}）: " + ("通过" if passed else "未通过")]

    if passed:
        stamp("[test] 门槛通过，运行 test 阶段...")
        te_rows = []
        for name in ["raw", best_ph]:
            tau_full = float(np.mean([tau_store[(name, s)] for s in SEEDS]))
            auc_t, f1_t = test_eval(mats[name][0], ytr, mats[name][1], yte,
                                    tau_full)
            te_rows.append((name, auc_t, f1_t, tau_full))
            print(f"  {name:<5} test AUC={auc_t:.4f} 宏F1={f1_t:.4f} "
                  f"τ={tau_full:.3f}", flush=True)
        lines += ["", f"test（全 train 拟合，τ=各臂 OOF 最优均值；"
                      f"历史锚点 0.8138±0.0022 仅参考）:",
                  "-" * 72]
        for name, auc_t, f1_t, tau_full in te_rows:
            lines.append(f"{name:<8} test AUC={auc_t:.4f}  宏F1={f1_t:.4f}  τ={tau_full:.3f}")
        d_auc = te_rows[1][1] - te_rows[0][1]
        d_f1 = te_rows[1][2] - te_rows[0][2]
        lines.append(f"配对差（{best_ph} − raw）: ΔAUC={d_auc:+.4f} "
                     f"Δ宏F1={d_f1:+.4f}")

    report = "\n".join(lines)
    print(report)
    (out / "calibrate_report.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] calibrate_report.txt / calibrate_cv.csv 已写出")


if __name__ == "__main__":
    main()
