# -*- coding: utf-8 -*-
"""v15b Stage2：答案就绪位置逐层线性探针——投影瓶颈假设的干净检验。

预登记（train 全量，test 零参与）：
  臂 = scaffold_train.npz 的全部层槽 S0..S{L-1}（S0=embedding 输出，
       末槽=last_hidden_state）逐层线性探针
       + 上下文臂 bare_mean（v15 遗产最佳臂，仅参照不参与门）
  探针 = StandardScaler + L2-LR(C=1.0)，5 折 OOF
  流程（预先声明）= 第一步：全层 × seed42 扫描曲线；第二步：top-3 层
       补 seed43/44 → 最终层报 3 种子 mean±std
  求解器匹配（同 probe_hidden_v15）：torch LBFGS float64 GPU + M4 良态
       认证（max|Δp|<0.02 且 |ΔAUC|<0.003），不达标整体回退 sklearn；
       参照与臂恒同求解器
  参照 = M4 fresh OOF（torch GPU，同求解器；历史 0.8906/0.8948 仅参考）
  门槛（预先声明）：
    主门：最佳层 OOF AUC ≥ 0.910 → 投影瓶颈证实，CBM 2.0（概念=方向）复活
    次门：最佳层 ≥ 参照 AUC → "同位置全向量 ≥ 152 查询蒸馏矩阵"
  判读：主门过 → 进入 CBM 2.0；主门不过次门过 → 投影有损但集成测量仍强；
       双不过 → "隐藏的东西"叙事关闭，结论=教师需微调
产物：results/v15/scaffold/{probe_report.txt, sweep.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from readout_v12 import assemble_m4
from probe_hidden_v15 import (_torch_lr_proba, probe_oof, GATE_AUC,
                              CV_SEEDS, CERT_P_TOL, CERT_AUC_TOL,
                              GRAD_CONV_TOL, stamp)

T0 = time.time()


def main():
    ap = argparse.ArgumentParser(description="v15b 逐层探针（求解器匹配）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--scaffold_dir", default="results/v15/scaffold")
    ap.add_argument("--v15_hidden_dir", default="results/v15/hidden")
    ap.add_argument("--output_dir", default="results/v15/scaffold")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵与脚手架隐状态...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    assert {"item_polarity", "items_per_construct"} <= set(freeze.keys()), \
        "freeze 文件缺少必需键"
    y = tr["toxic"].astype(int).values
    ids = np.array([str(i) for i in tr["id"].values])
    zs = np.load(Path(args.scaffold_dir) / "scaffold_train.npz")
    assert (zs["ids"] == ids).all(), "scaffold npz id 与矩阵 id 不对齐"
    layers_all = zs["layers"]          # (n, L_total, H) float16，逐层切片用
    L_total, H = layers_all.shape[1], layers_all.shape[2]
    stamp(f"[加载] L_total={L_total} hidden={H} n={len(y)}")

    # v15 遗产臂：bare_mean（上下文参照，非门槛）
    z15 = np.load(Path(args.v15_hidden_dir) / "hidden_train.npz")
    assert (z15["ids"] == ids).all(), "v15 npz id 不对齐"
    bm = z15["bare_mean"].astype(np.float32)
    assert np.isfinite(bm).all()

    stamp("[装配] 口头矩阵参照 M4 fresh 重算...")
    Xtr_m4, _ = assemble_m4(tr, te, v3tr, v3te, freeze, y)
    assert len(Xtr_m4) == len(y) and np.isfinite(Xtr_m4).all()

    # ---- 参照基准线：sklearn 侧 M4（无论选路都算，供对照）----
    ref_sk = []
    for seed in CV_SEEDS:
        a, _, _, _ = probe_oof(Xtr_m4, y, seed, device=None)
        ref_sk.append(a)
    ref_sk = np.array(ref_sk)

    # ---- GPU 认证（M4 良态，与 v15 相同代码路径）----
    device = None
    cert_m4, drift_bm = None, None
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            cv0 = list(StratifiedKFold(5, shuffle=True,
                                       random_state=CV_SEEDS[0]).split(Xtr_m4, y))
            trn, va = cv0[0]
            sc = StandardScaler().fit(Xtr_m4[trn])
            Xtr_s, Xva_s = sc.transform(Xtr_m4[trn]), sc.transform(Xtr_m4[va])
            p_sk = LogisticRegression(max_iter=2000, C=1.0).fit(
                Xtr_s, y[trn]).predict_proba(Xva_s)[:, 1]
            p_to, _ = _torch_lr_proba(Xtr_s, y[trn], Xva_s, dev)
            dp = float(np.abs(p_sk - p_to).max())
            da = abs(roc_auc_score(y[va], p_sk) - roc_auc_score(y[va], p_to))
            cert_m4 = (dp, da)
            ok = dp < CERT_P_TOL and da < CERT_AUC_TOL
            stamp(f"[认证] M4 良态 sklearn/torch 对照: max|Δp|={dp:.2e} "
                  f"|ΔAUC|={da:.2e} → " + ("通过，全量走 GPU" if ok
                                           else "未达标，整体回退 CPU"))
            if ok:
                device = dev
                # 诊断：bare_mean（v15 最佳臂）跨求解器漂移（不参与门）
                cvb = list(StratifiedKFold(5, shuffle=True,
                                           random_state=CV_SEEDS[0]).split(bm, y))
                trb, vab = cvb[0]
                scb = StandardScaler().fit(bm[trb])
                Btr_s, Bva_s = scb.transform(bm[trb]), scb.transform(bm[vab])
                pb_sk = LogisticRegression(max_iter=2000, C=1.0).fit(
                    Btr_s, y[trb]).predict_proba(Bva_s)[:, 1]
                pb_to, _ = _torch_lr_proba(Btr_s, y[trb], Bva_s, dev)
                drift_bm = (float(np.abs(pb_sk - pb_to).max()),
                            abs(roc_auc_score(y[vab], pb_sk)
                                - roc_auc_score(y[vab], pb_to)))
                stamp(f"[诊断] bare_mean 跨求解器漂移: max|Δp|={drift_bm[0]:.2e} "
                      f"|ΔAUC|={drift_bm[1]:.2e}（仅诊断）")
    except Exception as e:  # noqa: BLE001
        stamp(f"[认证] GPU 不可用（{type(e).__name__}: {e}），走 sklearn CPU")
        device = None

    # ---- 参照（与臂同求解器）----
    if device is not None:
        ref_main = []
        for seed in CV_SEEDS:
            a, _, _, _ = probe_oof(Xtr_m4, y, seed, device=device)
            ref_main.append(a)
        ref_main = np.array(ref_main)
    else:
        ref_main = ref_sk
    stamp(f"[参照] M4 fresh OOF（同求解器）: AUC = {ref_main.mean():.4f} "
          f"± {ref_main.std():.4f}（sklearn 侧对照 {ref_sk.mean():.4f}）")

    # ---- 第一步：全层 × seed42 扫描 ----
    stamp(f"[扫描] {L_total} 层 × seed{CV_SEEDS[0]}...")
    sweep_rows = []
    conv_max = 0.0
    for k in range(L_total):
        X = layers_all[:, k, :].astype(np.float32)
        assert np.isfinite(X).all(), f"层 S{k} 出现非有限值"
        auc, tau, f1m, conv = probe_oof(X, y, CV_SEEDS[0], device=device)
        conv_max = max(conv_max, float(np.max(conv)))
        sweep_rows.append({"slot": f"S{k}", "layer_idx": k,
                           "seed": CV_SEEDS[0], "oof_auc": auc,
                           "tau": tau, "oof_macro": f1m})
        print(f"  S{k:>2} AUC={auc:.4f} 宏F1@τ={f1m:.4f}", flush=True)
    sweep = pd.DataFrame(sweep_rows)
    top3 = sweep.nlargest(3, "oof_auc")

    # ---- 第二步：top-3 层补 seed43/44 ----
    stamp("[确认] top-3 层 × 3 种子...")
    final_rows = []
    for _, r in top3.iterrows():
        k = int(r["layer_idx"])
        X = layers_all[:, k, :].astype(np.float32)
        aucs = [float(r["oof_auc"])]
        for seed in CV_SEEDS[1:]:
            a, _, _, conv = probe_oof(X, y, seed, device=device)
            conv_max = max(conv_max, float(np.max(conv)))
            aucs.append(a)
        final_rows.append({"slot": f"S{k}", "layer_idx": k,
                           "auc_mean": float(np.mean(aucs)),
                           "auc_std": float(np.std(aucs)),
                           "aucs": aucs})
        print(f"  S{k:>2} AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
              flush=True)
    final = pd.DataFrame(final_rows).sort_values("auc_mean", ascending=False)
    best = final.iloc[0]
    best_auc = float(best["auc_mean"])

    # ---- 上下文臂：v15 bare_mean（同求解器 3 种子）----
    bm_aucs = []
    for seed in CV_SEEDS:
        a, _, _, _ = probe_oof(bm, y, seed, device=device)
        bm_aucs.append(a)
    bm_auc = float(np.mean(bm_aucs))

    sweep.to_csv(out / "sweep.csv", index=False, encoding="utf-8-sig")
    final.to_csv(out / "top3_final.csv", index=False, encoding="utf-8-sig")

    # ---- 报告 ----
    passed_main = best_auc >= GATE_AUC
    passed_sec = best_auc >= ref_main.mean()
    lines = ["=" * 72, "v15b 答案就绪位置逐层探针——投影瓶颈干净检验",
             "=" * 72,
             f"槽位数={L_total}（S0=embedding 输出，末槽=last_hidden_state），"
             f"hidden={H}，n={len(y)}",
             "脚手架: user = {text}\\n\\n这条评论是否有毒性？+ assistant 起始截停"
             "（末位置 = 口头探针首 token 输出位置；文献做法见 docstring）",
             "执行路径: " + ("GPU（torch LBFGS float64，M4 认证通过，参照与臂同求解器）"
                            if device is not None
                            else "CPU（sklearn，参照与臂同求解器）"), ""]
    if cert_m4 is not None:
        lines.append(f"认证（M4 良态）: max|Δp|={cert_m4[0]:.2e} "
                     f"|ΔAUC|={cert_m4[1]:.2e}")
    if drift_bm is not None:
        lines.append(f"诊断（bare_mean 跨求解器漂移，不参与门）: "
                     f"max|Δp|={drift_bm[0]:.2e} |ΔAUC|={drift_bm[1]:.2e}")
    lines += ["",
              f"口头矩阵参照（M4 fresh OOF，同求解器）: "
              f"{ref_main.mean():.4f} ± {ref_main.std():.4f}"
              f"（sklearn 侧 {ref_sk.mean():.4f}）",
              f"v15 bare_mean 上下文臂（同求解器）: {bm_auc:.4f}", "",
              "top-3 层（3 种子）:", "-" * 72]
    for _, r in final.iterrows():
        lines.append(f"{r['slot']:<6} AUC = {r['auc_mean']:.4f} ± "
                     f"{r['auc_std']:.4f}")
    lines += ["", f"最佳层: {best['slot']}（AUC {best_auc:.4f}）",
              f"相对 M4 参照差值: {best_auc - ref_main.mean():+.4f}",
              f"相对 v15 bare_mean 差值: {best_auc - bm_auc:+.4f}",
              f"主门（AUC ≥ {GATE_AUC}）: " + ("通过" if passed_main else "未通过"),
              f"次门（≥ 参照 {ref_main.mean():.4f}）: "
              + ("通过" if passed_sec else "未通过")]
    if device is not None:
        conv_warn = ("" if conv_max < GRAD_CONV_TOL else
                     f"；警告: 部分拟合梯度 ≥ {GRAD_CONV_TOL:.0e}"
                     "（未完全收敛，AUC 读数偏保守）")
        lines.append(f"收敛性（GPU）: 全部拟合最终梯度∞范数 max={conv_max:.1e}"
                     + conv_warn)
    else:
        lines.append(f"收敛性（CPU）: n_iter max={conv_max:.0f}"
                     + ("" if conv_max < 2000 * 0.95 else
                        "；警告: 部分拟合 n_iter 打满 2000（未完全收敛）"))
    lines += ["", "判读: 主门过 → 投影瓶颈证实，进入 CBM 2.0；",
              "      主门不过次门过 → 投影有损但 152 查询集成测量仍强；",
              "      双不过 → \"隐藏的东西\"叙事关闭，结论=教师需微调。"]
    report = "\n".join(lines)
    print(report)
    (out / "probe_report.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] probe_report.txt / sweep.csv / top3_final.csv 已写出")


if __name__ == "__main__":
    main()
