# -*- coding: utf-8 -*-
"""v15 Stage2：隐状态线性探针 vs 口头探针矩阵的"知道-说出"差距判读（GPU 全程）。

预登记（train 全量，test 零参与）：
  四臂 = bare_last / bare_mean / chat_last / concat(bare_last+chat_last)
  探针 = StandardScaler + L2-LR(C=1.0)，5 折 × 3 种子（42/43/44）OOF
  参照 = 口头矩阵 M4（264 维，assemble_m4 fresh 装配）同一 CV 划分下的 OOF AUC
  门槛 = 最佳臂 OOF AUC ≥ 0.910（绝对值，预先声明）
    过 → "LLM 知道的多于说出来的"证实，天花板打开，进入 CBM 2.0（概念方向分解）
    不过 → 状态内也无更多线性可读信息，结论修正为"教师需微调"
  同时报告各臂 τ@宏F1（仅参照，不参与门槛）。

【求解器匹配设计（v15b 修订）】此前版本 GPU 只跑探针臂、参照用 sklearn，
跨求解器比较混入求解器差异（4096 维近线性可分问题最优解处于病态平坦区，
不同求解器停止点不同，Δp 可达 0.9）。修订：参照与探针臂使用**同一个求解器**
——GPU 认证通过时全部走 torch LBFGS（float64，目标 = Σ BCE + ||w||²/(2C)，
截距不罚，与 sklearn 同损失同尺度），否则全部走 sklearn CPU。求解器差异
由此从参照-探针比较中彻底消除。

【等价性认证】在参照矩阵 M4（264 维，良态）的 seed42 第 0 折上双跑
sklearn 与 torch，要求 max|Δp| < 0.02 且 |ΔAUC| < 0.003。良态问题上两个
求解器都应收敛到同一凸最优解，认证通过即证明"工具一致"；不通过则整体
回退 sklearn CPU（参照与臂仍同求解器，比较仍然干净）。
另在 bare_last（4096 维，病态）第 0 折上报告跨求解器漂移——仅作诊断
（量化 4096 维病态的求解器敏感度），不参与任何门。

【收敛自检】GPU 路径记录每折最终梯度 ∞ 范数（< 1e-5 视为收敛）；CPU 路径
记录 n_iter（接近 max_iter=2000 视为未收敛、读数偏保守）。

产物：results/v15/hidden/probe_report.txt + probe_cv.csv
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

GATE_AUC = 0.910
CV_SEEDS = (42, 43, 44)
CERT_P_TOL = 0.02     # 认证阈值：单样本概率最大绝对差（M4 良态参照矩阵上）
CERT_AUC_TOL = 0.003  # 认证阈值：折内 AUC 绝对差
GRAD_CONV_TOL = 1e-5  # GPU 路径收敛判据：最终梯度 ∞ 范数
T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _torch_lr_proba(Xtr_s, ytr, Xva_s, device, C=1.0, max_iter=2000):
    """torch LBFGS 复刻 sklearn LogisticRegression(C=C, l2, lbfgs)：
    目标 = Σ BCE + ||w||²/(2C)（求和尺度与 sklearn 一致，截距不罚），
    float64 求解。返回 (验证折概率, 最终梯度∞范数)。"""
    import torch
    Xt = torch.tensor(Xtr_s, dtype=torch.float64, device=device)
    yt = torch.tensor(ytr.astype(np.float64), device=device)
    Xv = torch.tensor(Xva_s, dtype=torch.float64, device=device)
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, device=device,
                    requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    last = {"gnorm": float("nan")}

    def closure():
        opt.zero_grad()
        loss = bce(Xt @ w + b, yt, reduction="sum") + (w * w).sum() / (2.0 * C)
        loss.backward()
        with torch.no_grad():
            last["gnorm"] = float(w.grad.abs().max().item()
                                  + abs(b.grad.item()))
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (Xv @ w + b).sigmoid().cpu().numpy(), last["gnorm"]


def probe_oof(X, y, seed, device=None, max_iter=2000):
    """返回 (AUC, τ@宏F1, 宏F1, conv 列表)。
    conv：GPU 路径 = 每折最终梯度∞范数；CPU 路径 = 每折 n_iter。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    conv = []
    for trn, va in cv.split(X, y):
        sc = StandardScaler().fit(X[trn])
        Xtr_s, Xva_s = sc.transform(X[trn]), sc.transform(X[va])
        if device is not None:
            oof[va], g = _torch_lr_proba(Xtr_s, y[trn], Xva_s, device,
                                         max_iter=max_iter)
            conv.append(g)
        else:
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, y[trn])
            oof[va] = m.predict_proba(Xva_s)[:, 1]
            conv.append(int(np.ravel(m.n_iter_)[0]))
    tau = max(r83.TAU_GRID,
              key=lambda t: f1_score(y, (oof >= t).astype(int), average="macro"))
    return roc_auc_score(y, oof), float(tau), \
        f1_score(y, (oof >= tau).astype(int), average="macro"), conv


def main():
    ap = argparse.ArgumentParser(description="v15 隐状态线性探针（求解器匹配）")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--hidden_dir", default="results/v15/hidden")
    ap.add_argument("--output_dir", default="results/v15/hidden")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵与隐状态...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    assert {"item_polarity", "items_per_construct"} <= set(freeze.keys()), \
        "freeze 文件缺少必需键"
    y = tr["toxic"].astype(int).values
    ids = np.array([str(i) for i in tr["id"].values])
    z = np.load(Path(args.hidden_dir) / "hidden_train.npz")
    assert (z["ids"] == ids).all(), "npz id 与矩阵 id 不对齐"
    bl = z["bare_last"].astype(np.float32)
    bm = z["bare_mean"].astype(np.float32)
    cl = z["chat_last"].astype(np.float32)
    assert np.isfinite(bl).all() and np.isfinite(bm).all() and np.isfinite(cl).all()

    stamp("[装配] 口头矩阵参照 M4 fresh 重算...")
    Xtr_m4, _ = assemble_m4(tr, te, v3tr, v3te, freeze, y)
    assert len(Xtr_m4) == len(y) and np.isfinite(Xtr_m4).all()

    ARMS = [("bare_last", bl), ("bare_mean", bm), ("chat_last", cl),
            ("concat_lc", np.hstack([bl, cl]))]

    # ---- 参照基准线：sklearn 在 M4 上的 fresh OOF（无论选路都算，供对照）----
    ref_sk = []
    for seed in CV_SEEDS:
        a, _, _, _ = probe_oof(Xtr_m4, y, seed, device=None)
        ref_sk.append(a)
        print(f"  [ref-sklearn] M4 seed{seed} AUC={a:.4f}", flush=True)
    ref_sk = np.array(ref_sk)

    # ---- GPU 选取：认证发生在 M4（良态）上，双解应重合 → 工具一致性证明 ----
    device = None
    cert_m4, drift_bl = None, None
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
            stamp(f"[认证] M4 良态参照矩阵 sklearn/torch 对照: "
                  f"max|Δp|={dp:.2e} |ΔAUC|={da:.2e} → "
                  + ("通过，参照+全部臂统一走 GPU torch 求解器" if ok
                     else "未达标，整体回退 CPU sklearn（参照与臂仍同求解器）"))
            if ok:
                device = dev
                # 诊断：4096 维病态臂上的跨求解器漂移（不参与任何门）
                cvb = list(StratifiedKFold(5, shuffle=True,
                                           random_state=CV_SEEDS[0]).split(bl, y))
                trb, vab = cvb[0]
                scb = StandardScaler().fit(bl[trb])
                Btr_s, Bva_s = scb.transform(bl[trb]), scb.transform(bl[vab])
                pb_sk = LogisticRegression(max_iter=2000, C=1.0).fit(
                    Btr_s, y[trb]).predict_proba(Bva_s)[:, 1]
                pb_to, _ = _torch_lr_proba(Btr_s, y[trb], Bva_s, dev)
                drift_bl = (float(np.abs(pb_sk - pb_to).max()),
                            abs(roc_auc_score(y[vab], pb_sk)
                                - roc_auc_score(y[vab], pb_to)))
                stamp(f"[诊断] bare_last 病态臂跨求解器漂移: "
                      f"max|Δp|={drift_bl[0]:.2e} |ΔAUC|={drift_bl[1]:.2e}"
                      "（仅诊断，比较内部无跨求解器）")
    except Exception as e:  # noqa: BLE001 —— 无 torch/CUDA 或认证异常一律回退
        stamp(f"[认证] GPU 不可用（{type(e).__name__}: {e}），"
              "参照与臂统一走 sklearn CPU")
        device = None

    # ---- 参照（与臂同求解器）----
    if device is not None:
        stamp("[参照] M4 × torch GPU 求解器 fresh OOF...")
        ref_main = []
        for seed in CV_SEEDS:
            a, _, _, conv = probe_oof(Xtr_m4, y, seed, device=device)
            ref_main.append(a)
            print(f"  [ref-torch] M4 seed{seed} AUC={a:.4f} "
                  f"grad_max={max(conv):.1e}", flush=True)
        ref_main = np.array(ref_main)
    else:
        ref_main = ref_sk  # CPU 路径下 sklearn 参照即为同求解器参照

    # ---- 四臂 ----
    stamp("[探针] 四臂 × 3 种子（求解器与参照一致）...")
    rows = []
    for name, X in ARMS:
        for seed in CV_SEEDS:
            auc, tau, f1m, conv = probe_oof(X, y, seed, device=device)
            rows.append({"arm": name, "seed": seed, "oof_auc": auc,
                         "tau": tau, "oof_macro": f1m, "conv": float(np.max(conv))})
            print(f"  {name:<10} seed{seed} AUC={auc:.4f} 宏F1@τ={f1m:.4f} "
                  f"τ*={tau} conv={np.max(conv):.1e}", flush=True)
    cvdf = pd.DataFrame(rows)
    cvdf["path"] = "torch-gpu" if device is not None else "sklearn-cpu"
    cvdf.to_csv(out / "probe_cv.csv", index=False, encoding="utf-8-sig")

    g = cvdf.groupby("arm")[["oof_auc", "oof_macro"]].agg(["mean", "std"])
    best = g[("oof_auc", "mean")].idxmax()
    best_auc = g.loc[best, ("oof_auc", "mean")]
    passed = best_auc >= GATE_AUC

    lines = ["=" * 72, "v15 隐状态线性探针——\"知道-说出\"差距判读（求解器匹配版）",
             "=" * 72,
             f"hidden_size={bl.shape[1]}，n={len(y)}；探针=Scaler+L2LR(C=1.0)，"
             f"5折×3种子 OOF",
             "执行路径: " + ("GPU（torch LBFGS float64；M4 良态认证通过，"
                            "参照与臂同求解器）" if device is not None
                            else "CPU（sklearn；参照与臂同求解器）"), ""]
    if cert_m4 is not None:
        lines.append(f"认证（M4 良态）: max|Δp|={cert_m4[0]:.2e} "
                     f"|ΔAUC|={cert_m4[1]:.2e}")
    if drift_bl is not None:
        lines.append(f"诊断（bare_last 病态跨求解器漂移，不参与门）: "
                     f"max|Δp|={drift_bl[0]:.2e} |ΔAUC|={drift_bl[1]:.2e}")
    lines += ["",
              f"口头矩阵参照（{('torch GPU' if device is not None else 'sklearn CPU')}"
              f"，M4 fresh OOF）: AUC = {ref_main.mean():.4f} ± {ref_main.std():.4f}",
              f"  （sklearn 侧对照: {ref_sk.mean():.4f} ± {ref_sk.std():.4f}）", "",
              f"{'臂':<12}{'OOF AUC mean±std':<24}{'OOF 宏F1 mean±std':<24}",
              "-" * 72]
    for name in g.index:
        lines.append(f"{name:<12}"
                     f"{g.loc[name, ('oof_auc', 'mean')]:.4f} ± "
                     f"{g.loc[name, ('oof_auc', 'std')]:.4f}"
                     f"{'':<10}"
                     f"{g.loc[name, ('oof_macro', 'mean')]:.4f} ± "
                     f"{g.loc[name, ('oof_macro', 'std')]:.4f}")
    lines += ["", f"最佳臂: {best}（AUC {best_auc:.4f}）",
              f"相对口头矩阵参照差值: {best_auc - ref_main.mean():+.4f}",
              f"预登记门槛（AUC ≥ {GATE_AUC}）: " + ("通过" if passed else "未通过")]
    conv_max = cvdf["conv"].max()
    if device is not None:
        bad = cvdf[cvdf["conv"] >= GRAD_CONV_TOL]
        lines += [f"收敛性（GPU）: 全部拟合最终梯度∞范数 max={conv_max:.1e}"
                  + ("" if bad.empty else
                     f"；警告: {len(bad)} 次拟合梯度 ≥ {GRAD_CONV_TOL:.0e}"
                     "（未完全收敛，AUC 读数偏保守）")]
    else:
        hit = cvdf[cvdf["conv"] >= 2000 * 0.95]
        lines += [f"收敛性（CPU）: n_iter max={conv_max:.0f}"
                  + ("" if hit.empty else
                     f"；警告: {len(hit)} 次拟合 n_iter 打满 2000"
                     "（未完全收敛，AUC 读数偏保守）")]
    lines += ["", "判读: 通过 → \"LLM 知道的多于说出来的\"证实，进入 CBM 2.0；",
              "      未通过 → 状态内线性可读信息也已耗尽，结论修正为\"教师需微调\"。"]
    report = "\n".join(lines)
    print(report)
    (out / "probe_report.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] probe_report.txt / probe_cv.csv 已写出")


if __name__ == "__main__":
    main()
