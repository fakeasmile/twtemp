# -*- coding: utf-8 -*-
"""v16 CBM 概念重建诊断：冻结骨干 post-hoc CBM（线性概念头 + 线性读出）。

【文献依据】Koh et al. 2020 (ICML) CBM：x → 概念头 → ĉ → 最终分类器 → ŷ，
两段式训练（先概念头后读出）；Yuksekgonul et al. 2023 (ICLR) PCBM：骨干
冻结、只训概念头与读出，概念头输入用其训练集内预测（in-sample）训练读出
——本脚本照此标准两段式。概念监督来源 = 口头测量矩阵 M4（回归/蒸馏版，
L_concept = MSE(h→ĉ, M4)）。

【管线（train 全量，test 零参与；全部线性，CPU sklearn）】
  对每臂 h、每种子（42/43/44）的每外折：
    ① 概念头（线性变换）：StandardScaler + RidgeCV(alphas 网格 LOO-GCV)
       多目标拟合 h_tr → M4_tr（264 维），输出 ĉ_tr（in-sample）与 ĉ_va
    ② 读出（线性分类器）：StandardScaler + L2-LR(C=1.0) 在 ĉ_tr 上拟合
       → ĉ_va 的 OOF 概率（两段式，PCBM 同构）
    ③ 对照臂：A = M4 直接 LR（同折）；C = h 直接 LR（同折）；
       D = concat(h, ĉ) LR（同折）
  外折 = StratifiedKFold(5)；同一 (seed, fold) 划分跨臂共用 → 配对差值。

【预登记判读（无门槛竞争，描述性诊断）】
  Δ(B−A)：重建概念读出 vs 口头矩阵原值读出 = "单次前向 h 的线性内容"
       相对 152 查询集成测量的信息损失（0 = 集成信息全部线性存在于 h）
  Δ(B−C)：概念监督的结构增益（>0 = CBM 结构帮助读出）
  Δ(D−max(B,C))：联合增益
  决策：B ≥ A − 0.002 → 概念层可行，CBM 值得搭；
       B 更低 → 重建有损，损失量 = 上述差值，CBM 上限受限
  附：逐列 OOF R²（重建保真度），前 K1 列为宏观构念块（带名，逐值校验）。

产物：results/v16/cbm_recon/{recon_report.txt, recon_detail.csv,
recon_summary.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
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
ALPHAS = np.logspace(-2, 4, 13)
GATE_DELTA = -0.002  # B 相对 A 的可接受损失


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def lr_probe(Xtr, ytr, Xva):
    m = Pipeline_LR()
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], int(np.ravel(m[-1].n_iter_)[0])


def Pipeline_LR():
    from sklearn.pipeline import Pipeline
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=2000))])


def macro_names(tr, freeze, Xtr):
    """复算宏观构念块列名并逐值校验块边界；不匹配则退化通用名（仅影响
    报告命名，不影响任何测量）。"""
    try:
        item_cols = [c for c in tr.columns
                     if c not in ("id", "toxic", "content")]
        Atr = tr[item_cols].copy()
        for c in item_cols:
            if freeze["item_polarity"].get(c, 1) == -1:
                Atr[c] = 1.0 - tr[c]
        macro = r83.construct_raw(tr, freeze, Atr, tr[item_cols])
        mcols = [c for c in macro.columns if c != "id"]
        M = macro[mcols].values
        if M.shape[1] <= Xtr.shape[1] and np.allclose(
                Xtr[:, :M.shape[1]], M, atol=1e-8, equal_nan=True):
            return mcols, True
        stamp("[命名] 宏观块校验不匹配，退化通用列名")
        return [], False
    except Exception as e:  # noqa: BLE001
        stamp(f"[命名] 校验异常（{type(e).__name__}: {e}），退化通用列名")
        return [], False


def main():
    ap = argparse.ArgumentParser(description="v16 CBM 概念重建诊断")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v15_hidden_dir", default="results/v15/hidden")
    ap.add_argument("--scaffold_dir", default="results/v15/scaffold")
    ap.add_argument("--scaffold_layer", type=int, default=20)
    ap.add_argument("--output_dir", default="results/v16/cbm_recon")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    assert {"item_polarity", "items_per_construct"} <= set(freeze.keys()), \
        "freeze 文件缺少必需键"
    y = tr["toxic"].astype(int).values
    ids = np.array([str(i) for i in tr["id"].values])

    stamp("[装配] M4 264 维 fresh 重算（概念监督目标 + 参照臂 A）...")
    Xtr_m4, _ = assemble_m4(tr, te, v3tr, v3te, freeze, y)
    assert len(Xtr_m4) == len(y) and np.isfinite(Xtr_m4).all()
    names, ok = macro_names(tr, freeze, Xtr_m4)
    colnames = (list(names) + [f"c{j}" for j in range(len(names),
                                                     Xtr_m4.shape[1])])
    stamp(f"[装配] M4 dim={Xtr_m4.shape[1]}（宏观构念块 K1={len(names)} "
          f"{'已校验' if ok else '未校验'}）")

    # ---- 臂加载 ----
    arms = {}
    z15 = np.load(Path(args.v15_hidden_dir) / "hidden_train.npz")
    assert (z15["ids"] == ids).all(), "v15 npz id 不对齐"
    for key in ("bare_mean", "bare_last"):
        X = z15[key].astype(np.float32)
        assert np.isfinite(X).all()
        arms[key] = X
    sc_path = Path(args.scaffold_dir) / "scaffold_train.npz"
    if sc_path.exists():
        zs = np.load(sc_path)
        assert (zs["ids"] == ids).all(), "scaffold npz id 不对齐"
        L = zs["layers"].shape[1]
        k = args.scaffold_layer if L > args.scaffold_layer else L // 2
        stamp(f"[加载] scaffold 层槽 S{k}/{L - 1}（中层代表）")
        X = zs["layers"][:, k, :].astype(np.float32)
        assert np.isfinite(X).all()
        arms[f"scaffold_S{k}"] = X
    else:
        stamp("[加载] scaffold npz 不存在，跳过中层臂")
    for name, X in arms.items():
        stamp(f"[臂] {name} dim={X.shape[1]}")

    # ---- 预生成交折划分（跨臂共用 → 配对）----
    folds_by_seed = {}
    for seed in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        folds_by_seed[seed] = list(cv.split(Xtr_m4, y))

    # ---- 参照臂 A：M4 直接 LR ----
    stamp("[臂A] M4 直接 LR（参照）...")
    oof_A = {seed: np.zeros(len(y)) for seed in SEEDS}
    nits_A = []
    folds_A = [(s, trn, va) for s in SEEDS
               for trn, va in folds_by_seed[s]]
    for s, trn, va in tqdm(folds_A, desc="臂A[M4直接LR]", dynamic_ncols=True):
        p, nit = lr_probe(Xtr_m4[trn], y[trn], Xtr_m4[va])
        oof_A[s][va] = p
        nits_A.append(nit)
    auc_A = {seed: roc_auc_score(y, oof_A[seed]) for seed in SEEDS}
    stamp(f"[臂A] AUC={np.mean(list(auc_A.values())):.4f} "
          f"(n_iter均={np.mean(nits_A):.0f})")

    # ---- 各 h 臂：重建 + 两段式读出 ----
    results = []
    detail_rows = []
    for name, X in tqdm(arms.items(), desc="臂循环", dynamic_ncols=True):
        stamp(f"[{name}] 重建（RidgeCV 多目标）+ 两段式读出...")
        oof_B = {seed: np.zeros(len(y)) for seed in SEEDS}
        oof_D = {seed: np.zeros(len(y)) for seed in SEEDS}
        oof_C = {seed: np.zeros(len(y)) for seed in SEEDS}
        c_full = {seed: np.zeros((len(y), Xtr_m4.shape[1]))
                  for seed in SEEDS}  # 逐折 OOF 重建，供 R²
        r2_cols = {seed: np.zeros(Xtr_m4.shape[1]) for seed in SEEDS}
        alphas_used, nits_B, nits_C = [], [], []
        folds_h = [(s, trn, va) for s in SEEDS
                   for trn, va in folds_by_seed[s]]
        pbar = tqdm(folds_h, desc=f"[{name}]重建+读出", dynamic_ncols=True)
        for s, trn, va in pbar:
            sc = StandardScaler().fit(X[trn])
            ridge = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[trn]),
                                               Xtr_m4[trn])
            alphas_used.append(float(ridge.alpha_))
            c_tr = ridge.predict(sc.transform(X[trn]))
            c_va = ridge.predict(sc.transform(X[va]))
            c_full[s][va] = c_va
            # ② 读出（两段式：LR 在 in-sample ĉ_tr 上拟合，PCBM 同构）
            p, nit = lr_probe(c_tr, y[trn], c_va)
            oof_B[s][va] = p
            nits_B.append(nit)
            # ③ 对照：h 直接 / concat
            p, nit = lr_probe(X[trn], y[trn], X[va])
            oof_C[s][va] = p
            nits_C.append(nit)
            p, _ = lr_probe(np.hstack([X[trn], c_tr]), y[trn],
                            np.hstack([X[va], c_va]))
            oof_D[s][va] = p
            pbar.set_postfix({"B_auc": f"{roc_auc_score(y, oof_B[s]):.4f}",
                              "C_auc": f"{roc_auc_score(y, oof_C[s]):.4f}"})
        # 逐列 OOF R²（逐折 OOF 重建值，按种子池化后跨种子平均）
        for seed in SEEDS:
            ss_res = ((c_full[seed] - Xtr_m4) ** 2).sum(0)
            ss_tot = ((Xtr_m4 - Xtr_m4.mean(0)) ** 2).sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2_cols[seed] = np.where(ss_tot > 1e-12,
                                         1.0 - ss_res / ss_tot, np.nan)
        r2_mean = np.nanmean(np.vstack([r2_cols[s] for s in SEEDS]), axis=0)
        for j in range(len(r2_mean)):
            detail_rows.append({"arm": name, "col": colnames[j],
                                "col_idx": j, "oof_r2": r2_mean[j]})
        auc_B = {seed: roc_auc_score(y, oof_B[seed]) for seed in SEEDS}
        auc_C = {seed: roc_auc_score(y, oof_C[seed]) for seed in SEEDS}
        auc_D = {seed: roc_auc_score(y, oof_D[seed]) for seed in SEEDS}
        r = {"arm": name,
             "auc_A": float(np.mean(list(auc_A.values()))),
             "auc_B": float(np.mean(list(auc_B.values()))),
             "auc_C": float(np.mean(list(auc_C.values()))),
             "auc_D": float(np.mean(list(auc_D.values()))),
             "auc_B_std": float(np.std(list(auc_B.values()))),
             "delta_B_minus_A": float(np.mean(list(auc_B.values()))
                                      - np.mean(list(auc_A.values()))),
             "delta_B_minus_C": float(np.mean(list(auc_B.values()))
                                      - np.mean(list(auc_C.values()))),
             "delta_D": float(np.mean(list(auc_D.values()))
                              - max(np.mean(list(auc_B.values())),
                                    np.mean(list(auc_C.values())))),
             "r2_median": float(np.nanmedian(r2_mean)),
             "r2_mean": float(np.nanmean(r2_mean)),
             "alpha_median": float(np.median(alphas_used)),
             "n_iter_B": float(np.mean(nits_B)),
             "n_iter_C": float(np.mean(nits_C))}
        results.append(r)
        stamp(f"[{name}] B={r['auc_B']:.4f}(±{r['auc_B_std']:.4f}) "
              f"C={r['auc_C']:.4f} D={r['auc_D']:.4f} "
              f"R2中位={r['r2_median']:.3f} α中位={r['alpha_median']:.2g}")

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(out / "recon_detail.csv", index=False, encoding="utf-8-sig")
    res = pd.DataFrame(results).sort_values("auc_B", ascending=False)
    res.to_csv(out / "recon_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 报告 ----
    best = res.iloc[0]
    verdict = ("概念层可行：最佳臂重建信息损失可忽略（Δ(B−A) ≥ "
               f"{GATE_DELTA}），CBM 值得搭"
               if best["delta_B_minus_A"] >= GATE_DELTA else
               "重建有损：口头集成测量中存在单次前向 h 线性不含的信息，"
               "CBM 概念层上限受限（损失量见 Δ(B−A)）")
    lines = ["=" * 72,
             "v16 CBM 概念重建诊断（冻结骨干 post-hoc，线性概念头+线性读出）",
             "=" * 72,
             f"概念监督 = M4（{Xtr_m4.shape[1]} 维，fresh 重算）；两段式"
             f"（RidgeCV 概念头 → LR 读出，PCBM 同构）；5折×3种子 OOF；"
             f"test 零参与",
             f"参照臂 A（M4 直接 LR，同折配对）: {best['auc_A']:.4f}", ""]
    lines.append(f"{'臂':<16}{'B=重建读出':<18}{'C=h直接':<18}"
                 f"{'D=concat':<18}{'R2中位':<10}{'Δ(B−A)':<10}")
    lines.append("-" * 72)
    for _, r in res.iterrows():
        lines.append(f"{r['arm']:<16}{r['auc_B']:.4f}±{r['auc_B_std']:.4f}"
                     f"{'':<4}{r['auc_C']:.4f}{'':<8}{r['auc_D']:.4f}"
                     f"{'':<6}{r['r2_median']:.3f}{'':<4}"
                     f"{r['delta_B_minus_A']:+.4f}")
    top = detail.sort_values("oof_r2", ascending=False).head(5)
    bot = detail.sort_values("oof_r2").head(5)
    lines += ["", "重建最好 5 列: " + ", ".join(
        f"{r['arm']}/{r['col']}({r['oof_r2']:.2f})"
        for _, r in top.iterrows()),
        "重建最差 5 列: " + ", ".join(
            f"{r['arm']}/{r['col']}({r['oof_r2']:.2f})"
            for _, r in bot.iterrows()),
        "", f"判读: {verdict}",
        "注: Δ(B−C) = 概念监督结构增益；Δ(D) = concat 联合增益（见 summary csv）"]
    report = "\n".join(lines)
    print(report)
    (out / "recon_report.txt").write_text(report + "\n", encoding="utf-8")
    stamp("[完成] recon_report.txt / recon_detail.csv / recon_summary.csv "
          "已写出")


if __name__ == "__main__":
    main()
