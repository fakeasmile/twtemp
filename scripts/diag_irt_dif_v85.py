# -*- coding: utf-8 -*-
"""v8.5 理论诊断表：测量理论侧（IRT/DIF/信息函数）四张表。test 零参与。

背景：v9 证伪了"教师缺意图维度"假设（6 条意图条目全灭门槛），但这只否定
了"新测量模式"路线；"中段专家条目存在但被全局 LR 权重机制埋没"是另一个
未检验假设。本脚本从测量理论侧出四张诊断表，为 R4（DIF + 中带专家交互）
提供预登记证据：

  表1 条目区分度（来自 GRM）：a_i 分布、弱区分条目——CTT 冻结权重 vs
      IRT 信息权重的差距来源。
  表2 θ 信息覆盖：测验信息函数 I(θ) 分带——定位"信息洞"（θ 某段几乎没有
      条目提供信息 → 该段样本的毒性判定本质不可分）。
  表3 中带专家搜索：全局 AUC<0.62 但 θ tercile 带内 AUC≥0.70 的条目——
      被全局机制埋没的中段判别力的直接证据。
  表4 DIF（项目功能差异）：按 expression 组拆分后条目判别方向是否一致——
      全局裁定隐含"组不变"假设，DIF 条目即违反该假设者。

分组（OOF 稳定错误掩码，3 种子均值概率，同 v8.3 审计法）：
  BEN = 全部无毒样本；TP_g = y=1 且判定正确且 expression=g（g=1,2,3）；
  E0FP = y=0 且 expression=0 且误判为毒；TN = y=0 且判定正确。
DIF 对：expr1/2/3-TP vs BEN、E0FP vs TN（组 <30 跳过）。AUC 取折叠方向，
符号 '+' 表示组 1 取值更高。

产物：results/v85/theory_diag/
  {metrics.txt, dif_table.csv, midband_experts.csv, info_bands.csv}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import readout_v83 as r83

MIN_GROUP = 30
MIN_BAND = 100
EXPERT_GLOBAL = 0.62
EXPERT_BAND = 0.70
DIF_DEV = 0.10
INFO_HOLE_RATIO = 0.5   # 带内平均测验信息 < 全局均值一半 → 信息洞


def fold_auc(x1, x2):
    """两组取值的折叠 AUC 与方向符号（'+' = 组 1 更高）。"""
    y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])
    x = np.concatenate([x1, x2])
    a = roc_auc_score(y, x)
    return (max(a, 1.0 - a), 1 if a >= 0.5 else -1)


def main():
    ap = argparse.ArgumentParser(description="v8.5 理论诊断：IRT/DIF/中带专家")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--v3_rings_json", required=True, help="train rings JSON（expression）")
    ap.add_argument("--grm_dir", required=True, help="irt_grm.py 输出目录")
    ap.add_argument("--output_dir", default="results/v85/theory_diag")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    v3tr = pd.read_csv(args.v3_train)
    assert (tr["id"].values == v3tr["id"].values).all()
    ytr = tr["toxic"].astype(int).values

    # ---- GRM 参数与 θ（train）----
    gdir = Path(args.grm_dir)
    gm = np.load(gdir / "grm_params.npz")
    gm_items = [str(x) for x in gm["items"]]
    a_gm = gm["a"]
    th_gm = gm["th"]
    info = gm["info"]                       # (I, Q)
    tot_info = info.sum(axis=0)             # (Q,) 测验信息
    th_tr_df = pd.read_csv(gdir / "theta_train.csv")
    assert (th_tr_df["id"].values == tr["id"].values).all()
    eap = th_tr_df["theta_eap"].values

    con_of = {}
    for cid, kept in freeze["items_per_construct"].items():
        for c in kept:
            con_of[c] = cid
    flip_of = {c: int(freeze["item_polarity"].get(c, 1) == -1)
               for c in gm_items}

    # ---- 条目值（对齐翻转，与 GRM 同尺度）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr = tr[item_cols].copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    assert kept_items == gm_items, "GRM 条目顺序与 freeze 不一致，请检查 freeze 文件"
    Ait = Atr[kept_items].values

    # ---- OOF 稳定错误掩码（Q1 锚点复刻，train 侧）----
    print("复刻 Q1 锚点装配 ...")
    Xtr_raw = tr[item_cols]
    v3tr_all = [c for c in v3tr.columns if c not in ("id", "toxic")]
    concept_cols = [c for c in v3tr_all if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    sample_cols = [c for c in v3tr_all if c.startswith(("s1a_", "s1b_", "cat_"))]
    var_ok = [c for c in concept_cols if v3tr[c].std() > 1e-9]
    pol, auc_v3 = {}, {}
    for c in var_ok:
        a = roc_auc_score(ytr, v3tr[c].values)
        auc_v3[c] = max(a, 1.0 - a)
        pol[c] = 1 if a >= 0.5 else -1
    valid_c = [c for c in var_ok if auc_v3[c] >= r83.V3_MIN_AUC]
    Ftr = v3tr[valid_c].copy()
    for c in valid_c:
        if pol[c] == -1:
            Ftr[c] = 1.0 - v3tr[c]
    kept_c = []
    for c in valid_c:
        r = np.abs(np.corrcoef(Ftr[c].values, Atr[kept_items].values.T)[0, 1:])
        if np.nanmax(r) < r83.V3_MAX_CORR:
            kept_c.append(c)
    kept_s = [c for c in sample_cols if v3tr[c].std() > 1e-9]
    v3_names = kept_c + kept_s
    v3_X = np.hstack([Ftr[kept_c].values, v3tr[kept_s].values])

    prob_names = kept_items + v3_names
    raw_tr = np.hstack([Atr[kept_items].values, v3_X])
    qs = np.linspace(0.0, 1.0, r83.N_QUANT)
    Rall_tr = raw_tr.copy()
    for j in range(raw_tr.shape[1]):
        edges = np.quantile(raw_tr[:, j], qs)
        Rall_tr[:, j] = np.interp(raw_tr[:, j], edges, qs)
    n_items = len(kept_items)
    Atr_items = Atr[kept_items].values

    Mtr, _ = r83.meta_base(tr["content"].tolist(), tr["content"].tolist())
    pbr = np.array([abs(np.corrcoef(Atr_items[:, j], ytr)[0, 1])
                    for j in range(n_items)])
    order_i = np.argsort(-pbr)[:r83.N_TOP_INTERACT]
    meta_tr = np.hstack([Mtr, Mtr[:, 0:1] * Atr_items[:, order_i]])

    macro_raw = r83.construct_raw(tr, freeze, Atr, Xtr_raw)
    mcols = [c for c in macro_raw.columns if c != "id"]
    q1_tr = np.hstack([macro_raw[mcols].values, Rall_tr, meta_tr])

    print("Q1 锚点 OOF（3 种子）...")
    oofs = np.column_stack([r83.cv_select(q1_tr, ytr, s, "diag")[2]
                            for s in r83.CV_SEEDS])
    oof_prob = oofs.mean(axis=1)
    tau_e = max(r83.TAU_GRID, key=lambda t: r83.f1_score(
        ytr, (oof_prob >= t).astype(int), average="macro"))
    err = (oof_prob >= tau_e).astype(int) != ytr

    # ---- 分组掩码 ----
    with open(args.v3_rings_json, "r", encoding="utf-8") as f:
        rings = json.load(f)
    expr_map = {r["id"]: (-1 if r.get("expression") is None else int(r["expression"]))
                for r in rings["results"]}
    expr = np.array([expr_map.get(i, -1) for i in tr["id"].values])
    ben = ytr == 0
    tp = (ytr == 1) & (~err)
    fp0 = (ytr == 0) & err & (expr == 0)
    tn = (ytr == 0) & (~err)
    groups = {"BEN": ben,
              "expr1TP": tp & (expr == 1),
              "expr2TP": tp & (expr == 2),
              "expr3TP": tp & (expr == 3),
              "E0FP": fp0, "TN": tn}

    lines = ["=" * 72, "v8.5 理论诊断（测量理论侧）——train 内，test 零参与", "=" * 72,
             f"Q1 锚点 OOF（τ={tau_e:.3f}）：FN {int((err & (ytr == 1)).sum())}/"
             f"{int((ytr == 1).sum())}  FP {int((err & (ytr == 0)).sum())}/"
             f"{int((ytr == 0).sum())}", ""]

    # ================= 表 1：条目区分度 =================
    pct = np.percentile(a_gm, [10, 25, 50, 75, 90])
    lines += ["【表1】GRM 条目区分度 a_i", "-" * 72,
              f"P10/25/50/75/90: " + "/".join(f"{v:.3f}" for v in pct),
              f"a<0.4: {int((a_gm < 0.4).sum())} 条   0.4-0.7: "
              f"{int(((a_gm >= 0.4) & (a_gm < 0.7)).sum())} 条   0.7-1.2: "
              f"{int(((a_gm >= 0.7) & (a_gm < 1.2)).sum())} 条   >=1.2: "
              f"{int((a_gm >= 1.2).sum())} 条", ""]
    for lbl, sel in (("区分度最高 12", np.argsort(-a_gm)[:12]),
                     ("区分度最低 12", np.argsort(a_gm)[:12])):
        lines.append(f"{lbl}：")
        for i in sel:
            lines.append(f"    {gm_items[i]:<8} ({con_of.get(gm_items[i], '?'):<4}"
                         f"{', 翻转' if flip_of[gm_items[i]] else ''}) a={a_gm[i]:.3f}")
    lines.append("")

    # ================= 表 2：θ 信息覆盖 =================
    bands = [(-4, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 4)]
    rows_band = []
    gmean_info = float(np.mean(tot_info))
    for lo, hi in bands:
        sel_n = (eap >= lo) & (eap < hi)
        gsel = (th_gm >= lo) & (th_gm < hi)
        rows_band.append({"band": f"[{lo},{hi})", "n_train": int(sel_n.sum()),
                          "mean_total_info": float(tot_info[gsel].mean()) if gsel.any() else np.nan,
                          "mean_theta_se_train": float(
                              th_tr_df["theta_se"].values[sel_n].mean()) if sel_n.any() else np.nan})
    ib = pd.DataFrame(rows_band)
    ib["info_hole"] = ib["mean_total_info"] < INFO_HOLE_RATIO * gmean_info
    ib.to_csv(out / "info_bands.csv", index=False, encoding="utf-8-sig")
    lines += ["【表2】θ 信息覆盖（测验信息 I(θ)=ΣI_i(θ)，θ 带 = EAP 分带）", "-" * 72,
              f"全局平均测验信息 {gmean_info:.2f}；信息洞判定：带均值 < "
              f"{INFO_HOLE_RATIO:.0%} × 全局均值", ib.to_string(index=False), ""]

    # ================= 表 3：中带专家搜索 =================
    q33 = np.quantile(eap, [1 / 3, 2 / 3])
    band_masks = [eap <= q33[0], (eap > q33[0]) & (eap <= q33[1]), eap > q33[1]]
    band_tags = ["低带", "中带", "高带"]
    rows_exp = []
    for i, name in enumerate(gm_items):
        x = Ait[:, i]
        g_auc, _ = fold_auc(x[ytr == 1], x[ytr == 0])
        band_aucs = []
        for bm in band_masks:
            if (ytr[bm] == 1).sum() >= MIN_BAND and (ytr[bm] == 0).sum() >= MIN_BAND:
                ba, _ = fold_auc(x[bm & (ytr == 1)], x[bm & (ytr == 0)])
            else:
                ba = np.nan
            band_aucs.append(ba)
        bmax = np.nanmax(band_aucs)
        if g_auc < EXPERT_GLOBAL and bmax >= EXPERT_BAND:
            rows_exp.append({"item": name, "construct": con_of.get(name, "?"),
                             "flip": flip_of[name], "global_auc": round(g_auc, 4),
                             "auc_low": round(band_aucs[0], 4),
                             "auc_mid": round(band_aucs[1], 4),
                             "auc_high": round(band_aucs[2], 4),
                             "max_band_auc": round(bmax, 4)})
    exp_df = pd.DataFrame(rows_exp)
    if len(exp_df):
        exp_df = exp_df.sort_values("max_band_auc", ascending=False)
    exp_df.to_csv(out / "midband_experts.csv", index=False, encoding="utf-8-sig")
    n_mid_only = int(((exp_df["auc_mid"] >= EXPERT_BAND) &
                      (exp_df["auc_low"] < EXPERT_BAND) &
                      (exp_df["auc_high"] < EXPERT_BAND)).sum()) if len(exp_df) else 0
    lines += [f"【表3】中带专家条目（全局 AUC<{EXPERT_GLOBAL} 且某 θ 三分带 AUC≥"
              f"{EXPERT_BAND}；带 n≥{MIN_BAND}）", "-" * 72,
              f"θ 三分位切点: {q33[0]:.3f} / {q33[1]:.3f}   命中 {len(exp_df)} 条"
              f"（其中仅中带达标 {n_mid_only} 条）"]
    if len(exp_df):
        lines.append(exp_df.head(20).to_string(index=False))
    lines.append("")

    # ================= 表 4：DIF =================
    pairs = [("expr1TP", "BEN"), ("expr2TP", "BEN"), ("expr3TP", "BEN"),
             ("E0FP", "TN")]
    rows_dif = []
    for i, name in enumerate(gm_items):
        x = Ait[:, i]
        g_auc, _ = fold_auc(x[ytr == 1], x[ytr == 0])
        row = {"item": name, "construct": con_of.get(name, "?"),
               "flip": flip_of[name], "global_auc": round(g_auc, 4)}
        max_dev = 0.0
        for g1, g2 in pairs:
            m1, m2 = groups[g1], groups[g2]
            if m1.sum() < MIN_GROUP or m2.sum() < MIN_GROUP:
                row[f"dif_{g1}_vs_{g2}"] = np.nan
                row[f"sign_{g1}_vs_{g2}"] = ""
                continue
            a1, s1 = fold_auc(x[m1], x[m2])
            row[f"dif_{g1}_vs_{g2}"] = round(a1, 4)
            row[f"sign_{g1}_vs_{g2}"] = "+" if s1 == 1 else "-"
            max_dev = max(max_dev, abs(a1 - 0.5))
        row["max_dif_dev"] = round(max_dev, 4)
        rows_dif.append(row)
    dif = pd.DataFrame(rows_dif).sort_values("max_dif_dev", ascending=False)
    dif.to_csv(out / "dif_table.csv", index=False, encoding="utf-8-sig")
    lines += [f"【表4】DIF（expression 组间条目功能差异；组 n≥{MIN_GROUP}）", "-" * 72,
              f"{'对':<24}{'n组1':<8}{'n组2':<8}{'|AUC-0.5|≥0.10':<16}{'≥0.15':<8}{'top 条目'}"]
    for g1, g2 in pairs:
        col = f"dif_{g1}_vs_{g2}"
        if col not in dif.columns or dif[col].isna().all():
            lines.append(f"  {g1+' vs '+g2:<24}组过小，跳过")
            continue
        sub = dif.dropna(subset=[col])
        n10 = int((sub[col] - 0.5).abs().ge(DIF_DEV).sum())
        n15 = int((sub[col] - 0.5).abs().ge(0.15).sum())
        top = sub.iloc[0]
        lines.append(f"  {g1+' vs '+g2:<22}{int(groups[g1].sum()):<8}"
                     f"{int(groups[g2].sum()):<8}{n10:<16}{n15:<8}"
                     f"{top['item']}({top[col]:.3f}{top[f'sign_{g1}_vs_{g2}']})")
    n_dif = int((dif["max_dif_dev"] >= DIF_DEV).sum())
    n_flip_cnt = int(((dif["max_dif_dev"] >= DIF_DEV) & (dif["flip"] == 1)).sum())
    lines += ["",
              f"全表：DIF 条目（max_dev≥{DIF_DEV}）{n_dif} 条；"
              f"其中极性翻转条目 {n_flip_cnt} 条",
              "",
              "判读指南（预登记）：",
              "  表2 出现信息洞 → 该 θ 段样本不可分，需中带专家条目或新测量补洞；",
              "  表3 命中>=5 条 → '中段专家被全局权重埋没'假设成立，R4 交互项值得做；",
              "  表4 高 DIF 集中在 expr2TP vs BEN 且符号混杂 → 组条件化（分组裁定）有据。"]

    report = "\n".join(lines)
    print(report)
    (out / "metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
