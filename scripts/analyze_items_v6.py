# -*- coding: utf-8 -*-
"""v6 条目分析脚本（train 上唯一裁决，test 零接触）。

设计文档 §5 全部产出：构念内 α 门槛 ≥ 0.70、校正条目-总分相关 r < 0.30 淘汰、
反向条目表决（作为构念条目 vs 仅效度探针）、19×19 分层构念相关矩阵、
§7.1 全部经验检查项、构念判死表决、输出 v6.1 条目子集冻结文件。

使用：python scripts/analyze_items_v6.py \\
  --matrix data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv \\
  --items_csv data/concepts/cbm_concept_set_v6.csv \\
  --output_dir results/v6/item_analysis_glm_train
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def auc(score, yy):
    """Mann-Whitney U AUC，score 大→toxic1。"""
    r = pd.Series(score).rank().values
    n1, n0 = (yy == 1).sum(), (yy == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cronbach_alpha(M: pd.DataFrame) -> float:
    k = M.shape[1]
    if k < 2:
        return np.nan
    var_rows = M.sum(axis=1).var(ddof=1)
    if var_rows <= 0:
        return np.nan
    return k / (k - 1) * (1 - M.var(ddof=1).sum() / var_rows)


def corrected_item_total_corr(M: pd.DataFrame):
    """校正条目-总分相关：删该条目后剩余均值与该条目的 Pearson r。"""
    out = {}
    for col in M.columns:
        rest = M.drop(columns=[col]).mean(axis=1)
        out[col] = np.corrcoef(M[col].values, rest.values)[0, 1]
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="v6 条目分析（train）")
    parser.add_argument("--matrix", required=True, help="train matrix CSV（条目测量脚本输出）")
    parser.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    parser.add_argument("--output_dir", default="results/v6/item_analysis")
    parser.add_argument("--alpha_threshold", type=float, default=0.70)
    parser.add_argument("--citc_threshold", type=float, default=0.30)
    parser.add_argument("--high_trigger", type=float, default=0.80,
                        help="无条件高触发条目阈值（设计文档 §7.1 项 4）")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items_df = pd.read_csv(args.items_csv, encoding="utf-8-sig")
    items = items_df[["条目id", "构念id", "构念名", "区块", "题式", "方向"]].copy()
    items.columns = ["item", "construct", "cname", "block", "form", "direction"]
    item_ids = items["item"].tolist()

    df = pd.read_csv(args.matrix)
    for c in item_ids:
        df[c] = df[c].astype(float)
    y = df["toxic"].astype(int).values
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    length = df["content"].astype(str).str.len().values
    X = df[item_ids]
    n = len(df)
    rev_set = set(items[items["direction"] == "反向"]["item"])
    print(f"=== v6 条目分析（train 唯一裁决点） ===")
    print(f"n={n}  toxic1={n1}  toxic0={n0}  α门槛≥{args.alpha_threshold}  CITC门槛≥{args.citc_threshold}")

    # 反向计分矩阵（不改动原矩阵，各构念内部分别处理）
    R = X.copy()
    R.loc[:, X.columns.isin(rev_set)] = 1.0 - R.loc[:, X.columns.isin(rev_set)]

    # ---------- 1. 构念级诊断 ----------
    construct_scores = {}  # cid -> array
    construct_keep = {}    # cid -> list of item_ids 被保留
    decisions = []         # 每构念的裁决记录

    all_constructs = list(items.groupby("construct", sort=False))

    for cid, g in all_constructs:
        g_items = g["item"].tolist()
        pos_items = g[g["direction"] == "正向"]["item"].tolist()
        rev_items = g[g["direction"] == "反向"]["item"].tolist()

        Mall = R[g_items]          # 反向已计分
        Mpos = R[pos_items]        # 正向（计分与未计分一致）
        Mrev = R[rev_items]        # 反向（已计分：1-p）

        a_all = cronbach_alpha(Mall)
        a_pos = cronbach_alpha(Mpos)
        a_rev = cronbach_alpha(Mrev) if len(rev_items) >= 2 else np.nan

        citc_all = corrected_item_total_corr(Mall)
        citc_pos = corrected_item_total_corr(Mpos)

        # 表决：反向条目是进构念分还是仅效度探针？
        # 条件：①正向-only α ≥ 0.70 且 ②全条目 α 显著低于正向-only（< 正向-only - 0.10 或 < 0.70）
        reverse_invalid = (a_pos >= args.alpha_threshold) and (
            (np.isnan(a_all) or a_all < args.alpha_threshold)
            or a_all < a_pos - 0.10
        )

        # 候选条目：若反向失败则只用正向，否则全条目
        candidate_items = pos_items if reverse_invalid else g_items
        Mcand = (R[pos_items] if reverse_invalid else Mall)

        # 再用 CITC 在候选集中淘汰（r < 0.30）
        citc_cand = corrected_item_total_corr(Mcand)
        keep_items = [it for it in candidate_items if citc_cand.get(it, 0) >= args.citc_threshold]

        if len(keep_items) < 2:
            # 极端情况：构念判死（条目太少没法聚合）
            construct_scores[cid] = np.full(n, np.nan)
            construct_keep[cid] = []
            decisions.append({
                "construct": cid,
                "cname": g["cname"].iloc[0],
                "block": g["block"].iloc[0],
                "n_pos": len(pos_items),
                "n_rev": len(rev_items),
                "alpha_full": round(a_all, 3) if not np.isnan(a_all) else None,
                "alpha_pos": round(a_pos, 3),
                "alpha_rev": round(a_rev, 3) if not np.isnan(a_rev) else None,
                "reverse_downgraded": reverse_invalid,
                "citc_weak": [it for it, v in citc_cand.items() if v < args.citc_threshold],
                "kept_items": [],
                "n_kept": 0,
                "alpha_final": None,
                "auc_full_items": round(auc(Mall.mean(axis=1).values, y), 3),
                "auc_pos_only": round(auc(Mpos.mean(axis=1).values, y), 3),
                "auc_final": None,
                "decision": "DEAD（保留条目<2）",
            })
            continue

        Mkeep = R[keep_items]
        a_final = cronbach_alpha(Mkeep)
        S_final = Mkeep.mean(axis=1).values
        construct_scores[cid] = S_final
        construct_keep[cid] = keep_items
        auc_final = auc(S_final, y)

        # 构念判死表决：保留后 α < 0.70 或 AUC ≤ 0.55
        dead = (a_final < args.alpha_threshold) or (auc_final <= 0.55)
        decision = "KEEP" if not dead else f"DEAD（α={a_final:.3f} AUC={auc_final:.3f}）"

        decisions.append({
            "construct": cid,
            "cname": g["cname"].iloc[0],
            "block": g["block"].iloc[0],
            "n_pos": len(pos_items),
            "n_rev": len(rev_items),
            "alpha_full": round(a_all, 3) if not np.isnan(a_all) else None,
            "alpha_pos": round(a_pos, 3),
            "alpha_rev": round(a_rev, 3) if not np.isnan(a_rev) else None,
            "reverse_downgraded": reverse_invalid,
            "citc_weak": [it for it, v in citc_cand.items() if v < args.citc_threshold],
            "kept_items": keep_items,
            "n_kept": len(keep_items),
            "alpha_final": round(a_final, 3),
            "auc_full_items": round(auc(Mall.mean(axis=1).values, y), 3),
            "auc_pos_only": round(auc(Mpos.mean(axis=1).values, y), 3),
            "auc_final": round(auc_final, 3),
            "decision": decision,
        })

    dec_df = pd.DataFrame(decisions)
    dec_path = out / "01_construct_decisions.csv"
    dec_df.to_csv(dec_path, index=False, encoding="utf-8-sig")

    # ---------- 2. 打印构念裁决 ----------
    print("\n" + "=" * 80)
    print("构念裁决总表")
    print(f"{'构念':4s} {'区块':1s} {'α全':>6s} {'α正':>6s} {'α反':>6s} {'反向降级':>6s} "
          f"{'保留条数':>8s} {'α保留':>6s} {'AUC保留':>7s} {'决策':20s}")
    for _, r in dec_df.iterrows():
        a_all = f"{r['alpha_full']:.3f}" if r["alpha_full"] is not None else "  n/a"
        a_rev = f"{r['alpha_rev']:.3f}" if r["alpha_rev"] is not None else "  n/a"
        auc_f = f"{r['auc_final']:.3f}" if r["auc_final"] is not None else "  n/a"
        print(f"{r['construct']:4s} {r['block']:1s} {a_all:>6s} {r['alpha_pos']:>6.3f} {a_rev:>6s} "
              f"{'是' if r['reverse_downgraded'] else '否':>6s} {r['n_kept']:>8d} "
              f"{r['alpha_final'] if r['alpha_final'] is not None else -1:>6.3f} "
              f"{auc_f:>7s} {r['decision']:20s}")

    # ---------- 3. 条目级详情 ----------
    item_rows = []
    for cid, g in all_constructs:
        ckeep = construct_keep[cid]
        cand_items = (g[g["direction"] == "正向"]["item"].tolist()
                      if dec_df[dec_df["construct"] == cid]["reverse_downgraded"].iloc[0]
                      else g["item"].tolist())
        Mcand = (R[g[g["direction"] == "正向"]["item"].tolist()]
                 if dec_df[dec_df["construct"] == cid]["reverse_downgraded"].iloc[0]
                 else R[g["item"].tolist()])
        citc = corrected_item_total_corr(Mcand)
        cands_set = set(cand_items)
        for _, it in g.iterrows():
            raw_p = X[it["item"]]
            reversed_p = R[it["item"]]
            is_kept = it["item"] in set(ckeep)
            cits = citc.get(it["item"], np.nan) if it["item"] in cands_set else np.nan
            item_rows.append({
                "construct": cid,
                "item": it["item"],
                "form": it["form"],
                "direction": it["direction"],
                "mean_p": round(raw_p.mean(), 4),
                "std_p": round(raw_p.std(), 4),
                "mean_t1": round(raw_p[y == 1].mean(), 4),
                "mean_t0": round(raw_p[y == 0].mean(), 4),
                "d_t1_t0": round(raw_p[y == 1].mean() - raw_p[y == 0].mean(), 4),
                "reversed_mean": round(reversed_p.mean(), 4),
                "citc": round(cits, 4) if not np.isnan(cits) else None,
                "auc_single": round(auc(reversed_p.values, y), 3),
                "kept": is_kept,
            })
    item_df = pd.DataFrame(item_rows)
    item_path = out / "02_item_details.csv"
    item_df.to_csv(item_path, index=False, encoding="utf-8-sig")
    print(f"\n条目详情 → {item_path}")

    # ---------- 4. §7.1 检查项 ----------
    checks = {}

    # 4a. 反向条目反常触发
    rev_rows = item_df[item_df["direction"] == "反向"].copy()
    rev_anom = rev_rows[rev_rows["mean_t1"] > rev_rows["mean_t0"] + 0.02]
    checks["reverse_anomaly"] = {
        "n_anomalous": len(rev_anom),
        "n_total_reverse": len(rev_rows),
        "items": rev_anom[["item", "construct", "mean_t1", "mean_t0", "d_t1_t0"]].to_dict(orient="records"),
    }

    # 4b. 无条件高触发
    high_trig = item_df[item_df["mean_p"] >= args.high_trigger]
    checks["high_trigger"] = {
        "threshold": args.high_trigger,
        "n_items": len(high_trig),
        "items": high_trig[["item", "construct", "form", "direction", "mean_p", "auc_single"]]
            .sort_values("mean_p", ascending=False).to_dict(orient="records"),
    }

    # 4c. 长度混杂
    length_all = float(np.corrcoef(length, y)[0, 1])
    cons_len = []
    alive = [c for c, s in construct_scores.items() if not np.isnan(s).all()]
    for c in alive:
        cons_len.append({
            "construct": c,
            "r_length": round(float(np.corrcoef(length, construct_scores[c])[0, 1]), 4),
            "auc": round(auc(construct_scores[c], y), 3),
        })
    cons_len.sort(key=lambda r: -abs(r["r_length"]))
    checks["length_confounding"] = {
        "length_toxic_r": round(length_all, 4),
        "t1_mean_len": round(float(length[y == 1].mean()), 1),
        "t0_mean_len": round(float(length[y == 0].mean()), 1),
        "top_items_by_len_corr": cons_len[:5],
        "all": cons_len,
    }

    # 4d. 语用框架条目失效：B 区 + D1 中高触发条目的 AUC 检查
    prag_items = item_df[(item_df["construct"].str.startswith("B")) | (item_df["construct"] == "D1")]
    prag_fail = prag_items[(prag_items["mean_p"] >= args.high_trigger)
                           & (prag_items["auc_single"] < 0.70)]
    checks["pragmatic_framework_check"] = {
        "n_high_trig_low_auc": len(prag_fail),
        "items": prag_fail[["item", "construct", "mean_p", "auc_single"]].to_dict(orient="records"),
    }

    # 4e. 反向条目整体方向（原始 p_yes，不反转）
    rev_raw_mean = X[list(rev_set)].mean(axis=1).values
    checks["reverse_overall_raw_direction"] = {
        "auc_raw": round(auc(rev_raw_mean, y), 3),
        "t1_mean_raw": round(float(rev_raw_mean[y == 1].mean()), 4),
        "t0_mean_raw": round(float(rev_raw_mean[y == 0].mean()), 4),
        "interpretation": ("反向问句整体方向混乱或反了（v3 guard 共线类）"
                           if auc(rev_raw_mean, y) <= 0.55
                           else "反向问句整体方向偏正确但弱"),
    }

    checks_path = out / "03_v7_1_checks.json"
    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    # ---------- 5. 构念相关矩阵（全样本 + 分层） ----------
    alive_cids = [c for c, s in construct_scores.items() if not np.isnan(s).all()]
    Sall = np.stack([construct_scores[c] for c in alive_cids], axis=1)
    C_all = np.corrcoef(Sall.T)
    pd.DataFrame(C_all, index=alive_cids, columns=alive_cids).to_csv(
        out / "04_construct_corr_all.csv", encoding="utf-8-sig")

    for tag, mask in [("toxic1", y == 1), ("toxic0", y == 0)]:
        Ss = np.stack([construct_scores[c][mask] for c in alive_cids], axis=1)
        Cs = np.corrcoef(Ss.T)
        pd.DataFrame(Cs, index=alive_cids, columns=alive_cids).to_csv(
            out / f"04_construct_corr_{tag}.csv", encoding="utf-8-sig")

    # ---------- 6. 输出 v6.1 条目子集 ----------
    v61_rows = []
    for _, it in items_df.iterrows():
        cid = it["构念id"]
        if cid not in construct_keep:
            continue
        v61_rows.append({
            "构念id": cid,
            "构念名": it["构念名"],
            "区块": it["区块"],
            "条目id": it["条目id"],
            "题式": it["题式"],
            "方向": it["方向"],
            "条目问句": it["条目问句"],
            "kept_in_v6_1": it["条目id"] in set(construct_keep[cid]),
            "reverse_downgraded": dec_df[dec_df["construct"] == cid]["reverse_downgraded"].iloc[0],
        })
    v61_df = pd.DataFrame(v61_rows)
    v61_path = out / "v6_1_item_subset.csv"
    v61_df.to_csv(v61_path, index=False, encoding="utf-8-sig")

    # 冻结摘要 JSON（进后续聚合脚本）
    freeze = {
        "version": "v6.1",
        "basis": "train 唯一裁决点，test 零接触",
        "alpha_threshold": args.alpha_threshold,
        "citc_threshold": args.citc_threshold,
        "constructs_alive": alive_cids,
        "constructs_dead": [r["construct"] for _, r in dec_df.iterrows() if "DEAD" in r["decision"]],
        "items_per_construct": {c: construct_keep[c] for c in alive_cids},
        "reverse_items_role": {
            r["construct"]: bool(r["reverse_downgraded"])
            for _, r in dec_df.iterrows() if r["construct"] in alive_cids
        },
        "meta": {
            "n_train": n,
            "n_toxic1": int(n1),
            "n_toxic0": int(n0),
            "total_items_in_v6_1": sum(len(vs) for vs in construct_keep.values()),
            "reverse_items_not_in_scores": sorted(rev_set - {it for vs in construct_keep.values() for it in vs}),
        },
    }
    freeze_path = out / "v6_1_freeze.json"
    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(freeze, f, ensure_ascii=False, indent=2)

    # ---------- 7. 打印摘要 ----------
    n_alive = len(alive_cids)
    n_dead = 19 - n_alive
    n_total_items = sum(len(vs) for vs in construct_keep.values())
    n_rev_downgraded = int(dec_df["reverse_downgraded"].sum())
    print("\n" + "=" * 80)
    print(f"v6.1 冻结摘要：存活构念 {n_alive}/19  判死 {n_dead}  "
          f"保留条目总计 {n_total_items}  反向降级为效度探针的构念 {n_rev_downgraded}")
    print(f"裁决文件 → {dec_path}")
    print(f"条目详情 → {item_path}")
    print(f"§7.1 检查 → {checks_path}")
    print(f"v6.1 条目子集 → {v61_path}")
    print(f"v6.1 冻结摘要 → {freeze_path}")


if __name__ == "__main__":
    main()
