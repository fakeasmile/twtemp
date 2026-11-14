# -*- coding: utf-8 -*-
"""v11 Stage3：use/mention 测量的预登记门槛判读（CPU）。

聚合（每样本，未命中样本填中性值 has_occ=0 / p_use=0.5）：
  has_occ, n_occ, p_use_max/mean/min/first, n_use, n_mention, mention_ratio

预登记门槛（train 全量，test 零参与）：
  G1 主门：命中子集内 E0FP（expr0 无毒）vs 有毒 的判别力——
     最优单特征 AUC ≥ 0.75 且新特征联合 LR（3 种子 5 折 CV）AUC ≥ 0.75
     （理论预期方向：无毒提及 → p_use 低；AUC 取 max(a,1-a) 并报告符号）
  G2 正交门：9 个 um 特征与现有概率特征（141 翻转条目 + v3p_ + S1A/S1B/CAT）
     的 max|corr| < 0.7（v8.3/v8.8 条目准入规则）
  机制检查：命中子集 p_use 分布按组（无毒/expr1/expr2/expr3）、按长度；
     mention_ratio 在无毒命中子集应显著偏低（提及假设的直接验证）

产物：results/v11/usemention/analysis_metrics.txt + gate.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ["has_occ", "n_occ", "p_use_max", "p_use_mean", "p_use_min",
         "p_use_first", "n_use", "n_mention", "mention_ratio"]


def aggregate(occ_items: dict, all_ids):
    """occ_items: id -> list of {p_use, occ_idx,...}；返回逐样本特征 DataFrame。"""
    rows = []
    for _id in all_ids:
        occs = occ_items.get(_id, [])
        valid = [o for o in occs if o["p_use"] == o["p_use"]]
        n_occ = len(occs)
        if n_occ == 0:
            rows.append({"id": _id, "has_occ": 0.0, "n_occ": 0.0,
                         "p_use_max": 0.5, "p_use_mean": 0.5, "p_use_min": 0.5,
                         "p_use_first": 0.5, "n_use": 0.0, "n_mention": 0.0,
                         "mention_ratio": 0.5})
            continue
        ps = [o["p_use"] for o in valid]
        first = sorted(valid, key=lambda o: o["occ_idx"])[0]["p_use"] if valid else 0.5
        if ps:
            n_use = float(sum(p >= 0.5 for p in ps))
            n_mention = float(len(ps)) - n_use
            mention_ratio = n_mention / len(ps)
        else:
            n_use, n_mention, mention_ratio = 0.0, 0.0, 0.5
        rows.append({"id": _id, "has_occ": 1.0, "n_occ": float(n_occ),
                     "p_use_max": max(ps) if ps else 0.5,
                     "p_use_mean": float(np.mean(ps)) if ps else 0.5,
                     "p_use_min": min(ps) if ps else 0.5,
                     "p_use_first": first,
                     "n_use": n_use, "n_mention": n_mention,
                     "mention_ratio": mention_ratio})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="v11 Stage3 门槛判读")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json", required=True)
    ap.add_argument("--um_train", default="results/v11/usemention/um_train.json")
    ap.add_argument("--output_dir", default="results/v11/usemention")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.train_matrix)
    y = tr["toxic"].astype(int).values
    ids = tr["id"].values
    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids])

    with open(args.um_train, encoding="utf-8") as f:
        um = json.load(f)["results"]
    occ_items = {}
    for r in um:
        occ_items.setdefault(r["id"], []).append(r)
    F = aggregate(occ_items, ids)
    assert (F["id"].values == ids).all()
    n_invalid = sum(1 for r in um if r["p_use"] != r["p_use"])

    lines = ["=" * 72, "v11 词项级 use/mention 测量——预登记门槛判读", "=" * 72,
             f"train 样本 {len(ids)}，出现位置 {len(um)}（p_use 无效 {n_invalid}），"
             f"命中样本 {int(F['has_occ'].sum())}（{F['has_occ'].mean():.3f}）", ""]

    # ---- 组分布与机制检查 ----
    hit = F["has_occ"].values == 1
    groups = [("E0无毒", (y == 0) & (expr == 0)), ("expr1有毒", (y == 1) & (expr == 1)),
              ("expr2有毒", (y == 1) & (expr == 2)), ("expr3有毒", (y == 1) & (expr == 3))]
    lines += [f"{'组':<10}{'n':<7}{'命中':<7}{'命中内p_use均值':<16}"
              f"{'mention_ratio均值':<18}", "-" * 72]
    for name, m in groups:
        hm = m & hit
        if hm.sum():
            lines.append(f"{name:<10}{m.sum():<7}{hm.sum():<7}"
                         f"{F.loc[hm, 'p_use_mean'].mean():<16.4f}"
                         f"{F.loc[hm, 'mention_ratio'].mean():<18.4f}")
    lines.append("")

    # ---- G1 主门：命中子集 E0FP vs 有毒 ----
    e0fp = np.where((y == 0) & (expr == 0) & hit)[0]
    tox = np.where((y == 1) & hit)[0]
    lines += ["G1 主门（命中子集 E0FP vs 有毒，预登记 ≥0.75）", "-" * 72]
    sub = np.concatenate([e0fp, tox])
    ysub = np.where(np.arange(len(sub)) < len(e0fp), 0, 1)
    best = (None, -1.0, 0)
    for f in FEATS:
        v = F[f].values[sub]
        if np.std(v) < 1e-9:
            continue
        a = roc_auc_score(ysub, v)
        sign = "+" if a >= 0.5 else "-"
        a = max(a, 1 - a)
        if a > best[1]:
            best = (f, a, sign)
        lines.append(f"  {f:<14} AUC={a:.4f} 方向'{sign}'")
    lines.append(f"  最优单特征: {best[0]} AUC={best[1]:.4f} 方向'{best[2]}'"
                 f"  [门槛 {'PASS' if best[1] >= 0.75 else 'FAIL'}]")

    aucs = []
    for seed in (42, 43, 44):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(sub))
        for trn, va in cv.split(F.loc[sub, FEATS].values, ysub):
            m = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=0.01, max_iter=2000))])
            m.fit(F.loc[sub, FEATS].values[trn], ysub[trn])
            oof[va] = m.predict_proba(F.loc[sub, FEATS].values[va])[:, 1]
        aucs.append(max(roc_auc_score(ysub, oof), 1 - roc_auc_score(ysub, oof)))
    joint = float(np.mean(aucs))
    lines.append(f"  联合 LR AUC = {joint:.4f} ± {np.std(aucs):.4f}"
                 f"  [门槛 {'PASS' if joint >= 0.75 else 'FAIL'}]")

    # ---- G2 正交门 ----
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr = tr[item_cols].copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    v3 = pd.read_csv(args.v3_train)
    v3_cols = [c for c in v3.columns
               if c.startswith(("v3p_", "s1a_", "s1b_", "cat_"))]
    exist = np.hstack([Atr[kept_items].values, v3[v3_cols].values])
    maxc = {}
    for f in FEATS:
        r = np.abs(np.corrcoef(F[f].values, exist.T)[0, 1:])
        maxc[f] = float(np.nanmax(r))
    worst_f, worst_v = max(maxc.items(), key=lambda x: x[1])
    lines += ["", "G2 正交门（与现有概率特征 max|corr| < 0.7）", "-" * 72]
    for f in FEATS:
        lines.append(f"  {f:<14} max|corr|={maxc[f]:.4f}")
    g2_pass = worst_v < 0.7
    lines.append(f"  最大: {worst_f} {worst_v:.4f}  [门槛 {'PASS' if g2_pass else 'FAIL'}]")

    # ---- 裁决 ----
    g1_pass = best[1] >= 0.75 and joint >= 0.75
    lines += ["", "=" * 72,
              f"[裁决] G1 主门 {'PASS' if g1_pass else 'FAIL'}，"
              f"G2 正交门 {'PASS' if g2_pass else 'FAIL'}",
              "两门皆过 -> 写 readout_v11（M4 + um 特征，CV 门槛 ≥0.002）；"
              "任一 FAIL -> 词项级通道判死并归档机制结论", "=" * 72]

    (out / "gate.json").write_text(json.dumps(
        {"g1_single": {"feat": best[0], "auc": best[1], "sign": best[2]},
         "g1_joint": joint, "g1_pass": g1_pass, "g2_maxcorr": worst_v,
         "g2_pass": g2_pass}, ensure_ascii=False, indent=1), encoding="utf-8")
    report = "\n".join(lines)
    print(report)
    (out / "analysis_metrics.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
