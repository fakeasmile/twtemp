# -*- coding: utf-8 -*-
"""v10 指令消融分析（CPU，预登记门槛）。

问题：v6 模板锚定条款（"变形出现同样算"）是否在指令层面制造 use/mention
误差（FP 池全部为 expr0 引用型）。判读（预登记）：
  主门：V1 或 V2 下，E0FP vs TP 的判别（最优单条目 / 回避通道 / 联合 LR）
        任一 AUC >= 0.75（突破现有矩阵上限 0.7335）→ PASS
  机制证据：V0→V2 的 E0FP 样本 p_yes 均值是否下移而 TP 不变（指令因果可视化）

产物：results/v10/instr_ablation/{metrics.txt, gate.json}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cv_auc_sub(X, y01, seed=0):
    """子集内 5 折 CV L2-LR AUC（均值±std）。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in cv.split(X, y01):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(penalty="l2", C=0.01, max_iter=2000)
        m.fit(sc.transform(X[tr]), y01[tr])
        p = m.predict_proba(sc.transform(X[va]))[:, 1]
        aucs.append(roc_auc_score(y01[va], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def sf_ceil(x_pos, x_neg):
    """单特征方向折叠 AUC：返回 (auc, sign)，sign=+1 表示组 pos 更高。"""
    x = np.nan_to_num(np.concatenate([x_pos, x_neg]), nan=0.5)
    y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    if np.std(x) < 1e-12:
        return 0.5, 0
    a = roc_auc_score(y, x)
    if a >= 0.5:
        return a, 1
    return 1.0 - a, -1


def main():
    ap = argparse.ArgumentParser(description="v10 指令消融门槛分析")
    ap.add_argument("--measure", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--output_dir", default="results/v10/instr_ablation")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    polarity = freeze.get("item_polarity", {})

    df = pd.read_json(args.measure, lines=True)
    variants = sorted(df["variant"].unique())
    ids = sorted(df["id"].unique())
    items = sorted(df["item_id"].unique())
    id_ix = {i: k for k, i in enumerate(ids)}
    it_ix = {i: k for k, i in enumerate(items)}
    groups = df.drop_duplicates("id").set_index("id")["group"].to_dict()

    lines = ["v10 指令消融——预登记门槛分析", "=" * 70,
             f"样本 {len(ids)}，条目 {len(items)}，变体 {variants}"]
    gate = {"pass": False}
    for v in variants:
        dv = df[df["variant"] == v]
        P = np.full((len(ids), len(items)), np.nan)
        D = np.full((len(ids), len(items)), np.nan)
        for r in dv.itertuples(index=False):
            P[id_ix[r.id], it_ix[r.item_id]] = r.p_yes
            D[id_ix[r.id], it_ix[r.item_id]] = r.deflect
        # 极性对齐（反向条目 1-p）
        for j, it in enumerate(items):
            if polarity.get(it, 1) == -1:
                P[:, j] = 1.0 - P[:, j]
        g = np.array([groups[i] for i in ids])

        def sub(m, a, b):
            ia = np.where(g == a)[0]
            ib = np.where(g == b)[0]
            return m[ia], m[ib]

        fp, tp = sub(P, "E0FP", "TP")
        fn, tn = sub(P, "E2FN", "TN")
        fpD, tpD = sub(D, "E0FP", "TP")

        # 单条目折叠 AUC（p_yes 与 deflect 两个通道）
        best_item, best_auc = "", 0.0
        for j, it in enumerate(items):
            if np.isnan(fp[:, j]).all():
                continue
            a, _ = sf_ceil(fp[:, j], tp[:, j])
            if a > best_auc:
                best_auc, best_item = a, it
        best_defl, best_defl_item = 0.0, ""
        for j, it in enumerate(items):
            a, _ = sf_ceil(fpD[:, j], tpD[:, j])
            if a > best_defl:
                best_defl, best_defl_item = a, it
        # 联合 LR：141 对齐条目 + 条目均值回避
        Xa = np.column_stack([np.nan_to_num(np.vstack([fp, tp]), nan=0.5),
                              np.nanmean(np.vstack([fpD, tpD]), axis=1, keepdims=True)])
        ya = np.concatenate([np.ones(len(fp)), np.zeros(len(tp))])
        joint_m, joint_s = cv_auc_sub(Xa, ya)

        # 机制表：E0FP/TP 的 p_yes 均值（对齐后）
        m_fp = np.nanmean(fp)
        m_tp = np.nanmean(tp)
        lines.append("-" * 70)
        lines.append(f"[{v}]  E0FP n={len(fp)}  TP n={len(tp)}")
        lines.append(f"  单条目最优 AUC(E0FP vs TP) = {best_auc:.4f}  ({best_item})"
                     f"   [现有矩阵上限 0.7335]")
        lines.append(f"  回避通道最优 AUC           = {best_defl:.4f}  ({best_defl_item})")
        lines.append(f"  联合 LR AUC = {joint_m:.4f}±{joint_s:.4f}  (141 对齐条目+回避均值)")
        lines.append(f"  p_yes 均值:  E0FP={m_fp:.4f}  TP={m_tp:.4f}")
        if v == "V0":
            base_fp, base_tp = m_fp, m_tp
        else:
            lines.append(f"  相对 V0 均值漂移:  E0FP {m_fp - base_fp:+.4f}  TP {m_tp - base_tp:+.4f}")

        fnb, tnb = sub(P, "E2FN", "TN")
        b_best = max((sf_ceil(fnb[:, j], tnb[:, j])[0]
                      for j in range(len(items))
                      if not np.isnan(fnb[:, j]).all()), default=0.0)
        lines.append(f"  辅助: E2FN vs TN 单条目最优 = {b_best:.4f}")

        hit = max(best_auc, best_defl, joint_m) >= 0.75
        gate[f"{v}_pass"] = bool(hit)
        gate[f"{v}_best"] = round(max(best_auc, best_defl, joint_m), 4)
        gate[f"{v}_best_item"] = best_item if best_auc >= max(best_defl, joint_m) else (
            best_defl_item if best_defl >= joint_m else "joint_LR")
        if hit:
            gate["pass"] = True

    lines.append("-" * 70)
    lines.append("[裁决] " + ("主门 PASS：指令消融突破 0.7335 上限——指令是 FP 池成因之一，"
                             "用胜出变体做全量重测" if gate["pass"] else
                             "主门 FAIL：三种指令变体均未突破上限，'指令制造误差'假设证伪"))
    (out / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
