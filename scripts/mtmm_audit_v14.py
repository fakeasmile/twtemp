# -*- coding: utf-8 -*-
"""v14 Stage 0 辅助线：MTMM 式构念效度审计（CPU，全 fresh，零门槛描述性诊断）。

理论背景：Campbell & Fiske (1959) 多质多法矩阵。v8.7 终审（历史结论）
点名"MMPI/心理测量传统内未使用的理论层=反应模式理论+MTMM"；本脚本把
MTMM 的会聚/判别效度逻辑落到现有数据上，回答一个设计问题：
  —— 若 v14 Thurstone 成对比较只重测部分构念，该测谁？

特质（trait）= v6 freeze 的 19 个构念（items_per_construct）
方法A（method A） = v6 条目池：条目按 freeze 经验极性对齐后等权均值聚合
                    为构念分（CTT 等权版；与 v6.4 的 CITC 加权口径不同，
                    此处 fresh 重算，差异已在表中并列 CITC 供对照）
方法B（method B） = v3 环概念通道（v3p_/v3pd_/v3d_，train AUC 极性对齐；
                    s1a_/s1b_/cat_ 样本级效度通道不属特质测量，排除）

四项审计（全部 descriptive，无判死门）：
【1】构念内质量：Cronbach α（对齐后）、平均 CITC、条目数
【2】会聚/方法重叠：构念分与 v3 全通道 max|corr| / mean|corr|
     ——max 高 = 该构念已被环测量捕获（重测价值低）
【3】判别效度：19×19 构念分相关矩阵（off-diag 均值/最大值/top5 共线对）
【4】效标关联与污染：构念分 AUC@全train、AUC@E0无毒vs有毒、|corr|与文本长度
     ——长度相关高 = 构念分被 FP 池长度机制污染
候选短名单（供人工判读，非预登记门）：α≥0.70 且 AUC@E0≥0.60 且 max|corr|<0.80

产物：results/v14/mtmm/{mtmm_audit.txt, construct_scores_train.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cronbach_alpha(X):
    """X: (n, k) 对齐后条目矩阵。k<2 返回 nan。"""
    k = X.shape[1]
    if k < 2:
        return np.nan
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 1e-12:
        return np.nan
    return k / (k - 1) * (1.0 - item_var.sum() / total_var)


def mean_citc(X):
    """平均校正条目-总分相关：corr(item_i, total − item_i)。"""
    k = X.shape[1]
    if k < 2:
        return np.nan
    citc = []
    for j in range(k):
        rest = X[:, np.arange(k) != j].sum(axis=1)
        if rest.std() < 1e-12 or X[:, j].std() < 1e-12:
            continue
        citc.append(np.corrcoef(X[:, j], rest)[0, 1])
    return float(np.mean(citc)) if citc else np.nan


def feat_auc(score, ysub):
    """单特征 AUC（方向自由）。"""
    a = roc_auc_score(ysub, score)
    return float(max(a, 1.0 - a))


def main():
    ap = argparse.ArgumentParser(description="v14 MTMM 式构念效度审计")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json")
    ap.add_argument("--output_dir", default="results/v14/mtmm")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 读入矩阵与冻结文件...")
    tr = pd.read_csv(args.train_matrix)
    v3tr = pd.read_csv(args.v3_train)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    y = tr["toxic"].astype(int).values
    ids = tr["id"].values
    lens = tr["content"].astype(str).str.len().values
    assert (tr["id"].values == v3tr["id"].values).all()

    with open(args.rings_train_json, encoding="utf-8") as f:
        expr_map = {r["id"]: (-1 if r.get("expression") is None
                              else int(r["expression"]))
                    for r in json.load(f)["results"]}
    expr = np.array([expr_map.get(i, -1) for i in ids])

    # ---- 方法A：条目极性对齐 + 构念分（等权均值）----
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr = tr[item_cols].copy()
    n_flip = 0
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - tr[c]
            n_flip += 1
    stamp(f"[对齐] 条目极性翻转 {n_flip}/{len(item_cols)}")

    constructs = list(freeze["items_per_construct"].keys())
    scores = {}
    alpha_rows = []
    for cname in constructs:
        cols = [c for c in freeze["items_per_construct"][cname] if c in Atr.columns]
        if not cols:
            continue
        Xc = Atr[cols].values
        scores[cname] = Xc.mean(axis=1)
        # CITC 用设计极性原条目（对齐前）与对齐后等价（单调翻转不改变 |corr|，
        # 此处用对齐后矩阵保持与 α 一致口径）
        alpha_rows.append({"construct": cname, "n_items": len(cols),
                           "alpha": cronbach_alpha(Xc),
                           "citc_mean": mean_citc(Xc),
                           "score_std": float(Xc.mean(axis=1).std())})
    names = list(scores.keys())
    S = np.column_stack([scores[c] for c in names])
    stamp(f"[聚合] {len(names)} 个构念分完成")

    # ---- 方法B：v3 概念通道极性对齐（与 assemble_m4 同逻辑）----
    v3_cols = [c for c in v3tr.columns
               if c.startswith(("v3p_", "v3pd_", "v3d_")) and v3tr[c].std() > 1e-9]
    B = v3tr[v3_cols].values.copy()
    for j, c in enumerate(v3_cols):
        a = roc_auc_score(y, B[:, j])
        if a < 0.5:
            B[:, j] = 1.0 - B[:, j]
    stamp(f"[对齐] v3 概念通道 {len(v3_cols)} 个（极性按 train AUC 对齐）")

    # ---- E0 无毒 vs 有毒 配对 ----
    e0fp = np.where((y == 0) & (expr == 0))[0]
    tox = np.where(y == 1)[0]
    idx_e0 = np.concatenate([e0fp, tox])
    ysub = np.where(np.arange(len(idx_e0)) < len(e0fp), 0, 1)

    # ---- 【1】+【2】+【4】逐构念表 ----
    stamp("[审计] 逐构念指标计算中...")
    rows = []
    for i, cname in enumerate(names):
        s = S[:, i]
        cv = np.abs(np.corrcoef(s, B.T)[0, 1:])
        cv = cv[np.isfinite(cv)]
        rows.append({
            "construct": cname,
            "n_items": alpha_rows[i]["n_items"],
            "alpha": alpha_rows[i]["alpha"],
            "citc": alpha_rows[i]["citc_mean"],
            "auc_full": feat_auc(s, y),
            "auc_e0": feat_auc(s[idx_e0], ysub),
            "v3_max": float(np.max(cv)) if cv.size else np.nan,
            "v3_mean": float(np.mean(cv)) if cv.size else np.nan,
            "len_corr": float(abs(np.corrcoef(s, lens)[0, 1])),
        })
    tb = pd.DataFrame(rows).sort_values("auc_e0", ascending=False)

    # ---- 【3】构念间判别效度 ----
    C = np.corrcoef(S.T)
    off = C[np.triu_indices(len(names), k=1)]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], C[i, j]))
    pairs.sort(key=lambda t: -abs(t[2]))

    # ---- 方法B 覆盖率 ----
    Xs = np.abs(np.corrcoef(np.hstack([S, B]).T))[:len(names), len(names):]
    cover_80 = float((np.nanmax(Xs, axis=0) >= 0.8).mean())
    cover_90 = float((np.nanmax(Xs, axis=0) >= 0.9).mean())

    short = tb[(tb["alpha"] >= 0.70) & (tb["auc_e0"] >= 0.60)
               & (tb["v3_max"] < 0.80)]

    L = ["=" * 90, "v14 MTMM 式构念效度审计（全 fresh；描述性诊断，无判死门）",
         "=" * 90,
         f"特质=构念（{len(names)}），方法A=v6条目池（极性对齐等权），"
         f"方法B=v3概念通道（{len(v3_cols)} 个，AUC 极性对齐）",
         f"配对：E0无毒(expr0) n={len(e0fp)} vs 有毒 n={len(tox)}", "",
         "【1】【2】【4】逐构念表（按 AUC@E0 降序）",
         "  构念                        条目  α      CITC   AUC全   AUC@E0  "
         "v3max  v3mean  |corr|长度",
         "  " + "-" * 86]
    for _, r in tb.iterrows():
        L.append(f"  {r['construct']:<24}{int(r['n_items']):>4}  "
                 f"{r['alpha']:.3f}  {r['citc']:.3f}  "
                 f"{r['auc_full']:.3f}   {r['auc_e0']:.3f}   "
                 f"{r['v3_max']:.3f}  {r['v3_mean']:.3f}   {r['len_corr']:.3f}")
    L += ["",
          "【3】判别效度（构念分 19×19 相关矩阵）",
          f"  off-diag |corr|: mean={np.nanmean(np.abs(off)):.3f}  "
          f"max={np.nanmax(np.abs(off)):.3f}",
          "  top5 共线构念对: " + "; ".join(
              f"{a}~{b}({c:+.2f})" for a, b, c in pairs[:5]),
          "",
          f"方法B 覆盖率：被某构念 max|corr|≥0.8 覆盖的 v3 通道占比 = {cover_80:.2f}"
          f"（≥0.9 = {cover_90:.2f}）", "",
          "候选短名单（α≥0.70 且 AUC@E0≥0.60 且 v3max<0.80 —— 供人工判读）：",
          "  " + (", ".join(short["construct"].tolist()) if len(short)
                 else "（无构念同时满足三条件）"), "",
          "=" * 90,
          "判读指引：AUC@E0 高+v3max 低 = 环测量未捕获的独立构念 → Thurstone",
          "重测价值最高；v3max 高 = 已被捕获，重测价值低；len_corr 高 = 长度",
          "污染构念分，重测时需在探针设计中规避长度混淆。"]

    report = "\n".join(L)
    print(report)
    (out / "mtmm_audit.txt").write_text(report + "\n", encoding="utf-8")

    pd.DataFrame({"id": ids, "toxic": y, "expr": expr, "len": lens,
                  **{f"score_{c}": scores[c] for c in names}}
                 ).to_csv(out / "construct_scores_train.csv", index=False,
                          encoding="utf-8-sig")
    stamp(f"[完成] 构念分已导出 construct_scores_train.csv "
          f"（{len(ids)} 行 × {len(names)} 构念）")


if __name__ == "__main__":
    main()
