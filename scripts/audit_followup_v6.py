# -*- coding: utf-8 -*-
"""审计补充验证（只读）：
  A. 全部 39 条反向条目在 train 上的方向与翻转后 AUC
  B. 反向条目翻转后与同构念正向条目的相关（解释 α 崩塌）
  C. 全题饱和文本统计（全条目均值 >0.9 / <0.1 的文本量与标签分布）
输出 results/v6/audit/audit_followup.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = "data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv"
ITEMS_CSV = "data/concepts/cbm_concept_set_v6.csv"
OUT = Path("results/v6/audit/audit_followup.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

items = pd.read_csv(ITEMS_CSV, encoding="utf-8-sig")
df = pd.read_csv(TRAIN)
y = df["toxic"].astype(int).values
item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
X = df[item_cols]
rev_items = items[items["方向"] == "反向"]["条目id"].tolist()

rep = []


def p(s=""):
    rep.append(str(s))


p("=" * 72)
p("审计补充：反向条目方向复查 + 全题饱和统计（train 9600）")
p("=" * 72)

# A. 反向条目方向
p(f"\n[A] 反向条目 {len(rev_items)} 条：benign 触发更高的条数（=翻转后方向正确）")
correct, wrong, rows = 0, 0, []
for c in rev_items:
    v = X[c].values
    auc_raw = roc_auc_score(y, v)
    flip_ok = auc_raw < 0.5  # raw p 在 benign 更高 → 翻转后与 toxic 同向
    correct += flip_ok
    wrong += (not flip_ok)
    rows.append((c, v[y == 1].mean(), v[y == 0].mean(), auc_raw, 1 - auc_raw if flip_ok else auc_raw))
p(f"  翻转后方向正确 {correct}/{len(rev_items)}，方向错误 {wrong}/{len(rev_items)}")
p("  条目     tox触发 ben触发  原始AUC  翻转后AUC")
for c, mt, mb, ar, af in sorted(rows, key=lambda r: -r[4]):
    p(f"  {c:<8} {mt:.2f}    {mb:.2f}    {ar:.3f}    {af:.3f}")

# B. 反向条目与正向条目的构念内相关（翻转后）
p("\n[B] 翻转后反向条目 vs 同构念正向条目的 Pearson 相关（解释 v6.1 的 α 崩塌）")
for cid, g in items.groupby("构念id"):
    pos = g[g["方向"] == "正向"]["条目id"].tolist()
    rev = [c for c in g[g["方向"] == "反向"]["条目id"] if c in X.columns]
    if not pos or not rev:
        continue
    posM = X[pos].values
    cors = []
    for c in rev:
        fv = 1.0 - X[c].values
        cors += [abs(np.corrcoef(fv, posM[:, j])[0, 1]) for j in range(posM.shape[1])]
    p(f"  {cid:<4} |r| 均值 {np.mean(cors):.3f}  最大 {np.max(cors):.3f}")

# C. 全题饱和
p("\n[C] 全题饱和统计：全 152 条目均值（未翻转原始 p）")
am = X[item_cols].mean(axis=1).values
p(f"  mean>0.90 的文本 {int((am > 0.90).sum())} 条  (toxic {int(((am > 0.90) & (y == 1)).sum())} / "
  f"benign {int(((am > 0.90) & (y == 0)).sum())})")
p(f"  mean>0.80        {int((am > 0.80).sum())} 条  (toxic {int(((am > 0.80) & (y == 1)).sum())} / "
  f"benign {int(((am > 0.80) & (y == 0)).sum())})")
p(f"  mean<0.10        {int((am < 0.10).sum())} 条  (toxic {int(((am < 0.10) & (y == 1)).sum())} / "
  f"benign {int(((am < 0.10) & (y == 0)).sum())})")
p(f"  全体 mean 分位:  P10 {np.percentile(am, 10):.2f}  P50 {np.percentile(am, 50):.2f}  "
  f"P90 {np.percentile(am, 90):.2f}")
p(f"  全题均值本身的 AUC: {roc_auc_score(y, am):.3f}  （高→更可能 benign，即玩梗/反讽文本全题触发）")

OUT.write_text("\n".join(rep), encoding="utf-8")
print(f"补充审计已写入 {OUT}")
