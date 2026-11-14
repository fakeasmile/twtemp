# -*- coding: utf-8 -*-
"""v6 概念向量效度审计（只读诊断，不改任何产物）。

问题（用户提出）：概念向量是否在真实反映文本的状态？
审计面：
  1. 条目分布：每条目 toxic/benign 触发差、AUC、饱和度、与长度的相关
  2. 良性高触发审计：benign 触发率异常高的条目 + 真实文本样例（mention≠attack 探查）
  3. 构念分与毒性标签的单调性 + FP/FN 极端样例阅读
  4. 长度混杂：构念分-长度相关
输出：results/v6/audit/audit_report.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = "data/processed/TOXICN/glm-4-9b-chat_seed42/items_glm-4-9b-chat_train_matrix.csv"
FREEZE = "results/v6/item_analysis_glm_train/v6_1_freeze.json"
ITEMS_CSV = "data/concepts/cbm_concept_set_v6.csv"
OUT = Path("results/v6/audit/audit_report.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

freeze = json.loads(Path(FREEZE).read_text(encoding="utf-8"))
items = pd.read_csv(ITEMS_CSV, encoding="utf-8-sig")
df = pd.read_csv(TRAIN)

y = df["toxic"].astype(int).values
texts = df["content"].astype(str).values
L = texts.astype(object)
length = np.array([len(t) for t in texts])
item_cols = [c for c in df.columns if c not in ("id", "toxic", "content")]
X = df[item_cols]

rep = []


def p(s=""):
    rep.append(str(s))


def ex(i, n=56):
    t = texts[i].replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


p("=" * 72)
p("v6 概念向量效度审计（train 9600，GLM 单教师）")
p("=" * 72)

# ---------- 1. 条目级分布 ----------
p("\n[1] 条目级诊断（152 条）")
rows = []
for c in item_cols:
    v = X[c].values
    rows.append({
        "item": c,
        "mean_all": v.mean(),
        "mean_tox": v[y == 1].mean(),
        "mean_ben": v[y == 0].mean(),
        "auc": roc_auc_score(y, v),
        "std": v.std(),
        "r_len": np.corrcoef(v, length)[0, 1],
    })
st = pd.DataFrame(rows)
st["gap"] = st["mean_tox"] - st["mean_ben"]

p(f"\n1a. 方向正确且强（|AUC-0.5| 前列，toxic 触发更高）前 12：")
top = st.sort_values("auc", ascending=False).head(12)
for _, r in top.iterrows():
    p(f"  {r['item']:<8} auc {r['auc']:.3f}  tox {r['mean_tox']:.2f} ben {r['mean_ben']:.2f}  "
      f"mean_all {r['mean_all']:.2f}  r_len {r['r_len']:+.2f}")

p(f"\n1b. 方向反常（benign 触发 ≥ toxic，按 benign-toxic 差排序）前 12：")
rev = st.sort_values("gap").head(12)
for _, r in rev.iterrows():
    p(f"  {r['item']:<8} auc {r['auc']:.3f}  tox {r['mean_tox']:.2f} ben {r['mean_ben']:.2f}  "
      f"mean_all {r['mean_all']:.2f}")

p(f"\n1c. 饱和/近死条目（mean_all>0.85 或 std<0.08）{int(((st['mean_all']>0.85)|(st['std']<0.08)).sum())} 条：")
sat = st[(st["mean_all"] > 0.85) | (st["std"] < 0.08)].sort_values("mean_all", ascending=False)
for _, r in sat.head(15).iterrows():
    q = items[items["条目id"] == r["item"]]
    qtext = q.iloc[0]["条目问句"][:38] if len(q) else "?"
    p(f"  {r['item']:<8} mean {r['mean_all']:.2f} std {r['std']:.3f} auc {r['auc']:.3f}  {qtext}")

p(f"\n1d. 长度混杂最强的条目（|r_len| 前 8）：")
for _, r in st.reindex(st["r_len"].abs().sort_values(ascending=False).index).head(8).iterrows():
    p(f"  {r['item']:<8} r_len {r['r_len']:+.3f}  auc {r['auc']:.3f}  mean_all {r['mean_all']:.2f}")

# ---------- 2. 良性高触发样例（mention≠attack 探查） ----------
p("\n[2] 良性高触发样例阅读（benign 且综合分最高，疑似 mention≠attack）")
dirmap = dict(zip(items["条目id"], items["构念id"]))
cons = list(freeze["items_per_construct"].keys())
cmean = pd.DataFrame({"id": df["id"]})
for cid in cons:
    kept = freeze["items_per_construct"][cid]
    cols = [c for c in kept if c in X.columns]
    if cid == "D2":
        cols = [c for c in ["D2-01", "D2-02", "D2-03", "D2-04", "D2-05"] if c in X.columns]
    if not cols:
        continue
    Mv = X[cols].copy()
    for c in cols:
        if dirmap.get(c) == "反向":
            Mv[c] = 1.0 - Mv[c]
    cmean[cid] = Mv.mean(axis=1).values
cm_cols = cons
comp = cmean[cm_cols].values.max(axis=1)

ben_idx = np.where(y == 0)[0]
order_b = ben_idx[np.argsort(-comp[ben_idx])]
p("benign 中综合触发分最高 12 条（附其最高 3 构念）：")
for i in order_b[:12]:
    topc = pd.Series(comp[i], index=cm_cols).sort_values(ascending=False).head(3)
    cs = " ".join(f"{k}:{v:.2f}" for k, v in topc.items())
    p(f"  id {df.iloc[i]['id']}  max构念分 {comp[i]:.2f}  [{cs}]")
    p(f"      {ex(i)}")

# ---------- 3. FN 样例：toxic 但触发分最低 ----------
p("\n[3] 漏检样例阅读（toxic 且综合触发分最低 12 条）")
tox_idx = np.where(y == 1)[0]
order_t = tox_idx[np.argsort(comp[tox_idx])]
for i in order_t[:12]:
    p(f"  id {df.iloc[i]['id']}  max构念分 {comp[i]:.2f}  {ex(i)}")

# ---------- 4. 构念分与长度相关 ----------
p("\n[4] 构念分-长度相关（长度混杂）")
for cid in cons:
    r = np.corrcoef(cmean[cid].values, length)[0, 1]
    auc = roc_auc_score(y, cmean[cid].values)
    p(f"  {cid:<4} r_len {r:+.3f}  auc {auc:.3f}")

# ---------- 5. 触发面形态：benign 内高触发的构念 ----------
p("\n[5] benign 内构念触发形态（benign 构念分 P90，越低越干净）")
for cid in cm_cols:
    v = cmean[cid].values
    p(f"  {cid:<4} benign P50 {np.percentile(v[y==0],50):.2f}  P90 {np.percentile(v[y==0],90):.2f}  "
      f"toxic P50 {np.percentile(v[y==1],50):.2f}")

OUT.write_text("\n".join(rep), encoding="utf-8")
print(f"审计报告已写入 {OUT}")
