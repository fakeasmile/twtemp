# -*- coding: utf-8 -*-
"""v10 pilot 门槛分析：从 rationale 解析言语行为力量特征，检验预登记门槛（CPU）。

预登记门槛（v10 方案定稿）：
  G-A（FP 池，主门）  E0FP-pilot vs TP-pilot   CV-LR AUC >= 0.75
                      （现有矩阵上限 0.7335，v8.3 审计 + v8.7 非线性探针双重确认）
  G-B（FN 池）        E2FN-pilot vs BEN-pilot  CV-LR AUC >= 0.75
                      （现有上限：线性 0.687 / 非线性 0.729）
两门任一通过 → 解析特征有资格进入概念层（再走全量测量）；全败 → 投射模式
引不出缺失维度，证据链闭合。

特征：结构化槽位解析（行为 7 类 / 指向 4 类 / 严肃性 3 类 one-hot）+ 力量
关键词旗标。辅助诊断：rationale 文本 char n-gram TF-IDF 的 CV AUC（检验自由
文本里是否存在结构化解析丢失的信息）。
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BEHAVIOR = ["攻击辱骂", "贬损蔑视", "引用转述", "讨论评价", "反讽挖苦", "玩笑调侃", "其他"]
DIRECTION = ["文中具体对象", "直接对话者", "泛指群体", "无特定对象"]
SERIOUS = ["严肃恶意", "玩笑夸张", "无法判断"]
KEYWORDS = ["引用", "转述", "谈论", "讨论", "复述", "摘录", "反讽", "讽刺", "玩笑",
            "调侃", "玩梗", "举例", "假设"]


def parse_rationale(text):
    """解析结构化槽位 + 关键词旗标。返回 (parsed_ok, feat_dict)。"""
    text = str(text)
    def grab(slot, options):
        m = re.search(slot + r"[:：]\s*([^\n]+)", text)
        if not m:
            return None
        seg = m.group(1)
        for opt in options:
            if opt in seg:
                return opt
        return None
    beh = grab("行为", BEHAVIOR)
    dr = grab("指向", DIRECTION)
    se = grab("严肃性", SERIOUS)
    feats = {}
    for k, v in [("beh_" + b, b == beh) for b in BEHAVIOR] + \
                [("dir_" + d, d == dr) for d in DIRECTION] + \
                [("ser_" + s, s == se) for s in SERIOUS]:
        feats[k] = 1.0 if v else 0.0
    feats["unparsed"] = 0.0 if (beh and dr and se) else 1.0
    for kw in KEYWORDS:
        feats[f"kw_{kw}"] = 1.0 if kw in text else 0.0
    return (beh is not None), feats


def cv_auc(pos_X, neg_X, seed=42):
    """子集内 5 折 CV L2-LR AUC（与 v8.3 审计同法）。"""
    X = np.vstack([pos_X, neg_X])
    y = np.concatenate([np.ones(len(pos_X)), np.zeros(len(neg_X))])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        prob = m.predict_proba(sc.transform(X[va]))[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y[va], prob))
    return float(np.mean(aucs)), float(np.std(aucs))


def tfidf_auc(pos_texts, neg_texts, seed=42):
    """rationale 全文 char n-gram TF-IDF 的 CV AUC（诊断解析损失）。"""
    texts = list(pos_texts) + list(neg_texts)
    y = np.concatenate([np.ones(len(pos_texts)), np.zeros(len(neg_texts))])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in cv.split(texts, y):
        from sklearn.pipeline import make_pipeline
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=500),
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000))
        pipe.fit([texts[i] for i in tr], y[tr])
        prob = pipe.predict_proba([texts[i] for i in va])[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y[va], prob))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser(description="v10 pilot 门槛分析")
    ap.add_argument("--rationales", required=True)
    ap.add_argument("--pilot_csv", required=True)
    ap.add_argument("--output_dir", default="results/v10/pilot")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rat = {}
    with open(args.rationales, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rat[d["id"]] = d
    pilot = pd.read_csv(args.pilot_csv)
    assert len(pilot) == len(rat), "rationale 数量与 pilot 样本不一致"

    rows = []
    for r in pilot.itertuples(index=False):
        d = rat[r.id]
        ok, feats = parse_rationale(d["rationale"])
        rows.append({"id": r.id, "group": r.group, "toxic": int(r.toxic),
                     "parsed_ok": int(ok), "rationale": d["rationale"], **feats})
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith(("beh_", "dir_", "ser_", "kw_"))
                 or c == "unparsed"]
    print(f"槽位解析成功率：{df['parsed_ok'].mean():.3f}")
    if df["parsed_ok"].mean() < 0.9:
        print("[警告] 解析率过低，打印 2 条原始输出排查：")
        for r in df.head(2).itertuples(index=False):
            print(f"  --- id={r.id} group={r.group} ---")
            print("  " + str(r.rationale)[:400].replace("\n", "\n  "))
    for c, pref in [("行为", "beh_"), ("指向", "dir_"), ("严肃性", "ser_")]:
        cols = [x for x in feat_cols if x.startswith(pref)]
        dist = df[cols].sum()
        print(f"  {c} 分布: " + "  ".join(f"{k.split('_', 1)[1]}={int(v)}" for k, v in dist.items() if v > 0))

    F = df[feat_cols].values

    def grp(g):
        return df["group"] == g

    fp_X = F[grp("E0FP").values]
    tp_X = F[grp("TP").values]
    e2fn_X = F[grp("E2FN").values]
    ben_X = F[(grp("TN") | grp("E0FP")).values]
    tn_X = F[grp("TN").values]

    rt = {r.id: r.rationale for r in df.itertuples(index=False)}
    fp_t = [rt[i] for i in df.loc[grp("E0FP"), "id"]]
    tp_t = [rt[i] for i in df.loc[grp("TP"), "id"]]
    e2fn_t = [rt[i] for i in df.loc[grp("E2FN"), "id"]]
    ben_t = [rt[i] for i in df.loc[grp("TN") | grp("E0FP"), "id"]]

    lines = []
    lines.append("v10 pilot 言语行为力量特征——预登记门槛分析")
    lines.append("=" * 70)
    lines.append(f"样本：E0FP {len(fp_X)}  E2FN {len(e2fn_X)}  TP {len(tp_X)}  TN {len(tn_X)}")
    lines.append(f"槽位解析率 {df['parsed_ok'].mean():.3f}；特征 {F.shape[1]} 维")
    lines.append("")

    a, s = cv_auc(fp_X, tp_X)
    lines.append(f"[G-A 主门] E0FP vs TP        CV-LR AUC = {a:.4f} ± {s:.4f}"
                 f"   （现有矩阵上限 0.7335；门槛 >=0.75）   裁决: {'PASS' if a >= 0.75 else 'FAIL'}")
    b, s2 = cv_auc(e2fn_X, ben_X)
    lines.append(f"[G-B]      E2FN vs BEN(pilot)  CV-LR AUC = {b:.4f} ± {s2:.4f}"
                 f"   （现有上限 线性0.687/非线性0.729；门槛 >=0.75）   裁决: {'PASS' if b >= 0.75 else 'FAIL'}")
    lines.append("")
    c1, _ = cv_auc(fp_X, tn_X)
    c2, _ = cv_auc(e2fn_X, tp_X)
    d1, _ = tfidf_auc(fp_t, tp_t)
    d2, _ = tfidf_auc(e2fn_t, ben_t)
    lines.append("[辅助诊断]")
    lines.append(f"  E0FP vs TN     AUC = {c1:.4f}   （审计参照 0.986：力量特征是否保留表面可分性）")
    lines.append(f"  E2FN vs TP     AUC = {c2:.4f}   （审计参照 0.968）")
    lines.append(f"  TF-IDF 全文:   E0FP vs TP = {d1:.4f}   E2FN vs BEN = {d2:.4f}"
                 f"   （>结构化特征 → 解析有信息损失，自由文本含更多信息）")

    verdict = []
    if a >= 0.75:
        verdict.append("G-A 通过：力量特征对 FP 池有资格进入概念层 → 走全量测量")
    if b >= 0.75:
        verdict.append("G-B 通过：力量特征对 FN 池有资格进入概念层 → 走全量测量")
    if not verdict:
        verdict.append("两门均未通过：投射模式（开放生成）引不出缺失维度，"
                       "v9 封闭探测 + v10 开放引出双证伪，测量侧证据链闭合")
    lines.append("")
    lines.append("[裁决] " + "；".join(verdict))

    report = "\n".join(lines)
    print(report)
    (out / "gate_report.txt").write_text(report, encoding="utf-8")
    df.drop(columns=["rationale"]).to_csv(out / "pilot_features.csv", index=False,
                                          encoding="utf-8-sig")


if __name__ == "__main__":
    main()
