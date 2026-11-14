# -*- coding: utf-8 -*-
"""v11 Stage1：词项级定位（确定性词表匹配，CPU，零 GPU）。

理论动机（v8.8 后全案复盘结论）：use/mention 是词项级局部现象——冒犯词项与其
局部句法语境的关系；而 v6/v3 全部测量都是文本级问句，强制教师完成辖域解析后
坍缩成一个数，长文本中"提及"被平均稀释（FP 错误率随长度 9.3%→35.4% 单调上升
= 粒度错配指纹）。本方案把测量对象单位从"文本"改为"词项+辖域"：
  Stage1（本脚本）：词表匹配定位文本中全部冒犯词项出现位置（确定性、可复现）
  Stage2（run_usemention_v11.py）：教师对每个出现位置回答 使用/提及（词项条件化
    单概念问句，两选一，不违反多概念禁令）
  Stage3（analyze_usemention_v11.py）：预登记门槛判读

词表来源：cbm_concept_set_v6.csv 条目问句的括号锚定清单（例词已验证有效），
过滤规则（确定性，被删词全部打印供人工复核）：
  - 保留 2 ≤ len ≤ 6、无空格的词形（单字"打/丑/呸"误命中率高，排除）
  - 删除含元描述子串的词（写法/类/代号/emoji/XX/表达/行为/内容 等）
覆盖率诊断（写脚本前已验证）：expr0 无毒命中率随长度 6.2%→55.3% 单调上升，
与 FP 错误率指纹同构——词表精确覆盖"提及结构型 FP"风险区。
上下文窗口：出现位置 ±50 字符，目标出现以「」标记（确定性代码标记，
满足"聚合/呈现不由 LLM 完成"原则）。

产物：results/v11/usemention/occurrences_{train,test}.csv
      列：id, term, occ_idx, n_occ, pos_start, ctx
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEXICON_CSV = "data/concepts/cbm_concept_set_v6.csv"
META_SUBSTR = ("写法", "代号", "emoji", "XX", "升高", "表达", "行为",
               "内容", "说法", "想法", "做法", "字眼", "词汇", "称呼")
CTX_RADIUS = 50

# 人工策展：从 v6 锚定词表（自动过滤后 214 词）中只保留 use/mention 问句
# 对其成立的词形——可指称的冒犯性语言单位（脏话贬称/动物化疾病化蔑称/
# 暴力威胁诅咒/群体标签/排斥歧视词形）。删去：句式模式（"他们纵容的"）、
# 成语（"替天行道"）、虚词与标点（"都是""！！！"）、元描述（"负面评价"）、
# 情绪短语（"气死我了"）——这些词形上问句不成立，测量输出为纯噪声。
CURATED_TERMS = {
    # 脏话与贬称
    "傻逼", "人渣", "贱人", "脑残", "废物", "畜生", "牲口", "蟑螂", "蝗虫",
    "败类", "货物", "耗材", "工具人", "生育机器", "垃圾", "垃圾人",
    "蠢得像猪", "连畜生都不如", "又蠢又坏",
    # 动物化/疾病化群体蔑称
    "病毒", "细菌", "病灶", "传染源", "毒瘤",
    # 暴力威胁与诅咒
    "打死", "打死活该", "弄死", "冲爆", "收拾", "动手", "威胁", "想打人",
    "真想动手", "手痒", "该被打", "谁去教训一下", "见一次骂一次", "不得好死",
    "全家爆炸", "出门被车撞", "怎么还不死", "暴毙最好", "处理掉", "人道毁灭",
    "关到死", "直接判", "送走", "滚回去", "滚滚滚", "顶格处理", "清理门户",
    "驯服", "驱逐", "禁入", "消灭", "清除", "清洗", "清理",
    # 群体标签（争议议题词，可被提及）
    "地域", "民族", "宗教", "性别",
    # 排斥与歧视行为
    "不欢迎", "别搭理", "别来往", "别理他们", "拒绝接触", "离他们远点",
    "划清界限", "抵制", "碰瓷", "抹黑", "羞辱", "优化人口", "减少劣质基因",
    "伤风败俗", "以牙还牙", "一个不留", "抢工作", "抢资源", "带坏风气",
    "污染文化", "毁传统",
}


def build_lexicon(csv_path: str):
    """从 v6 问句括号锚定清单抽取词表，返回 (保留词表, 被删词及原因)。"""
    df = pd.read_csv(csv_path)
    kept, dropped = {}, {}
    for _, r in df.iterrows():
        for grp in re.findall(r"（([^）]{2,60})）", str(r["条目问句"])):
            for w in grp.split("/"):
                w = w.strip()
                if not w or w in kept or w in dropped:
                    continue
                if len(w) < 2 or len(w) > 6 or " " in w:
                    dropped[w] = "长度/空格"
                elif any(s in w for s in META_SUBSTR):
                    dropped[w] = "元描述"
                else:
                    kept[w] = str(r["构念id"])
    n_auto = len(kept)
    kept = {w: c for w, c in kept.items() if w in CURATED_TERMS}
    print(f"人工策展: 自动过滤 {n_auto} 词 -> 保留 {len(kept)} 词"
          f"（删 {n_auto - len(kept)}，问句不成立的句式/虚词/成语/元描述）")
    return kept, dropped


def find_occurrences(text: str, pattern: re.Pattern):
    """返回 [(term, pos)]，正则交替按词长降序排列以优先匹配最长词形。"""
    return [(m.group(0), m.start()) for m in pattern.finditer(text)]


def main():
    ap = argparse.ArgumentParser(description="v11 Stage1 词项定位")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--lexicon_csv", default=LEXICON_CSV)
    ap.add_argument("--output_dir", default="results/v11/usemention")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lex, dropped = build_lexicon(args.lexicon_csv)
    print(f"词表构建：保留 {len(lex)} 词项，删除 {len(dropped)} 项")
    print("  被删词（供人工复核）:")
    for w, why in sorted(dropped.items(), key=lambda x: x[1]):
        print(f"    [{why}] {w}")
    # 词长降序交替，长词形优先（避免"傻逼玩意"里短词先吞掉长词形）
    alts = sorted(lex.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(w) for w in alts))

    stats = []
    for split in ("train", "test"):
        src = args.train_matrix if split == "train" else args.test_matrix
        df = pd.read_csv(src)
        rows = []
        for _id, text in zip(df["id"].values, df["content"].astype(str).values):
            occ = find_occurrences(text, pattern)
            n_occ = len(occ)
            for k, (term, pos) in enumerate(occ):
                lo, hi = max(0, pos - CTX_RADIUS), pos + len(term) + CTX_RADIUS
                ctx = (text[lo:pos] + "「" + term + "」" + text[pos + len(term):hi])
                rows.append({"id": _id, "term": term, "occ_idx": k, "n_occ": n_occ,
                             "pos_start": pos, "ctx": ctx})
        occ_df = pd.DataFrame(rows)
        dst = out / f"occurrences_{split}.csv"
        occ_df.to_csv(dst, index=False, encoding="utf-8-sig")
        hit = df["id"].isin(set(occ_df["id"])) if len(occ_df) else np.zeros(len(df), bool)
        L = df["content"].astype(str).str.replace(" ", "").str.len()
        y = df["toxic"].values
        stats.append((split, len(df), int(hit.sum()), hit.mean(), int(len(occ_df))))
        print(f"[{split}] 样本 {len(df)}，命中样本 {int(hit.sum())}（{hit.mean():.3f}），"
              f"出现位置 {len(occ_df)}  -> {dst}")
        for lo, hi in ((0, 40), (40, 10**9)):
            m = (L >= lo) & (L < hi)
            if m.sum():
                print(f"    len[{lo},{hi}) n={m.sum():<5} 命中={hit[m].mean():.3f} "
                      f"(无毒 {hit[m & (y == 0)].mean():.3f} / 有毒 {hit[m & (y == 1)].mean():.3f})")

    with open(out / "lexicon.txt", "w", encoding="utf-8") as f:
        for w, cid in sorted(lex.items()):
            f.write(f"{w}\t{cid}\n")


if __name__ == "__main__":
    main()
