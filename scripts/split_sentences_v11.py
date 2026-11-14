# -*- coding: utf-8 -*-
"""v11.1 Stage1：句子切分（确定性代码，零依赖，零 GPU）。

理论动机（v11 复盘后用户裁定）：词表定位器绑定语料词形分布，跨数据集
（如 COLD 泛化实验）不可迁移；改用测量对象单位的结构化推广——
断言状态（veridicality）测量。use/mention 的结构化本质是句子的嵌入层级：
  断言 > 引用转述 > 假设反事实 > 反问质疑
教师出错机制 = 文本级测量把"嵌入的冒犯性命题"与"被断言的冒犯性命题"
同等计权（Kintsch 命题分解：概念测量从文本级降到句子级）。

切分规则：按句末标点（。！？!?…）切分，仅在引号深度为零时切——
引语（「」『』“”‘’《》()）内部的句末标点不切分，保住引用归属结构。
纯代码实现，无词表、无解析器依赖，跨数据集直接可用。

产物：results/v11/assert/sentences_{train,test}.csv
      列：id, sent_idx, n_sent, sent, char_len
"""

import argparse
import re
from pathlib import Path

import pandas as pd

SENT_END = "。！？!?…"
PAIRS = {"「": "」", "『": "』", "“": "”", "‘": "’", "《": "》", "(": ")",
         "（": "）"}
CLOSE = set(PAIRS.values())


def split_sentences(text: str):
    """按句末标点切分，引号深度>0 时不切；返回非空句列表。"""
    sents, start, depth = [], 0, 0
    for i, ch in enumerate(text):
        if ch in PAIRS:
            depth += 1
        elif ch in CLOSE:
            depth = max(0, depth - 1)
        elif ch in SENT_END and depth == 0:
            s = text[start:i + 1].strip()
            if s:
                sents.append(s)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents


def main():
    ap = argparse.ArgumentParser(description="v11.1 Stage1 句子切分")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--output_dir", default="results/v11/assert")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src = args.train_matrix if split == "train" else args.test_matrix
        df = pd.read_csv(src)
        rows = []
        for _id, text in zip(df["id"].values, df["content"].astype(str).values):
            sents = split_sentences(text)
            for k, s in enumerate(sents):
                rows.append({"id": _id, "sent_idx": k, "n_sent": len(sents),
                             "sent": s, "char_len": len(s)})
        sdf = pd.DataFrame(rows,
                           columns=["id", "sent_idx", "n_sent", "sent", "char_len"])
        dst = out / f"sentences_{split}.csv"
        sdf.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"[{split}] 样本 {len(df)}，句子 {len(sdf)}"
              f"（均 {len(sdf) / max(len(df), 1):.2f} 句/样本，"
              f"最长 {sdf['char_len'].max() if len(sdf) else 0} 字）-> {dst}")


if __name__ == "__main__":
    main()
