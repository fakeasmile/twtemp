# -*- coding: utf-8 -*-
"""解析 v3 rings JSON → GLM 教师 soft 编码完整宽表（对齐 train_concept_lr 的 99 维配方）。

train_concept_lr 的冻结编码（encode_features）：
  p_yes（全部概念）+ P(直接)=p_yes·p_B（stage2 触发概念）+ p_A（stage2 触发概念）
  + S1A 六档概率 + S1B 六档概率 + CAT 六通道 p_yes

本脚本抽取全部原始通道，未触发概念 p_B/p_A/P直接 填 0（等价于 concept_lr 的
s2_cids 触发集 + 后续方差过滤）。列前缀：
  v3p_<id>   stage1 p_yes
  v3b_<id>   stage2 p_B（未触发 0）
  v3d_<id>   stage2 p_A（未触发 0）
  v3pd_<id>  P(直接) = p_yes × p_B（未触发 0）
  s1a_p0..5  样本级 S1A 六档概率
  s1b_p0..5  样本级 S1B 六档概率
  cat_<k>    CAT 六通道 p_yes（CAT1A/CAT1B/CAT2A/CAT2B/CAT3/CAT4 中存在者）

输出：data/processed/TOXICN/glm-4-9b-chat_v3/v3_soft_{train,test}_matrix.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CAT_ORDER = ("CAT1A", "CAT1B", "CAT2A", "CAT2B", "CAT3", "CAT4")


def parse_one(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    order = [c["id"] for c in meta["concepts"]]
    cat_names = [k for k in CAT_ORDER if k in data["results"][0]]
    print(f"  {json_path.name}: {meta['n_samples']} 样本, {len(order)} 概念, "
          f"CAT 通道 {cat_names}")

    rows = []
    for r in data["results"]:
        row = {"id": r["id"], "toxic": int(r["toxic"])}
        cpts = r["concepts"]
        for cid in order:
            c = cpts.get(cid, {})
            s1 = c.get("stage1")
            s2 = c.get("stage2")
            p_yes = s1["p_yes"] if s1 else None
            row[f"v3p_{cid}"] = p_yes
            p_b = s2["p_B"] if s2 and s2.get("p_B") is not None else 0.0
            p_a = s2["p_A"] if s2 and s2.get("p_A") is not None else 0.0
            row[f"v3b_{cid}"] = p_b
            row[f"v3d_{cid}"] = p_a
            row[f"v3pd_{cid}"] = (p_yes * p_b) if p_yes is not None else 0.0
        s1a = (r.get("S1A") or {}).get("probs", {})
        s1b = (r.get("S1B") or {}).get("probs", {})
        for k in range(6):
            row[f"s1a_p{k}"] = float(s1a.get(str(k), 0.0))
            row[f"s1b_p{k}"] = float(s1b.get(str(k), 0.0))
        for k in cat_names:
            row[f"cat_{k}"] = r[k]["p_yes"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="解析 v3 rings JSON 为 soft 完整宽表")
    ap.add_argument("--v3_dir", default="data/processed/TOXICN/glm-4-9b-chat_v3")
    ap.add_argument("--model_name", default="glm-4-9b-chat",
                    help="rings 文件名中的模型名（如 Qwen3.5-9B）")
    args = ap.parse_args()
    v3_dir = Path(args.v3_dir)

    for split in ("train", "test"):
        src = v3_dir / f"rings_{args.model_name}_{split}.json"
        dst = v3_dir / f"v3_soft_{split}_matrix.csv"
        if not src.exists():
            print(f"跳过（不存在）: {src}")
            continue
        df = parse_one(src)
        feat_cols = [c for c in df.columns if c not in ("id", "toxic")]
        n_const = int(sum(df[c].nunique() <= 1 for c in feat_cols))
        print(f"    特征列 {len(feat_cols)}，恒定列 {n_const}")
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  -> {dst}  shape={df.shape}")


if __name__ == "__main__":
    main()
