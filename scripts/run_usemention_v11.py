# -*- coding: utf-8 -*-
"""v11 Stage2：词项条件化 use/mention 测量（GPU，GLM 单教师）。

对 Stage1 定位的每个出现位置，向教师提词项条件化二选一问句：
  【使用】说话者用该词指称、攻击或评价某人/群体
  【提及】引用、转述、举例、批评该词本身或讨论相关话题，而非用它攻击谁
这是两阶段比较式提问（stage2 强度分类已验证 9B 可稳定执行的形态），
单概念、单位置，不违反多概念禁令；上下文窗口与「」标记由确定性代码给出。

测量协议与 v10 指令消融同构（无掩码协议）：max_tokens=1, temperature=0,
logprobs=50 自然分布，p_use = P(使用)/(P(使用)+P(提及)) 事后归一化，
deflect = 1 − P(使用) − P(提及)。候选取标签词首 token（"使用"→"使"、
"提及"→"提"，兼容多 token 词形；首 token 不同即可分离）。

推理链路与 run_items_v6.py 逐字同源：load_vllm_model / detect_chat_kwargs /
MODEL_PROMPT_SUFFIX / render_prompts 均直接导入复用。

产物：results/v11/usemention/um_{train,test}.json + um_flat_{train,test}.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from run_items_v6 import (detect_chat_kwargs, load_vllm_model,
                          render_prompts)

UM_TEMPLATE = """你是语言学标注助手。判断文本片段中「」标注的词项的用法。只依据片段本身判断，不评价内容对错，不解释。

片段：{ctx}

问句：「」中的词「{term}」在片段中是被【使用】（说话者用该词指称、攻击或评价某人或群体），还是被【提及】（引用、转述、举例、批评该词本身或讨论相关话题，而非用它攻击谁）？

只回答：使用/提及。回答："""


def first_token_id(word, tokenizer):
    """标签词首 token id（兼容多 token 词形；不同标签首 token 不同即分离）。"""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"标签词 {word!r} 编码为空")
    return ids[0], len(ids)


def main():
    ap = argparse.ArgumentParser(description="v11 Stage2 use/mention 测量")
    ap.add_argument("--model_path", required=True, help="教师模型本地权重目录")
    ap.add_argument("--model_name", default="glm-4-9b-chat")
    ap.add_argument("--occ_train", default="results/v11/usemention/occurrences_train.csv")
    ap.add_argument("--occ_test", default="results/v11/usemention/occurrences_test.csv")
    ap.add_argument("--output_dir", default="results/v11/usemention")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    occs = []
    for split, path in (("train", args.occ_train), ("test", args.occ_test)):
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            df["split"] = split
            occs.append(df)
            print(f"[{split}] 出现位置 {len(df)}")
        else:
            print(f"[{split}] 不存在，跳过: {p}")
    if not occs:
        raise RuntimeError("无 occurrences 输入，先运行 locate_terms_v11.py")
    occ = pd.concat(occs, ignore_index=True)
    print(f"合计出现位置 {len(occ)}，独立样本 "
          f"{occ.groupby('split')['id'].nunique().to_dict()}")

    model_dir = Path(args.model_path)
    tokenizer, llm = load_vllm_model(model_dir, args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    use_ids, mention_ids = [], []
    for word, bucket in (("使用", use_ids), ("提及", mention_ids)):
        tid, n_tok = first_token_id(word, tokenizer)
        bucket.append(tid)
        print(f"标签 {word!r}: 首 token id={tid}（词共 {n_tok} token）")
    if set(use_ids) & set(mention_ids):
        raise ValueError("使用/提及首 token 重叠，候选不可分离")

    prompts = []
    for _, r in occ.iterrows():
        content = UM_TEMPLATE.format(ctx=r["ctx"], term=r["term"])
        prompts.append((r["split"], r["id"], r["term"], int(r["occ_idx"]), content))
    rendered = render_prompts(tokenizer, [p[4] for p in prompts], chat_kwargs,
                              args.model_name)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50)

    results, flat = [], []
    BATCH = 20000
    for i in range(0, len(rendered), BATCH):
        outs = llm.generate(rendered[i:i + BATCH], sp)
        for (split, _id, term, k, _), o in zip(prompts[i:i + BATCH], outs):
            lp_list = o.outputs[0].logprobs
            lp_dict = lp_list[0] if lp_list else {}
            m_use = sum(math.exp(lp_dict[t].logprob) for t in use_ids
                        if t in lp_dict)
            m_ment = sum(math.exp(lp_dict[t].logprob) for t in mention_ids
                         if t in lp_dict)
            tot = m_use + m_ment
            p_use = m_use / tot if tot > 1e-8 else float("nan")
            gen = o.outputs[0].text
            results.append({"split": split, "id": _id, "term": term,
                            "occ_idx": k, "p_use": p_use,
                            "deflect": 1.0 - m_use - m_ment,
                            "m_use": m_use, "m_mention": m_ment,
                            "first_token": gen})
            flat.append({"split": split, "id": _id, "term": term, "occ_idx": k,
                         "p_use": p_use, "first_token": gen})

    for split in ("train", "test"):
        rs = [r for r in results if r["split"] == split]
        if not rs:
            continue
        with open(out / f"um_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": {"model": args.model_name, "n_occ": len(rs),
                                "template": "v11_um_v1"},
                       "results": rs}, f, ensure_ascii=False)
        pd.DataFrame([r for r in flat if r["split"] == split]).to_csv(
            out / f"um_flat_{split}.csv", index=False, encoding="utf-8-sig")
        n_nan = sum(1 for r in rs if r["p_use"] != r["p_use"])
        print(f"[{split}] 测量 {len(rs)} 条，p_use 无效 {n_nan} 条"
              f"（两候选质量均≈0） -> um_{split}.json")


if __name__ == "__main__":
    main()
