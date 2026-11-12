# -*- coding: utf-8 -*-
"""v11.2 Stage2：攻击目标探针（GPU，GLM 单教师）。

v11.1 机制诊断的后续：无毒文本中说话者自己的"骂回/谴责"也是"断言+攻击"
（1155 句实证），缺失的判别维度是【攻击谁】。本脚本对 v11.1 测得
p_tox>=0.5 的句子（train 9906 + test 2492 = 12398 句）加测探针C：

  探针C 攻击目标（三选一）：
    当事人  = 被攻击、被贬损的群体或个人（仇恨表达的真实受害者）
    言论者  = 说出/传播被引述或讨论言论的人（骂回、谴责、嘲讽发言者）
    其他    = 无明确攻击对象、仅针对物/事/行为本身、自嘲

预测机制：直接攻击文本的断言毒性集中在【当事人】；无毒引用+谴责文本的
断言毒性集中在【言论者】。特征 = 断言句中"当事人向攻击"的最大毒性。

每句 1 token，无掩码 logprobs；候选取标签首 token（当/言/其，互不重叠，
自动校验）。推理链路与 run_items_v6.py / run_assert_v11.py 逐字同源。

产物：results/v11/target/target_{train,test}.json + target_flat_{train,test}.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from run_items_v6 import detect_chat_kwargs, load_vllm_model, render_prompts

TARGET_TEMPLATE = """你是语言学标注助手。判断单个句子的攻击指向。只依据句子本身判断，不评价内容对错，不解释。

句子：{sent}

问句：若这个句子中存在攻击、贬损或仇恨表达，其攻击对象是【当事人】（被攻击、被贬损的群体或个人），还是【言论者】（说出或传播被引述/讨论言论的人，例如骂回、谴责、嘲讽这类人），还是【其他】（无明确攻击对象、仅针对物/事/行为本身、自嘲等）？

只回答：当事人/言论者/其他。回答："""

TARGET_KEYS = ("当事人", "言论者", "其他")


def first_token_id(word, tokenizer):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"标签词 {word!r} 编码为空")
    return ids[0]


def measure(llm, tokenizer, chat_kwargs, model_name, rows, desc):
    prompts = [TARGET_TEMPLATE.format(sent=s) for s in rows["sent"]]
    rendered = render_prompts(tokenizer, prompts, chat_kwargs, model_name)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50)
    cands = {w: first_token_id(w, tokenizer) for w in TARGET_KEYS}
    if len(set(cands.values())) != len(cands):
        raise ValueError(f"标签首 token 重叠: {cands}")
    out_mass = []
    BATCH = 20000
    for i in range(0, len(rendered), BATCH):
        outs = llm.generate(rendered[i:i + BATCH], sp)
        for o in outs:
            lp_list = o.outputs[0].logprobs
            lp_dict = lp_list[0] if lp_list else {}
            mass = {w: (math.exp(lp_dict[t].logprob) if t in lp_dict else 0.0)
                    for w, t in cands.items()}
            out_mass.append((mass, o.outputs[0].text))
    return out_mass


def main():
    ap = argparse.ArgumentParser(description="v11.2 攻击目标探针测量")
    ap.add_argument("--model_path", required=True, help="教师模型本地权重目录")
    ap.add_argument("--model_name", default="glm-4-9b-chat")
    ap.add_argument("--assert_dir", default="results/v11/assert",
                    help="v11.1 断言测量输出目录（读 assert_{split}.json）")
    ap.add_argument("--output_dir", default="results/v11/target")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("train", "test"):
        src = Path(args.assert_dir) / f"assert_{split}.json"
        if not src.exists():
            print(f"[{split}] 不存在，跳过: {src}")
            continue
        with open(src, encoding="utf-8") as f:
            recs = json.load(f)["results"]
        sel = [r for r in recs
               if r["p_tox"] == r["p_tox"] and r["p_tox"] >= 0.5]
        df = pd.DataFrame(sel)
        df["split"] = split
        frames.append(df)
        print(f"[{split}] 毒性句 p_tox>=0.5: {len(sel)} / {len(recs)}")
    if not frames:
        raise RuntimeError("无 assert 输入，先运行 run_assert_v11.py")
    sents = pd.concat(frames, ignore_index=True)

    tokenizer, llm = load_vllm_model(Path(args.model_path),
                                     args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    print(f"测量探针C（攻击目标，n={len(sents)}）...")
    tgt = measure(llm, tokenizer, chat_kwargs, args.model_name, sents, "target")

    from collections import Counter
    cnt_all = Counter()
    results_by_split = {"train": [], "test": []}
    flat_by_split = {"train": [], "test": []}
    for (_, r), (tm, ttok) in zip(sents.iterrows(), tgt):
        tot = sum(tm.values())
        if tot > 1e-8:
            w = {k: tm[k] / tot for k in TARGET_KEYS}
        else:
            w = {k: 1.0 / len(TARGET_KEYS) for k in TARGET_KEYS}
        argmax = max(w, key=w.get)
        cnt_all[argmax] += 1
        rec = {"split": r["split"], "id": r["id"],
               "sent_idx": int(r["sent_idx"]), "n_sent": int(r["n_sent"]),
               "sent": r["sent"], "p_tox": float(r["p_tox"]),
               "status_argmax": r["status_argmax"],
               "w_断言": float(r["status_masses"]["断言"]),
               "target_argmax": argmax, "target_masses": w, "tok": ttok}
        results_by_split[r["split"]].append(rec)
        flat_by_split[r["split"]].append({
            "split": r["split"], "id": r["id"],
            "sent_idx": int(r["sent_idx"]), "n_sent": int(r["n_sent"]),
            "p_tox": float(r["p_tox"]), "status_argmax": r["status_argmax"],
            "w_断言": float(r["status_masses"]["断言"]),
            "target_argmax": argmax,
            **{f"w_{k}": w[k] for k in TARGET_KEYS}, "tok": ttok})

    for split in ("train", "test"):
        rs = results_by_split[split]
        if not rs:
            continue
        with open(out / f"target_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": {"model": args.model_name, "n_sent": len(rs),
                                "template": "v11_target_v1"},
                       "results": rs}, f, ensure_ascii=False)
        pd.DataFrame(flat_by_split[split]).to_csv(
            out / f"target_flat_{split}.csv", index=False, encoding="utf-8-sig")
        print(f"[{split}] 测量 {len(rs)} 句，目标分布 {dict(cnt_all)}"
              f" -> target_{split}.json")


if __name__ == "__main__":
    main()
