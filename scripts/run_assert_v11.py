# -*- coding: utf-8 -*-
"""v11.1 Stage2：句子级双探针测量（GPU，GLM 单教师）。

对 Stage1 切出的每个句子测两个维度（各 1 token，无掩码 logprobs 协议）：
  探针A 句级毒性（内容层）：该句内容是否攻击/贬损现实中的某人或群体——
        只判内容本身，不管谁说的（与断言层正交分解）
  探针B 断言状态（结构层）：该句的判断是说话者【断言】的，还是【引用】
        转述他人、【假设】反事实、【反问】质疑——veridicality 四档
  p_yes(A) = P(是)/(P(是)+P(否))；B 输出四档质量分布，argmax 为硬标签。
候选取标签词首 token（是/否、断/引/假/反，互不重叠，自动校验）。

推理链路与 run_items_v6.py 逐字同源：load_vllm_model / detect_chat_kwargs /
render_prompts 直接导入复用。

产物：results/v11/assert/assert_{train,test}.json + assert_flat_{train,test}.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from run_items_v6 import detect_chat_kwargs, load_vllm_model, render_prompts

TOX_TEMPLATE = """你是语言学标注助手。判断单个句子的内容性质。只依据句子本身判断，不评价内容对错，不解释。

句子：{sent}

问句：这个句子的内容是否对现实中的某人或某群体构成攻击、贬损或仇恨表达（无论该内容是被谁说出的）？

只回答：是/否。回答："""

STATUS_TEMPLATE = """你是语言学标注助手。判断单个句子的表述方式。只依据句子本身判断，不评价内容对错，不解释。

句子：{sent}

问句：这个句子中的判断或评价，是说话者自己【断言】的，还是在【引用】或转述他人的话，或是【假设】性的、非现实的说法，或是【反问】/质疑性的说法？

只回答：断言/引用/假设/反问。回答："""

STATUS_KEYS = ("断言", "引用", "假设", "反问")


def first_token_id(word, tokenizer):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"标签词 {word!r} 编码为空")
    return ids[0]


def measure(llm, tokenizer, chat_kwargs, model_name, rows, template, labels,
            desc):
    """对 rows（sent 文本列表）跑单探针；返回逐句质量记录。"""
    prompts = [template.format(sent=s) for s in rows["sent"]]
    rendered = render_prompts(tokenizer, prompts, chat_kwargs, model_name)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50)
    cands = {w: first_token_id(w, tokenizer) for w in labels}
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
    ap = argparse.ArgumentParser(description="v11.1 Stage2 断言状态测量")
    ap.add_argument("--model_path", required=True, help="教师模型本地权重目录")
    ap.add_argument("--model_name", default="glm-4-9b-chat")
    ap.add_argument("--sent_train", default="results/v11/assert/sentences_train.csv")
    ap.add_argument("--sent_test", default="results/v11/assert/sentences_test.csv")
    ap.add_argument("--output_dir", default="results/v11/assert")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for split, path in (("train", args.sent_train), ("test", args.sent_test)):
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            df["split"] = split
            frames.append(df)
            print(f"[{split}] 句子 {len(df)}")
        else:
            print(f"[{split}] 不存在，跳过: {p}")
    if not frames:
        raise RuntimeError("无 sentences 输入，先运行 split_sentences_v11.py")
    sents = pd.concat(frames, ignore_index=True)

    tokenizer, llm = load_vllm_model(Path(args.model_path),
                                     args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    print(f"测量探针A（句级毒性，n={len(sents)}）...")
    tox = measure(llm, tokenizer, chat_kwargs, args.model_name, sents,
                  TOX_TEMPLATE, ("是", "否"), "tox")
    print(f"测量探针B（断言状态，n={len(sents)}）...")
    status = measure(llm, tokenizer, chat_kwargs, args.model_name, sents,
                     STATUS_TEMPLATE, STATUS_KEYS, "status")

    results, flat = [], []
    for (_, r), (tm, ttok), (sm, stok) in zip(sents.iterrows(), tox, status):
        tot_yes = tm["是"] + tm["否"]
        p_yes = tm["是"] / tot_yes if tot_yes > 1e-8 else float("nan")
        tot_s = sum(sm.values())
        if tot_s > 1e-8:
            w = {k: sm[k] / tot_s for k in STATUS_KEYS}
        else:
            w = {k: 0.25 for k in STATUS_KEYS}
        argmax = max(w, key=w.get)
        rec = {"split": r["split"], "id": r["id"], "sent_idx": int(r["sent_idx"]),
               "n_sent": int(r["n_sent"]), "char_len": int(r["char_len"]),
               "sent": r["sent"], "p_tox": p_yes,
               "deflect": 1.0 - tm["是"] - tm["否"], "m_yes": tm["是"],
               "m_no": tm["否"], "status_argmax": argmax,
               "status_masses": w, "tok_tox": ttok, "tok_status": stok}
        results.append(rec)
        flat.append({"split": r["split"], "id": r["id"],
                     "sent_idx": int(r["sent_idx"]), "n_sent": int(r["n_sent"]),
                     "p_tox": p_yes, "status_argmax": argmax,
                     **{f"w_{k}": w[k] for k in STATUS_KEYS},
                     "tok_tox": ttok, "tok_status": stok})

    for split in ("train", "test"):
        rs = [r for r in results if r["split"] == split]
        if not rs:
            continue
        with open(out / f"assert_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": {"model": args.model_name, "n_sent": len(rs),
                                "template": "v11_assert_v1"},
                       "results": rs}, f, ensure_ascii=False)
        fs = [r for r in flat if r["split"] == split]
        pd.DataFrame(fs).to_csv(out / f"assert_flat_{split}.csv", index=False,
                                encoding="utf-8-sig")
        n_nan = sum(1 for r in rs if r["p_tox"] != r["p_tox"])
        from collections import Counter
        cnt = Counter(r["status_argmax"] for r in rs)
        print(f"[{split}] 测量 {len(rs)} 句，p_tox 无效 {n_nan}，状态分布 "
              f"{dict(cnt)} -> assert_{split}.json")


if __name__ == "__main__":
    main()
