# -*- coding: utf-8 -*-
"""v17 臂③ Stage1（远端 GPU）：DC-PMI 校准的先验读数重测。

机制（Holtzman et al., EMNLP 2021 "Surface Form Competition" 的 DC-PMI；
Zhao et al. 2021 contentless 校准的占位符做法）：
  校准 = logit p(是|x, item) − logit p(是|contentless, item)
先验读数与 run_items_v6.py 的条件测量**逐字同构**（同一 ITEM_TEMPLATE、
同一 chat 渲染、同一 受限解码 + 是/否 质量归一化、同一 verbalizer），
唯一区别：{text} 槽换成中性占位符。占位符两套（预登记）：
  na = "N/A"（Zhao et al. 2021 原版）
  wu = "无"（中文对应物）
每 (占位符, 条目) 1 次前向：2 × 152 = 304 次，分钟级。
产物：results/v17/dcpmi/prior_{model_name}.json
  （每次推理记录首 token，含 {item_id: {占位符: p_prior}} + 首token登记）
复用方式：import run_items_v6（同目录），测量函数零复制。
"""

import argparse
import json
import time
from pathlib import Path

import run_items_v6 as rim

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()
PLACEHOLDERS = {"na": "N/A", "wu": "无"}


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="v17b DC-PMI 先验读数")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--items_csv", default="data/concepts/cbm_concept_set_v6.csv")
    ap.add_argument("--output_dir", default="results/v17/dcpmi")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name

    items = rim.load_v6_items(Path(args.items_csv))
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids)), "条目 id 重复"
    stamp(f"[条目] {len(items)} 条（{args.items_csv}）")

    tokenizer, llm = rim.load_vllm_model(Path(args.model_path),
                                         args.gpu_memory_utilization)
    chat_kwargs = rim.detect_chat_kwargs(tokenizer)
    yes_ids = rim.get_candidate_ids(["是"], tokenizer)
    no_ids = rim.get_candidate_ids(["否"], tokenizer)
    sp_yn = rim.build_sampling_params(yes_ids + no_ids)
    stamp(f"verbalizer: 是={yes_ids} 否={no_ids}")

    # 先验 prompt：占位符 × 条目（每条目与条件测量同模板，仅换 text 槽）
    prompts, keys = [], []
    for ph_name, ph_text in PLACEHOLDERS.items():
        for it in items:
            prompts.append(rim.ITEM_TEMPLATE.format(text=ph_text,
                                                    question=it["question"]))
            keys.append((ph_name, it["item_id"]))
    stamp(f"[先验] {len(prompts)} 次前向...")
    outs = llm.generate(rim.render_prompts(tokenizer, prompts, chat_kwargs,
                                           model_name),
                        sp_yn, use_tqdm=True)

    prior = {ph: {} for ph in PLACEHOLDERS}
    first_tok = {ph: {} for ph in PLACEHOLDERS}
    n_invalid = 0
    for (ph, iid), o in zip(keys, outs):
        lp = o.outputs[0].logprobs[0]
        tok = tokenizer.decode(list(o.outputs[0].token_ids))
        masses, norm = rim.candidate_mass(lp, [yes_ids, no_ids])
        first_tok[ph][iid] = tok
        if sum(masses) > 0:
            prior[ph][iid] = round(norm[0], 4)
        else:
            n_invalid += 1
    assert n_invalid == 0, f"先验读数出现无效判定 {n_invalid}（受限解码后应为 0）"
    for ph in PLACEHOLDERS:
        assert len(prior[ph]) == len(items), f"{ph} 先验条目数不足"

    for ph in PLACEHOLDERS:
        vals = list(prior[ph].values())
        stamp(f"[{ph}] p_prior: min={min(vals):.4f} 中位="
              f"{sorted(vals)[len(vals) // 2]:.4f} max={max(vals):.4f}")

    meta = {"model": model_name, "model_path": args.model_path,
            "items_csv": args.items_csv, "n_items": len(items),
            "placeholders": PLACEHOLDERS,
            "mechanism": "ITEM_TEMPLATE 同模板换 text 槽；受限解码+是/否质量归一化，"
                         "与 run_items_v6 逐字同源",
            "verbalizer_yes": yes_ids, "verbalizer_no": no_ids,
            "n_invalid": n_invalid}
    (out / f"prior_{model_name}.json").write_text(
        json.dumps({"meta": meta, "prior": prior, "first_token": first_tok},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    stamp(f"[完成] prior_{model_name}.json 已写出")


if __name__ == "__main__":
    main()
