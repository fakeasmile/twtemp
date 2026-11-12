# -*- coding: utf-8 -*-
"""v10 指令消融测量（GPU vLLM，机制与 run_items_v6.py 逐字同源）。

假设：v6 测量模板的锚定条款（"变形出现同样算"）在指令层面命令教师把引用/
转述中的毒性内容计为命中——工具说明书本身在制造 use/mention 错误（FP 池
全部为 expr0 引用型长文本）。本脚本在同一批 pilot 样本 × 141 条目上测 3 种
指令变体，检验指令是否是 FP 池的成因：
  V0 对照：原 v6 模板（含锚定条款）
  V1      ：删除锚定条款，其余逐字相同
  V2      ：替换为直施条款（仅直接实施/表达时算，引用/转述/批评/讨论不算）

与 v6 协议的唯一机制差异：不做首 token 受限解码，取无约束 logprobs=50，
p_yes 事后归一化 = P(是)/(P(是)+P(否))（数值上与受限版近等价），并免费获得
回避质量 deflect = 1 - P(是) - P(否)（教师"想说别的"的程度，置信度通道）。

产物：results/v10/instr_ablation/ablation_measure.jsonl
  每行 {variant, id, group, item_id, p_yes, yes_mass, no_mass, deflect, valid}
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import math
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
from vllm import SamplingParams
from tqdm import tqdm

from run_items_v6 import (load_vllm_model, detect_chat_kwargs,
                          MODEL_PROMPT_SUFFIX, get_candidate_ids, load_v6_items)

project_root = Path(__file__).parent.parent

# 三个指令变体：第一句为共同角色定义，第二句为被消融的条款
COMMON = "你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。"
ANCHOR = "问句指出的现象以讽刺、委婉、谐音、缩写等变形出现同样算。"
DIRECT = "问句指出的现象仅在文本直接实施或表达该行为时才算；文本仅在引用、转述、批评、举例或讨论该现象时，不算。"

VARIANTS = {
    "V0": f"{COMMON} {ANCHOR}",
    "V1": COMMON,
    "V2": f"{COMMON} {DIRECT}",
}

TEMPLATE = """{instruction}

文本：{text}

问句：{question}

只回答“是/否”。回答："""


def main():
    ap = argparse.ArgumentParser(description="v10 指令消融测量（无掩码 logprobs）")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pilot_csv", required=True)
    ap.add_argument("--items_csv",
                    default=str(project_root / "data" / "concepts" / "cbm_concept_set_v6.csv"))
    ap.add_argument("--output_file",
                    default=str(project_root / "results/v10/instr_ablation/ablation_measure.jsonl"))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pilot = pd.read_csv(args.pilot_csv)
    assert {"id", "content"}.issubset(pilot.columns)
    items = load_v6_items(Path(args.items_csv))
    print(f"pilot 样本 {len(pilot)}，条目 {len(items)}，变体 {len(VARIANTS)}")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    model_name = Path(args.model_path).name
    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    yes_id, no_id = yes_ids[0], no_ids[0]
    print(f"是 token={yes_id}  否 token={no_id}")

    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50)
    suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    if suffix is None:
        print(f"[提示] 模型 {model_name} 未登记后缀，走自动修正（非换行结尾则补 \\n）")

    n_records = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for vname, instruction in VARIANTS.items():
            # 逐条目展开（问句槽在模板中部，必须逐条填充）
            prompts = []
            for row in pilot.itertuples(index=False):
                for it in items:
                    t = TEMPLATE.format(instruction=instruction,
                                        text=str(row.content), question=it["question"])
                    messages = [{"role": "user", "content": t}]
                    p = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        **chat_kwargs)
                    if suffix is not None:
                        p += suffix
                    elif not p.endswith("\n"):
                        p += "\n"
                    prompts.append(p)
            print(f"[{vname}] prompts = {len(prompts)}")
            outs = llm.generate(prompts, sp)
            # 顺序对应：pilot 行 × items 顺序
            k = 0
            for row in pilot.itertuples(index=False):
                for it in items:
                    lp_list = outs[k].outputs[0].logprobs
                    lp_dict = lp_list[0] if lp_list else {}
                    yes_m = math.exp(lp_dict[yes_id].logprob) if yes_id in lp_dict else 0.0
                    no_m = math.exp(lp_dict[no_id].logprob) if no_id in lp_dict else 0.0
                    denom = yes_m + no_m
                    p_yes = (yes_m / denom) if denom > 1e-12 else 0.5
                    rec = {"variant": vname, "id": row.id,
                           "group": getattr(row, "group", ""),
                           "item_id": it["item_id"], "p_yes": round(p_yes, 6),
                           "yes_mass": round(yes_m, 6), "no_mass": round(no_m, 6),
                           "deflect": round(1.0 - yes_m - no_m, 6),
                           "valid": denom > 1e-12}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    k += 1
                    n_records += 1
    print(f"完成：{n_records} 条记录 → {out_path}")


if __name__ == "__main__":
    main()
