# -*- coding: utf-8 -*-
"""v10 pilot 测量：开放生成引出言语行为判断（vLLM，机制与 run_items_v6.py 同源）。

与封闭单 token 探测的本质区别：不约束首 token，temperature=0，
让教师先自由展开分析再给出结构化结论（投射式开放反应——MMPI 传统中
自陈量表之外的测量模式）。生成文本整体保存（供 TF-IDF 诊断），
结构化结论（行为/指向/严肃性）由解析脚本提取。

提示词设计：要求"先简要分析再下结论"，自由分析部分允许教师展开推理
（意图判断只有在此处才可能浮出），结论部分用固定槽位保证可解析。

输出：results/v10/pilot/rationales_glm.jsonl（id/group/rationale）
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}

PROMPT = """你是语言学标注助手。请仔细阅读下面这条网络评论，分析说话者的言语行为。先简要分析（不超过80字），再按以下格式给出结论。

文本：{text}

分析后按如下格式回答：
行为：[攻击辱骂/贬损蔑视/引用转述/讨论评价/反讽挖苦/玩笑调侃/其他]
指向：[文中具体对象/直接对话者/泛指群体/无特定对象]
严肃性：[严肃恶意/玩笑夸张/无法判断]"""


def load_vllm_model(llm_path: Path, gpu_memory_utilization: float):
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading vLLM model from {llm_path}")
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,      # 每条文本只问一次，无前缀复用收益
        max_model_len=2048,
        max_num_seqs=64,
        enforce_eager=True,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def render_prompts(tokenizer, texts, chat_kwargs, model_name):
    manual_suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    prompts = []
    for t in texts:
        p = tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False,
            add_generation_prompt=True, **chat_kwargs)
        if manual_suffix is not None:
            p += manual_suffix
        elif not p.endswith("\n"):
            p += "\n"
        prompts.append(p)
    return prompts


def main():
    ap = argparse.ArgumentParser(description="v10 pilot 开放生成测量")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pilot_csv", required=True)
    ap.add_argument("--output_file", default="results/v10/pilot/rationales_glm.jsonl")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = ap.parse_args()

    pilot = pd.read_csv(args.pilot_csv)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    model_name = Path(args.model_path).name

    prompts = render_prompts(tokenizer, pilot["content"].tolist(), chat_kwargs, model_name)
    sp = SamplingParams(max_tokens=160, temperature=0)
    outs = llm.generate(prompts, sp)

    with open(out, "w", encoding="utf-8") as f:
        for row, o in zip(pilot.itertuples(index=False), outs):
            text = o.outputs[0].text.strip()
            f.write(json.dumps({"id": row.id, "group": row.group,
                                "toxic": int(row.toxic), "rationale": text},
                               ensure_ascii=False) + "\n")
    print(f"{len(pilot)} 条 rationale 写入 {out}")


if __name__ == "__main__":
    main()
