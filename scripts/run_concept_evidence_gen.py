#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0-gen 概念证据生成（服务器，本地 GLM + vLLM）。

验证"生成 vs 打分"：让 LLM 逐字引用证据片段 + 判断立场 + 归类概念（自由生成），
对比 M0 的 A/B 首 token 打分（其 use/mention 在 stage1="是" 子集 AUC 0.53 ≈ 随机）。
若本脚本生成出的"立场"能区分 TP（使用）与 FP（提及），则"生成"能力强于"打分"，
正向路径（概念证据生成 + 梯度对齐）成立。

输入：_m0gen_samples.json（id/toxic/group/content，本地 prep_m0gen_samples.py 产出）
输出：data/processed/TOXICN/m0gen_<model>_result.csv
      （id, toxic, group, content, evidence, stance, concept, reason, raw）

用法（服务器）：
  python scripts/run_concept_evidence_gen.py \
    --model_path /root/autodl-tmp/glm-4-9b-chat \
    --samples _m0gen_samples.json \
    --limit 8          # 先冒烟
  python scripts/run_concept_evidence_gen.py \
    --model_path /root/autodl-tmp/glm-4-9b-chat \
    --samples _m0gen_samples.json
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import csv
import json
import re
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PROMPT_SUFFIX = {"glm-4-9b-chat": "\n", "Qwen3.5-9B": "\n"}

GEN_TEMPLATE = """你是中文有害言论分析助手。对文本执行分析：逐字引用证据、判断立场、归类概念。

文本：{text}

请只输出一个 JSON（不要多余文字）：
{{"证据": "逐字引用文本中针对某人/某群体的负面或贬损表达的片段；无则空字符串", "立场": "使用 或 提及 或 无", "概念": "负面评价词 / 去人化隐喻 / 身份锚定 / 立场承诺 / 行动号召 / 转述引用 / 反歧视驳斥 / 语境取效 / 无", "理由": "一句话"}}

判定规则：
- 立场"使用"=这处贬损是说话人自己的表态（直接断言，或转述后明确背书）。
- 立场"提及"=这处贬损是转述/引用/驳斥他人，或假设/虚构情境里的贬损。
- 文本没有针对人/群体的贬损时：证据为空、立场"无"、概念"无"。

示例：
文本："我朋友说河南人偷井盖，这是偏见"
输出：{{"证据":"河南人偷井盖","立场":"提及","概念":"转述引用","理由":"转述他人说法并驳斥为偏见"}}

现在请分析："""


def load_vllm_model(model_path: Path, gpu_memory_utilization: float, max_model_len: int):
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_path}")
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading local vLLM model from: {model_path}", flush=True)
    llm = LLM(
        model=str(model_path), trust_remote_code=True, dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
        max_model_len=max_model_len, enforce_eager=True,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer) -> dict:
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def render_prompts(tokenizer, prompt_texts, chat_kwargs, model_name):
    suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    rendered = []
    for text in prompt_texts:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, **chat_kwargs
        )
        if suffix is not None:
            prompt += suffix
        elif not prompt.endswith("\n"):
            prompt += "\n"
        rendered.append(prompt)
    return rendered


def parse_output(raw: str) -> dict:
    """从 LLM 生成文本里提取 JSON 字段；容错（json.loads -> 首对花括号 -> 正则）。"""
    raw = (raw or "").strip()
    # 去掉可能的 ```json ... ``` 包裹
    raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()

    def _extract(obj):
        return {
            "evidence": str(obj.get("证据", obj.get("evidence", ""))).strip(),
            "stance": str(obj.get("立场", obj.get("stance", ""))).strip(),
            "concept": str(obj.get("概念", obj.get("concept", ""))).strip(),
            "reason": str(obj.get("理由", obj.get("reason", ""))).strip(),
        }

    # 1) 直接 json.loads
    try:
        return _extract(json.loads(raw))
    except Exception:
        pass
    # 2) 首对花括号内
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return _extract(json.loads(m.group(0)))
        except Exception:
            pass
    # 3) 正则逐字段
    def re_field(key):
        m = re.search(r'["\']?' + key + r'["\']?\s*[:：]\s*["\']?([^"\',}\]]+)', raw)
        return m.group(1).strip() if m else ""
    return {
        "evidence": re_field("证据") or re_field("evidence"),
        "stance": re_field("立场") or re_field("stance"),
        "concept": re_field("概念") or re_field("concept"),
        "reason": re_field("理由") or re_field("reason"),
    }


def parse_args():
    p = argparse.ArgumentParser(description="M0-gen 概念证据生成（本地 vLLM）")
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--samples", required=True, type=Path, help="_m0gen_samples.json")
    p.add_argument("--output_csv", type=Path, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条冒烟")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    samples_path = args.samples.expanduser().resolve()
    if not samples_path.is_file():
        raise FileNotFoundError(f"samples 不存在：{samples_path}")

    samples = json.load(open(samples_path, encoding="utf-8"))
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"冒烟限制：{len(samples)} 条")
    print(f"样本 {len(samples)} 条")

    tokenizer, llm = load_vllm_model(
        model_path, args.gpu_memory_utilization, args.max_model_len)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    model_name = model_path.name

    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

    rows = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="概念证据生成"):
        batch = samples[start:start + args.batch_size]
        texts = [GEN_TEMPLATE.format(text=s["content"]) for s in batch]
        rendered = render_prompts(tokenizer, texts, chat_kwargs, model_name)
        outputs = llm.generate(rendered, sampling_params, use_tqdm=False)
        for s, out in zip(batch, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            parsed = parse_output(raw)
            rows.append([s["id"], s["toxic"], s["group"], s["content"],
                         parsed["evidence"], parsed["stance"], parsed["concept"],
                         parsed["reason"], raw])

    out_csv = args.output_csv.expanduser().resolve() if args.output_csv else (
        samples_path.with_name(f"m0gen_{model_name}_result.csv"))
    header = ["id", "toxic", "group", "content", "evidence", "stance", "concept", "reason", "raw"]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # 简单统计：各 group 的立场分布
    from collections import Counter
    print("\n立场分布（按 group）：")
    for g in ("TP", "FP", "TN"):
        subset = [r for r in rows if r[2] == g]
        stance_counts = Counter(r[5] for r in subset)
        print(f"  {g} (n={len(subset)}): {dict(stance_counts)}")

    print(f"\nCSV -> {out_csv}")


if __name__ == "__main__":
    main()
