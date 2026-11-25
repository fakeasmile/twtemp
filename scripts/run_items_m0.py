#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToxiCN CBM M0 受控"使用 vs 提及"测量脚本（本地 GLM + vLLM）。

目标
----
按 docs/M0_受控使用提及测量_v1.md 的两阶段协议，产出单一分数 s_use ∈ [0,1]：
  阶段1（存在性，出现性锚定）：文本是否出现针对某人/某群体的负面/贬损态度
        （转述/引用/驳斥中的"提及"也算）→ 首 token 是/否，记 p_exists。
  阶段2（强制选择，仅阶段1判"是"时触发）：
        该处贬损是 A 说话人自己的表态（使用）还是 B 转述/引用/驳斥他人的表态（提及）
        → 首 token A/B，记 s_use = p(A)/(p(A)+p(B))。
  阶段1判"否"的样本：s_use = 0（无贬损可谈，不进阶段2）。

s_use 即主张文档 §4 中 M2（概念对齐正则）的对齐目标；其验收标准见 M0 设计文档 §4.1
（反相关检验：mean(s_use|TP) − mean(s_use|FP) ≥ 0.15 且 FP≈TN）。

设计原则（沿用 run_items_v8.py）
--------------------------------
1. 模型加载 / chat 模板 / 首 token 概率提取 / 受限解码 沿用 v8。
2. 两阶段分两遍跑：先全量阶段1，再对"是"样本阶段2；各自独立 checkpoint，可断点续跑。
3. 仅接受本地模型目录；离线模式，禁止远程下载。

用法（服务器，vLLM）
--------------------
# 冒烟测试（8 条）
python scripts/run_items_m0.py --model_path /root/autodl-tmp/glm-4-9b-chat \
    --data_path data/raw/TOXICN --split test --limit 8

# 全量（test 2411 条；train 9600 条同理，--split train）
python scripts/run_items_m0.py --model_path /root/autodl-tmp/glm-4-9b-chat \
    --data_path data/raw/TOXICN --split test

产物：data/processed/TOXICN/m0_<model>_<split>_use.csv
      data/processed/TOXICN/m0_<model>_<split>_use.json
      （列：id, toxic, content, p_exists, exists_token, s_use, use_token,
            cand_mass_stage1, cand_mass_stage2）
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import math
import os.path
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}

STAGE1_TEMPLATE = """你是语言学标注助手。判断文本中是否出现针对某人或某群体的负面/贬损态度（鄙夷、厌恶、敌意、威胁、道德定罪、限制其应处位置等）。转述、引用、驳斥他人观点时的“提及”同样算出现。只依据文本本身判断，禁止用世界知识推测作者意图。只回答“是”或“否”。

文本：{text}
回答："""

STAGE2_TEMPLATE = """你是语言学标注助手。文本中出现了一处针对某人/某群体的负面或贬损态度。判断这处态度是：
A = 说话人自己的表态（使用）——说话人直接断言，或转述他人后明确背书；
B = 转述/引用/驳斥他人的表态（提及）——纯转述无背书、转述后驳斥、评价指向攻击者本身、或假设/虚构情境里的贬损。

判断示例：
“黑真是太丑” → A
“我朋友说河南人偷井盖” → B
“我朋友说河南人偷井盖，确实如此” → A
“我朋友说河南人偷井盖，这是偏见” → B
“骂黑人的人才是真劣等” → B

只回答 A 或 B。

文本：{text}
回答："""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sample(item: dict, source_name: str, split: str, idx: int) -> dict:
    if not isinstance(item, dict):
        raise TypeError(f"数据第 {idx + 1} 条不是 JSON object")
    if "content" not in item or item["content"] is None:
        raise ValueError(f"数据第 {idx + 1} 条缺少 content")
    content = str(item["content"])
    if not content.strip():
        raise ValueError(f"数据第 {idx + 1} 条 content 为空")
    sid = item.get("id")
    if sid is None or str(sid).strip() == "":
        sid = f"{source_name}_{split}_{idx:04d}"
    return {
        "id": str(sid),
        "source": str(item.get("source") or source_name),
        "split": split,
        "content": content,
        "toxic": item.get("toxic"),
        "topic": item.get("topic"),
        "toxic_type": item.get("toxic_type"),
        "expression": item.get("expression"),
        "target": item.get("target"),
    }


def load_dataset(data_path: Path, split: str) -> Tuple[List[dict], List[str], str, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"data_path 不存在：{data_path}")
    if data_path.is_file():
        json_path = data_path
        dataset_tag = data_path.parent.name or data_path.stem
        actual_split = data_path.stem
        files = [data_path.name]
    else:
        json_path = data_path / f"{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"{data_path} 下找不到 {split}.json")
        dataset_tag = data_path.name
        actual_split = split
        files = [json_path.name]

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"数据文件顶层必须是 JSON list：{json_path}")
    samples = [normalize_sample(x, dataset_tag, actual_split, i) for i, x in enumerate(raw)]
    if not samples:
        raise ValueError(f"数据集为空：{json_path}")
    ids = [x["id"] for x in samples]
    if len(ids) != len(set(ids)):
        duplicates = sorted({x for x in ids if ids.count(x) > 1})[:20]
        raise ValueError(f"数据样本 id 不唯一（前 20 个重复值）：{duplicates}")
    return samples, files, dataset_tag, json_path


def load_vllm_model(model_path: Path, gpu_memory_utilization: float,
                    max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json：{model_path}")

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer 的 pad_token 与 eos_token 均为空")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading local vLLM model from: {model_path}", flush=True)
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=True,
        max_logprobs=50,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer) -> dict:
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


def get_candidate_ids(words: List[str], tokenizer) -> List[int]:
    """只接受单 token verbalizer。"""
    ids = []
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"verbalizer {word!r} 在当前 tokenizer 下不是单 token：{token_ids}。"
                "不能继续读取首 token 概率。"
            )
        ids.extend(token_ids)
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError(f"verbalizer 无有效 token：{words}")
    return ids


def make_mask_processor(allowed_ids: List[int]):
    allowed = sorted(set(allowed_ids))

    def _processor(*args, **kwargs):
        import torch
        logits = kwargs.get("logits")
        if logits is None:
            logits = next(a for a in args if isinstance(a, torch.Tensor))
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked

    return _processor


def build_sampling_params(allowed_ids: List[int]):
    from vllm import SamplingParams

    allowed = sorted(set(allowed_ids))
    candidates = [
        ("logits_processors", {"logits_processors": [make_mask_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束回退", {}),
    ]
    errors = []
    for mode, extra in candidates:
        try:
            params = SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra)
            print(f"首 token 约束方式：{mode}；候选 token ids={allowed}")
            return params, mode
        except TypeError as exc:
            errors.append(f"{mode}: {exc}")
    raise RuntimeError("当前 vLLM 无法建立 SamplingParams：" + " | ".join(errors))


def render_prompts(tokenizer, prompt_texts: List[str], chat_kwargs: dict, model_name: str) -> List[str]:
    manual_suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    rendered = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if manual_suffix is not None:
            prompt += manual_suffix
        elif not prompt.endswith("\n"):
            prompt += "\n"
        rendered.append(prompt)
    return rendered


def _logprob_value(value) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def get_two_way_result(request_output, tokenizer, ids_a: List[int], ids_b: List[int],
                       label_a: str, label_b: str) -> dict:
    """提取首 token 的 p_a（在 a/b 候选内归一化）与候选概率质量。"""
    if not request_output.outputs:
        raise RuntimeError("vLLM 返回空 outputs")
    completion = request_output.outputs[0]
    if not completion.token_ids:
        raise RuntimeError("模型没有生成首 token")
    if not completion.logprobs or not completion.logprobs[0]:
        raise RuntimeError("模型没有返回首 token logprobs；请检查 SamplingParams(logprobs=50)")

    token_id = int(completion.token_ids[0])
    first_token = (label_a if token_id in ids_a else
                   (label_b if token_id in ids_b else
                    tokenizer.decode([token_id], skip_special_tokens=False)))
    logprob_map = completion.logprobs[0]

    def mass_for(ids):
        total = 0.0
        for tid in ids:
            entry = logprob_map.get(tid)
            if entry is not None:
                total += math.exp(_logprob_value(entry))
        return total

    mass_a = mass_for(ids_a)
    mass_b = mass_for(ids_b)
    cand_mass = mass_a + mass_b
    if cand_mass <= 0.0 or not math.isfinite(cand_mass):
        p_a = None
    else:
        p_a = mass_a / cand_mass
    return {
        "first_token": first_token,
        "first_token_id": token_id,
        "p_a": None if p_a is None else round(float(p_a), 6),
        "cand_mass": round(float(cand_mass), 6),
    }


def read_jsonl_checkpoint(path: Path, config_hash: str) -> Dict[str, dict]:
    """读取 checkpoint：返回 {sample_id: record}。容忍末尾半行。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", "config_hash": config_hash}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {}

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed = []
    valid_end = 0
    for i, line in enumerate(lines):
        try:
            parsed.append(json.loads(line.decode("utf-8")))
            valid_end += len(line)
        except Exception:
            if i == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                break
            raise ValueError(f"checkpoint 第 {i + 1} 行损坏：{path}")

    if not parsed or parsed[0].get("record_type") != "meta":
        raise ValueError(f"checkpoint 缺少 meta 首行：{path}")
    if parsed[0].get("config_hash") != config_hash:
        raise ValueError(
            "已有 checkpoint 的配置与本次运行不一致。\n"
            f"checkpoint: {path}\n如需从头开始，请加 --overwrite；不要混用不同实验结果。"
        )

    results = {}
    for obj in parsed[1:]:
        results[obj["id"]] = obj
    return results


def append_jsonl(path: Path, records: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_csv(path: Path, header: List[str], rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description="ToxiCN CBM M0 受控使用/提及测量（本地 vLLM）")
    p.add_argument("--model_path", required=True, type=Path, help="本地 GLM-4-9B-Chat 模型目录")
    p.add_argument("--data_path", required=True, type=Path, help="含 train.json/test.json 的目录")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="输出目录；默认 data/processed/<dataset>/m0_<model>/")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument("--batch_samples", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条作冒烟测试")
    p.add_argument("--overwrite", action="store_true", help="删除同名 checkpoint 从头运行")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_samples <= 0:
        raise ValueError("--batch_samples 必须为正整数")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if not (0.05 <= args.gpu_memory_utilization <= 0.98):
        raise ValueError("--gpu_memory_utilization 建议且仅允许 0.05 到 0.98")

    model_path = args.model_path.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    dataset, input_files, dataset_tag, json_path = load_dataset(data_path, args.split)
    if args.limit is not None:
        dataset = dataset[:args.limit]
        print(f"技术冒烟限制生效：{len(dataset)} 条样本")

    model_name = model_path.name
    split_tag = json_path.stem
    suffix = split_tag + (f"_limit{args.limit}" if args.limit is not None else "")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "data" / "processed" / dataset_tag / f"m0_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_out = output_dir / f"m0_{model_name}_{suffix}_use.csv"
    json_out = output_dir / f"m0_{model_name}_{suffix}_use.json"
    ckpt1 = output_dir / f"m0_{model_name}_{suffix}_stage1.checkpoint.jsonl"
    ckpt2 = output_dir / f"m0_{model_name}_{suffix}_stage2.checkpoint.jsonl"

    script_path = Path(__file__).resolve()
    base_config = {
        "model_path": str(model_path),
        "input_path": str(json_path),
        "input_sha256": sha256_file(json_path),
        "split": split_tag,
        "n_samples": len(dataset),
        "script_sha256": sha256_file(script_path),
        "prompt_version": "m0-v1-two-stage-forced-choice",
        "temperature": 0,
        "max_tokens": 1,
        "seed": args.seed,
    }
    config_hash = hashlib.sha256(
        json.dumps(base_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if args.overwrite:
        for stale in (ckpt1, ckpt2, csv_out, json_out):
            if stale.exists():
                stale.unlink()

    all_ids = {x["id"] for x in dataset}
    s1_done = read_jsonl_checkpoint(ckpt1, config_hash)
    s2_done = read_jsonl_checkpoint(ckpt2, config_hash)

    tokenizer, llm = load_vllm_model(
        model_path, args.gpu_memory_utilization, args.max_model_len,
        args.max_num_seqs, args.max_num_batched_tokens
    )
    chat_kwargs = detect_chat_kwargs(tokenizer)

    # ---- 阶段1：存在性（是/否） ----
    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    if set(yes_ids) & set(no_ids):
        raise ValueError(f"是/否 verbalizer token id 重叠：是={yes_ids}, 否={no_ids}")
    s1_params, s1_mode = build_sampling_params(yes_ids + no_ids)

    todo1 = [x for x in dataset if x["id"] not in s1_done]
    print(f"阶段1 待处理 {len(todo1)}/{len(dataset)}", flush=True)
    if todo1:
        pbar = tqdm(total=len(dataset), initial=len(s1_done), desc="M0 阶段1 存在性", unit="sample")
        for start in range(0, len(todo1), args.batch_samples):
            batch = todo1[start:start + args.batch_samples]
            texts = [STAGE1_TEMPLATE.format(text=s["content"]) for s in batch]
            rendered = render_prompts(tokenizer, texts, chat_kwargs, model_name)
            outputs = llm.generate(rendered, s1_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(f"阶段1 输出数不一致：{len(outputs)} vs {len(batch)}")
            recs = []
            for s, out in zip(batch, outputs):
                v = get_two_way_result(out, tokenizer, yes_ids, no_ids, "是", "否")
                recs.append({"record_type": "sample", "id": s["id"], "stage": 1,
                             "p_exists": v["p_a"], "first_token": v["first_token"],
                             "cand_mass": v["cand_mass"], "restricted_decoding": s1_mode})
            append_jsonl(ckpt1, recs)
            for r in recs:
                s1_done[r["id"]] = r
            pbar.update(len(batch))
        pbar.close()

    # ---- 阶段2：使用/提及（A/B），仅阶段1="是" ----
    yes_samples = [x for x in dataset
                   if s1_done.get(x["id"], {}).get("first_token") == "是"]
    a_ids = get_candidate_ids(["A"], tokenizer)
    b_ids = get_candidate_ids(["B"], tokenizer)
    if set(a_ids) & set(b_ids):
        raise ValueError(f"A/B verbalizer token id 重叠：A={a_ids}, B={b_ids}")
    s2_params, s2_mode = build_sampling_params(a_ids + b_ids)

    todo2 = [x for x in yes_samples if x["id"] not in s2_done]
    print(f"阶段2（阶段1='是' 的 {len(yes_samples)} 条）待处理 {len(todo2)}", flush=True)
    if todo2:
        pbar = tqdm(total=len(yes_samples), initial=len(s2_done), desc="M0 阶段2 使用/提及", unit="sample")
        for start in range(0, len(todo2), args.batch_samples):
            batch = todo2[start:start + args.batch_samples]
            texts = [STAGE2_TEMPLATE.format(text=s["content"]) for s in batch]
            rendered = render_prompts(tokenizer, texts, chat_kwargs, model_name)
            outputs = llm.generate(rendered, s2_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(f"阶段2 输出数不一致：{len(outputs)} vs {len(batch)}")
            recs = []
            for s, out in zip(batch, outputs):
                v = get_two_way_result(out, tokenizer, a_ids, b_ids, "A", "B")
                recs.append({"record_type": "sample", "id": s["id"], "stage": 2,
                             "s_use": v["p_a"], "first_token": v["first_token"],
                             "cand_mass": v["cand_mass"], "restricted_decoding": s2_mode})
            append_jsonl(ckpt2, recs)
            for r in recs:
                s2_done[r["id"]] = r
            pbar.update(len(batch))
        pbar.close()

    # ---- 汇总输出 ----
    rows = []
    for s in dataset:
        r1 = s1_done[s["id"]]
        p_exists = r1.get("p_exists")
        exists_token = r1.get("first_token")
        cm1 = r1.get("cand_mass")
        if exists_token == "是":
            r2 = s2_done.get(s["id"], {})
            s_use = r2.get("s_use", "")
            use_token = r2.get("first_token", "")
            cm2 = r2.get("cand_mass", "")
        else:
            s_use, use_token, cm2 = 0.0, "", ""
        rows.append([s["id"], s["toxic"], s["content"], p_exists, exists_token,
                     s_use, use_token, cm1, cm2])

    header = ["id", "toxic", "content", "p_exists", "exists_token",
              "s_use", "use_token", "cand_mass_stage1", "cand_mass_stage2"]
    atomic_write_csv(csv_out, header, rows)

    n_yes = sum(1 for r in rows if r[4] == "是")
    meta = {
        "script": "run_items_m0.py",
        "prompt_version": base_config["prompt_version"],
        "config": base_config,
        "config_hash": config_hash,
        "data_files": input_files,
        "n_samples": len(dataset),
        "n_exists_yes": n_yes,
        "n_use": sum(1 for r in rows if r[6] == "A"),
        "n_mention": sum(1 for r in rows if r[6] == "B"),
        "stage1_mode": s1_mode,
        "stage2_mode": s2_mode,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    tmp = json_out.with_suffix(json_out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, json_out)

    print("\n" + "=" * 72)
    print(f"完成：样本 {len(dataset)}；阶段1='是' {n_yes} 条；阶段2 完成 {len(s2_done)} 条")
    print(f"CSV：{csv_out}")
    print(f"JSON：{json_out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
