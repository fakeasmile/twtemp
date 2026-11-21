#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToxiCN N01/N02 候选概念独立测量脚本（本地 GLM-4-9B-Chat + vLLM）。

本脚本只测量两个候选概念，不重算、不覆盖 v8 的 26 维结果。
离线加载本地模型；每个样本分别询问 N01、N02；记录首 token、p_yes、cand_mass；
逐批 checkpoint，支持断点续跑；输出 JSON、矩阵 CSV、首 token 审计 CSV。
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

from tqdm import tqdm

SCRIPT_VERSION = "n01-n02-pilot-1.0.0"
PROMPT_VERSION = "n01-n02-observable-evidence-v1"
REQUIRED_COLUMNS = [
    "条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句",
    "概念定义", "边界说明", "优先级",
]
EXPECTED = {
    "N01": "非攻击性身份提及与事实性群体讨论",
    "N02": "平等尊重与包容的肯定立场",
}
PROMPT_TEMPLATE = """你是严格的中文文本现象判定助手。请根据文本中明确可观察到的内容，判断指定概念是否出现。
只依据文本证据判断，不推测作者未表达的意图，不补充外部事实。不要根据文本的整体毒性标签、主题或身份词是否出现来猜测答案。
如果证据不足，回答“否”。文本中被引用或转述的内容，只有在符合该概念定义及边界时才计入；涉及当前发言者立场的概念，只判断能够归属于当前发言者的立场。

文本：{text}

概念：{concept_name}
判定问题：{question}
概念定义：{definition}
边界说明：{boundary}

只回答“是”或“否”。回答："""
MODEL_PROMPT_SUFFIX = {"glm-4-9b-chat": "\n", "Qwen3.5-9B": "\n"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到候选词典：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("候选词典缺少表头")
        reader.fieldnames = [(x or "").lstrip("\ufeff\x00").strip() for x in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"候选词典缺少列：{missing}")
        items = []
        for line_no, row in enumerate(reader, start=2):
            item = {k: (row.get(k) or "").strip() for k in REQUIRED_COLUMNS}
            if any(not item[k] for k in ("条目id", "构念名", "条目问句", "概念定义", "边界说明")):
                raise ValueError(f"词典第 {line_no} 行存在必需字段为空")
            items.append(item)
    ids = [x["条目id"] for x in items]
    if len(items) != 2 or set(ids) != set(EXPECTED):
        raise ValueError(f"本脚本只接受 N01、N02 两项，实际条目：{ids}")
    for item in items:
        if item["构念名"] != EXPECTED[item["条目id"]]:
            raise ValueError(f"{item['条目id']} 名称与审定候选版本不一致")
        if item["方向"] != "正向" or item["优先级"] != "候选核心":
            raise ValueError(f"{item['条目id']} 的方向/优先级不符合候选词典约定")
    return sorted(items, key=lambda x: x["条目id"])


def normalize_sample(obj, dataset_tag, split, idx):
    if not isinstance(obj, dict):
        raise TypeError(f"数据第 {idx + 1} 条不是 JSON object")
    if obj.get("content") is None or not str(obj["content"]).strip():
        raise ValueError(f"数据第 {idx + 1} 条缺少有效 content")
    sid = obj.get("id")
    if sid is None or not str(sid).strip():
        sid = f"{dataset_tag}_{split}_{idx:04d}"
    return {
        "id": str(sid), "content": str(obj["content"]), "toxic": obj.get("toxic"),
        "source": str(obj.get("source") or dataset_tag), "split": split,
        "topic": obj.get("topic"), "toxic_type": obj.get("toxic_type"),
        "expression": obj.get("expression"), "target": obj.get("target"),
    }


def load_dataset(data_path: Path, split: str):
    if not data_path.exists():
        raise FileNotFoundError(f"data_path 不存在：{data_path}")
    if data_path.is_file():
        json_path = data_path
        dataset_tag, actual_split = data_path.parent.name or data_path.stem, data_path.stem
    else:
        json_path = data_path / f"{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"{data_path} 下找不到 {split}.json")
        dataset_tag, actual_split = data_path.name, split
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError("数据文件顶层必须是非空 JSON list")
    samples = [normalize_sample(x, dataset_tag, actual_split, i) for i, x in enumerate(raw)]
    ids = [x["id"] for x in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("数据集样本 id 不唯一")
    return samples, json_path


def build_prompt(sample, item):
    return PROMPT_TEMPLATE.format(
        text=sample["content"], concept_name=item["构念名"],
        question=item["条目问句"], definition=item["概念定义"], boundary=item["边界说明"],
    )


def load_model(model_path: Path, gpu_memory_utilization: float, max_model_len: int,
               max_num_seqs: int, max_num_batched_tokens: int):
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"必须提供含 config.json 的本地模型目录：{model_path}")
    from transformers import AutoTokenizer
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer 的 pad_token 与 eos_token 均为空")
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=str(model_path), trust_remote_code=True, dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
        max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens, enforce_eager=True, max_logprobs=50,
    )
    return tokenizer, llm


def candidate_ids(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"verbalizer {word!r} 不是单 token：{ids}；停止以避免首 token 概率定义错误")
    return ids


def make_logits_processor(allowed_ids):
    allowed = sorted(set(allowed_ids))
    def processor(*args, **kwargs):
        import torch
        logits = kwargs.get("logits")
        if logits is None:
            logits = next(a for a in args if isinstance(a, torch.Tensor))
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked
    return processor


def make_sampling_params(allowed_ids):
    from vllm import SamplingParams
    allowed = sorted(set(allowed_ids))
    attempts = [
        ("logits_processors", {"logits_processors": [make_logits_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束回退", {}),
    ]
    errors = []
    for mode, kwargs in attempts:
        try:
            params = SamplingParams(max_tokens=1, temperature=0, logprobs=50, **kwargs)
            print(f"首 token 约束方式：{mode}")
            if mode == "无约束回退":
                print("警告：vLLM 不支持本脚本尝试的受限解码方式；请重点审查 cand_mass。")
            return params, mode
        except TypeError as exc:
            errors.append(f"{mode}: {exc}")
    raise RuntimeError("无法建立 SamplingParams：" + " | ".join(errors))


def render_prompts(tokenizer, texts, model_name):
    template = getattr(tokenizer, "chat_template", None) or ""
    chat_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
    suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    rendered = []
    for text in texts:
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


def lp_value(value):
    return float(value.logprob) if hasattr(value, "logprob") else float(value)


def extract_result(output, tokenizer, yes_ids, no_ids):
    if not output.outputs or not output.outputs[0].token_ids:
        raise RuntimeError("vLLM 未返回首 token")
    completion = output.outputs[0]
    if not completion.logprobs or not completion.logprobs[0]:
        raise RuntimeError("vLLM 未返回首 token logprobs")
    token_id = int(completion.token_ids[0])
    token = "是" if token_id in yes_ids else ("否" if token_id in no_ids else tokenizer.decode([token_id]))
    logprob_map = completion.logprobs[0]
    def mass(ids):
        total = 0.0
        for tid in ids:
            entry = logprob_map.get(tid)
            if entry is not None:
                total += math.exp(lp_value(entry))
        return total
    yes_mass, no_mass = mass(yes_ids), mass(no_ids)
    cand_mass = yes_mass + no_mass
    p_yes = yes_mass / cand_mass if cand_mass > 0 and math.isfinite(cand_mass) else None
    return {"first_token": token, "first_token_id": token_id,
            "p_yes": None if p_yes is None else round(float(p_yes), 6),
            "cand_mass": round(float(cand_mass), 6)}


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path, header, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_checkpoint(path, config, sample_ids, item_ids, overwrite):
    if overwrite and path.exists():
        path.unlink()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", "config": config}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        return {}
    raw = path.read_bytes(); lines = raw.splitlines(keepends=True); parsed = []; valid_end = 0
    for i, line in enumerate(lines):
        try:
            parsed.append(json.loads(line.decode("utf-8"))); valid_end += len(line)
        except Exception as exc:
            if i == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as f: f.truncate(valid_end)
                print("检测到 checkpoint 末尾半行，已截断并恢复")
                break
            raise ValueError(f"checkpoint 第 {i+1} 行损坏：{exc}") from exc
    if raw and not raw.endswith(b"\n") and parsed:
        # 进程可能在完整 JSON 记录写完、换行符落盘前退出；补齐后再追加。
        with path.open("ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    if not parsed or parsed[0].get("record_type") != "meta" or parsed[0].get("config") != config:
        raise ValueError("checkpoint 配置/版本/输入/词典不匹配；请核对后使用 --overwrite 重新开始")
    results = {}
    expected = set(item_ids)
    for line_no, rec in enumerate(parsed[1:], 2):
        sid = rec.get("id")
        if rec.get("record_type") != "sample" or sid not in sample_ids or sid in results:
            raise ValueError(f"checkpoint 第 {line_no} 行样本记录非法或重复")
        for key in ("p_yes", "first_token", "cand_mass", "first_token_id"):
            if set(rec.get(key, {})) != expected:
                raise ValueError(f"checkpoint 样本 {sid} 的 {key} 字段不完整")
        results[sid] = rec
    return results


def main():
    ap = argparse.ArgumentParser(description="ToxiCN N01/N02 候选概念独立测量")
    ap.add_argument("--model_path", type=Path, required=True, help="本地 GLM-4-9B-Chat 模型目录，不接受远程模型名")
    ap.add_argument("--data_path", type=Path, required=True, help="包含 train.json/test.json 的目录，或单个 JSON 文件")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--items_csv", type=Path, default=Path(__file__).resolve().parent / "cbm_non_toxic_context_candidates_v1.csv")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--max_num_seqs", type=int, default=32)
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--batch_samples", type=int, default=8, help="每批样本数；默认8，即每批16个概念请求，适合8GB显存先行测试")
    ap.add_argument("--limit", type=int, default=None, help="仅取前N条作技术冒烟测试")
    ap.add_argument("--validate_only", action="store_true", help="只校验输入、词典和任务规模，不加载模型、不推理")
    ap.add_argument("--overwrite", action="store_true", help="删除本候选实验的 checkpoint 和输出后从头运行")
    args = ap.parse_args()

    if args.batch_samples <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--batch_samples 和 --limit 必须为正整数")
    if not 0.05 <= args.gpu_memory_utilization <= 0.98:
        raise ValueError("--gpu_memory_utilization 必须在 0.05 到 0.98 之间")
    model_path = args.model_path.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    items_path = args.items_csv.expanduser().resolve()
    samples, json_path = load_dataset(data_path, args.split)
    if args.limit is not None:
        samples = samples[:args.limit]
    items = load_items(items_path)
    item_ids = [x["条目id"] for x in items]
    input_hash, items_hash = sha256_file(json_path), sha256_file(items_path)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    print("=" * 68)
    print(f"划分：{json_path} | 样本：{len(samples)} | 概念：{item_ids} | 请求：{len(samples)*len(items)}")
    print(f"数据 SHA256：{input_hash}")
    print(f"候选词典 SHA256：{items_hash}")
    print(f"脚本 SHA256：{script_hash}")
    print(f"模型目录：{model_path}")
    if args.validate_only:
        print("validate_only：校验通过；未加载模型，未执行推理。")
        return

    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"本地模型目录不存在或缺少 config.json：{model_path}")
    dataset_tag = json_path.parent.name or json_path.stem
    suffix = json_path.stem + (f"_limit{args.limit}" if args.limit is not None else "")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  Path.cwd() / "outputs_n01_n02_pilot" / f"{dataset_tag}_{model_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"items_n01_n02_{model_path.name}_{suffix}"
    json_out, matrix_out = output_dir / f"{stem}.json", output_dir / f"{stem}_matrix.csv"
    audit_out, checkpoint = output_dir / f"{stem}_firsttoken.csv", output_dir / f"{stem}.checkpoint.jsonl"
    if args.overwrite:
        for p in (json_out, matrix_out, audit_out):
            if p.exists(): p.unlink()

    config = {
        "script_version": SCRIPT_VERSION, "script_sha256": script_hash,
        "model_path": str(model_path), "model_name": model_path.name,
        "input_path": str(json_path), "input_sha256": input_hash,
        "split": json_path.stem, "n_samples": len(samples),
        "items_path": str(items_path), "items_sha256": items_hash,
        "item_ids": item_ids, "prompt_version": PROMPT_VERSION,
        "temperature": 0, "max_tokens": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "batch_samples": args.batch_samples,
    }
    results = read_checkpoint(checkpoint, config, {x["id"] for x in samples}, item_ids, args.overwrite)
    todo = [x for x in samples if x["id"] not in results]
    modes = {x.get("restricted_decoding") for x in results.values()}
    if len(modes) > 1:
        raise ValueError(f"checkpoint 中受限解码机制不一致：{modes}")
    restrict_mode = next(iter(modes)) if modes else "not_started"

    if todo:
        tokenizer, llm = load_model(model_path, args.gpu_memory_utilization, args.max_model_len,
                                    args.max_num_seqs, args.max_num_batched_tokens)
        yes_ids, no_ids = candidate_ids(tokenizer, "是"), candidate_ids(tokenizer, "否")
        if set(yes_ids) & set(no_ids):
            raise ValueError("是/否 verbalizer token ID 重叠")
        sampling, restrict_mode = make_sampling_params(yes_ids + no_ids)
        progress = tqdm(total=len(samples), initial=len(results), desc="N01/N02 测量", unit="sample")
        try:
            for start in range(0, len(todo), args.batch_samples):
                batch = todo[start:start + args.batch_samples]
                prompts, keys = [], []
                for sample in batch:
                    for item in items:
                        prompts.append(build_prompt(sample, item))
                        keys.append((sample["id"], item["条目id"]))
                rendered = render_prompts(tokenizer, prompts, model_path.name)
                outputs = llm.generate(rendered, sampling, use_tqdm=False)
                if len(outputs) != len(keys):
                    raise RuntimeError(f"vLLM 输出数不匹配：{len(outputs)} != {len(keys)}")
                batch_records = {
                    s["id"]: {"record_type": "sample", "id": s["id"], "p_yes": {},
                              "first_token": {}, "cand_mass": {}, "first_token_id": {},
                              "restricted_decoding": restrict_mode}
                    for s in batch
                }
                for (sid, iid), output in zip(keys, outputs):
                    val = extract_result(output, tokenizer, yes_ids, no_ids)
                    rec = batch_records[sid]
                    for k in ("p_yes", "first_token", "cand_mass", "first_token_id"):
                        field = {"p_yes":"p_yes", "first_token":"first_token",
                                 "cand_mass":"cand_mass", "first_token_id":"first_token_id"}[k]
                        rec[k][iid] = val[field]
                with checkpoint.open("a", encoding="utf-8") as f:
                    for sample in batch:
                        rec = batch_records[sample["id"]]
                        if any(set(rec[k]) != set(item_ids) for k in
                               ("p_yes", "first_token", "cand_mass", "first_token_id")):
                            raise RuntimeError(f"样本 {sample['id']} 的两个概念结果不完整，拒绝写入 checkpoint")
                        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        results[sample["id"]] = rec
                    f.flush(); os.fsync(f.fileno())
                progress.update(len(batch))
                progress.set_postfix_str(f"完成 {len(results)}/{len(samples)}")
        finally:
            progress.close()
    else:
        print("checkpoint 已包含全部样本，直接重建输出文件。")

    missing = [s["id"] for s in samples if s["id"] not in results]
    if missing:
        raise RuntimeError(f"结果缺失 {len(missing)} 条，示例：{missing[:10]}")

    matrix = {s["id"]: results[s["id"]]["p_yes"] for s in samples}
    invalid, audit_rows, masses = [], [], []
    for s in samples:
        rec = results[s["id"]]
        for item in items:
            iid = item["条目id"]
            p, tok, cm, tid = rec["p_yes"][iid], rec["first_token"][iid], rec["cand_mass"][iid], rec["first_token_id"][iid]
            if p is None: invalid.append({"id": s["id"], "item_id": iid, "first_token": tok, "cand_mass": cm})
            if cm is not None: masses.append(float(cm))
            audit_rows.append([s["id"], iid, item["构念名"], tok, tid, "" if p is None else p, "" if cm is None else cm])
    n_requests = len(samples) * len(items)
    mass_mean = sum(masses) / len(masses) if masses else None
    token_stats = {
        "total": n_requests,
        "recognized_yes_no": sum(1 for row in audit_rows if row[3] in {"是", "否"}),
        "recognized_yes_no_pct": round(sum(1 for row in audit_rows if row[3] in {"是", "否"}) / n_requests, 6),
        "cand_mass_mean": round(mass_mean, 6) if mass_mean is not None else None,
        "invalid_p_yes": len(invalid), "restricted_decoding": restrict_mode,
    }
    meta = {
        "script": "run_items_n01_n02.py", "script_version": SCRIPT_VERSION,
        "script_sha256": script_hash, "prompt_version": PROMPT_VERSION,
        "config": config, "config_sha256": hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "n_samples": len(samples), "n_items": len(items), "item_definitions": items,
        "restricted_decoding": restrict_mode, "first_token_stats": token_stats,
        "n_invalid": len(invalid), "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_json(json_out, {"meta": meta, "matrix": matrix, "invalid": invalid})
    matrix_rows = [[s["id"], s["toxic"], s["content"], *[matrix[s["id"]][iid] if matrix[s["id"]][iid] is not None else "" for iid in item_ids]] for s in samples]
    atomic_csv(matrix_out, ["id", "toxic", "content", *item_ids], matrix_rows)
    atomic_csv(audit_out, ["sample_id", "item_id", "construct_name", "first_token", "first_token_id", "p_yes", "cand_mass"], audit_rows)
    print("=" * 68)
    print(f"完成：{len(samples)} 样本 × {len(items)} 概念 = {n_requests} 次判定")
    print(f"是/否首 token 识别率：{token_stats['recognized_yes_no_pct']:.2%}")
    print(f"cand_mass 均值：{token_stats['cand_mass_mean']}")
    print(f"无效 p_yes：{len(invalid)}")
    print(f"JSON：{json_out}\n矩阵：{matrix_out}\n首 token 审计：{audit_out}\ncheckpoint：{checkpoint}")
    if restrict_mode == "无约束回退" or (mass_mean is not None and mass_mean < 0.999):
        print("警告：受限解码/候选概率质量未达到预期，请先审查 firsttoken.csv，不要直接合并分析。")
    print("=" * 68)


if __name__ == "__main__":
    main()
