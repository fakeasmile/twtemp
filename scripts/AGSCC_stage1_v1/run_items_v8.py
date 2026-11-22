#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToxiCN CBM v8 概念测量脚本（本地 GLM + vLLM）。

设计原则
--------
1. 模型加载/对话模板/是-否 verbalizer/首 token 概率提取沿用 run_items_v7.py。
2. 读取 cbm_concept_set_v8.csv。默认只测量 21 个“核心”概念与 5 个“目标属性”；
   3 个“储备”概念不会进入主特征矩阵，只有显式传入 --include_reserve 才会测量。
3. 每个样本 × 每个概念单独询问一次；记录 p_yes、实际首 token 及候选概率质量。
4. 按样本小批量推理并逐批 checkpoint，支持断点续跑，避免一次性构造/持有全部请求。
5. 仅接受本地模型目录；离线模式，不允许 Transformers 从远端下载模型。

建议目录结构（与 v7 一致）
------------------------
project_root/
  scripts/run_items_v8.py
  data/concepts/cbm_concept_set_v8.csv
  data/raw/TOXICN/train.json
  data/raw/TOXICN/test.json

示例
----
# 先做 8 条技术冒烟测试（只验证加载、解码与输出，不代表概念质量评估）
python scripts/run_items_v8.py \
  --model_path /root/autodl-tmp/glm-4-9b-chat \
  --data_path data/raw/TOXICN --split train --limit 8

# 全量提取训练集概念向量
python scripts/run_items_v8.py \
  --model_path /root/autodl-tmp/glm-4-9b-chat \
  --data_path data/raw/TOXICN --split train

# 全量提取测试集概念向量（独立输出目录/文件）
python scripts/run_items_v8.py \
  --model_path /root/autodl-tmp/glm-4-9b-chat \
  --data_path data/raw/TOXICN --split test

# 可选：额外测量 3 个储备概念（不要与默认 v8 核心主实验混淆）
python scripts/run_items_v8.py ... --include_reserve

注意
----
- 首次运行建议先 --limit 8，核对输出后再运行全量。
- checkpoint 与最终输出放在同一输出目录；同一配置会自动续跑。配置/输入/词典不匹配
  时会停止，不会把不同实验拼接起来。需要从头重跑时使用 --overwrite。
- 默认不做量化，与 v7 相同。显存不足应先调整运行环境/模型加载配置，不要静默改变
  精度设置，否则会改变 logits 概率的可比性。
"""

import os

# 在导入 transformers/vLLM 前设置离线约束。
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import math
import os.path
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# 项目根目录：假定脚本放在 project_root/scripts/ 下。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ITEMS_CANDIDATES = [
    PROJECT_ROOT / "data" / "concepts" / "cbm_concept_set_v8.csv",
    Path(__file__).resolve().parent / "data" / "concepts" / "cbm_concept_set_v8.csv",
    Path(__file__).resolve().parent / "cbm_concept_set_v8.csv",
    Path.cwd() / "cbm_concept_set_v8.csv",
]

ITEM_TEMPLATE_V8 = """你是严格的中文文本现象判定助手。请根据文本判断指定概念是否出现。
只依据文本中可观察到的内容判断，不推测作者未表达的意图，不补充外部事实。
对内容现象概念，文本中被引用或转述的表达也按其实际内容判断；对发言者立场概念，
只判断能够归属于当前发言者的立场，不能因为文本没有反对就推定其支持。

文本：{text}

概念：{concept_name}
判定问题：{question}
概念定义：{definition}
边界说明：{boundary}

只回答“是”或“否”。回答："""

# 沿用 v7 对 GLM-4-9B-Chat 的 chat-template 后缀处理。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}

REQUIRED_COLUMNS = [
    "条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句",
    "概念定义", "边界说明", "优先级",
]
ACTIVE_PRIORITIES = {"核心", "目标属性"}
ALL_PRIORITIES = ACTIVE_PRIORITIES | {"储备"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_items_path() -> Path:
    for path in DEFAULT_ITEMS_CANDIDATES:
        if path.is_file():
            return path
    # 返回项目约定位置；不存在时由词典加载函数报出清晰错误。
    return DEFAULT_ITEMS_CANDIDATES[0]


def load_v8_items(csv_path: Path, include_reserve: bool = False) -> List[dict]:
    """读取并严格校验 v8 词典；默认只返回核心项和目标属性项。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"找不到 v8 词典：{csv_path}\n请将文件命名为 cbm_concept_set_v8.csv，"
            "放在 data/concepts/ 下，或通过 --items_csv 指定路径。"
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"词典没有表头：{csv_path}")
        reader.fieldnames = [(x or "").lstrip("\ufeff\x00").strip() for x in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"v8 词典缺少必需列：{missing}")
        all_items = []
        for line_no, row in enumerate(reader, start=2):
            item = {k: (row.get(k) or "").strip() for k in REQUIRED_COLUMNS}
            if not item["条目id"]:
                raise ValueError(f"词典第 {line_no} 行条目id为空")
            for col in ("构念名", "条目问句", "概念定义", "边界说明", "优先级"):
                if not item[col]:
                    raise ValueError(f"词典第 {line_no} 行 {col} 为空（条目 {item['条目id']}）")
            if item["优先级"] not in ALL_PRIORITIES:
                raise ValueError(
                    f"词典第 {line_no} 行优先级非法：{item['优先级']}（条目 {item['条目id']}）"
                )
            if item["优先级"] == "目标属性":
                if item["方向"] != "目标类别" or not item["区块"].startswith("T"):
                    raise ValueError(
                        f"目标属性行的区块/方向不匹配：{item['条目id']} "
                        f"({item['区块']}, {item['方向']})"
                    )
            elif item["方向"] not in {"正向", "反向"}:
                raise ValueError(
                    f"内容/立场概念方向非法：{item['条目id']} -> {item['方向']}"
                )
            all_items.append(item)

    if not all_items:
        raise ValueError(f"词典中没有概念：{csv_path}")
    all_ids = [x["条目id"] for x in all_items]
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted({x for x in all_ids if all_ids.count(x) > 1})
        raise ValueError(f"词典条目id重复：{duplicates}")

    # 核对 v8 当前设计的预期结构，防止误读其他版本或错误 CSV。
    priority_counts = {p: sum(x["优先级"] == p for x in all_items) for p in ALL_PRIORITIES}
    if priority_counts != {"核心": 21, "目标属性": 5, "储备": 3}:
        raise ValueError(
            "词典结构与已审定的 v8 不一致："
            f"核心={priority_counts['核心']}、目标属性={priority_counts['目标属性']}、"
            f"储备={priority_counts['储备']}。请确认没有误用其他版本。"
        )

    selected = [x for x in all_items if include_reserve or x["优先级"] != "储备"]
    if not selected:
        raise ValueError("没有可测量条目")
    return selected


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
        # 保留数据集辅助元数据供结果审计；不作为模型提示输入。
        "topic": item.get("topic"),
        "toxic_type": item.get("toxic_type"),
        "expression": item.get("expression"),
        "target": item.get("target"),
    }


def load_dataset(data_path: Path, split: str) -> Tuple[List[dict], List[str], str, Path]:
    """支持单个 JSON 文件，或含 train.json/test.json 的目录。"""
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
    """按 v7 已用方式加载本地模型；禁止远程下载。"""
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
    """只接受单 token verbalizer，避免首 token 概率切片含义不明确。"""
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
    """沿用 v7 的受限解码优先级，并记录最终实际生效机制。"""
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
            if mode == "无约束回退":
                print("警告：当前 vLLM 未启用受限解码；后续必须检查 cand_mass 与首 token 覆盖率。")
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


def get_probability_result(request_output, tokenizer, yes_ids: List[int], no_ids: List[int]) -> dict:
    """提取首 token 的 p_yes 与候选原始概率质量。"""
    if not request_output.outputs:
        raise RuntimeError("vLLM 返回空 outputs")
    completion = request_output.outputs[0]
    if not completion.token_ids:
        raise RuntimeError("模型没有生成首 token")
    if not completion.logprobs or not completion.logprobs[0]:
        raise RuntimeError("模型没有返回首 token logprobs；请检查 SamplingParams(logprobs=50)")

    token_id = int(completion.token_ids[0])
    first_token = "是" if token_id in yes_ids else ("否" if token_id in no_ids else
                    tokenizer.decode([token_id], skip_special_tokens=False))
    logprob_map = completion.logprobs[0]

    def mass_for(ids):
        total = 0.0
        for token_id_ in ids:
            entry = logprob_map.get(token_id_)
            if entry is not None:
                total += math.exp(_logprob_value(entry))
        return total

    yes_mass = mass_for(yes_ids)
    no_mass = mass_for(no_ids)
    cand_mass = yes_mass + no_mass
    if cand_mass <= 0.0 or not math.isfinite(cand_mass):
        p_yes = None
    else:
        # 在“是/否”候选集合内归一化；cand_mass 另行记录，以诊断未受限/降级情况。
        p_yes = yes_mass / cand_mass
    return {
        "first_token": first_token,
        "first_token_id": token_id,
        "p_yes": None if p_yes is None else round(float(p_yes), 6),
        "cand_mass": round(float(cand_mass), 6),
    }


def make_prompt(sample: dict, item: dict) -> str:
    return ITEM_TEMPLATE_V8.format(
        text=sample["content"],
        concept_name=item["构念名"],
        question=item["条目问句"],
        definition=item["概念定义"],
        boundary=item["边界说明"],
    )


def read_checkpoint(path: Path, config: dict, dataset_ids: set, item_ids: List[str], overwrite: bool):
    """读取/校验逐样本 JSONL checkpoint；容忍并修复崩溃造成的末尾半行。"""
    if overwrite and path.exists():
        path.unlink()

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", "config": config}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {}

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed = []
    valid_end = 0
    truncated_partial = False
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line.decode("utf-8"))
            parsed.append(obj)
            valid_end += len(line)
        except Exception as exc:
            is_last_partial = (i == len(lines) - 1 and not line.endswith((b"\n", b"\r")))
            if is_last_partial:
                print("检测到 checkpoint 末尾存在未写完记录，已安全截断该半行。")
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                truncated_partial = True
                break
            raise ValueError(f"checkpoint 第 {i + 1} 行损坏，拒绝继续：{exc}") from exc

    if raw and not raw.endswith(b"\n") and parsed and not truncated_partial:
        # 如果进程恰好在完整 JSON 写完、换行符落盘前退出，补齐换行再允许追加。
        with path.open("ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())

    if not parsed or parsed[0].get("record_type") != "meta":
        raise ValueError(f"checkpoint 缺少 meta 首行：{path}")
    if parsed[0].get("config") != config:
        raise ValueError(
            "已有 checkpoint 的模型/数据/词典/提示词配置与本次运行不一致。\n"
            f"checkpoint: {path}\n如需从头开始，请加 --overwrite；不要混用不同实验结果。"
        )

    results = {}
    item_id_set = set(item_ids)
    for line_no, obj in enumerate(parsed[1:], start=2):
        if obj.get("record_type") != "sample":
            raise ValueError(f"checkpoint 第 {line_no} 行记录类型非法")
        sid = obj.get("id")
        if sid not in dataset_ids:
            raise ValueError(f"checkpoint 含本数据集不存在的样本 id：{sid}")
        if sid in results:
            raise ValueError(f"checkpoint 样本重复：{sid}")
        p_map = obj.get("p_yes", {})
        t_map = obj.get("first_token", {})
        m_map = obj.get("cand_mass", {})
        id_map = obj.get("first_token_id", {})
        if (set(p_map) != item_id_set or set(t_map) != item_id_set
                or set(m_map) != item_id_set or set(id_map) != item_id_set):
            raise ValueError(f"checkpoint 样本 {sid} 的条目集合不完整/不匹配")
        results[sid] = obj
    return results


def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    parser = argparse.ArgumentParser(description="ToxiCN CBM v8 概念测量（本地 vLLM）")
    parser.add_argument("--model_path", required=True, type=Path,
                        help="本地 GLM-4-9B-Chat 模型目录；禁止填写远程模型名")
    parser.add_argument("--data_path", required=True, type=Path,
                        help="含 train.json/test.json 的目录，或单个 JSON 文件")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="目录模式读取的划分；默认 train。单 JSON 文件模式忽略")
    parser.add_argument("--items_csv", type=Path, default=default_items_path(),
                        help="v8 词典路径，默认 project_root/data/concepts/cbm_concept_set_v8.csv")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="输出目录；默认 data/processed/<dataset>/v8_<model>/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--batch_samples", type=int, default=8,
                        help="每批样本数；默认 8，即默认每批约 8×26=208 个请求")
    parser.add_argument("--limit", type=int, default=None,
                        help="只取前 N 条样本作技术冒烟测试；全量运行不传")
    parser.add_argument("--include_reserve", action="store_true",
                        help="额外测量 3 个储备概念；默认不纳入主实验")
    parser.add_argument("--overwrite", action="store_true",
                        help="删除同名 checkpoint 并从头运行；不要用于意外恢复")
    parser.add_argument("--seed", type=int, default=42,
                        help="保留在运行元数据中；temperature=0 时不主动改变输出")
    return parser.parse_args()


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
    items_path = args.items_csv.expanduser().resolve()

    dataset, input_files, dataset_tag, json_path = load_dataset(data_path, args.split)
    if args.limit is not None:
        dataset = dataset[:args.limit]
        print(f"技术冒烟限制生效：{len(dataset)} 条样本")

    items = load_v8_items(items_path, include_reserve=args.include_reserve)
    item_ids = [x["条目id"] for x in items]
    if len({x["id"] for x in dataset}) != len(dataset):
        raise ValueError("截取后的数据集 id 不唯一")

    model_name = model_path.name
    split_tag = json_path.stem
    suffix = split_tag + (f"_limit{args.limit}" if args.limit is not None else "")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "data" / "processed" / dataset_tag / f"v8_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path_out = output_dir / f"items_v8_{model_name}_{suffix}.json"
    matrix_path_out = output_dir / f"items_v8_{model_name}_{suffix}_matrix.csv"
    firsttoken_path_out = output_dir / f"items_v8_{model_name}_{suffix}_firsttoken.csv"
    checkpoint_path = output_dir / f"items_v8_{model_name}_{suffix}.checkpoint.jsonl"
    if args.overwrite:
        for stale_path in (json_path_out, matrix_path_out, firsttoken_path_out):
            if stale_path.exists():
                stale_path.unlink()

    input_hash = sha256_file(json_path)
    items_hash = sha256_file(items_path)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    config = {
        "model_path": str(model_path),
        "model_name": model_name,
        "input_path": str(json_path),
        "input_sha256": input_hash,
        "split": split_tag,
        "n_samples": len(dataset),
        "items_path": str(items_path),
        "items_sha256": items_hash,
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "selected_item_ids": item_ids,
        "include_reserve": bool(args.include_reserve),
        "prompt_version": "v8.0-question-definition-boundary",
        "temperature": 0,
        "max_tokens": 1,
        "seed_recorded": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "batch_samples": args.batch_samples,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    results = read_checkpoint(
        checkpoint_path, config, {x["id"] for x in dataset}, item_ids, args.overwrite
    )
    todo = [x for x in dataset if x["id"] not in results]

    core_n = sum(x["优先级"] == "核心" for x in items)
    target_n = sum(x["优先级"] == "目标属性" for x in items)
    reserve_n = sum(x["优先级"] == "储备" for x in items)
    print("=" * 72)
    print(f"数据集：{dataset_tag} / {split_tag}；样本：{len(dataset)}；待处理：{len(todo)}")
    print(f"测量条目：{len(items)}（核心 {core_n}，目标属性 {target_n}，储备 {reserve_n}）")
    print(f"总问询：{len(dataset) * len(items)}；本次剩余问询：{len(todo) * len(items)}")
    print(f"模型：{model_path}")
    print(f"词典 SHA256：{items_hash}")
    print(f"输入 SHA256：{input_hash}")
    print(f"输出目录：{output_dir}")
    print(f"checkpoint：{checkpoint_path}")
    if reserve_n == 0:
        print("说明：默认主矩阵不包含 A04、D03、D04 三个储备概念。")
    print("=" * 72, flush=True)

    # 若所有样本已完成，不必重新加载模型；直接从 checkpoint 汇总最终文件。
    # 受限解码方式写入每条已完成样本记录，恢复时可准确还原原始运行机制。
    prior_modes = {x.get("restricted_decoding") for x in results.values()}
    if len(prior_modes) > 1:
        raise ValueError(f"checkpoint 内受限解码方式不一致：{prior_modes}")
    restrict_mode = next(iter(prior_modes)) if prior_modes else "not_started"
    if todo:
        tokenizer, llm = load_vllm_model(
            model_path, args.gpu_memory_utilization, args.max_model_len,
            args.max_num_seqs, args.max_num_batched_tokens
        )
        chat_kwargs = detect_chat_kwargs(tokenizer)
        yes_ids = get_candidate_ids(["是"], tokenizer)
        no_ids = get_candidate_ids(["否"], tokenizer)
        if set(yes_ids) & set(no_ids):
            raise ValueError(f"是/否 verbalizer token id 重叠：是={yes_ids}, 否={no_ids}")
        print(f"verbalizer：是={yes_ids}；否={no_ids}")
        sampling_params, restrict_mode = build_sampling_params(yes_ids + no_ids)

        progress = tqdm(total=len(dataset), initial=len(results), desc="v8 概念测量", unit="sample")
        try:
            for start in range(0, len(todo), args.batch_samples):
                batch = todo[start:start + args.batch_samples]
                prompt_texts, request_keys = [], []
                for sample in batch:
                    for item in items:
                        prompt_texts.append(make_prompt(sample, item))
                        request_keys.append((sample["id"], item["条目id"]))

                rendered = render_prompts(tokenizer, prompt_texts, chat_kwargs, model_name)
                outputs = llm.generate(rendered, sampling_params, use_tqdm=False)
                if len(outputs) != len(request_keys):
                    raise RuntimeError(
                        f"vLLM 输出数与请求数不一致：outputs={len(outputs)}, requests={len(request_keys)}"
                    )

                batch_records = {
                    sample["id"]: {
                        "record_type": "sample",
                        "id": sample["id"],
                        "p_yes": {},
                        "first_token": {},
                        "cand_mass": {},
                        "first_token_id": {},
                        "restricted_decoding": restrict_mode,
                    }
                    for sample in batch
                }
                for (sid, iid), output in zip(request_keys, outputs):
                    value = get_probability_result(output, tokenizer, yes_ids, no_ids)
                    rec = batch_records[sid]
                    rec["p_yes"][iid] = value["p_yes"]
                    rec["first_token"][iid] = value["first_token"]
                    rec["cand_mass"][iid] = value["cand_mass"]
                    rec["first_token_id"][iid] = value["first_token_id"]

                # 只有整个批次完整得到结果后才落盘，避免半批结果被误判为完成。
                with checkpoint_path.open("a", encoding="utf-8") as f:
                    for sample in batch:
                        record = batch_records[sample["id"]]
                        if any(set(record[k]) != set(item_ids) for k in
                               ("p_yes", "first_token", "cand_mass", "first_token_id")):
                            raise RuntimeError(f"样本 {sample['id']} 的条目结果不完整，不写 checkpoint")
                        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        results[sample["id"]] = record
                    f.flush()
                    os.fsync(f.fileno())
                progress.update(len(batch))
                progress.set_postfix_str(f"完成 {len(results)}/{len(dataset)}")
        finally:
            progress.close()
    else:
        print("checkpoint 已包含本次配置的全部样本，无需重新推理。")

    # 确保 checkpoint 中样本次序/覆盖完整。
    missing_ids = [x["id"] for x in dataset if x["id"] not in results]
    if missing_ids:
        raise RuntimeError(f"推理结束后仍缺少 {len(missing_ids)} 个样本，首批：{missing_ids[:10]}")

    # 重建矩阵与逐请求首 token 审计表。
    matrix = {}
    invalid = []
    firsttoken_rows = []
    n_in_range = 0
    masses = []
    for sample in dataset:
        sid = sample["id"]
        record = results[sid]
        matrix[sid] = record["p_yes"]
        for item in items:
            iid = item["条目id"]
            p_yes = record["p_yes"][iid]
            token = record["first_token"][iid]
            cand_mass = record["cand_mass"][iid]
            token_id = record["first_token_id"][iid]
            if token in {"是", "否"}:
                n_in_range += 1
            if cand_mass is not None:
                masses.append(float(cand_mass))
            if p_yes is None:
                invalid.append({"id": sid, "item_id": iid, "first_token": token,
                                "first_token_id": token_id, "cand_mass": cand_mass})
            firsttoken_rows.append([
                sid, iid, item["构念名"], token, token_id,
                "" if p_yes is None else p_yes,
                "" if cand_mass is None else cand_mass,
            ])

    n_requests = len(dataset) * len(items)
    cand_mass_mean = sum(masses) / len(masses) if masses else None
    cand_mass_ge_999 = (sum(x >= 0.999 for x in masses) / len(masses)) if masses else None
    first_token_stats = {
        "total": n_requests,
        "recognized_yes_no": n_in_range,
        "recognized_yes_no_pct": round(n_in_range / n_requests, 6) if n_requests else None,
        "cand_mass_mean": round(cand_mass_mean, 6) if cand_mass_mean is not None else None,
        "cand_mass_ge_0_999_pct": round(cand_mass_ge_999, 6) if cand_mass_ge_999 is not None else None,
        "invalid_p_yes": len(invalid),
        "restricted_decoding": restrict_mode,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "script": "run_items_v8.py",
        "script_version": "8.0.0",
        "prompt_version": config["prompt_version"],
        "config_sha256": config_hash,
        "script_sha256": script_hash,
        "config": config,
        "data_files": input_files,
        "n_samples": len(dataset),
        "n_items": len(items),
        "item_definitions": [
            {"item_id": x["条目id"], "construct_id": x["构念id"],
             "construct_name": x["构念名"], "priority": x["优先级"],
             "direction": x["方向"], "question": x["条目问句"],
             "definition": x["概念定义"], "boundary": x["边界说明"]}
            for x in items
        ],
        "restricted_decoding": restrict_mode,
        "first_token_stats": first_token_stats,
        "timestamp": timestamp,
        "n_invalid": len(invalid),
    }

    atomic_write_json(json_path_out, {"meta": meta, "matrix": matrix, "invalid": invalid})

    # 保持 v7 宽表的前三列与后续特征列布局，方便下游脚本最小改动读取。
    matrix_rows = []
    for sample in dataset:
        sid = sample["id"]
        matrix_rows.append([
            sid, sample["toxic"], sample["content"],
            *[matrix[sid][iid] if matrix[sid][iid] is not None else "" for iid in item_ids],
        ])
    atomic_write_csv(matrix_path_out, ["id", "toxic", "content", *item_ids], matrix_rows)

    atomic_write_csv(
        firsttoken_path_out,
        ["sample_id", "item_id", "construct_name", "first_token", "first_token_id", "p_yes", "cand_mass"],
        firsttoken_rows,
    )

    print("\n" + "=" * 72)
    print(f"完成：样本 {len(dataset)} × 条目 {len(items)} = {n_requests} 次问询")
    print(f"首 token 是/否覆盖率：{first_token_stats['recognized_yes_no_pct'] * 100:.2f}%")
    if cand_mass_mean is not None:
        print(f"候选概率质量均值 cand_mass：{cand_mass_mean:.6f}")
        print(f"cand_mass ≥ 0.999 比例：{cand_mass_ge_999 * 100:.2f}%")
    print(f"无效 p_yes：{len(invalid)}")
    print(f"JSON：{json_path_out}")
    print(f"矩阵 CSV：{matrix_path_out}")
    print(f"首 token 审计 CSV：{firsttoken_path_out}")
    print(f"断点文件：{checkpoint_path}")
    if restrict_mode == "无约束回退" or (cand_mass_mean is not None and cand_mass_mean < 0.999):
        print("警告：首 token 候选质量未达到受限解码预期；正式使用前应检查 firsttoken.csv。")
    print("=" * 72)


if __name__ == "__main__":
    main()
