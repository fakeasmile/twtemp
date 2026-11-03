"""环1a 提示词 v2.8 两阶段推理脚本（vLLM 加速，CCC 验证集）。

流程（每次调用 = 一条文本 × 一个判断项，模板取自 data/concepts/prompt_ring1a_v1.md）：
  阶段 1  存在性二值判断（候选 是/否），全部 文本×概念 调用
  阶段 2  强度两档判断（候选 A/B），仅阶段 1 判"是"的对触发
  CAT1a   类别整体单问（候选 是/否），每条文本一次
提取：首 token logprobs → verbalizer 概率归一化。
数据层映射：阶段1"否"→0，阶段2 A→1、B→2；结果含 ring1a 聚合（max 值与确认位）。

概念来源：data/concepts/cbm_concept_set_v3.csv 中 类别=="环1a 负面评价" 的全部概念。

使用示例：
  python scripts/run_ring1a.py --model_path D:/models/glm-4-9b-chat
  python scripts/run_ring1a.py --model_path D:/models/Qwen3.5-9B --gpu_memory_utilization 0.9

说明（均自动处理，无需命令行指定）：
  首 token 位置修正：优先查代码内 MODEL_PROMPT_SUFFIX 表（按模型目录名手工登记后缀，
  如 glm-4-9b-chat: '\n'）；未登记的模型走自动修正——生成前缀不以换行结尾则补 '\n'。
  思考模式探测：chat template 含 enable_thinking 变量（Qwen3 系混合思考模型）时自动传
  enable_thinking=False——思考前缀会占据首 token，破坏 verbalizer 提取。脚本不判定支持哪些模型。

性能设定（针对 32G vGPU + 只要首 token 的负载）：
  每请求 max_tokens=1，负载以 prefill 为主：enforce_eager 省 CUDA graph 显存与启动时间、
  swap_space=0、prefix caching 开、阶段 1 与 CAT1a 合并单次批量调用、阶段 2 一次批量调用。
  全程共 2 次 llm.generate 往返。9B bf16 权重约 18G，0.85 利用率下 KV 约 9G，
  单模型一次装不下可降 --gpu_memory_utilization 前先确认无其他进程占卡。
  不启用量化：量化改变 logits 分布，影响概率读数效度（测量类任务不可妥协）。
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# 模型加载
# =============================================================================
def load_vllm_model(llm_path: Path, gpu_memory_utilization: float):
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {llm_path}")
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,     # 同文本的多个概念调用共享指令前缀，prefill 复用
        max_model_len=2048,             # 指令+定义+文本最长约 600-700 token，2048 留足余量
        max_num_seqs=64,
        max_num_batched_tokens=16384,   # 批量以 prefill 为主（每请求只解码 1 token），大 batched token 提高 GPU 利用率
        enforce_eager=True,             # 只解码 1 个 token，CUDA graph 主要加速多步 decode 循环，此处禁用可省启动时间与显存（腾给 KV cache）
        swap_space=0,                   # 无长序列换出需求
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    """chat template 支持 enable_thinking 时关闭思考模式，否则返回空。"""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


# 生成前缀后缀手工指定表（按模型目录名）。不同模型 chat template 结尾不一致，
# 需要特定后缀的模型在此登记；未登记的模型走自动修正（前缀不以换行结尾则补 '\n'）。
MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": " "
}


# =============================================================================
# 提示词模板（与 prompt_ring1a_v1.md v2.8 定稿逐字一致）
# =============================================================================
STAGE1_TEMPLATE = """你是语言学标注助手。任务：判断文本中是否出现下述判断项。注意：只依据文本中实际出现的内容判断，不评价文本内容对错，不解释。讽刺、委婉、暗示等隐式出现也算出现。

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

该判断项在文本中出现了吗？只回答“是/否”。回答："""

STAGE2_TEMPLATE = """你是语言学标注助手。任务：下述判断项已被确认出现在文本中，现在只判断文本中判断项的强度档位，不要重新判断是否出现。注意：只依据文本中实际出现的内容判断，不评价文本内容对错，不解释。
强度档位：
A = 出现但不直接：需要结合上下文才能读出来（如讽刺、委婉、暗示），或被评价的人/群体不明确
B = 出现且直接：明确出现，且能从原文逐字引出对应片段

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

文本中判断项的强度档位？只回答 A/B。回答："""

CAT1A_TEMPLATE = """文本：{text} 对某人或某群体表达了鄙夷、厌恶、敌意、威胁或道德定罪的态度。
该说法成立吗？
只回答 是/否。答："""


# =============================================================================
# 概念词典
# =============================================================================
def load_ring1a_concepts(csv_path: Path):
    concepts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["类别"].strip() == "环1a 负面评价":
                concepts.append({
                    "id": row["id"].strip(),
                    "name": row["概念"].strip(),
                    "definition": row["定义"].strip(),
                })
    if not concepts:
        raise ValueError(f"未在 {csv_path} 中找到 类别==环1a 负面评价 的概念")
    return concepts


# =============================================================================
# Verbalizer 工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    token_ids = list(dict.fromkeys(token_ids))
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效 token")
    return token_ids


def extract_candidate_probs(first_token_logprobs, candidate_ids):
    """首 token logprobs → 候选 token 概率（未归一化）+ 归一化。"""
    probs = [math.exp(first_token_logprobs[tid].logprob) if tid in first_token_logprobs else 0.0
             for tid in candidate_ids]
    total = sum(probs) + 1e-8
    return probs, [p / total for p in probs]


def render_prompts(tokenizer, texts, chat_kwargs, model_name):
    """texts: 已填槽的完整模板字符串列表 → chat template 后的 prompt 列表。

    后缀优先取 MODEL_PROMPT_SUFFIX[model_name]；未登记则自动修正
    （生成前缀不以换行结尾时补 '\\n'，GLM-4-9B-chat 等模型不加换行会扭曲首 token 分布）。
    """
    manual_suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    prompts = []
    for t in texts:
        messages = [{"role": "user", "content": t}]
        p = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if manual_suffix is not None:
            p += manual_suffix
        elif not p.endswith("\n"):
            p += "\n"
        prompts.append(p)
    return prompts


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="环1a 两阶段 + CAT1a vLLM 推理")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--dataset", type=str, default=str(project_root / "data" / "raw" / "CCC" / "ccc.json"))
    parser.add_argument("--concepts_csv", type=str, default=str(project_root / "data" / "concepts" / "cbm_concept_set_v3.csv"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/CCC/<model_name>/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name  # 输出目录/文件名用模型目录名
    dataset = json.load(open(args.dataset, "r", encoding="utf-8"))
    concepts = load_ring1a_concepts(Path(args.concepts_csv))
    print(f"样本 {len(dataset)} 条 × 概念 {len(concepts)} 个 "
          f"({'/'.join(c['id'] for c in concepts)})，另加 CAT1a {len(dataset)} 次")

    tokenizer, llm = load_vllm_model(
        Path(args.model_path), args.gpu_memory_utilization
    )
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_first_token_ids(["是"], tokenizer)
    no_ids = get_first_token_ids(["否"], tokenizer)
    a_ids = get_first_token_ids(["A"], tokenizer)
    b_ids = get_first_token_ids(["B"], tokenizer)
    print(f"verbalizer: 是={yes_ids} 否={no_ids} A={a_ids} B={b_ids}")

    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # ---------- 阶段 1 + CAT1a：合并为一次批量调用 ----------
    # 两者互不依赖（阶段 2 才依赖阶段 1），合并成一个 generate 请求批次，
    # 让 vLLM 调度器统一拼 batch，省一次引擎往返、提高 prefill 吞吐。
    s1_prompts, s1_keys = [], []
    for sample in dataset:
        for c in concepts:
            s1_prompts.append(STAGE1_TEMPLATE.format(
                text=sample["content"], concept_name=c["name"], definition=c["definition"]))
            s1_keys.append((sample["id"], c["id"]))
    cat_prompts = [CAT1A_TEMPLATE.format(text=s["content"]) for s in dataset]
    n_s1 = len(s1_prompts)
    print(f"阶段 1 + CAT1a 推理（合并批量 {n_s1}+{len(cat_prompts)} 条）...")
    all_outputs = llm.generate(
        render_prompts(tokenizer, s1_prompts + cat_prompts, chat_kwargs, model_name),
        sp, use_tqdm=True)
    s1_outputs = all_outputs[:n_s1]
    cat_outputs = all_outputs[n_s1:]

    # ---------- 阶段 2：仅阶段 1 判"是" ----------
    s1_results = {}
    s2_tasks = []
    for (sid, cid), out in zip(s1_keys, s1_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        raw, norm = extract_candidate_probs(lp, yes_ids + no_ids)
        p_yes, p_no = norm[0], norm[1]
        pred_yes = raw[0] >= raw[1]
        s1_results[(sid, cid)] = {"p_yes": p_yes, "p_no": p_no, "pred_yes": pred_yes,
                                  "gen_token": gen_tok}
        if pred_yes:
            sample = next(s for s in dataset if s["id"] == sid)
            concept = next(c for c in concepts if c["id"] == cid)
            s2_tasks.append((sid, cid, STAGE2_TEMPLATE.format(
                text=sample["content"], concept_name=concept["name"],
                definition=concept["definition"])))
    trigger_rate = len(s2_tasks) / max(1, len(s1_keys))
    print(f"阶段 2 触发 {len(s2_tasks)}/{len(s1_keys)} 对（{trigger_rate:.1%}）")

    s2_results = {}
    if s2_tasks:
        s2_outputs = llm.generate(
            render_prompts(tokenizer, [t[2] for t in s2_tasks], chat_kwargs, model_name),
            sp, use_tqdm=True)
        for (sid, cid, _), out in zip(s2_tasks, s2_outputs):
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            raw, norm = extract_candidate_probs(lp, a_ids + b_ids)
            s2_results[(sid, cid)] = {"p_A": norm[0], "p_B": norm[1],
                                      "pred_B": raw[1] >= raw[0],
                                      "gen_token": gen_tok}

    # ---------- 汇总与映射 ----------
    results = []
    for sample, cat_out in zip(dataset, cat_outputs):
        lp = cat_out.outputs[0].logprobs[0]
        _, cat_norm = extract_candidate_probs(lp, yes_ids + no_ids)
        cat = {"p_yes": cat_norm[0], "p_no": cat_norm[1],
               "gen_token": tokenizer.decode(list(cat_out.outputs[0].token_ids))}

        concept_values, entry_concepts = {}, {}
        for c in concepts:
            s1 = s1_results[(sample["id"], c["id"])]
            s2 = s2_results.get((sample["id"], c["id"]))
            if not s1["pred_yes"]:
                value = 0
            elif s2 is None:
                value = 0  # 不应发生
            else:
                value = 2 if s2["pred_B"] else 1
            concept_values[c["id"]] = value
            entry_concepts[c["id"]] = {
                "stage1": s1,
                "stage2": s2,
                "value": value,
            }

        max_value = max(concept_values.values())
        results.append({
            "id": sample["id"],
            "source": sample["source"],
            "content": sample["content"],
            "toxic": sample["toxic"],
            "topic": sample.get("topic"),
            "expression": sample.get("expression"),
            "concepts": entry_concepts,
            "ring1a_max_value": max_value,
            "ring1a_confirmed": max_value >= 1,
            "CAT1a": cat,
        })

    # ---------- 保存 ----------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "data" / "processed" / "CCC" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ring1a_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "prompt_version": "ring1a v2.8",
                "dataset": args.dataset,
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "stage2_trigger_rate": round(trigger_rate, 4),
                "concepts": [c["id"] for c in concepts],
            },
            "results": results,
        }, f, ensure_ascii=False, indent=4)

    # 扁平 CSV（WPS 审阅用）
    csv_path = out_dir / f"ring1a_{model_name}_flat.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "toxic", "expression", "concept",
                    "p_yes", "stage1_pred", "stage1_gen", "p_B", "stage2_gen", "value",
                    "ring1a_max", "ring1a_confirmed", "CAT1a_p_yes", "CAT1a_gen", "content"])
        for r in results:
            for cid, e in r["concepts"].items():
                w.writerow([r["id"], r["source"], r["toxic"], r.get("expression"), cid,
                            round(e["stage1"]["p_yes"], 4), int(e["stage1"]["pred_yes"]),
                            e["stage1"]["gen_token"],
                            round(e["stage2"]["p_B"], 4) if e["stage2"] else "",
                            e["stage2"]["gen_token"] if e["stage2"] else "",
                            e["value"], r["ring1a_max_value"], int(r["ring1a_confirmed"]),
                            round(r["CAT1a"]["p_yes"], 4), r["CAT1a"]["gen_token"], r["content"]])

    # ---------- 概览统计 ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(results)}")
    print(f"阶段2触发率: {trigger_rate:.1%}")
    for c in concepts:
        vs = [r["concepts"][c["id"]]["value"] for r in results]
        s1y = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["pred_yes"])
        print(f"  {c['id']}: 阶段1'是' {s1y}/{len(results)}  值分布 0/1/2 = "
              f"{vs.count(0)}/{vs.count(1)}/{vs.count(2)}")
    confirmed = sum(1 for r in results if r["ring1a_confirmed"])
    print(f"环1a 确认: {confirmed}/{len(results)}")
    cat_mean = sum(r["CAT1a"]["p_yes"] for r in results) / len(results)
    print(f"CAT1a p(是) 均值: {cat_mean:.4f}")
    # 首 token 越界监控：生成 token 不在候选集（是/否 或 A/B）的调用数
    off_s1 = sum(1 for v in s1_results.values() if v["gen_token"].strip() not in ("是", "否"))
    off_s2 = sum(1 for v in s2_results.values() if v["gen_token"].strip() not in ("A", "B"))
    off_cat = sum(1 for r in results if r["CAT1a"]["gen_token"].strip() not in ("是", "否"))
    print(f"首token越界: 阶段1 {off_s1}/{len(s1_results)}  阶段2 {off_s2}/{len(s2_results)}  CAT1a {off_cat}/{len(results)}")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
