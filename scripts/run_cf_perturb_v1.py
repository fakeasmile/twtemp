# -*- coding: utf-8 -*-
"""反事实扰动审计·远端测量 v1（vLLM，受限解码，首token逐条登记）。

目的（预登记审计）：在 1452 审计样本上测三类反事实扰动的 Δ毒性，
判定因果/对比模态的信息是否正交于共识轴、且方向符合真值：
  P1 身份替换   省籍/族群/性别/身份词表 → 某地/某些人（长词优先，命中才测）
  P2 缩写规范化 保守网络用语词表（命中才测）
  P3 真诚框架   问句前插入"该发言不是玩笑或引用，是发言者的真实观点"
  Δ = p(扰动) − p(基线)，基线=同一毒性问句的原文本测量
预登记判据（回本地计算）：Δ 特征在 FN-vs-TN / TP-vs-FP 配对真值检验
AUC>=0.65（CI 排除 0.5），且与 p_v6 |corr|<0.5。
产物：{output_dir}/cf_perturb_{model}.json / _matrix.csv / _firsttoken.csv
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent


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
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        enforce_eager=True,
        max_logprobs=50,
    )
    return tokenizer, llm


def detect_chat_kwargs(tokenizer):
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        print("chat template 支持 enable_thinking，已关闭思考模式")
        return {"enable_thinking": False}
    return {}


MODEL_PROMPT_SUFFIX = {"glm-4-9b-chat": "\n", "Qwen3.5-9B": "\n"}


# =============================================================================
# 毒性基线问句（句式与 v6/v7 电池一致）
# =============================================================================
TOX_TEMPLATE = """你是语言学标注助手。判断文本是否为毒性言论，即是否包含对他人的贬损、侮辱、歧视或攻击性内容。只依据文本本身判断，不评价内容对错，不解释。文本以讽刺、委婉、谐音、缩写等变形出现同样算。

文本：{text}

只回答"是/否"。回答："""

# P3 真诚框架：在问句体与文本之间插入框架行（其余逐字不变）
TOX_TEMPLATE_P3 = """你是语言学标注助手。判断文本是否为毒性言论，即是否包含对他人的贬损、侮辱、歧视或攻击性内容。只依据文本本身判断，不评价内容对错，不解释。文本以讽刺、委婉、谐音、缩写等变形出现同样算。

注意：该发言不是玩笑或反讽，也不是引用或转述他人言论，而是发言者的真实观点。

文本：{text}

只回答"是/否"。回答："""

# P1 身份词表（长词优先贪心替换）
P1_MAP = [
    ("维吾尔族人", "某些人"), ("内蒙古人", "某地人"), ("黑龙江人", "某地人"),
    ("维吾尔", "某些人"), ("穆斯林", "某些人"), ("农民工", "某些人"),
    ("低端人口", "某些人"), ("同性恋", "某些人"), ("跨性别", "某些人"),
    ("河南人", "某地人"), ("江西人", "某地人"), ("上海人", "某地人"),
    ("北京人", "某地人"), ("东北人", "某地人"), ("山东人", "某地人"),
    ("广东人", "某地人"), ("山西人", "某地人"), ("安徽人", "某地人"),
    ("苏北人", "某地人"), ("新疆人", "某地人"), ("西藏人", "某地人"),
    ("广西人", "某地人"), ("湖南人", "某地人"), ("湖北人", "某地人"),
    ("四川人", "某地人"), ("重庆人", "某地人"), ("天津人", "某地人"),
    ("河北人", "某地人"), ("江苏人", "某地人"), ("浙江人", "某地人"),
    ("福建人", "某地人"), ("贵州人", "某地人"), ("云南人", "某地人"),
    ("甘肃人", "某地人"), ("海南人", "某地人"), ("宁夏人", "某地人"),
    ("青海人", "某地人"), ("台湾人", "某地人"), ("香港人", "某地人"),
    ("辽宁人", "某地人"), ("吉林人", "某地人"), ("陕西人", "某地人"),
    ("回族人", "某些人"), ("乡下人", "某些人"), ("城里人", "某些人"),
    ("农村人", "某些人"), ("外地人", "某些人"), ("本地人", "某些人"),
    ("蒙古族", "某些人"), ("日本人", "某些人"), ("韩国人", "某些人"),
    ("美国人", "某些人"), ("上等人", "某些人"), ("下等人", "某些人"),
    ("农村", "某地"), ("北京户口", "某地户口"), ("上海户口", "某地户口"),
    ("农民", "某些人"),
    ("老人", "某些人"), ("大妈", "某些人"), ("大爷", "某些人"),
    ("小孩", "某些人"), ("学生", "某些人"), ("女人", "某些人"),
    ("男人", "某些人"), ("黑人", "某些人"), ("白人", "某些人"),
    ("回民", "某些人"), ("维族", "某些人"), ("藏族", "某些人"),
    ("壮族", "某些人"), ("苗族", "某些人"), ("老外", "某些人"),
    ("洋人", "某些人"), ("gay", "某些人"), ("txl", "某些人"),
    ("基佬", "某些人"), ("拉拉", "某些人"), ("娘炮", "某些人"),
    ("女司机", "某些人"), ("小仙女", "某些人"), ("女拳", "某些人"),
    ("直男", "某些人"), ("直女", "某些人"), ("舔狗", "某些人"),
    ("河南", "某地"), ("江西", "某地"), ("上海", "某地"), ("北京", "某地"),
    ("东北", "某地"), ("山东", "某地"), ("广东", "某地"), ("安徽", "某地"),
    ("苏北", "某地"), ("新疆", "某地"), ("西藏", "某地"), ("广西", "某地"),
    ("湖南", "某地"), ("湖北", "某地"), ("四川", "某地"), ("重庆", "某地"),
    ("天津", "某地"), ("河北", "某地"), ("江苏", "某地"), ("浙江", "某地"),
    ("福建", "某地"), ("贵州", "某地"), ("云南", "某地"), ("甘肃", "某地"),
    ("海南", "某地"), ("宁夏", "某地"), ("青海", "某地"), ("内蒙古", "某地"),
    ("山西", "某地"), ("陕西", "某地"), ("辽宁", "某地"), ("吉林", "某地"),
    ("黑龙江", "某地"),
]

# P2 网络用语规范化（保守词表：多字符或拉丁缩写）
P2_MAP = [
    ("nmsl", "你妈死了"), ("u1s1", "有一说一"), ("yysy", "有一说一"),
    ("xswl", "笑死我了"), ("dddd", "懂的都懂"), ("yyds", "永远的神"),
    ("dbq", "对不起"), ("bhys", "不好意思"), ("srds", "虽然但是"),
    ("zqsg", "真情实感"), ("awsl", "啊我死了"), ("gkd", "搞快点"),
    ("nsdd", "你说得对"), ("pyq", "朋友圈"), ("bdjw", "不懂就问"),
    ("kdl", "磕到了"), ("jms", "姐妹们"), ("txl", "同性恋"),
    ("蚌埠住了", "绷不住了"), ("栓Q", "谢谢"), ("emo了", "情绪低落了"),
    ("集美", "姐妹"), ("老铁", "朋友"), ("铁子", "朋友"),
    ("干饭", "吃饭"), ("摆烂", "自暴自弃"), ("躺平", "消极不抗争"),
    ("内卷", "过度竞争"), ("破防", "情绪失控"), ("giegie", "哥哥"),
    ("yygq", "阴阳怪气"), ("u can u up", "你行你上"), ("no can no bb", "不行别说话"),
]


def make_perturb_text(text, mapping):
    """长词优先贪心替换（显式按词长降序，稳定排序保留同长原序）；
    返回 (新文本, 命中数)。无命中返回 None。"""
    hits = 0
    out = text
    for a, b in sorted(mapping, key=lambda kv: -len(kv[0])):
        if a in out:
            hits += out.count(a)
            out = out.replace(a, b)
    return (out, hits) if hits > 0 else None


# =============================================================================
# 受限解码机芯（与 run_items_v7.py 逐字一致）
# =============================================================================
def render_prompts(tokenizer, texts, chat_kwargs, model_name):
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


def candidate_mass(first_token_logprobs, id_groups):
    masses = [sum(math.exp(first_token_logprobs[t].logprob) for t in g if t in first_token_logprobs)
              for g in id_groups]
    total = sum(masses) + 1e-8
    return masses, [m / total for m in masses]


def make_mask_processor(allowed_ids):
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


def build_sampling_params(allowed_ids):
    allowed = sorted(set(allowed_ids))
    candidates = [
        ("logits_processors", {"logits_processors": [make_mask_processor(allowed)]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("无约束", {}),
    ]
    for name, extra in candidates:
        try:
            sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra)
        except TypeError as e:
            print(f"受限解码 {name} 不可用（{e}），尝试下一种")
            continue
        print(f"受限解码方式: {name}，候选 id = {allowed}")
        return sp, name
    raise RuntimeError("当前 vLLM 既不支持 logits_processors 也不支持 allowed_token_ids")


def resolve_candidates(tokenizer, options, tag):
    ids = []
    for w in options:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"[{tag}] 标签 {w!r} 不是单 token（{enc}）")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"[{tag}] 标签 token id 重复: {options}")
    return ids


# =============================================================================
# 数据
# =============================================================================
def load_samples(samples_csv, raw_json):
    srows = []
    with open(samples_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            srows.append({"id": row["id"].strip(), "group": row["group"].strip(),
                          "toxic": int(row["toxic"]), "expression": int(row["expression"])})
    if not srows:
        raise ValueError(f"{samples_csv} 无样本")
    raw = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    content_of = {}
    for r in srows:
        i = int(r["id"].split("_")[-1])
        if not (0 <= i < len(raw)):
            raise ValueError(f"{r['id']} 越界")
        content_of[r["id"]] = str(raw[i]["content"])
    return srows, content_of


def parse_args():
    parser = argparse.ArgumentParser(description="反事实扰动审计远端测量 v1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--samples_csv", type=str,
                        default=str(project_root / "data" / "processed" / "probe_audit_v1" / "audit_samples.csv"))
    parser.add_argument("--raw_json", type=str,
                        default=str(project_root / "data" / "raw" / "TOXICN" / "train.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    srows, content_of = load_samples(Path(args.samples_csv), Path(args.raw_json))
    if args.limit is not None and args.limit > 0:
        srows = srows[:args.limit]
        print(f"--limit 生效：仅测量前 {len(srows)} 个样本")

    n_p1 = sum(1 for r in srows if make_perturb_text(content_of[r["id"]], P1_MAP))
    n_p2 = sum(1 for r in srows if make_perturb_text(content_of[r["id"]], P2_MAP))
    print(f"样本 {len(srows)}；P1 命中 {n_p1}，P2 命中 {n_p2}")
    n_calls = len(srows) * 2 + n_p1 + n_p2
    print(f"预计调用 {n_calls} 次（基线 {len(srows)} + P3 {len(srows)} + P1 {n_p1} + P2 {n_p2}）")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    yn_ids = resolve_candidates(tokenizer, ["是", "否"], "是/否")
    print(f"verbalizer: 是/否 -> {yn_ids}")
    sp, restrict_mode = build_sampling_params(yn_ids)

    # ---------- 组装问询 ----------
    prompts, keys = [], []
    for r in srows:
        text = content_of[r["id"]]
        prompts.append(TOX_TEMPLATE.format(text=text))
        keys.append((r["id"], "base"))
        prompts.append(TOX_TEMPLATE_P3.format(text=text))
        keys.append((r["id"], "P3"))
        p1 = make_perturb_text(text, P1_MAP)
        if p1 is not None:
            prompts.append(TOX_TEMPLATE.format(text=p1[0]))
            keys.append((r["id"], "P1"))
        p2 = make_perturb_text(text, P2_MAP)
        if p2 is not None:
            prompts.append(TOX_TEMPLATE.format(text=p2[0]))
            keys.append((r["id"], "P2"))
    print(f"扰动审计推理（批量 {len(prompts)} 条）...")
    outputs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                           sp, use_tqdm=True)

    # ---------- 提取 + 首token登记 ----------
    res = {r["id"]: {} for r in srows}
    invalid = []
    n_total = len(outputs)
    n_in_range = 0
    mass_sum = 0.0
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / "cf_perturb_v1" / f"{model_name}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_path = out_dir / f"cf_perturb_{model_name}_firsttoken.csv"
    with open(ft_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "cond", "first_token", "p_yes", "cand_mass"])
        for (sid, cond), out in zip(keys, outputs):
            lp = out.outputs[0].logprobs[0]
            tok_ids = list(out.outputs[0].token_ids)
            gen_tok = tokenizer.decode(tok_ids)
            masses, norm = candidate_mass(lp, [[i] for i in yn_ids])
            p_yes = norm[0]
            res[sid][cond] = p_yes
            w.writerow([sid, cond, gen_tok, round(p_yes, 6), round(sum(masses), 4)])
            if gen_tok.strip() in ("是", "否"):
                n_in_range += 1
            mass_sum += sum(masses)
            if gen_tok.strip() not in ("是", "否"):
                invalid.append({"id": sid, "cond": cond, "gen_token": gen_tok})

    ft_stats = {
        "total": n_total,
        "in_range": n_in_range,
        "in_range_pct": round(n_in_range / n_total, 6) if n_total else float("nan"),
        "cand_mass_mean": round(mass_sum / n_total, 6) if n_total else float("nan"),
    }

    # ---------- 明细 JSON ----------
    json_path = out_dir / f"cf_perturb_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name, "model_path": str(args.model_path),
                "n_samples": len(srows),
                "conditions": ["base", "P1", "P2", "P3"],
                "restricted_decoding": restrict_mode,
                "first_token_stats": ft_stats,
                "n_invalid": len(invalid),
                "p1_hits": n_p1, "p2_hits": n_p2,
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
            },
            "results": res,
            "invalid": invalid,
        }, f, ensure_ascii=False)

    # ---------- 宽表矩阵 ----------
    csv_path = out_dir / f"cf_perturb_{model_name}_matrix.csv"
    head = ["id", "group", "toxic", "expression",
            "p_base", "p_P1", "dP1", "p_P2", "dP2", "p_P3", "dP3",
            "p_v6"]
    p6 = {}
    p6_path = project_root / "results" / "v7" / "diag_v7_v6_gap" / "oof_error_diff.csv"
    if p6_path.exists():
        with open(p6_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                p6[row["id"].strip()] = row["p_v6"]
        print(f"已读入 p_v6 {len(p6)} 条（本地审计用，远端可能缺此文件则留空）")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in srows:
            rr = res[r["id"]]
            pb = rr.get("base")
            p1 = rr.get("P1")
            p2 = rr.get("P2")
            p3 = rr.get("P3")
            row = [r["id"], r["group"], r["toxic"], r["expression"],
                   round(pb, 6), p1, (round(p1 - pb, 6) if p1 is not None else ""),
                   p2, (round(p2 - pb, 6) if p2 is not None else ""),
                   round(p3, 6), round(p3 - pb, 6),
                   p6.get(r["id"], "")]
            w.writerow(row)

    # ---------- 概览 ----------
    print("\n" + "=" * 60)
    print(f"样本 {len(srows)}  问询 {n_total}")
    print(f"首token: 合法 {n_in_range} ({ft_stats['in_range_pct']*100:.2f}%)  "
          f"cand_mass均值 {ft_stats['cand_mass_mean']:.6f}  无效 {len(invalid)}")
    print("\n-- 分组均值 --")
    print("group     n     p_base  dP1     dP2     dP3")
    for g in ("FN", "TP", "FP", "TN"):
        sids = [r["id"] for r in srows if r["group"] == g]
        if not sids:
            continue
        vals = {}
        for c in ("base", "P1", "P2", "P3"):
            xs = [res[s][c] for s in sids if c in res[s]]
            vals[c] = sum(xs) / len(xs) if xs else float("nan")
        d1 = [res[s]["P1"] - res[s]["base"] for s in sids if "P1" in res[s]]
        d2 = [res[s]["P2"] - res[s]["base"] for s in sids if "P2" in res[s]]
        d3 = [res[s]["P3"] - res[s]["base"] for s in sids]
        f1m = sum(d1) / len(d1) if d1 else float("nan")
        f2m = sum(d2) / len(d2) if d2 else float("nan")
        f3m = sum(d3) / len(d3) if d3 else float("nan")
        print(f"{g:<8}  {len(sids):<4}  {vals['base']:6.3f}  {f1m:6.3f}  {f2m:6.3f}  {f3m:6.3f}")

    print(f"\n输出: {json_path}\n      {csv_path}\n      {ft_path}")


if __name__ == "__main__":
    main()
