# -*- coding: utf-8 -*-
"""新信息来源探针审计·远端测量 v1（vLLM，单 token 受限解码，首 token 逐条登记）。

目的（预登记审计，非正式管线）：在 E 集 1452 样本（FN 526/TP 526/FP 200/TN 200，
配对对照设计）上测两类候选新信息，判定其是否正交于现有"毒性存在性"轴：
  C1 对象指向（4 类强制选择：无特定对象/个体/群体/抽象议题或机构）
  C2 言语行为（6 类强制选择：贬损性评价/侮辱威胁/调侃玩梗/引述转述/诉求批评/中立描述）
  R1/R2/R3 锚定强度评级（3 锚：A=强毒/B=边缘毒/C=无毒；0-9 序数，0=远弱于锚）
判据（回本地后计算）：新特征在 E 内 FN vs FP 的 AUC>=0.65，且与共识激活 |corr|<0.5。

测量协议（与 v6/v7 逐字一致的机芯）：
  受限解码 logits_processors→allowed_token_ids→无约束 三级降级；temperature=0，max_tokens=1
  类别/数字标签均要求单 token（字母 A-F 与数字 0-9），首 token + 完整候选质量分布逐条登记
样本清单 audit_samples.csv 需先同步到远端（同相对路径）。
产物：{output_dir}/probe_audit_{model}.json / _matrix.csv / _firsttoken.csv
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent


# =============================================================================
# 模型加载与生成前缀修正（与 run_items_v7.py 逐字一致）
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


MODEL_PROMPT_SUFFIX = {
    "glm-4-9b-chat": "\n",
    "Qwen3.5-9B": "\n",
}


# =============================================================================
# 问句模板（探针审计 v1；语用健壮条款与 v6/v7 预演练句式一致）
# =============================================================================
C1_TEMPLATE = """你是语言学标注助手。判断文本的言语指向。只依据文本本身判断，不评价内容对错，不解释。文本以讽刺、反讽、委婉等变形表达的内容同样计入判断。

文本：{text}

问题：该文本的言语所针对的对象最接近哪一类？
A=无特定对象（泛泛而谈，没有具体指向）
B=个体（针对特定个人）
C=群体（针对某类人群、某地域人、某性别、某族群等）
D=抽象议题或机构（制度、公司、观点、事件等）

只回答选项字母。回答："""

C2_TEMPLATE = """你是语言学标注助手。判断文本的言语行为。只依据文本本身判断，不评价内容对错，不解释。文本以讽刺、反讽、委婉等变形表达的内容同样计入判断。

文本：{text}

问题：该文本主要的言语行为最接近哪一类？
A=贬损性评价（贬低、丑化、歧视对象）
B=侮辱或威胁（辱骂、诅咒、恐吓）
C=调侃玩梗（玩笑、反讽、自嘲、逗乐）
D=引述转述（引用或转述他人言论）
E=诉求或批评（提出要求、批评现象、表达立场）
F=中立描述（客观陈述信息）

只回答选项字母。回答："""

R_TEMPLATE = """你是语言学标注助手。比较两段文本的攻击性强度。只依据文本本身判断，不评价内容对错，不解释。文本以讽刺、反讽、委婉等变形表达的攻击性同样计入。

锚文本：{anchor}

待测文本：{text}

问题：待测文本的攻击性强度（贬损、冒犯、敌意的程度）与锚文本相比如何？
0=远弱于锚文本，9=远强于锚文本，其余按程度取中间值。只回答一个数字。回答："""

# 三个锚（选自主本 train，选点标准见 select_probe_audit_samples.py）
ANCHORS = [
    ("R1", "A_strong", "TOXICN_train_0079",
     "黑二代真是黑，偷渡来中国，连个回答都要学你碟黑着来匿名不敢明面回答"),
    ("R2", "B_edge", "TOXICN_train_0264",
     "支持正义执行！txl再不管管，气候变暖又要加重了。"),
    ("R3", "C_benign", "TOXICN_train_0135",
     "日本为何分区实行，不一起呢"),
]

C1_OPTIONS = ["A", "B", "C", "D"]
C2_OPTIONS = ["A", "B", "C", "D", "E", "F"]
DIGITS = [str(i) for i in range(10)]


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


# =============================================================================
# 受限解码（与 run_items_v7.py 逐字一致）
# =============================================================================
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
    raise RuntimeError("当前 vLLM 既不支持 logits_processors 也不支持 allowed_token_ids，"
                       "无法约束首 token 到候选集")


def resolve_candidates(tokenizer, options, tag):
    """每个标签必须是单 token（严格版：缺一即报错，不静默丢弃）。"""
    ids = []
    for w in options:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"[{tag}] 标签 {w!r} 不是单 token（{enc}），需换标签或核对 tokenizer")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"[{tag}] 标签 token id 重复: {options}")
    return ids


# =============================================================================
# 数据加载
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
    n_raw = len(raw)
    content_of = {}
    for r in srows:
        i = int(r["id"].split("_")[-1])
        if not (0 <= i < n_raw):
            raise ValueError(f"{r['id']} 越界（raw n={n_raw}）")
        content_of[r["id"]] = str(raw[i]["content"])
    # 锚文本防漂移校验
    for _, _, aid, atext in ANCHORS:
        i = int(aid.split("_")[-1])
        actual = str(raw[i]["content"])
        if actual != atext:
            raise ValueError(f"锚 {aid} 文本与预登记不一致，终止：\n预登记: {atext!r}\n数据: {actual!r}")
    return srows, content_of


def parse_args():
    parser = argparse.ArgumentParser(description="探针审计远端测量 v1（C1/C2 类别 + R1-3 锚定评级）")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--samples_csv", type=str,
                        default=str(project_root / "data" / "processed" / "probe_audit_v1" / "audit_samples.csv"))
    parser.add_argument("--raw_json", type=str,
                        default=str(project_root / "data" / "raw" / "TOXICN" / "train.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="只测前 N 个样本（冒烟用）")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    srows, content_of = load_samples(Path(args.samples_csv), Path(args.raw_json))
    if args.limit is not None and args.limit > 0:
        srows = srows[:args.limit]
        print(f"--limit 生效：仅测量前 {len(srows)} 个样本")

    groups = {}
    for r in srows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    print(f"样本 {len(srows)}（{groups}），问句 5 条/样本，总问询 {len(srows) * 5} 次")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    c1_ids = resolve_candidates(tokenizer, C1_OPTIONS, "C1")
    c2_ids = resolve_candidates(tokenizer, C2_OPTIONS, "C2")
    d_ids = resolve_candidates(tokenizer, DIGITS, "R")
    print(f"verbalizer: C1={C1_OPTIONS}->{c1_ids}  C2={C2_OPTIONS}->{c2_ids}  数字0-9={d_ids}")
    sp, restrict_mode = build_sampling_params(c1_ids + c2_ids + d_ids)

    # ---------- 批量推理：样本外层 × 问句内层 ----------
    prompts, keys = [], []
    for r in srows:
        text = content_of[r["id"]]
        prompts.append(C1_TEMPLATE.format(text=text))
        keys.append((r["id"], "C1"))
        prompts.append(C2_TEMPLATE.format(text=text))
        keys.append((r["id"], "C2"))
        for qid, _, _, atext in ANCHORS:
            prompts.append(R_TEMPLATE.format(anchor=atext, text=text))
            keys.append((r["id"], qid))
    print(f"探针测量推理（批量 {len(prompts)} 条）...")
    outputs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                           sp, use_tqdm=True)

    # ---------- 提取分布 + 首token逐条登记 ----------
    opt_ids = {"C1": c1_ids, "C2": c2_ids, **{q: d_ids for q, _, _, _ in ANCHORS}}
    opts = {"C1": C1_OPTIONS, "C2": C2_OPTIONS, **{q: DIGITS for q, _, _, _ in ANCHORS}}
    res = {r["id"]: {} for r in srows}
    invalid = []
    n_total = len(outputs)
    n_in_range = 0
    mass_sum = 0.0
    mass_ge_999 = 0
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / "probe_audit_v1" / f"{model_name}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_path = out_dir / f"probe_audit_{model_name}_firsttoken.csv"
    with open(ft_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "q_id", "first_token", "argmax", "cand_mass"])
        for (sid, qid), out in zip(keys, outputs):
            lp = out.outputs[0].logprobs[0]
            tok_ids = list(out.outputs[0].token_ids)
            gen_tok = tokenizer.decode(tok_ids)
            ids_q = opt_ids[qid]
            masses, norm = candidate_mass(lp, [[i] for i in ids_q])
            raw_mass = sum(masses)
            valid = raw_mass > 0
            arg = opts[qid][int(np.argmax(norm))] if valid else ""
            res[sid][qid] = {"mass": [round(m, 6) for m in masses],
                             "norm": [round(p, 6) for p in norm] if valid else None,
                             "argmax": arg}
            w.writerow([sid, qid, gen_tok, arg, round(raw_mass, 4)])
            if gen_tok.strip() in opts[qid]:
                n_in_range += 1
            mass_sum += raw_mass
            if raw_mass >= 0.999:
                mass_ge_999 += 1
            if not valid:
                invalid.append({"id": sid, "q": qid, "gen_token": gen_tok})

    ft_stats = {
        "total": n_total,
        "in_range": n_in_range,
        "in_range_pct": round(n_in_range / n_total, 6) if n_total else float("nan"),
        "cand_mass_mean": round(mass_sum / n_total, 6) if n_total else float("nan"),
        "cand_mass_ge_999_pct": round(mass_ge_999 / n_total, 6) if n_total else float("nan"),
    }

    # ---------- JSON 明细 ----------
    json_path = out_dir / f"probe_audit_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "samples_csv": str(args.samples_csv),
                "n_samples": len(srows),
                "questions": ["C1", "C2", "R1", "R2", "R3"],
                "c1_options": C1_OPTIONS, "c2_options": C2_OPTIONS, "rating_scale": DIGITS,
                "anchors": [{"q": q, "name": nm, "id": aid, "text": at}
                            for q, nm, aid, at in ANCHORS],
                "restricted_decoding": restrict_mode,
                "first_token_stats": ft_stats,
                "n_invalid": len(invalid),
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
            },
            "results": res,
            "invalid": invalid,
        }, f, ensure_ascii=False)

    # ---------- 宽表矩阵 ----------
    csv_path = out_dir / f"probe_audit_{model_name}_matrix.csv"
    head = (["id", "group", "toxic", "expression"]
            + [f"C1_{o}" for o in C1_OPTIONS] + ["C1_argmax"]
            + [f"C2_{o}" for o in C2_OPTIONS] + ["C2_argmax"]
            + sum([[f"{q}_val", f"{q}_soft"] for q, _, _, _ in ANCHORS], [])
            + [f"{q}_d{d}" for q, _, _, _ in ANCHORS for d in DIGITS])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in srows:
            rr = res[r["id"]]
            row = [r["id"], r["group"], r["toxic"], r["expression"]]
            row += [rr["C1"]["norm"][k] if rr["C1"]["norm"] else "" for k in range(len(C1_OPTIONS))]
            row += [rr["C1"]["argmax"]]
            row += [rr["C2"]["norm"][k] if rr["C2"]["norm"] else "" for k in range(len(C2_OPTIONS))]
            row += [rr["C2"]["argmax"]]
            for q, _, _, _ in ANCHORS:
                d = rr[q]
                soft = sum(float(d["norm"][k]) * k for k in range(10)) if d["norm"] else ""
                row += [d["argmax"], soft]
            for q, _, _, _ in ANCHORS:
                d = rr[q]
                row += [d["mass"][k] for k in range(10)]
            w.writerow(row)

    # ---------- 概览统计（分组速览） ----------
    print("\n" + "=" * 60)
    print(f"样本: {len(srows)}  问询: {n_total}")
    print(f"首token质量: 首∈合法集 {n_in_range} ({ft_stats['in_range_pct']*100:.2f}%)  "
          f"cand_mass均值 {ft_stats['cand_mass_mean']:.6f}  "
          f"mass>=0.999占比 {ft_stats['cand_mass_ge_999_pct']*100:.2f}%")
    print(f"无效判定: {len(invalid)}（受限解码后应为 0）")

    def soft_of(sid, q):
        d = res[sid][q]
        return (sum(float(d["norm"][k]) * k for k in range(10)) if d["norm"] else float("nan"))

    print("\n-- R1-3 soft 评分分组均值（A强毒/B边缘/C无毒） --")
    print("group     n     R1(A)   R2(B)   R3(C)")
    for g in ("FN", "TP", "FP", "TN"):
        sids = [r["id"] for r in srows if r["group"] == g]
        if not sids:
            continue
        means = []
        for q, _, _, _ in ANCHORS:
            vals = [soft_of(s, q) for s in sids]
            vals = [v for v in vals if v == v]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        print(f"{g:<8}  {len(sids):<4}  " + "  ".join(f"{m:6.3f}" for m in means))

    print("\n-- C1/C2 argmax 分组分布 --")
    for q, option_list in (("C1", C1_OPTIONS), ("C2", C2_OPTIONS)):
        print(f"[{q}]")
        for g in ("FN", "TP", "FP", "TN"):
            sids = [r["id"] for r in srows if r["group"] == g]
            if not sids:
                continue
            cnt = {o: 0 for o in option_list}
            for s in sids:
                a = res[s][q]["argmax"]
                if a in cnt:
                    cnt[a] += 1
            tot = sum(cnt.values()) or 1
            print(f"  {g:<4} " + "  ".join(f"{o}={cnt[o] / tot:.3f}" for o in option_list))

    print(f"\n输出: {json_path}\n      {csv_path}\n      {ft_path}")


if __name__ == "__main__":
    main()
