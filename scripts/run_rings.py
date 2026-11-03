"""剩余四环（1b/2/3/4）+ S1 提示词 v1.0 推理脚本（vLLM 加速）。

与 run_ring1a.py 同一套机制（单概念两阶段 + 首 token verbalizer 提取 + 受限解码），
直接复用其模型加载、渲染与提取工具（import，不改动 1a 脚本）。

每次调用 = 一条文本 × 一个判断项：
  阶段 1  存在性（是/否）：除环1a 外的全部概念（环1b 3 个、环2 6 个、环3 12 个、环4 8 个）
          0-1 值域概念只走阶段 1；0-2 值域概念在判"是"后触发阶段 2（A/B 对称定义）
  环4     概念模板首句替换为角色框架（熟悉网络社群用语的标注助手），其余与 1a 一字不差
  CAT     每条文本 5 问：CAT1b / CAT2a（含"话题提及不算"排除语）/ CAT2b（不含）/ CAT3 / CAT4
          CAT2 双变体用于决策排除语去留；全部为出现性锚定问法（v2.11 经验）
  S1      严重度 ordinal 0-5，单次调用，首 token 在数字 0-5 上提质量，双变体：
          S1A 无例词 / S1B 定义带例词（决策例词去留）
映射：阶段1"否"→0，阶段2 A→1、B→2；0-1 概念 是→1；S1 取 argmax 数字。
环确认聚合（guard 压制逻辑）放在 analyze_rings.py，本脚本只落 raw p 与值。

使用示例：
  python scripts/run_rings.py --model_path D:/models/glm-4-9b-chat --data_path data/raw/CCC/ccc.json
  python scripts/run_rings.py --model_path D:/models/Qwen3.5-9B --data_path data/raw/CCC/ccc.json
data_path 支持单个 json 文件，或含 train.json/test.json 的目录（--split test|train|all）。
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from run_ring1a import (  # noqa: E402  复用 1a 已验证的机制，不复制代码
    build_sampling_params, candidate_mass, detect_chat_kwargs,
    get_candidate_ids, load_dataset, load_vllm_model, render_prompts,
    LETTER_VARIANTS, STAGE1_THRESHOLD, STAGE1_THRESHOLD_DEFAULT,
)


# =============================================================================
# 提示词模板（rings v1.3 定稿；v1.2→v1.3 模板与定义未动，仅登记阈值表：
# CAT_TAU / GUARD_TAU = analyze_rings.py 6c 统一校准（Youden's J）在 CCC 试点上的程序输出，
# 落 meta.cat_taus / meta.guard_tau。
# v1.1→v1.2 仅改 CSV 环3 guard 角色为 descriptive——ring_spec v3.4：环3 C5a/b/c、C6a/b 降为
# 描述性维度，不参与阻断，环3 确认 = evidence≥1。
# v1.0→v1.1 仅改 CSV 定义：环3 及 U4a/U4b 定义补"贬损存在"前提，C5b/C5c/U4a/E6a 加排除语，
# U1/U2 删首token架构不可执行的括号注）
# =============================================================================
STAGE1_TEMPLATE = """你是语言学标注助手。判断文本中是否出现下述判断项。只依据文本本身判断，不评价内容对错，不解释。判断项以讽刺、委婉、谐音、缩写等变形出现同样算。

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

该判断项在文本中出现了吗？只回答“是/否”。回答："""

STAGE2_TEMPLATE = """你是语言学标注助手。判断项已确认出现在文本中，现在判断它是怎样出现的。只依据文本本身判断，不评价内容对错，不解释。
A = 间接出现：需要结合上下文才能读出判断项的含义（讽刺、委婉、暗示）
B = 直接出现：文本字面内容本身就是判断项，无需结合上下文

文本：{text}

判断项：{concept_name}
判断项定义：{definition}

判断项是直接出现（B）还是间接出现（A）？只回答 A/B。回答："""

# 环4 概念模板：仅首句换成角色框架（ring_spec 4.5 的固定组件），其余与上面一字不差。
STAGE1_TEMPLATE_U = STAGE1_TEMPLATE.replace(
    "你是语言学标注助手。", "你是熟悉网络社群用语的标注助手。", 1)
STAGE2_TEMPLATE_U = STAGE2_TEMPLATE.replace(
    "你是语言学标注助手。", "你是熟悉网络社群用语的标注助手。", 1)

CAT_TEMPLATES = {
    # 原 CSV 问句"该文本是否通过保护、角色限定或地位贬低，把人或群体固定到从属/较低的位置？"
    # 为机制归因式，改出现性锚定。
    "CAT1B": """以下文本是否包含把人或群体固定到从属、较低位置的表达（以保护、角色限定或地位贬低的方式）。
文本：{text}
文本中有这类表达吗？
只回答 是/否。答：""",
    # 环2 双变体：CAT2A 含"话题提及≠锚定"排除语，CAT2B 不含，跑完 ccc.json 对比后定去留。
    "CAT2A": """以下文本是否把对人的评价指向某个身份群体（性别、地域、民族、宗教、性取向、残障、年龄、职业等），而非只针对具名个人。话题提到某群体但评价不落在群体上的，不算。
文本：{text}
文本把评价指向身份群体了吗？
只回答 是/否。答：""",
    "CAT2B": """以下文本是否把对人的评价指向某个身份群体（性别、地域、民族、宗教、性取向、残障、年龄、职业等），而非只针对具名个人。
文本：{text}
文本把评价指向身份群体了吗？
只回答 是/否。答：""",
    "CAT3": """以下文本中的贬损内容，是否是说话人自己的表态（不是引用他人说法、不是假设情境、不是在反驳这种观点）。
文本：{text}
文本中有说话人自己的贬损表态吗？
只回答 是/否。答：""",
    "CAT4": """假设你是熟悉该文本社区用语的网络社群成员。
以下文本会被你的社群读作对某人或某群体的攻击或贬损吗。
文本：{text}
会吗？
只回答 是/否。答：""",
}

S1_TEMPLATES = {
    # S1 严重度双变体：S1A 纯档位名，S1B 每档带例词（更可执行 vs 更简洁，跑完对比定去留）。
    "S1A": """你是语言学标注助手。按文本中出现的最高伤害档位评级。只依据文本本身判断，不评价内容对错，不解释。
0 = 无伤害内容
1 = 口头贬低
2 = 疏远回避
3 = 排斥歧视
4 = 人身攻击或煽动
5 = 清除灭绝

文本：{text}

档位：只回答一个数字。答：""",
    "S1B": """你是语言学标注助手。按文本中出现的最高伤害档位评级。只依据文本本身判断，不评价内容对错，不解释。
0 = 无伤害内容
1 = 口头贬低（骂人话、贬低性评价）
2 = 疏远回避（不理、拉黑、远离他们）
3 = 排斥歧视（不配享有同等权利、滚回去）
4 = 人身攻击或煽动（威胁、号召动手、去死）
5 = 清除灭绝（灭绝、清除、不该存在）

文本：{text}

档位：只回答一个数字。答：""",
}

DIGITS = "012345"
FULLWIDTH_DIGITS = {"0": "０", "1": "１", "2": "２", "3": "３", "4": "４", "5": "５"}

# 数字候选的表层变体（与字母同理：裸形式 / 前导空格 / 全角）。
for _d, _fw in FULLWIDTH_DIGITS.items():
    LETTER_VARIANTS[_d] = [_d, f" {_d}", _fw, f" {_fw}"]

# CAT/GUARD 判是阈值登记表（按模型目录名）。数值 = analyze_rings.py 6c 统一校准（Youden's J）
# 在 CCC 试点上的程序输出（2026-11-03 登记），非手工拍定；未登记的模型/通道用默认 0.50。
# 实际所用值记录在 meta.cat_taus / meta.guard_tau；正式实验需在独立校准集上重新校准（挂账）。
CAT_TAU = {
    "glm-4-9b-chat": {"CAT1B": 0.94, "CAT2A": 0.92, "CAT2B": 0.94, "CAT3": 0.96, "CAT4": 0.92},
    "Qwen3.5-9B":    {"CAT1B": 0.14, "CAT2A": 0.12, "CAT2B": 0.50, "CAT3": 0.34, "CAT4": 0.22},
}
CAT_TAU_DEFAULT = 0.50
# 环4 guard 阻断阈值：任一 guard 概念 p(是) ≥ τ_guard 即阻断（环3 guard 已降级 descriptive，不阻断）。
GUARD_TAU = {"glm-4-9b-chat": 0.74, "Qwen3.5-9B": 0.10}
GUARD_TAU_DEFAULT = 0.50


# =============================================================================
# 概念词典
# =============================================================================
RING_KEY = {
    "环1b 限制性/等级化评价": "ring1b",
    "环2 身份锚定": "ring2",
    "环3 立场承诺": "ring3",
    "环4 语境取效": "ring4",
}


def load_ring_concepts(csv_path: Path):
    """读取除 环1a 与 S1 外的全部概念，保留类别/角色/值域（决定是否走阶段 2）。"""
    concepts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 传输环节可能在表头引入多余 BOM/NUL（曾出现双重 BOM 致 KeyError），统一剥掉
        reader.fieldnames = [(fn or "").lstrip("\ufeff\x00").strip() for fn in reader.fieldnames]
        for row in reader:
            cat = row["类别"].strip()
            if cat not in RING_KEY:
                continue
            concepts.append({
                "id": row["id"].strip(),
                "name": row["概念"].strip(),
                "definition": row["定义"].strip(),
                "category": cat,
                "ring": RING_KEY[cat],
                "role": row["角色"].strip(),
                "range": row["值域"].strip(),
            })
    if not concepts:
        raise ValueError(f"未在 {csv_path} 中找到环1b/2/3/4 的概念")
    return concepts


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="环1b/2/3/4 + S1 + CAT vLLM 推理")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录（完整路径，如 D:/models/glm-4-9b-chat）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径：单个 json 文件，或含 train.json/test.json 的目录")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "all"],
                        help="目录模式下的划分（单文件模式忽略）")
    parser.add_argument("--concepts_csv", type=str,
                        default=str(project_root / "data" / "concepts" / "cbm_concept_set_v3.csv"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 data/processed/<数据集>/<model_name>/")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    dataset, files, dataset_tag = load_dataset(Path(args.data_path), args.split)
    concepts = load_ring_concepts(Path(args.concepts_csv))
    n_by_ring = {}
    for c in concepts:
        n_by_ring[c["ring"]] = n_by_ring.get(c["ring"], 0) + 1
    print(f"数据集 {dataset_tag}（{', '.join(files)}）样本 {len(dataset)} 条，概念 "
          + " ".join(f"{k}:{v}" for k, v in n_by_ring.items())
          + f"，另加 CAT×{len(CAT_TEMPLATES)}、S1×{len(S1_TEMPLATES)} 每条文本")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)

    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    a_ids = get_candidate_ids(["A"], tokenizer)
    b_ids = get_candidate_ids(["B"], tokenizer)
    digit_groups = [get_candidate_ids([d], tokenizer) for d in DIGITS]
    print(f"verbalizer: 是={yes_ids} 否={no_ids} A={a_ids} B={b_ids} "
          f"数字={[g for g in digit_groups]}")

    sp_yn = build_sampling_params(yes_ids + no_ids)
    sp_ab = build_sampling_params(a_ids + b_ids)
    sp_digit = build_sampling_params([t for g in digit_groups for t in g])

    # ---------- 批量 1：阶段 1（29 概念）+ CAT（5 问/条） ----------
    yn_prompts, yn_keys = [], []
    for sample in dataset:
        for c in concepts:
            tmpl = STAGE1_TEMPLATE_U if c["ring"] == "ring4" else STAGE1_TEMPLATE
            yn_prompts.append(tmpl.format(text=sample["content"],
                                          concept_name=c["name"], definition=c["definition"]))
            yn_keys.append(("s1", sample["id"], c["id"]))
        for cat_name, tmpl in CAT_TEMPLATES.items():
            yn_prompts.append(tmpl.format(text=sample["content"]))
            yn_keys.append(("cat", sample["id"], cat_name))
    n_s1_prompts = sum(1 for k in yn_keys if k[0] == "s1")
    print(f"阶段 1 + CAT 推理（合并批量 {n_s1_prompts}+{len(yn_keys) - n_s1_prompts} 条）...")
    yn_outputs = llm.generate(render_prompts(tokenizer, yn_prompts, chat_kwargs, model_name),
                              sp_yn, use_tqdm=True)

    s1_results, cat_results = {}, {}
    for key, out in zip(yn_keys, yn_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, [yes_ids, no_ids])
        valid = sum(masses) > 0
        rec = {"p_yes": norm[0], "p_no": norm[1],
               "raw": [round(masses[0], 6), round(masses[1], 6)],
               "valid": valid,
               "topk_hit": [any(t in lp for t in yes_ids), any(t in lp for t in no_ids)],
               "gen_token": gen_tok}
        if key[0] == "s1":
            s1_results[(key[1], key[2])] = rec
        else:
            cat_results[(key[1], key[2])] = rec

    # ---------- 阶段 2：仅 0-2 值域概念且阶段 1 判"是" ----------
    tau = STAGE1_THRESHOLD.get(model_name, STAGE1_THRESHOLD_DEFAULT)
    print(f"阶段 1 判是阈值 τ = {tau}（模型 {model_name}，复用 1a 登记表）")
    range2 = {c["id"] for c in concepts if c["range"] == "0-2"}
    concept_by_id = {c["id"]: c for c in concepts}
    sample_by_id = {s["id"]: s for s in dataset}
    s2_tasks = []
    for (sid, cid), rec in s1_results.items():
        if cid in range2 and rec["valid"] and rec["p_yes"] >= tau:
            c = concept_by_id[cid]
            tmpl = STAGE2_TEMPLATE_U if c["ring"] == "ring4" else STAGE2_TEMPLATE
            s2_tasks.append((sid, cid, tmpl.format(text=sample_by_id[sid]["content"],
                                                   concept_name=c["name"],
                                                   definition=c["definition"])))
    print(f"阶段 2 触发 {len(s2_tasks)}/{len(s1_results)} 对"
          f"（0-2 概念 {len(range2)} 个，0-1 概念不触发）")
    s2_results = {}
    if s2_tasks:
        s2_outputs = llm.generate(
            render_prompts(tokenizer, [t[2] for t in s2_tasks], chat_kwargs, model_name),
            sp_ab, use_tqdm=True)
        for (sid, cid, _), out in zip(s2_tasks, s2_outputs):
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            masses, norm = candidate_mass(lp, [a_ids, b_ids])
            valid = sum(masses) > 0
            s2_results[(sid, cid)] = {"p_A": norm[0], "p_B": norm[1],
                                      "pred_B": (masses[1] > masses[0]) if valid else False,
                                      "raw": [round(masses[0], 6), round(masses[1], 6)],
                                      "valid": valid,
                                      "topk_hit": [any(t in lp for t in a_ids),
                                                   any(t in lp for t in b_ids)],
                                      "gen_token": gen_tok}

    # ---------- S1：双变体 ----------
    sev_prompts, sev_keys = [], []
    for sample in dataset:
        for name, tmpl in S1_TEMPLATES.items():
            sev_prompts.append(tmpl.format(text=sample["content"]))
            sev_keys.append((sample["id"], name))
    print(f"S1 推理（批量 {len(sev_prompts)} 条）...")
    sev_outputs = llm.generate(render_prompts(tokenizer, sev_prompts, chat_kwargs, model_name),
                               sp_digit, use_tqdm=True)
    sev_results = {}
    for (sid, name), out in zip(sev_keys, sev_outputs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, digit_groups)
        valid = sum(masses) > 0
        pred = int(max(range(len(DIGITS)), key=lambda i: masses[i])) if valid else None
        sev_results[(sid, name)] = {
            "probs": {d: round(norm[i], 6) for i, d in enumerate(DIGITS)},
            "pred": pred,
            "value": pred if pred is not None else 0,
            "expected": round(sum(i * norm[i] for i in range(len(DIGITS))), 4),
            "valid": valid,
            "topk_hit": [any(t in lp for t in g) for g in digit_groups],
            "gen_token": gen_tok,
        }

    # ---------- 汇总 ----------
    cat_taus = {name: CAT_TAU.get(model_name, {}).get(name, CAT_TAU_DEFAULT)
                for name in CAT_TEMPLATES}
    guard_tau = GUARD_TAU.get(model_name, GUARD_TAU_DEFAULT)
    print("CAT 判是阈值 τ = " + ", ".join(f"{k}:{v}" for k, v in cat_taus.items())
          + f" | 环4 guard 阻断 τ = {guard_tau}")
    results = []
    for sample in dataset:
        sid = sample["id"]
        entry_concepts, ring_vals = {}, {}
        for c in concepts:
            s1r = s1_results[(sid, c["id"])]
            s2r = s2_results.get((sid, c["id"]))
            pred_yes = (s1r["p_yes"] >= tau) if s1r["valid"] else False
            if not pred_yes:
                value = 0
            elif c["range"] == "0-1":
                value = 1
            elif s2r is None or not s2r["valid"]:
                value = 1  # 阶段 1 判是但强度没测出，保守取 1
            else:
                value = 2 if s2r["pred_B"] else 1
            entry_concepts[c["id"]] = {"stage1": s1r, "stage2": s2r, "value": value}
            ring_vals.setdefault(c["ring"], {}).setdefault(c["role"], []).append(value)

        rings = {}
        for ring, roles in ring_vals.items():
            rings[ring] = {
                "evidence_max": max(roles.get("evidence", [0])),
                "guard_max": max(roles.get("guard", [0])),
            }
        results.append({
            "id": sid,
            "source": sample["source"],
            "split": sample["split"],
            "content": sample["content"],
            "toxic": sample["toxic"],
            "topic": sample.get("topic"),
            "expression": sample.get("expression"),
            "platform": sample.get("platform"),
            "concepts": entry_concepts,
            "rings": rings,
            **{name: cat_results[(sid, name)] for name in CAT_TEMPLATES},
            **{name: sev_results[(sid, name)] for name in S1_TEMPLATES},
        })

    # ---------- 保存 ----------
    out_dir = (Path(args.output_dir) if args.output_dir
               else project_root / "data" / "processed" / dataset_tag / model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"rings_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model": model_name,
                "model_path": str(args.model_path),
                "data_path": str(args.data_path),
                "files": files,
                "split": args.split,
                "n_samples": len(dataset),
                "prompt_version": "rings v1.3",
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "stage1_tau": tau,
                "cat_taus": cat_taus,
                "guard_tau": guard_tau,
                "concepts": [{"id": c["id"], "ring": c["ring"], "role": c["role"],
                              "range": c["range"]} for c in concepts],
                "stage2_trigger_rate": round(len(s2_tasks) / max(1, len(s1_results)), 4),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=4)

    # 扁平 CSV（WPS 审阅用）
    csv_path = out_dir / f"rings_{model_name}_flat.csv"
    cat_names = list(CAT_TEMPLATES)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "toxic", "expression", "concept", "ring", "role",
                    "value", "p_yes", "stage1_pred", "p_B", "stage2_valid", "stage2_gen",
                    "S1A_pred", "S1B_pred",
                    *[f"{n}_p" for n in cat_names], "content"])
        for r in results:
            cat_p = [round(r[n]["p_yes"], 4) for n in cat_names]
            for cid, e in r["concepts"].items():
                meta_c = concept_by_id[cid]
                w.writerow([r["id"], r["source"], r["split"], r["toxic"], r.get("expression"),
                            cid, meta_c["ring"], meta_c["role"], e["value"],
                            round(e["stage1"]["p_yes"], 4), int(e["stage1"]["p_yes"] >= tau),
                            round(e["stage2"]["p_B"], 4) if e["stage2"] else "",
                            int(e["stage2"]["valid"]) if e["stage2"] else "",
                            e["stage2"]["gen_token"] if e["stage2"] else "",
                            r["S1A"]["pred"], r["S1B"]["pred"], *cat_p, r["content"]])

    # ---------- 概览统计 ----------
    print("\n" + "=" * 60)
    print(f"模型: {model_name}  样本: {len(results)}")
    for c in concepts:
        vals = [r["concepts"][c["id"]]["value"] for r in results]
        yes = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["p_yes"] >= tau)
        dist = {v: vals.count(v) for v in (0, 1, 2) if vals.count(v)}
        print(f"  {c['id']}({c['ring']},{c['role']},{c['range']}): 阶段1'是' {yes}/{len(results)}"
              f"  值分布 {dist}")
    for name in cat_names:
        pm = sum(r[name]["p_yes"] for r in results) / len(results)
        y = sum(1 for r in results if r[name]["p_yes"] >= cat_taus[name])
        print(f"  {name}: 判是 {y}/{len(results)}  p(是)均 {pm:.4f}")
    for name in S1_TEMPLATES:
        preds = [r[name]["pred"] for r in results]
        dist = {d: preds.count(d) for d in range(6) if preds.count(d)}
        e_mean = sum(r[name]["expected"] for r in results) / len(results)
        print(f"  {name}: 档位分布 {dist}  E[S1]均 {e_mean:.3f}")
    off_yn = sum(1 for v in list(s1_results.values()) + list(cat_results.values())
                 if v["gen_token"].strip() not in ("是", "否"))
    off_s2 = sum(1 for v in s2_results.values() if v["gen_token"].strip() not in ("A", "B"))
    off_d = sum(1 for v in sev_results.values() if v["gen_token"].strip() not in DIGITS)
    inv_yn = sum(1 for v in list(s1_results.values()) + list(cat_results.values())
                 if not v["valid"])
    inv_s2 = sum(1 for v in s2_results.values() if not v["valid"])
    inv_d = sum(1 for v in sev_results.values() if not v["valid"])
    print(f"首token越界: 阶段1+CAT {off_yn}  阶段2 {off_s2}  S1 {off_d}")
    print(f"无效判定: 阶段1+CAT {inv_yn}  阶段2 {inv_s2}  S1 {inv_d}（受限解码后应为 0）")
    print(f"输出: {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
