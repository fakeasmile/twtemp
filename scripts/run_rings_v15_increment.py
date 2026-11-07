"""v1.5 增量测量：仅对 Q1-Q7 框架概念（概念集 v4）跑 stage1，合并进现有 rings JSON。

背景：2c 控制实验 + expr0 法证分析定论——402 个 FP 集中于 expr0，病根是环3/环4
缺"话语框架"仪器（use/mention 区分）。v4 为纯新增（不动 v3 的 35 概念），故无需
全量重跑：加载 v1.4 rings 输出，对每条文本补测 7 个 Q 概念的 stage1 p_yes，
合并后写入 *_v15/ 新目录（v1.4 文件保持原样）。

Q1-Q7 定义与设计依据：data/concepts/concept_set_v4_frame_atoms.md（用户已确认）。
提示词复用 run_rings.STAGE1_TEMPLATE（ring4 用 _U 变体），verbalizer/解析复用
run_ring1a 同源机制，保证与 v1.4 测量逐位同构。
注意：frame 压制（≥3 命中 → 环3/环4 不确认）在下游分析/探针阶段实施，
本脚本只负责测量与落盘，不改任何聚合逻辑。

用法（服务器）：
  python scripts/run_rings_v15_increment.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat/rings_glm-4-9b-chat_train.json
  python scripts/run_rings_v15_increment.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat/rings_glm-4-9b-chat_test.json
输出：<输入目录>_v15/<同名文件>，meta 追加 7 个概念条目并标记 version=1.5。
"""
import argparse
import json
import shutil
from pathlib import Path

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from run_ring1a import (  # noqa: E402
    build_sampling_params, candidate_mass, detect_chat_kwargs,
    get_candidate_ids, load_vllm_model, render_prompts,
    STAGE1_THRESHOLD, STAGE1_THRESHOLD_DEFAULT,
)
from run_rings import STAGE1_TEMPLATE, STAGE1_TEMPLATE_U  # noqa: E402

# Q1-Q7：定义与 concept_set_v4_frame_atoms.md 逐字一致（role=frame，值域 0-1，无阶段 2）。
FRAME_CONCEPTS = [
    {"id": "Q1", "name": "直接引语结构", "ring": "ring3",
     "definition": "文本中出现引号（如“”‘’「」《》等）包裹的他人话语或对话式引用结构。"
                   "只看结构是否存在，不问引语内容是什么。"},
    {"id": "Q2", "name": "转述框架", "ring": "ring3",
     "definition": "文本中出现用“说/认为/觉得/意思/观点/声称/表示”等转述动词引出他人立场的句子"
                   "（引出的是别人的看法，不是说话人自己的表态）。"},
    {"id": "Q3", "name": "反问质疑句式", "ring": "ring3",
     "definition": "文本中出现以“难道/怎么能/怎么就/为什么/哪有/何来”等开头或包含这些疑问词的反问、"
                   "质疑句（句尾常带问号，表达质问而非陈述）。"},
    {"id": "Q4", "name": "元语言指涉", "ring": "ring4",
     "definition": "文本中出现“这句话/这个词/这种说法/这种言论”等指涉话语本身的表达"
                   "（谈论某句话/某个词，而不是直接使用它攻击人）。"},
    {"id": "Q5", "name": "范围限定/分离", "ring": "ring3",
     "definition": "文本中出现“并不是所有/并非/不是所有/个别/少数/一部分/不讨厌”等"
                   "限定范围或把对象从群体中分离出来的表达。"},
    {"id": "Q6", "name": "个体指向", "ring": "ring4",
     "definition": "文本中的贬损或负面评价指向具体的个人（用“你/这人/答主/他”等指称），"
                   "而不是泛化到身份群体（性别/地域/民族/宗教/性取向/残障/年龄/职业等）。"},
    {"id": "Q7", "name": "反歧视立场声明", "ring": "ring3",
     "definition": "文本中出现说话人自己的反歧视/反偏见立场声明，如“不应该歧视/反对地域黑/"
                   "歧视是不对的/不能因为XX就XX”。"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rings", required=True, help="v1.4 rings json 路径")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    src = Path(args.rings)
    d = json.load(open(src, encoding="utf-8"))
    results, meta = d["results"], d["meta"]
    model_name = Path(args.model_path).name
    if meta["model"] != model_name:
        raise SystemExit(f"模型不一致：rings 由 {meta['model']} 测得，当前 {model_name}")
    dup = [c["id"] for c in FRAME_CONCEPTS if c["id"] in results[0]["concepts"]]
    if dup:
        raise SystemExit(f"输入已含 Q 概念 {dup}，疑似重复增量")
    texts = [r["content"] for r in results]
    print(f"加载 {src.name}：{len(texts)} 条，meta.model={meta['model']}，"
          f"补测 Q 概念 {len(FRAME_CONCEPTS)} 个 → {len(texts)*len(FRAME_CONCEPTS)} 条 stage1")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    sp_yn = build_sampling_params(yes_ids + no_ids)
    tau = STAGE1_THRESHOLD.get(model_name, STAGE1_THRESHOLD_DEFAULT)
    print(f"stage1 判是阈值 τ={tau}（模型 {model_name}，复用 1a 登记表）")

    prompts = []
    for r in results:
        for c in FRAME_CONCEPTS:
            tmpl = STAGE1_TEMPLATE_U if c["ring"] == "ring4" else STAGE1_TEMPLATE
            prompts.append(tmpl.format(text=r["content"], concept_name=c["name"],
                                       definition=c["definition"]))
    print("stage1 推理（Q 系列仅 0-1 值域，不触发阶段 2）...")
    outs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                        sp_yn, use_tqdm=True)

    # ---- 解析 + 合并 ----
    valid_n = {c["id"]: 0 for c in FRAME_CONCEPTS}
    hit_n = {c["id"]: 0 for c in FRAME_CONCEPTS}
    for r in results:
        for c in FRAME_CONCEPTS:
            out = outs.pop(0)
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            masses, norm = candidate_mass(lp, [yes_ids, no_ids])
            valid = sum(masses) > 0
            rec = {"p_yes": norm[0], "p_no": norm[1],
                   "raw": [round(masses[0], 6), round(masses[1], 6)],
                   "valid": valid,
                   "topk_hit": [any(t in lp for t in yes_ids), any(t in lp for t in no_ids)],
                   "gen_token": gen_tok}
            value = int(rec["p_yes"] >= tau) if valid else 0
            r["concepts"][c["id"]] = {"stage1": rec, "stage2": None, "value": value}
            valid_n[c["id"]] += int(valid)
            hit_n[c["id"]] += value

    # ---- meta 概念表追加 + 落盘 ----
    for c in FRAME_CONCEPTS:
        meta["concepts"].append({"id": c["id"], "ring": c["ring"], "role": "frame",
                                 "range": "0-1"})
    meta["version"] = "1.5-frame"

    out_dir = src.parent.parent / f"{src.parent.name}_v15"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print("\nQ 概念有效解析率 / 判是率（判是率仅供质检，压制阈值由校准程序选）：")
    for c in FRAME_CONCEPTS:
        n = len(results)
        print(f"  {c['id']} {c['name']:<8} valid {valid_n[c['id']]/n:.1%}  "
              f"判是 {hit_n[c['id']]/n:.1%}")
    print(f"\n已写出 {dst}")
    print("v1.4 原文件未改动；下游请用 *_v15 目录（probe_concepts / analyze 直接换目录即可）")


if __name__ == "__main__":
    main()
