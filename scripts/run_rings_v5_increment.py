"""v5 增量测量：仅对 T1-T5 表层原子（概念集 v5）跑 stage1+stage2，合并进现有 rings JSON。

背景：测量可靠性审计定论——瓶颈层天花板（E++ ~0.829 vs backbone 0.8331）病因是
词汇表覆盖缺口，非测量造假/校准失真。v5 为纯新增 5 个词汇-句法域原子（不动 v4 的
47 概念），故无需全量重跑：加载 v4 rings 输出，对每条文本补测 5 个 T 概念，
合并后写入 *_v5/ 新目录（v4 文件保持原样）。

T1-T5 定义与 concept_set_v5_surface_atoms.md 逐字一致（定稿自包含版）：
  T1 威胁恐吓程式  ring4 evidence 0-2（触发阶段 2）
  T2 诅咒祝愿程式  ring3 evidence 0-1
  T3 亲属牵连程式  ring1a evidence 0-1
  T4 性道德羞辱    ring1a evidence 0-2（触发阶段 2）
  T5 对线冲突标记  ring4 descriptive 0-1
提示词复用 run_rings.STAGE1/STAGE2 模板（ring4 用 _U 变体），verbalizer/解析复用
run_ring1a 同源机制，保证与 v1.4 测量逐位同构。本脚本只负责测量与落盘，
不改任何聚合逻辑；判死与验收在本地按预登记协议执行。

用法（服务器，先 test 后 train）：
  python scripts/run_rings_v5_increment.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat_v4/rings_glm-4-9b-chat_test.json
  python scripts/run_rings_v5_increment.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat_v4/rings_glm-4-9b-chat_train.json
输出：<输入目录>_v5/<同名文件>，meta 追加 5 个概念条目并标记 version=v5-surface。
"""
import argparse
import json
from pathlib import Path

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from run_ring1a import (  # noqa: E402
    build_sampling_params, candidate_mass, detect_chat_kwargs,
    get_candidate_ids, load_vllm_model, render_prompts,
    STAGE1_THRESHOLD, STAGE1_THRESHOLD_DEFAULT,
)
from run_rings import (  # noqa: E402
    STAGE1_TEMPLATE, STAGE1_TEMPLATE_U, STAGE2_TEMPLATE, STAGE2_TEMPLATE_U,
)

# T1-T5：定义与 concept_set_v5_surface_atoms.md 定稿逐字一致。
T_CONCEPTS = [
    {"id": "T1", "name": "威胁恐吓程式", "ring": "ring4", "role": "evidence", "range": "0-2",
     "definition": "文本对受话人或具名个人发出威胁恐吓：信不信我…/给我等着/弄死你/砍你/"
                   "收拾你/人肉你/地址发我。排除：号召对某个群体采取行动（滚出去/冲爆/"
                   "举报到底），或只有咒骂而没有威胁动作。"},
    {"id": "T2", "name": "诅咒祝愿程式", "ring": "ring3", "role": "evidence", "range": "0-1",
     "definition": "文本以祝愿或诅咒句式表达恶意（祝你出门被撞/希望你全家…/活不过今年/"
                   "早点消失/不得好死）。排除：直陈式骂人话（无祝愿/诅咒句式）；"
                   "号召采取行动的句子。"},
    {"id": "T3", "name": "亲属牵连程式", "ring": "ring1a", "role": "evidence", "range": "0-1",
     "definition": "文本把贬损或诅咒扩展到对方的亲属（问候家人/nmsl/全家不得好死/你妈/祖上）。"
                   "排除：仅骂受话人本人；亲属词只作普通脏话音节、语义已虚化的（卧槽/我靠）"
                   "不算。"},
    {"id": "T4", "name": "性道德羞辱", "ring": "ring1a", "role": "evidence", "range": "0-2",
     "definition": "文本以性行为或性道德贬损对象（婊/破鞋/绿茶/给他戴帽/被睡/不检点/滥交/母0）。"
                   "排除：非性维度的蔑称或群体标签；与性无关的一般骂人话。"},
    {"id": "T5", "name": "对线冲突标记", "ring": "ring4", "role": "descriptive", "range": "0-1",
     "definition": "文本出现网络对线/争执话语标记（破防/急了/对线/开团/嘴硬/回旋镖/赢麻了）。"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rings", required=True, help="v4 rings json 路径")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    src = Path(args.rings)
    d = json.load(open(src, encoding="utf-8"))
    results, meta = d["results"], d["meta"]
    model_name = Path(args.model_path).name
    if meta["model"] != model_name:
        raise SystemExit(f"模型不一致：rings 由 {meta['model']} 测得，当前 {model_name}")
    dup = [c["id"] for c in T_CONCEPTS if c["id"] in results[0]["concepts"]]
    if dup:
        raise SystemExit(f"输入已含 T 概念 {dup}，疑似重复增量")
    texts = [r["content"] for r in results]
    print(f"加载 {src.name}：{len(texts)} 条，meta.model={meta['model']}，"
          f"补测 T 概念 {len(T_CONCEPTS)} 个 → {len(texts)*len(T_CONCEPTS)} 条 stage1")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    yes_ids = get_candidate_ids(["是"], tokenizer)
    no_ids = get_candidate_ids(["否"], tokenizer)
    a_ids = get_candidate_ids(["A"], tokenizer)
    b_ids = get_candidate_ids(["B"], tokenizer)
    sp_yn = build_sampling_params(yes_ids + no_ids)
    sp_ab = build_sampling_params(a_ids + b_ids)
    tau = STAGE1_THRESHOLD.get(model_name, STAGE1_THRESHOLD_DEFAULT)
    print(f"stage1 判是阈值 τ={tau}（模型 {model_name}，复用 1a 登记表）")

    # ---- 阶段 1（全部 5 概念）----
    prompts = []
    for r in results:
        for c in T_CONCEPTS:
            tmpl = STAGE1_TEMPLATE_U if c["ring"] == "ring4" else STAGE1_TEMPLATE
            prompts.append(tmpl.format(text=r["content"], concept_name=c["name"],
                                       definition=c["definition"]))
    print("stage1 推理...")
    outs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                        sp_yn, use_tqdm=True)

    s1_by_key = {}
    for r in results:
        for c in T_CONCEPTS:
            out = outs.pop(0)
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            masses, norm = candidate_mass(lp, [yes_ids, no_ids])
            valid = sum(masses) > 0
            s1_by_key[(r["id"], c["id"])] = {
                "p_yes": norm[0], "p_no": norm[1],
                "raw": [round(masses[0], 6), round(masses[1], 6)],
                "valid": valid,
                "topk_hit": [any(t in lp for t in yes_ids), any(t in lp for t in no_ids)],
                "gen_token": gen_tok,
            }

    # ---- 阶段 2：仅 0-2 值域概念（T1/T4）且阶段 1 判"是" ----
    range2 = {c["id"] for c in T_CONCEPTS if c["range"] == "0-2"}
    concept_by_id = {c["id"]: c for c in T_CONCEPTS}
    sample_by_id = {r["id"]: r for r in results}
    s2_tasks = []
    for (sid, cid), rec in s1_by_key.items():
        if cid in range2 and rec["valid"] and rec["p_yes"] >= tau:
            c = concept_by_id[cid]
            tmpl = STAGE2_TEMPLATE_U if c["ring"] == "ring4" else STAGE2_TEMPLATE
            s2_tasks.append((sid, cid, tmpl.format(text=sample_by_id[sid]["content"],
                                                   concept_name=c["name"],
                                                   definition=c["definition"])))
    print(f"stage2 推理（0-2 概念 {sorted(range2)}，触发 {len(s2_tasks)} 对）...")
    s2_by_key = {}
    if s2_tasks:
        s2_outs = llm.generate(render_prompts(tokenizer, [t[2] for t in s2_tasks],
                                              chat_kwargs, model_name),
                               sp_ab, use_tqdm=True)
        for (sid, cid, _), out in zip(s2_tasks, s2_outs):
            lp = out.outputs[0].logprobs[0]
            gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
            masses, norm = candidate_mass(lp, [a_ids, b_ids])
            valid = sum(masses) > 0
            s2_by_key[(sid, cid)] = {
                "p_A": norm[0], "p_B": norm[1],
                "pred_B": (masses[1] > masses[0]) if valid else False,
                "raw": [round(masses[0], 6), round(masses[1], 6)],
                "valid": valid,
                "topk_hit": [any(t in lp for t in a_ids), any(t in lp for t in b_ids)],
                "gen_token": gen_tok,
            }

    # ---- 汇总值 + 合并 ----
    dist_n = {c["id"]: {} for c in T_CONCEPTS}
    for r in results:
        for c in T_CONCEPTS:
            s1r = s1_by_key[(r["id"], c["id"])]
            s2r = s2_by_key.get((r["id"], c["id"]))
            pred_yes = (s1r["p_yes"] >= tau) if s1r["valid"] else False
            if not pred_yes:
                value = 0
            elif c["range"] == "0-1":
                value = 1
            elif s2r is None or not s2r["valid"]:
                value = 1  # 阶段 1 判是但强度没测出，保守取 1
            else:
                value = 2 if s2r["pred_B"] else 1
            r["concepts"][c["id"]] = {"stage1": s1r, "stage2": s2r, "value": value}
            dist_n[c["id"]][value] = dist_n[c["id"]].get(value, 0) + 1

    # ---- meta 概念表追加 + 落盘 ----
    for c in T_CONCEPTS:
        meta["concepts"].append({"id": c["id"], "ring": c["ring"], "role": c["role"],
                                 "range": c["range"]})
    meta["version"] = "v5-surface"

    out_dir = src.parent.parent / f"{src.parent.name}_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)

    n = len(results)
    print("\nT 概念有效解析率 / 判是率 / 值分布（判是率仅供质检，判死与验收在本地）：")
    for c in T_CONCEPTS:
        valid_n = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["valid"])
        yes_n = sum(1 for r in results if r["concepts"][c["id"]]["stage1"]["p_yes"] >= tau)
        print(f"  {c['id']} {c['name']:<8} valid {valid_n/n:.1%}  判是 {yes_n/n:.1%}  "
              f"值分布 {dist_n[c['id']]}")
    print(f"\n已写出 {dst}")
    print("v4 原文件未改动；下游请用 *_v5 目录（probe_concepts / analyze 直接换目录即可）")


if __name__ == "__main__":
    main()
