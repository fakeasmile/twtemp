"""v1.5.1 增量测量：言语行为形态 5 分类（A断言/B引用/C假设/D反问/E无贬损）。

背景：v4 Q 系列判死后的下一步——三次失败的仪器全是"是/否 gate"问法
（要求 9B 做排除语义指派，三次证明做不到），而流水线中已验证有效的模式是
两阶段多分类（stage2 A/B 强度分类，22 概念全部有效）。本实验把环3"立场承诺"
从 gate 改写为言语行为形态分类（理论原形：Austin/Searle 言语行为学）：
贬损内容以何种形态出现——说话人自己的断言，还是引用/假设/反问中的出现。
压制候选规则：仅 A 类允许环3 确认（分类驱动聚合，区别于已死的 gate）。

验收协议（与 v4 相同，预登记）：train 上以 FP 池分离度作 J*（P(A|toxic1,链触发)
vs P(A|benign,链触发)），6c 程序判定；test 只碰一次。判死则降级记录，不硬调。

提示词要点：不预设文本含贬损内容（E 类承接）；混合形态按主导形态单选；
单 token 贪心解码，logprob 解析与 stage2 同源（run_ring1a 机制复用）。

用法（服务器）：
  python scripts/run_speech_act_v151.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat_v15/rings_glm-4-9b-chat_train.json
  python scripts/run_speech_act_v151.py --model_path <模型路径> \
      --rings data/processed/TOXICN/glm-4-9b-chat_v15/rings_glm-4-9b-chat_test.json
输出：<输入目录>_v151/<同名文件>，结果挂在 results[].speech_act，meta 标记 1.5.1。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from run_ring1a import (  # noqa: E402
    build_sampling_params, candidate_mass, detect_chat_kwargs,
    get_candidate_ids, load_vllm_model, render_prompts,
)
from run_rings import STAGE1_TEMPLATE_U  # noqa: E402  同款 ring4 社群 persona 前缀

ACT_TEMPLATE = """你是语言学标注助手。判断文本中贬损、攻击或负面评价内容的言语行为形态。只依据文本本身判断，不评价内容对错，不解释。文本没有明显贬损内容时选 E。

文本：{text}

A = 直接断言：贬损内容是说话人自己表达的观点，没有引用、假设等框架
B = 引用/转述：贬损内容出现在引用他人话语、转述他人观点或描述他人言论中
C = 假设/条件：贬损内容出现在假设、条件或虚拟情境中（如"如果…就…"）
D = 反问/质疑：贬损内容以反问、质疑或反驳的形式出现
E = 无明显贬损内容

若多种形态并存，按占主导地位的形态判断。文本中贬损内容的言语行为形态是？只回答一个字母。回答："""

CLASSES = ["A", "B", "C", "D", "E"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rings", required=True, help="v1.4/v1.5 rings json 路径")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    src = Path(args.rings)
    d = json.load(open(src, encoding="utf-8"))
    results, meta = d["results"], d["meta"]
    model_name = Path(args.model_path).name
    if meta["model"] != model_name:
        raise SystemExit(f"模型不一致：rings 由 {meta['model']} 测得，当前 {model_name}")
    if "speech_act" in results[0]:
        raise SystemExit("输入已含 speech_act 通道，疑似重复增量")
    has_q = "Q1" in results[0]["concepts"]
    print(f"加载 {src.name}：{len(results)} 条，model={model_name}，含Q通道={has_q}")

    tokenizer, llm = load_vllm_model(Path(args.model_path), args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    letter_groups = [get_candidate_ids([ch], tokenizer) for ch in CLASSES]
    sp = build_sampling_params([t for g in letter_groups for t in g])
    print(f"verbalizer: {[g for g in letter_groups]}")

    prompts = [ACT_TEMPLATE.format(text=r["content"]) for r in results]
    print("言语行为分类推理（单 token 贪心）...")
    outs = llm.generate(render_prompts(tokenizer, prompts, chat_kwargs, model_name),
                        sp, use_tqdm=True)

    dist = {"toxic0": Counter(), "toxic1": Counter()}
    invalid = 0
    for r, out in zip(results, outs):
        lp = out.outputs[0].logprobs[0]
        gen_tok = tokenizer.decode(list(out.outputs[0].token_ids))
        masses, norm = candidate_mass(lp, letter_groups)
        valid = sum(masses) > 0
        pred = CLASSES[max(range(len(CLASSES)), key=lambda i: masses[i])] if valid else None
        if not valid:
            invalid += 1
        r["speech_act"] = {
            "probs": {ch: round(norm[i], 6) for i, ch in enumerate(CLASSES)},
            "pred": pred, "valid": valid, "gen_token": gen_tok,
        }
        if "toxic" in r and pred:
            dist[f"toxic{r['toxic']}"][pred] += 1

    meta["speech_act"] = {"version": "1.5.1", "classes": CLASSES}

    out_dir = src.parent.parent / f"{src.parent.name}_v151"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)

    n = len(results)
    print(f"\n有效解析率 {1 - invalid / n:.1%}")
    if dist["toxic0"] or dist["toxic1"]:
        print("形态分布（质检，方向预期：toxic1 偏 A，toxic0 偏 B/C/D/E）：")
        for k in ("toxic0", "toxic1"):
            tot = sum(dist[k].values()) or 1
            row = "  ".join(f"{ch} {dist[k][ch] / tot:.1%}" for ch in CLASSES)
            print(f"  {k}（n={sum(dist[k].values())}）: {row}")
    print(f"\n已写出 {dst}（v1.5 原文件未改动）")
    print("本地验收：python scripts/probe_frames_v5.py 之后用 speech_act 通道做 J* 检验")


if __name__ == "__main__":
    main()
