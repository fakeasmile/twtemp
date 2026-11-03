"""环1a 提示词探针：在下方 CONFIG 区改文本/概念/定义/模式，直接看两样东西——
  1) LLM 实际收到的完整提示词（模板填槽 + chat template + 后缀，逐层打印）
  2) LLM 的首 token 输出（生成 token、top 候选概率、verbalizer 归一化概率）

复用 run_ring1a.py 的模板与渲染逻辑，保证探针所见即正式脚本所发。

使用示例：
  python scripts/probe_ring1a.py
"""

import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_ring1a import (
    CAT1A_TEMPLATE,
    MODEL_PROMPT_SUFFIX,
    STAGE1_TEMPLATE,
    STAGE2_TEMPLATE,
    build_sampling_params,
    candidate_mass,
    detect_chat_kwargs,
    get_candidate_ids,
    load_vllm_model,
    render_prompts,
)

# =============================================================================
# CONFIG —— 只改这里
# =============================================================================
MODEL_PATH = "/root/autodl-tmp/glm-4-9b-chat"   # 模型权重目录
MODE = "stage1"                          # stage1 | stage2 | cat
GPU_MEMORY_UTILIZATION = 0.85            # 显存占用比例

TEXT = "北京人计较？你笑死我了"            # 待探查文本

CONCEPT_NAME = "贬义评价词与骂人话"        # 判断项名（stage1/stage2 用）
CONCEPT_DEFINITION = (                    # 判断项定义（stage1/stage2 用）
    "文本出现针对人/群体的显性贬义判断词或辱骂（无耻/下流/恶心/讨厌/恨/看不惯/滚/死得好/"
    "连家人一起骂）。骂人话以谐音、拼音缩写、拆字或emoji变形出现同样算。"
    "注意：'不配/也配'类不算本项"
)
# =============================================================================


def build_filled_prompt():
    if MODE == "stage1":
        return STAGE1_TEMPLATE.format(text=TEXT, concept_name=CONCEPT_NAME,
                                      definition=CONCEPT_DEFINITION)
    if MODE == "stage2":
        return STAGE2_TEMPLATE.format(text=TEXT, concept_name=CONCEPT_NAME,
                                      definition=CONCEPT_DEFINITION)
    if MODE == "cat":
        return CAT1A_TEMPLATE.format(text=TEXT)
    raise ValueError(f"未知 MODE: {MODE}（应为 stage1/stage2/cat）")


def main():
    model_name = Path(MODEL_PATH).name
    filled = build_filled_prompt()

    print("=" * 70)
    print(f"【1】模板填槽后（发给 chat template 之前）  MODE={MODE}")
    print("=" * 70)
    print(filled)

    tokenizer, llm = load_vllm_model(Path(MODEL_PATH), GPU_MEMORY_UTILIZATION)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    final_prompt = render_prompts(tokenizer, [filled], chat_kwargs, model_name)[0]

    print()
    print("=" * 70)
    print("【2】LLM 实际收到的完整提示词（chat template + 后缀）")
    print("=" * 70)
    print(final_prompt)
    suffix = MODEL_PROMPT_SUFFIX.get(model_name)
    suffix_desc = repr(suffix) if suffix is not None else "未登记，走自动换行修正"
    print(f"[模型 {model_name} 后缀: {suffix_desc}]")

    # verbalizer 候选（按标签分组；同标签的多种表层变体合并求概率质量）
    if MODE == "stage2":
        cand_words, cand_labels = ["A", "B"], ["A", "B"]
    else:
        cand_words, cand_labels = ["是", "否"], ["是", "否"]
    cand_groups = [get_candidate_ids([w], tokenizer) for w in cand_words]
    all_ids = [tid for g in cand_groups for tid in g]
    print(f"[verbalizer 候选 token id: "
          f"{ {label: [tokenizer.decode([t]) for t in g] for label, g in zip(cand_labels, cand_groups)} }]")

    print()
    print("=" * 70)
    print("【3】LLM 首 token 输出")
    print("=" * 70)
    # 与 run_ring1a.py 一致的受限解码：候选集之外的 token 被屏蔽
    sp = build_sampling_params(all_ids)
    out = llm.generate([final_prompt], sp, use_tqdm=False)[0]

    first_tid = out.outputs[0].token_ids[0]
    print(f"生成 token: {tokenizer.decode([first_tid])!r} (id={first_tid})")

    lp = out.outputs[0].logprobs[0]
    top = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)[:10]
    print("首 token top-10 候选:")
    for tid, obj in top:
        print(f"  {tokenizer.decode([tid])!r:<12} id={tid:<8} p={math.exp(obj.logprob):.6f}")

    masses, norm = candidate_mass(lp, cand_groups)
    valid = sum(masses) > 0
    print("verbalizer 归一化概率:")
    for label, p, m, g in zip(cand_labels, norm, masses, cand_groups):
        hit = [t for t in g if t in lp]
        print(f"  P({label}) = {p:.6f}  原值={m:.6f}  进top-k={bool(hit)}")
    print(f"判定有效: {valid}")
    if not valid:
        print("判定: 无效（候选 token 均未进入 top-k，原值为 0，不能比较）")
    elif MODE == "stage2":
        print(f"判定: {'B' if masses[1] > masses[0] else 'A'}")
    else:
        print(f"判定: {'是' if masses[0] > masses[1] else '否'}")


if __name__ == "__main__":
    main()
