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
    detect_chat_kwargs,
    extract_candidate_probs,
    get_first_token_ids,
    load_vllm_model,
    render_prompts,
)
from vllm import SamplingParams

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

    # verbalizer 候选
    if MODE == "stage2":
        cand_words, cand_labels = ["A", "B"], ["A", "B"]
    else:
        cand_words, cand_labels = ["是", "否"], ["是", "否"]
    cand_ids = get_first_token_ids(cand_words, tokenizer)
    print(f"[verbalizer 候选 token id: {dict(zip(cand_labels, cand_ids))}]")

    print()
    print("=" * 70)
    print("【3】LLM 首 token 输出")
    print("=" * 70)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    out = llm.generate([final_prompt], sp, use_tqdm=False)[0]

    first_tid = out.outputs[0].token_ids[0]
    print(f"生成 token: {tokenizer.decode([first_tid])!r} (id={first_tid})")

    lp = out.outputs[0].logprobs[0]
    top = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)[:10]
    print("首 token top-10 候选:")
    for tid, obj in top:
        print(f"  {tokenizer.decode([tid])!r:<12} id={tid:<8} p={math.exp(obj.logprob):.6f}")

    raw, norm = extract_candidate_probs(lp, cand_ids)
    print("verbalizer 归一化概率:")
    for label, p in zip(cand_labels, norm):
        print(f"  P({label}) = {p:.6f}")
    if MODE == "stage1" or MODE == "cat":
        print(f"判定: {'是' if raw[0] >= raw[1] else '否'}")
    else:
        print(f"判定: {'B' if raw[1] >= raw[0] else 'A'}")


if __name__ == "__main__":
    main()
