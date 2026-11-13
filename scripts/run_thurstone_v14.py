# -*- coding: utf-8 -*-
"""v14 主线：Thurstone 成对比较测量（GPU，GLM 单教师，标签零接触）。

理论：Thurstone 比较判断定律 Case V——p(A≻B) 的差分测量天然消除
绝对"是/否"判据的加性偏置（v10 消融实锤：锚定条款只移校准不改排序）。
FP 与 TP 同处表面毒性轴中带（fresh B2=0.7382），心理物理学上相邻刺激
的比较分辨力最高——比较模式直接瞄准该边界。

星型锚点设计（控成本：O(n·k) 而非 O(n²)）：
  锚点 = 24 个，从 train 确定性选出——参考分 = reverify 的 seed42 OOF
  （oof_s42b，仅作设计参考，不接触标签），双侧各 12：
    高侧锚点 = 参考分最高的 12 条（候选池限 char_len≤200）
    低侧锚点 = 参考分最低的 12 条（同上）
  每条文本（train 9600 + test 2411）与两侧各 4 锚点比较：
    高侧锚点索引 = (i % 12 + j) for j in 0..3；低侧同理
    ——轮转分配保证每锚点每侧恰 3200 条比较，BT 图连通且均衡
  位置平衡：md5(text_id|anchor_id) 奇偶决定文本放甲位或乙位
  探针（1 token，无掩码 logprobs）：p_text = 文本被判为"更强攻击"的概率
  （文本在乙位时 p_text = 1 − P(甲)，同质量归一）

成本：9600×8 + 2411×8 = 96,088 次 1-token 调用（约为 v6 全量的 6.6%）。
预登记门槛（CPU 端 analyze_thurstone_v14 执行，引用 reverify fresh 值）：
  G1' M4+BT特征联合 AUC ≥ 0.8929(fresh B1) + 0.010 = 0.9029
  G2' BT特征与现有 264 维 max|corr| < 0.7
产物：results/v14/thurstone/{pairs_flat_train.csv, pairs_flat_test.csv,
      meta.json}
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from run_items_v6 import detect_chat_kwargs, load_vllm_model, render_prompts

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


COMP_TEMPLATE = """你是语言学标注助手。比较两条言论的攻击性。只依据两条文本本身比较，不评价内容对错，不解释。

言论甲：{a}

言论乙：{b}

问句：哪条言论对现实中的某人或某群体构成更强的攻击、贬损或仇恨表达？

只回答：甲/乙。回答："""


def first_token_id(word, tokenizer):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise ValueError(f"标签词 {word!r} 编码为空")
    return ids[0]


def select_anchors(train_df, anchor_pred, n_side=12, max_len=200):
    """确定性锚点选择：参考分 = reverify 的 seed42 OOF（设计参考，非标签）。
    候选池限 char_len≤max_len；同分用 id 排序破平，全程可复现。"""
    ref = dict(zip(anchor_pred["id"], anchor_pred["oof_s42b"]))
    cand = train_df[["id", "content"]].copy()
    cand["ref"] = cand["id"].map(ref)
    assert cand["ref"].notna().all(), "anchor_predictions_train.csv 覆盖不全"
    cand["len"] = cand["content"].astype(str).str.len()
    cand = cand[cand["len"] <= max_len]
    cand = cand.sort_values(["ref", "id"], ascending=[False, True])
    ids = cand["id"].tolist()
    assert len(ids) >= 2 * n_side, f"候选池 {len(ids)} 不足以双侧各 {n_side} 锚点"
    return ids[:n_side], ids[-n_side:]


def main():
    ap = argparse.ArgumentParser(description="v14 Thurstone 成对比较测量")
    ap.add_argument("--model_path", required=True, help="教师模型本地权重目录")
    ap.add_argument("--model_name", default="glm-4-9b-chat")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--anchor_pred", default="results/v14/reverify/anchor_predictions_train.csv")
    ap.add_argument("--output_dir", default="results/v14/thurstone")
    ap.add_argument("--n_side", type=int, default=12, help="每侧锚点数")
    ap.add_argument("--k_per_side", type=int, default=4, help="每文本每侧比较数")
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    anchor_pred = pd.read_csv(args.anchor_pred)
    assert set(anchor_pred["id"]) >= set(tr["id"]), "reverify 产物 id 未覆盖 train"

    # ---- fail-fast ----
    assert len(tr) == 9600 and len(te) == 2411, (len(tr), len(te))
    assert {"id", "content", "toxic"} <= set(tr.columns)
    assert {"id", "oof_s42b"} <= set(anchor_pred.columns)
    assert args.k_per_side <= args.n_side

    # ---- 锚点选择（确定性）----
    high_ids, low_ids = select_anchors(tr, anchor_pred, args.n_side)
    anchor_set = set(high_ids) | set(low_ids)
    stamp(f"[锚点] 高侧 {len(high_ids)} / 低侧 {len(low_ids)} 已确定")

    text_of = dict(zip(tr["id"], tr["content"].astype(str)))
    # test 文本不参与锚点；anchor 文本从 train 取
    texts_by_id = text_of

    # ---- 配对构建 ----
    pair_rows = []
    for split, df in (("train", tr), ("test", te)):
        ids_list = df["id"].tolist()
        contents = df["content"].astype(str).tolist()
        pos = {t: c for t, c in zip(ids_list, contents)}
        for i, tid in enumerate(ids_list):
            for side, anchor_ids in (("high", high_ids), ("low", low_ids)):
                n_a = len(anchor_ids)
                for j in range(args.k_per_side):
                    aid = anchor_ids[(i % n_a + j) % n_a]
                    h = hashlib.md5(f"{tid}|{aid}".encode("utf-8")).hexdigest()
                    swapped = int(h, 16) % 2 == 1
                    pair_rows.append({"split": split, "text_id": tid,
                                      "anchor_id": aid, "side": side, "j": j,
                                      "swapped": swapped,
                                      "text": pos[tid], "anchor": texts_by_id[aid]})
    pairs = pd.DataFrame(pair_rows)
    stamp(f"[配对] {len(pairs)} 对（train {int((pairs['split']=='train').sum())} / "
          f"test {int((pairs['split']=='test').sum())}），位置平衡比 "
          f"{1 - pairs['swapped'].mean():.3f} 甲位")

    # ---- GPU 测量 ----
    tokenizer, llm = load_vllm_model(Path(args.model_path),
                                     args.gpu_memory_utilization)
    chat_kwargs = detect_chat_kwargs(tokenizer)
    cands = {w: first_token_id(w, tokenizer) for w in ("甲", "乙")}
    assert len(set(cands.values())) == len(cands), f"标签首 token 重叠: {cands}"

    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=50)

    prompts = []
    for _, r in pairs.iterrows():
        a, b = (r["text"], r["anchor"]) if not r["swapped"] else (r["anchor"], r["text"])
        prompts.append(COMP_TEMPLATE.format(a=a, b=b))
    rendered = render_prompts(tokenizer, prompts, chat_kwargs, args.model_name)

    p_text, m_jia_all, m_yi_all = [], [], []
    tok_id_all, tok_text_all = [], []
    swapped_arr = pairs["swapped"].values
    for i in tqdm(range(0, len(rendered), args.batch),
                  desc="Thurstone 成对比较", dynamic_ncols=True):
        outs = llm.generate(rendered[i:i + args.batch], sp)
        for o in outs:
            k = len(p_text)
            out0 = o.outputs[0]
            lp_list = out0.logprobs
            lp_dict = lp_list[0] if lp_list else {}
            m_jia = math.exp(lp_dict[cands["甲"]].logprob) if cands["甲"] in lp_dict else 0.0
            m_yi = math.exp(lp_dict[cands["乙"]].logprob) if cands["乙"] in lp_dict else 0.0
            tot = m_jia + m_yi
            # 实际首 token（核算甲/乙覆盖度：首token恰为甲/乙的比例）
            _tids = getattr(out0, "token_ids", None)
            tok_id_all.append(int(_tids[0]) if _tids else None)
            tok_text_all.append(out0.text)
            m_jia_all.append(m_jia)
            m_yi_all.append(m_yi)
            if tot > 1e-8:
                p_jia = m_jia / tot
                p_text.append(p_jia if not swapped_arr[k] else 1.0 - p_jia)
            else:
                p_text.append(float("nan"))
    assert len(p_text) == len(pairs)
    pairs["p_text"] = p_text
    pairs["m_jia"] = m_jia_all
    pairs["m_yi"] = m_yi_all
    pairs["tok_id"] = tok_id_all
    pairs["tok"] = tok_text_all
    n_first_match = int(sum(1 for t in tok_id_all
                            if t in (cands["甲"], cands["乙"])))
    stamp(f"[测量] 完成，p_text 无效率 = {pairs['p_text'].isna().mean():.4f}，"
          f"首token∈{{甲,乙}} = {n_first_match / len(pairs):.4f}")

    # ---- 导出 ----
    for split in ("train", "test"):
        sub = pairs[pairs["split"] == split].drop(columns=["text", "anchor"])
        sub.to_csv(out / f"pairs_flat_{split}.csv", index=False,
                   encoding="utf-8-sig")
    meta = {"model": args.model_name, "template": "v14_thurstone_v1",
            "n_pairs": int(len(pairs)),
            "n_side": args.n_side, "k_per_side": args.k_per_side,
            "label_token_ids": {w: int(t) for w, t in cands.items()},
            "first_token_in_labels_rate": n_first_match / len(pairs),
            "high_anchors": high_ids, "low_anchors": low_ids,
            "p_text_nan_rate": float(pairs["p_text"].isna().mean())}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    stamp(f"[完成] 产物 -> {out}/pairs_flat_train.csv / pairs_flat_test.csv / meta.json")


if __name__ == "__main__":
    main()
