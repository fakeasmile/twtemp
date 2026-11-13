# -*- coding: utf-8 -*-
"""v15b Stage1：答案就绪位置全层隐状态抽取（GPU，用户执行）。

【文献依据（设计照此，不自创）】顶刊/顶会用 LLM 内部表示做判别的标准做法：
  - Azaria & Mitchell 2023 (EMNLP Findings) "The Internal State of an LLM
    Knows When It's Lying"：问答脚手架 "Question: ... Answer:"，取回答
    token 之前的最后位置隐状态 + 监督线性探针
  - Burns et al. 2023 (ICLR) CCS "Discovering Latent Knowledge"：问答
    脚手架 + 末位置状态；其正/负答案对比差值用于无监督场景——本实验
    脚手架对所有文本恒定，监督探针下脚手架分量被截距吸收，无需差值
  - Zou et al. 2023 (NeurIPS best paper) RepE：逐层读出，中层常优于末层
  共同经验：①在"答案出口前"的位置取状态（模型处于决策就绪态）；
  ②逐层扫描而非只用末层。

【v15 的教训】v15 三臂取的是裸文本/普通 chat 包裹的末位置——模型未处于
"准备回答毒性问题"的状态，与口头探针矩阵的测量位置不同构，比较被混杂。
v15b 修复：chat 模板 user = "{text}\n\n这条评论是否有毒性？"，
add_generation_prompt=True，末 token 位置 = 口头探针读首 token logits 的
同一位置 → "同位置向量 vs 标量"的干净对照。

【做法】每文本 1 次前向，捕获该位置全部层的隐状态（槽位约定：
S0 = embedding 输出，S1..S40 = 各 transformer 层输出，末槽 =
last_hidden_state（过最终 norm，作为锚定槽位固定追加）），float16 导出。
train/test 均抽取；test 不参与任何门槛（零参与协议）。
产物：results/v15/scaffold/{scaffold_train.npz, scaffold_test.npz,
extract_meta.json}，npz 键 = ids + layers (n, L_total, H)。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from extract_hidden_v15 import patch_tied_weights_attr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def scaffold_wrap(tok, text):
    """答案就绪脚手架：user = 文本 + 毒性问句，assistant 起始处截停。
    模板不可用时回退 GLM-4 手拼（与 extract_hidden_v15 同风格）。"""
    user = f"{text}\n\n这条评论是否有毒性？"
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[gMASK]<sop><|user|>\n{user}<|assistant|>\n"


@torch.no_grad()
def extract_all_layers(model, tok, texts, hidden_size, batch_size,
                       max_length, desc):
    """返回 (n, L_total, H) float16。槽位 = 各层在"末 token 位置"的隐状态，
    末槽固定为 last_hidden_state（过最终 norm）。left padding 保证
    末位置 = 真实最后 token（即口头探针的首 token 输出位置）。"""
    n = len(texts)
    order = np.argsort([-len(t) for t in texts])  # 长文本在前
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sorted_texts = [texts[i] for i in order]
    tok.padding_side = "left"
    layers_arr = None  # (n, L_total, H)，首批后按 L_total 分配
    for s in tqdm(range(0, n, batch_size), desc=desc, dynamic_ncols=True):
        chunk = sorted_texts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = list(out.hidden_states)                       # L 个 (B, seq, H)
        hs.append(out.last_hidden_state)                   # 末槽锚定
        # 各层末位置堆叠 → (B, L_total, H)
        stack = torch.stack([h[:, -1, :] for h in hs], dim=1)
        if layers_arr is None:
            layers_arr = np.zeros((n, stack.shape[1], hidden_size),
                                  dtype=np.float16)
        layers_arr[s:s + len(chunk)] = stack.float().cpu().numpy() \
            .astype(np.float16)
    assert layers_arr is not None and np.isfinite(layers_arr).all()
    return layers_arr[inv]


def main():
    ap = argparse.ArgumentParser(description="v15b 答案就绪位置全层抽取")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--output_dir", default="results/v15/scaffold")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    stamp(f"[加载] train n={len(tr)} / test n={len(te)}")

    stamp("[加载] 模型与 tokenizer...")
    patch_tied_weights_attr()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=dt, trust_remote_code=True).cuda().eval()
    hidden_size = model.config.hidden_size
    stamp(f"[加载] {type(model).__name__} hidden_size={hidden_size} "
          f"dtype={args.dtype}")

    sample = scaffold_wrap(tok, tr["content"].astype(str).iloc[0])
    stamp(f"[脚手架示例] ...{sample[-80:]}")

    for split, ids, texts in (("train", tr["id"].values,
                               tr["content"].astype(str).tolist()),
                              ("test", te["id"].values,
                               te["content"].astype(str).tolist())):
        stamp(f"[抽取] {split}（每文本 1 次前向，全层末位置）...")
        layers = extract_all_layers(model, tok, texts, hidden_size,
                                    args.batch_size, args.max_length,
                                    f"scaffold[{split}]")
        np.savez(out / f"scaffold_{split}.npz",
                 ids=np.array([str(i) for i in ids]), layers=layers)
        stamp(f"[导出] scaffold_{split}.npz 完成 shape={layers.shape}")

    import transformers as _tf
    meta = {"model_name": args.model_name, "model_path": args.model_path,
            "hidden_size": hidden_size, "dtype": args.dtype,
            "transformers_version": _tf.__version__,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "scaffold_user": "{text}\n\n这条评论是否有毒性？",
            "position": "chat 模板 add_generation_prompt=True 后的末 token"
                        "（=口头探针首 token logits 的输出位置）",
            "slots": "S0=embedding 输出；S1..S{L-2}=各 transformer 层输出；"
                     "末槽=last_hidden_state（过最终 norm）",
            "n_train": len(tr), "n_test": len(te)}
    (out / "extract_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp("[完成] 全部产物写出")


if __name__ == "__main__":
    main()
