# -*- coding: utf-8 -*-
"""v15 Stage1：GLM 末层隐状态抽取（GPU，用户执行）。

范式转移（v15）：测量接口从"口头 1-token 探针"换成"内部表示"。
文本 → GLM 一次前向 → 最后一层残差流 hidden_states[-1] →
  - bare_last : 裸文本、最后 token 位置向量（≈RoBERTa CLS 对应物）
  - bare_mean : 裸文本、非 pad token mean-pool
  - chat_last : chat 模板包裹（无问题无概念）、最后 token 位置向量

零新概念、零提示词工程——首实验只测这个接口的信息上限。
产物：results/v15/hidden/{hidden_train.npz, hidden_test.npz, extract_meta.json}
（float16，id 顺序与输入矩阵一致）
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def chat_wrap(tok, text):
    """chat 模板包裹（无问题无概念）。模板不可用时回退 GLM-4 手拼。"""
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[gMASK]<sop><|user|>\n{text}<|assistant|>\n"


@torch.no_grad()
def extract(model, tok, texts, hidden_size, batch_size, max_length, desc):
    """返回 (n, hidden_size) float16，顺序与 texts 一致。
    left padding → last token = h[:, -1]；mean-pool 用 attention mask。"""
    n = len(texts)
    last = np.zeros((n, hidden_size), dtype=np.float16)
    mean = np.zeros((n, hidden_size), dtype=np.float16)
    order = np.argsort([-len(t) for t in texts])  # 长文本在前
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sorted_texts = [texts[i] for i in order]
    tok.padding_side = "left"
    for s in tqdm(range(0, n, batch_size), desc=desc, dynamic_ncols=True):
        chunk = sorted_texts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # last_hidden_state = 过最终 layer norm 的末层输出（RoBERTa CLS 对应物）；
        # 老 remote-code 返回 dict/对象均兼容
        h = (out.last_hidden_state if hasattr(out, "last_hidden_state")
             and out.last_hidden_state is not None
             else out.hidden_states[-1]).float().cpu().numpy()   # (B, L, H)
        m = enc["attention_mask"].float().cpu().numpy()[..., None]
        last[s:s + len(chunk)] = h[:, -1, :].astype(np.float16)
        mean[s:s + len(chunk)] = ((h * m).sum(1) / m.sum(1)).astype(np.float16)
    return last[inv], mean[inv]


def patch_tied_weights_attr() -> None:
    """远端新 transformers + ChatGLM remote-code 的已知不兼容：
    _move_missing_keys_from_meta_to_device 读取 self.all_tied_weights_keys，
    而 remote-code 动态类未设置该属性。此处给 PreTrainedModel 打方法补丁，
    属性缺失时注入空 dict 后调用原方法。版本容忍：本地旧版无该方法则
    零操作；已打过补丁则幂等返回。对 v15 无副作用：只消费 transformer
    主体隐状态；若 lm_head 因绑定键处理变化而未被正确加载，logits 会
    错，但隐状态不受任何影响。"""
    try:
        from transformers import modeling_utils as _mu
        _orig = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
    except AttributeError:
        return  # 该 transformers 版本无此方法，无需补丁
    if getattr(_orig, "_v15_patched", False):
        return

    def _patched(self, *a, **k):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, *a, **k)

    _patched._v15_patched = True
    _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched


def main():
    ap = argparse.ArgumentParser(description="v15 GLM 末层隐状态抽取")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--output_dir", default="results/v15/hidden")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    texts_tr = tr["content"].astype(str).tolist()
    texts_te = te["content"].astype(str).tolist()
    stamp(f"[加载] train n={len(texts_tr)} / test n={len(texts_te)}")

    stamp("[加载] 模型与 tokenizer...")
    patch_tied_weights_attr()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # AutoModel = ChatGLMModel 主体（无 lm_head）：v15 只需末层隐状态，
    # 且绕开新 transformers 对 CausalLM remote-code 绑定权重加载的不兼容。
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=dt, trust_remote_code=True).cuda().eval()
    hidden_size = model.config.hidden_size
    stamp(f"[加载] {type(model).__name__} hidden_size={hidden_size} dtype={args.dtype}")

    npz_payload = {}
    for split, ids, texts in (("train", tr["id"].values, texts_tr),
                              ("test", te["id"].values, texts_te)):
        stamp(f"[抽取] {split} 裸文本...")
        bl, bm = extract(model, tok, texts, hidden_size,
                         args.batch_size, args.max_length, f"bare[{split}]")
        chat_texts = [chat_wrap(tok, t) for t in texts]
        stamp(f"[抽取] {split} chat 模板...")
        cl, _ = extract(model, tok, chat_texts, hidden_size,
                        args.batch_size, args.max_length, f"chat[{split}]")
        for arr in (bl, bm, cl):
            assert np.isfinite(arr.astype(np.float32)).all(), f"{split} 出现非有限值"
        npz_payload[f"ids_{split}"] = np.array([str(i) for i in ids])
        npz_payload[f"bare_last_{split}"] = bl
        npz_payload[f"bare_mean_{split}"] = bm
        npz_payload[f"chat_last_{split}"] = cl
        np.savez(out / f"hidden_{split}.npz",
                 ids=npz_payload[f"ids_{split}"], bare_last=bl,
                 bare_mean=bm, chat_last=cl)
        stamp(f"[导出] hidden_{split}.npz 完成 ({bl.shape})")

    import transformers as _tf
    meta = {"model_name": args.model_name, "model_path": args.model_path,
            "hidden_size": hidden_size, "dtype": args.dtype,
            "transformers_version": _tf.__version__,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "layer": -1, "pooling": ["bare_last", "bare_mean", "chat_last"],
            "n_train": len(texts_tr), "n_test": len(texts_te)}
    (out / "extract_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp("[完成] 全部产物写出")


if __name__ == "__main__":
    main()
