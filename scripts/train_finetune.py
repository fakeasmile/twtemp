"""纯微调基线：文本 → BERT/RoBERTa [CLS] → 稀疏线性分类器（单层 Linear + L1）。

架构：AutoModel 编码（BERT 与 RoBERTa 结构同源，--model_path 指定哪个都支持），
取 [CLS]（last_hidden_state 第 0 位）直接接单个 Linear(hidden,1)；
训练时对分类头权重加 L1 惩罚（--w_l1，0 关闭）产生稀疏——读出层权重可直接解释，
结束时报告非零权重数。

协议：官方 train/test；train 内部 10% 分层验证集按 val loss 早停；
test 仅在训练结束（载入最优轮权重）后评估一次。
输出目录强制带 _seed{N} 后缀，多种子互不覆盖。

环境：本地运行（BERT 类模型）。离线硬约束：--model_path 必须是已存在的本地
模型目录（否则报错退出），TRANSFORMERS_OFFLINE=1 + local_files_only=True，
任何情况不远程下载。

用法：
  python scripts/train_finetune.py --model_path d:/models/chinese-roberta-wwm-ext --data_dir data/raw/TOXICN --output_dir results/finetune/toxicn_roberta
"""

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def ensure_safetensors(model_path):
    """transformers 新版拒绝在 torch<2.6 下加载 .bin 权重（CVE-2025-32434）。

    本地模型目录若只有 pytorch_model.bin，一次性转换为 model.safetensors
    （原文件保留）。
    """
    p = Path(model_path)
    if not p.is_dir():
        return
    if (p / "model.safetensors").exists() or not (p / "pytorch_model.bin").exists():
        return
    import torch
    from safetensors.torch import save_file
    print(f"[convert] {p / 'pytorch_model.bin'} → model.safetensors（一次性，原文件保留）")
    sd = torch.load(p / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # clone 消除共享张量（embedding tie），save_file 不接受共享存储
    save_file({k: v.clone().contiguous() for k, v in sd.items()}, str(p / "model.safetensors"))


def load_split(data_dir, split):
    p = Path(data_dir) / f"{split}.json"
    if not p.exists():
        raise SystemExit(f"数据文件不存在: {p}")
    items = json.load(open(p, encoding="utf-8"))
    texts = [x["content"] for x in items]
    toxic = np.asarray([int(x["toxic"]) for x in items], dtype=np.int64)
    expr = np.asarray([-1 if x.get("expression") is None else int(x["expression"])
                       for x in items], dtype=np.int64)
    return texts, toxic, expr


def main():
    ap = argparse.ArgumentParser(description="BERT/RoBERTa CLS → 稀疏线性分类器微调")
    ap.add_argument("--model_path", required=True,
                    help="本地模型目录（如 d:/models/chinese-roberta-wwm-ext，BERT/RoBERTa 均支持）")
    ap.add_argument("--data_dir", default="data/raw/TOXICN",
                    help="含 train.json/test.json 的数据目录")
    ap.add_argument("--output_dir", default="results/finetune/toxicn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--w_l1", type=float, default=1e-4,
                    help="分类头权重 L1 惩罚系数（0=关闭）")
    ap.add_argument("--early_stop", type=int, default=2, help="val loss 连续不降轮数")
    ap.add_argument("--dump_pred", default=None, help="预测 CSV 落盘目录（默认随 output_dir）")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.is_dir():
        raise SystemExit(f"--model_path 必须是已存在的本地模型目录: {model_dir}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedShuffleSplit
    from tqdm import tqdm

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 种子目录隔离：无条件追加 _seed{N}，多种子互不覆盖
    out_dir = Path(str(args.output_dir).rstrip("/\\") + f"_seed{args.seed}")
    dump_dir = (Path(str(args.dump_pred).rstrip("/\\") + f"_seed{args.seed}")
                if args.dump_pred else out_dir)
    print(f"设备 {device}  模型 {model_dir}  seed={args.seed}")
    print(f"输出目录 {out_dir}")

    texts_tr, ytr, expr_tr = load_split(args.data_dir, "train")
    texts_te, yte, expr_te = load_split(args.data_dir, "test")
    print(f"train {len(texts_tr)}（toxic=1 {int(ytr.sum())}）  test {len(texts_te)}"
          f"（toxic=1 {int(yte.sum())}）")

    ensure_safetensors(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    class ClsLinear(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone, local_files_only=True)
            self.head = nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, enc):
            cls = self.encoder(**enc).last_hidden_state[:, 0]
            return self.head(cls).squeeze(-1)

    model = ClsLinear(model_dir).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.1f}M  分类头 Linear({model.encoder.config.hidden_size},1)"
          f"  L1={args.w_l1}")

    # 内部 10% 分层验证集（早停/选轮），test 不参与
    tr_i, va_i = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=args.seed)
                      .split(np.zeros(len(ytr)), ytr))

    def collate(texts, ys, idx):
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt")
        return enc, torch.tensor(np.asarray(ys)[idx])

    dl_tr = DataLoader(range(len(tr_i)), batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda idx: collate(texts_tr, ytr, list(np.asarray(tr_i)[idx])),
                       num_workers=0, drop_last=True)
    dl_va = DataLoader(range(len(va_i)), batch_size=args.batch_size * 2, shuffle=False,
                       collate_fn=lambda idx: collate(texts_tr, ytr, list(np.asarray(va_i)[idx])))

    bce = nn.functional.binary_cross_entropy_with_logits
    no_decay = ("bias", "LayerNorm.weight")
    opt = torch.optim.AdamW(
        [{"params": [p for n_, p in model.named_parameters() if not any(nd in n_ for nd in no_decay)],
          "weight_decay": 0.01},
         {"params": [p for n_, p in model.named_parameters() if any(nd in n_ for nd in no_decay)],
          "weight_decay": 0.0}],
        lr=args.lr)
    steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val, best_state, patience, history = float("inf"), None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        run, probs_tr, ys_tr = 0.0, [], []
        pbar = tqdm(dl_tr, desc=f"    epoch {ep}/{args.epochs}", leave=False, dynamic_ncols=True)
        for enc, y in pbar:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(enc)
                loss = bce(logits.float(), y.float())
                if args.w_l1 > 0:
                    loss = loss + args.w_l1 * model.head.weight.abs().sum()
            probs_tr.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
            ys_tr.append(y.detach().cpu().numpy())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            p_f1 = f1_score(np.concatenate(ys_tr), np.concatenate(probs_tr) >= 0.5)
            pbar.set_postfix(loss=f"{loss.item():.3f}", F1=f"{p_f1:.4f}")

        # 验证（早停依据 = val loss）
        model.eval()
        vl, n_b, probs_va, ys_va = 0.0, 0, [], []
        with torch.no_grad():
            for enc, y in tqdm(dl_va, desc="    val", leave=False, dynamic_ncols=True):
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(enc)
                    vl += bce(logits.float(), y.to(device).float()).item() * len(y)
                    n_b += len(y)
                probs_va.append(torch.sigmoid(logits.float()).cpu().numpy())
                ys_va.append(y.numpy())
        vl /= max(n_b, 1)
        vf1 = f1_score(np.concatenate(ys_va), np.concatenate(probs_va) >= 0.5)
        print(f"  epoch {ep}/{args.epochs}  train loss {run / len(dl_tr):.4f}  train F1 {p_f1:.4f}"
              f"  val loss {vl:.4f}  val F1@0.5 {vf1:.4f}")
        history.append((ep, run / len(dl_tr), p_f1, vl, vf1))
        if vl < best_val - 1e-4:
            best_val, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"  早停于 epoch {ep}（最优 val loss {best_val:.4f}）")
                break
    if best_state:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1"])
        for h in history:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in h])

    # ---------- test 评估（仅此一次） ----------
    dl_te = DataLoader(range(len(texts_te)), batch_size=args.batch_size * 2, shuffle=False,
                       collate_fn=lambda idx: collate(texts_te, yte, list(idx)))
    model.eval()
    probs_te = []
    with torch.no_grad():
        for enc, _ in tqdm(dl_te, desc="    test", leave=False, dynamic_ncols=True):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                probs_te.append(torch.sigmoid(model(enc).float()).cpu().numpy())
    proba = np.concatenate(probs_te)

    def metric_block(name, p, y):
        """单组指标：Acc / Macro F1 / Weighted F1 / 分类别 P R F1 / AUC。"""
        pred = (p >= 0.5).astype(int)
        pr, rc, f1c, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
        return [f"[{name}]  n={len(y)}  阈值 0.50",
                f"  Acc          {accuracy_score(y, pred):.4f}",
                f"  Macro F1     {f1_score(y, pred, average='macro'):.4f}",
                f"  Weighted F1  {f1_score(y, pred, average='weighted'):.4f}",
                f"  有毒(=1)     P {pr[0]:.4f}  R {rc[0]:.4f}  F1 {f1c[0]:.4f}",
                f"  无毒(=0)     P {pr[1]:.4f}  R {rc[1]:.4f}  F1 {f1c[1]:.4f}",
                f"  AUC          {roc_auc_score(y, p):.4f}"]

    w_head = model.head.weight.detach().cpu().numpy()
    nz = int((np.abs(w_head[0]) > 1e-6).sum())
    lines = ["=" * 60,
             "train_finetune 纯微调基线（CLS → 稀疏线性）",
             f"模型: {model_dir}   seed: {args.seed}",
             f"test: n={len(yte)}  toxic=1: {int(yte.sum())}",
             "=" * 60, "",
             *metric_block("全部", proba, yte), ""]
    lines.append(f"分类头非零权重: {nz}/{w_head.shape[1]}（L1={args.w_l1}，|w|>1e-6 计非零）")
    report = "\n".join(lines)

    print("\n===== test 评估（@0.5，仅此一次）=====")
    print(report)
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"指标已落盘 {out_dir / 'metrics.txt'}")

    if not args.no_save:
        torch.save({"state_dict": model.state_dict(), "model_path": str(model_dir),
                    "args": vars(args)}, out_dir / "best.pt")
        print(f"checkpoint 已保存 {out_dir / 'best.pt'}")

    dump_dir.mkdir(parents=True, exist_ok=True)
    with open(dump_dir / "finetune_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "expression", "toxic", "proba", "content"])
        for i in range(len(texts_te)):
            w.writerow([i, int(expr_te[i]), int(yte[i]), f"{proba[i]:.4f}",
                        texts_te[i][:50]])
    print(f"预测已落盘 {dump_dir / 'finetune_predictions.csv'}")


if __name__ == "__main__":
    main()
