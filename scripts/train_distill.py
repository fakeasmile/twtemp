"""概念蒸馏：LLM 概念教师 → BERT/RoBERTa 学生 [CLS] 拟合概念通道。

架构：学生 AutoModel（--model_path，BERT 与 RoBERTa 均支持）取 [CLS]，接概念族
线性头：全部概念 p_yes（47）+ 0-2 概念 P(直接) + p_A + S1A/S1B 六档 + CAT 六通道
+ toxic 头；soft 目标 = 教师值（run_rings 输出），toxic 头以 --w_toxic 联合监督。
推理不再调用 LLM。

冻结读出路径（瓶颈）：学生预测的概念通道用 encode_features（train_concept_lr 的
唯一实现，与教师特征逐位同构）编码 → 教师 train 特征上拟合的 L1 LR
（alpha 5 折 CV + OOF 阈值，fit_sparse_lr）→ test 预测。
概念瓶颈保留：任何预测都能沿"学生概念值 → 读出权重"解释。

协议：官方 train/test；train 内部 10% 分层验证集按 val loss 早停；
test 仅在训练结束（载入最优轮权重）后评估一次，含概念保真度。
输出目录强制带 _seed{N} 后缀。

环境：本地运行（BERT 类模型）。离线硬约束同 train_finetune：
--model_path 必须是已存在的本地模型目录，不远程下载。

用法：
  python scripts/train_distill.py --model_path d:/models/chinese-roberta-wwm-ext --rings_dir data/processed/TOXICN/glm-4-9b-chat_v3
（rings_dir 下自动发现 rings_{模型名}_train/test.json；全部产物落 results/distill/...）
"""

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_concept_lr import (ConceptArrays, encode_features, fit_sparse_lr,  # noqa: E402
                              metric_block)
from train_concept_lr import discover_rings, load_rings  # noqa: E402
from train_finetune import ensure_safetensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="概念教师 → BERT/RoBERTa [CLS] 蒸馏")
    ap.add_argument("--model_path", required=True,
                    help="本地模型目录（如 d:/models/chinese-roberta-wwm-ext，BERT/RoBERTa 均支持）")
    ap.add_argument("--rings_dir", required=True,
                    help="含 rings_{模型名}_train.json / rings_{模型名}_test.json 的目录")
    ap.add_argument("--output_dir", default="results/distill/toxicn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--w_toxic", type=float, default=1.0)
    ap.add_argument("--w_py", type=float, default=1.0, help="概念 p_yes 头")
    ap.add_argument("--w_stage2", type=float, default=0.5, help="P(直接) 与 p_A 头")
    ap.add_argument("--w_s1", type=float, default=0.5, help="S1A/S1B 六档分布头")
    ap.add_argument("--w_cat", type=float, default=0.5, help="CAT 六通道头")
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
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from tqdm import tqdm

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(str(args.output_dir).rstrip("/\\") + f"_seed{args.seed}")
    dump_dir = (Path(str(args.dump_pred).rstrip("/\\") + f"_seed{args.seed}")
                if args.dump_pred else out_dir)
    print(f"设备 {device}  学生 {model_dir}  seed={args.seed}")
    print(f"输出目录 {out_dir}")

    tr_path, te_path = discover_rings(args.rings_dir)
    tr_results, meta = load_rings(tr_path)
    te_results, meta_te = load_rings(te_path)
    assert [c["id"] for c in meta["concepts"]] == [c["id"] for c in meta_te["concepts"]], \
        "train/test 概念集 id 不一致（概念顺序必须一致）"
    assert meta["model"] == meta_te["model"], "违反严格单模型原则：train/test 来自不同教师"

    tea_tr, tea_te = ConceptArrays(tr_results, meta), ConceptArrays(te_results, meta)
    assert tea_tr.s2_cids == tea_te.s2_cids, "train/test 0-2 概念集合不一致"
    texts = tea_tr.texts
    n_s2 = len(tea_tr.s2_cids)
    print(f"教师 {meta['model']}  通道：p_yes {len(tea_tr.cids)} + P(直接) {n_s2}"
          f" + p_A {n_s2} + S1A 6 + S1B 6 + CAT {len(tea_tr.cat_names)} + toxic")
    print(f"train {len(texts)}（toxic=1 {int(tea_tr.toxic.sum())}）  "
          f"test {len(tea_te.texts)}（toxic=1 {int(tea_te.toxic.sum())}）")

    ensure_safetensors(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    class Student(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone, local_files_only=True)
            h = self.encoder.config.hidden_size
            self.head_toxic = nn.Linear(h, 1)
            self.head_py = nn.Linear(h, len(tea_tr.cids))
            self.head_pd = nn.Linear(h, n_s2)
            self.head_pa = nn.Linear(h, n_s2)
            self.head_s1a = nn.Linear(h, 6)
            self.head_s1b = nn.Linear(h, 6)
            self.head_cat = nn.Linear(h, len(tea_tr.cat_names))

        def forward(self, enc):
            cls = self.encoder(**enc).last_hidden_state[:, 0]
            return {"toxic": self.head_toxic(cls).squeeze(-1),
                    "py": self.head_py(cls), "pd": self.head_pd(cls),
                    "pa": self.head_pa(cls), "s1a": self.head_s1a(cls),
                    "s1b": self.head_s1b(cls), "cat": self.head_cat(cls)}

    model = Student(model_dir).to(device)

    def soft_ce(logits, target_probs):
        logp = torch.log_softmax(logits, dim=-1)
        return -(target_probs * logp).sum(-1).mean()

    def loss_fn(out, t):
        bce = nn.functional.binary_cross_entropy_with_logits
        loss = args.w_toxic * bce(out["toxic"], t[0].float())
        loss = loss + args.w_py * bce(out["py"], t[1])
        loss = loss + args.w_stage2 * bce(out["pd"], t[2])
        loss = loss + args.w_stage2 * bce(out["pa"], t[3])
        loss = loss + args.w_s1 * soft_ce(out["s1a"], t[4])
        loss = loss + args.w_s1 * soft_ce(out["s1b"], t[5])
        loss = loss + args.w_cat * bce(out["cat"], t[6])
        return loss

    # 内部 10% 分层验证集（早停/选轮），test 不参与
    tr_i, va_i = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=args.seed)
                      .split(np.zeros(len(texts)), tea_tr.toxic))

    class TDS:
        """文本 + 教师通道切片的数据集视图。"""

        def __init__(self, idx):
            self.idx = idx
            self.texts = [texts[i] for i in idx]
            self.toxic = tea_tr.toxic[idx]
            self.py = tea_tr.p_yes[idx]
            self.pd = tea_tr.p_direct[idx]
            self.pa = tea_tr.p_a[idx]
            self.s1a = tea_tr.s1a[idx]
            self.s1b = tea_tr.s1b[idx]
            self.cat = tea_tr.cat[idx]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            t = self.texts[i]
            cols = (self.toxic[i], self.py[i], self.pd[i], self.pa[i],
                    self.s1a[i], self.s1b[i], self.cat[i])
            return t, cols

    def collate(batch):
        texts_b = [b[0] for b in batch]
        enc = tokenizer(texts_b, padding=True, truncation=True, max_length=args.max_len,
                        return_tensors="pt")
        cols = list(zip(*[b[1] for b in batch]))
        return enc, [torch.tensor(np.stack(c)) for c in cols]

    dl_tr = DataLoader(TDS(tr_i), batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=0, drop_last=True)
    dl_va = DataLoader(TDS(va_i), batch_size=args.batch_size * 2, shuffle=False,
                       collate_fn=collate)

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
        for enc, t in pbar:
            enc = {k: v.to(device) for k, v in enc.items()}
            t = [x.to(device) for x in t]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(enc)
                loss = loss_fn(out, t)
            probs_tr.append(torch.sigmoid(out["toxic"].float()).detach().cpu().numpy())
            ys_tr.append(t[0].detach().cpu().numpy())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            p_f1 = f1_score(np.concatenate(ys_tr).astype(int),
                            np.concatenate(probs_tr) >= 0.5)
            pbar.set_postfix(loss=f"{loss.item():.3f}", F1=f"{p_f1:.4f}")

        model.eval()
        vl, n_b, probs_va, ys_va = 0.0, 0, [], []
        with torch.no_grad():
            for enc, t in tqdm(dl_va, desc="    val", leave=False, dynamic_ncols=True):
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(enc)
                    vl += loss_fn(out, [x.to(device) for x in t]).item() * len(t[0])
                    n_b += len(t[0])
                probs_va.append(torch.sigmoid(out["toxic"].float()).cpu().numpy())
                ys_va.append(t[0].numpy())
        vl /= max(n_b, 1)
        vf1 = f1_score(np.concatenate(ys_va).astype(int),
                       np.concatenate(probs_va) >= 0.5)
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

    # ---------- test 通道推理（单次前向） ----------
    model.eval()
    outs = {k: [] for k in ("toxic", "py", "pd", "pa", "s1a", "s1b", "cat")}
    with torch.no_grad():
        for i0 in tqdm(range(0, len(tea_te.texts), args.batch_size * 2),
                       desc="    test 推理", leave=False, dynamic_ncols=True):
            chunk = tea_te.texts[i0:i0 + args.batch_size * 2]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_len,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc)
            for k in outs:
                outs[k].append(out[k].float().cpu().numpy())
    pred = {k: np.concatenate(v, 0) for k, v in outs.items()}
    pred["s1a"] = torch.softmax(torch.tensor(pred["s1a"]), -1).numpy()
    pred["s1b"] = torch.softmax(torch.tensor(pred["s1b"]), -1).numpy()
    pred["toxic"] = 1 / (1 + np.exp(-pred["toxic"]))
    pred["py"] = np.clip(1 / (1 + np.exp(-pred["py"])), 0, 1)
    pred["pd"] = np.clip(1 / (1 + np.exp(-pred["pd"])), 0, 1)
    pred["pa"] = np.clip(1 / (1 + np.exp(-pred["pa"])), 0, 1)
    pred["cat"] = np.clip(1 / (1 + np.exp(-pred["cat"])), 0, 1)
    yte = tea_te.toxic
    proba_direct = pred["toxic"]

    # ---------- 冻结读出（瓶颈路径）：教师特征拟合 L1 LR，学生通道编码进读出 ----------
    Xtr_tea = encode_features(tea_tr.p_yes, tea_tr.p_direct, tea_tr.p_a,
                              tea_tr.s1a, tea_tr.s1b, tea_tr.cat)
    Xte_stu = encode_features(pred["py"], pred["pd"], pred["pa"],
                              pred["s1a"], pred["s1b"], pred["cat"])
    Xte_tea = encode_features(tea_te.p_yes, tea_te.p_direct, tea_te.p_a,
                              tea_te.s1a, tea_te.s1b, tea_te.cat)
    readout, tau = fit_sparse_lr(Xtr_tea, tea_tr.toxic, random_state=0)[:2]
    proba_bn = readout.predict_proba(Xte_stu)[:, 1]

    print("\n===== test 评估（@0.5，仅此一次）=====")
    proba_teacher = readout.predict_proba(Xte_tea)[:, 1]
    lines = ["=" * 60,
             "train_distill 概念蒸馏（CLS 拟合教师概念通道）",
             f"学生: {model_dir}   seed: {args.seed}   教师: {meta['model']}",
             f"test: n={len(yte)}  toxic=1: {int(yte.sum())}",
             "=" * 60, "",
             *metric_block("直接路径（学生 toxic 头）", yte, proba_direct), "",
             *metric_block("瓶颈路径（学生概念 → 冻结 LR）", yte, proba_bn), "",
             f"瓶颈路径 @τ*={tau:.2f}（train OOF 选）  "
             f"F1 {f1_score(yte, (proba_bn >= tau).astype(int)):.4f}", "",
             *metric_block("教师参考（教师特征 → 同一 LR）", yte, proba_teacher), ""]

    mae_py = float(np.abs(pred["py"] - tea_te.p_yes).mean())
    mae_pd = float(np.abs(pred["pd"] - tea_te.p_direct).mean())
    mae_cat = float(np.abs(pred["cat"] - tea_te.cat).mean())
    lines += ["概念保真度 MAE（学生 vs 教师）:",
              f"  p_yes {mae_py:.4f}   P(直接) {mae_pd:.4f}   CAT {mae_cat:.4f}"]
    report = "\n".join(lines)
    print(report)

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"指标已落盘 {out_dir / 'metrics.txt'}")

    if not args.no_save:
        torch.save({"state_dict": model.state_dict(), "model_path": str(model_dir),
                    "cids": tea_tr.cids, "s2_cids": tea_tr.s2_cids,
                    "cat_names": tea_tr.cat_names, "args": vars(args)},
                   out_dir / "best.pt")
        print(f"\ncheckpoint 已保存 {out_dir / 'best.pt'}")

    dump_dir.mkdir(parents=True, exist_ok=True)
    with open(dump_dir / "distill_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "expression", "toxic", "proba_direct", "proba_bottleneck",
                    "content"])
        for i, r in enumerate(te_results):
            w.writerow([r.get("id", i), int(tea_te.expr[i]), int(yte[i]),
                        f"{proba_direct[i]:.4f}", f"{proba_bn[i]:.4f}", r["content"][:50]])
    print(f"预测已落盘 {dump_dir / 'distill_predictions.csv'}")


if __name__ == "__main__":
    main()
