# -*- coding: utf-8 -*-
"""v14 学生 CBM：RoBERTa 学 v7 教师概念向量 → k 维预测 → 下游读出（标准 CBM 范式）。

范式（Koh et al. 2020 标准 CBM）：
  教师测量 = v7 概念向量（GLM-4-9B 对 180 条目问句的 p_yes，data/processed/TOXICN/
             v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_{train,test}_matrix.csv，
             自带 id/toxic/content 列）
  学生     = RoBERTa 微调，输入 content，经 [CLS](768)→概念头输出 k=180 维概念
             预测（默认非线性头 CLS→256→ReLU→Dropout(0.2)→180，--linear_head 可
             退化为线性 CLS→180；BCEWithLogits，软目标=教师 p_yes，非 0/1 硬标签）
             。direct 臂概念头同构（→256→ReLU→Dropout(0.2)→1），保持两臂头结构一致
  下游     = 学生预测的 k 维概念向量 → {FC Linear(d→1)+L1, 两层 MLP(d→64→1)}
臂：
  direct_bert       RoBERTa 直接判毒（num_labels=1），τ 在验证子集筛选
  student_flat_fc   学生 180 维概念 → FC
  student_flat_mlp  学生 180 维概念 → MLP

控制变量（direct 与 student 严格同条件）：
  - 同 backbone（--model_name，默认 hfl/chinese-roberta-wwm-ext，可传本地路径）
  - 同 9:1 分层划分（同 repeat 同 seed，secrets.randbits(31) 随机生成并写入结果）
  - 同微调预算：max_len=128 / batch=64 / AdamW lr=3e-5 / warmup 10% / 梯度裁剪 1.0
    / 最多 6 epoch / 早停 patience=2（指标=验证子集损失，恢复最优权重）
  - 下游与 flat3 完全同协议：α∈{0.01,0.03,0.1,0.3,1.0} 与 τ∈[0.30,0.70]（步长
    0.005）全部在验证子集按宏F1筛选；早停 patience=20；test 仅训练完成后评一次
  - 同指标电池：宏F1/加权F1/ACC/精确宏/召回宏/有毒F1/无毒F1/AUC

无泄漏保证：
  - 学生只在训练子集上训练（教师目标也只取训练子集行）；val/test 的概念向量由
    训练好的学生前向预测，教师对 val/test 的概念读数不进入任何训练环节
  - 教师自身读出（GLM v7 原值 → FC/MLP）的对照数字已在 results/flat3/v7/
    summary_v7.txt，本脚本不重跑

产物：results/student_cbm_v14/<arm>/{selection.csv, test_eval.csv, metrics.txt}
      results/student_cbm_v14/summary_all.csv + summary_v14.txt

快速自检（约几分钟，不训练完整轮）：--smoke 200（截取前200样本、1 epoch、
只跑 direct+student 各一次，验证链路连通，结果不作数）。

运行（本机，toxicnew 环境）：
  D:/miniconda3/envs/toxicnew/python.exe scripts/run_student_cbm_v14.py --smoke 200   # 先自检
  D:/miniconda3/envs/toxicnew/python.exe scripts/run_student_cbm_v14.py               # 全量
"""

import argparse
import math
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit(f"[FATAL] 缺少 torch：{e}")

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup
except ImportError as e:
    raise SystemExit(f"[FATAL] 缺少 transformers：{e}")

try:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise SystemExit(f"[FATAL] 缺少 sklearn：{e}")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ---- 协议常量 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data/processed/TOXICN/v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_train_matrix.csv"
TEST_CSV = PROJECT_ROOT / "data/processed/TOXICN/v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_test_matrix.csv"

FT_MAX_EPOCHS = 6        # 学生/直接臂微调 epoch 上限（r0 实测 4~6 epoch 早停）
FT_PATIENCE = 2          # 微调早停耐心（验证损失）
FT_BATCH = 64
FT_LR = 3e-5
FT_MAX_LEN = 128  # 覆盖 p95=125 字；>128 字仅 4.8%，截断对 direct/student 两臂对称
FT_WARMUP_RATIO = 0.1
FT_GRAD_CLIP = 1.0

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
TAU_GRID = np.round(np.arange(0.30, 0.7005, 0.005), 3)
MAX_EPOCHS = 150         # 下游训练 epoch 上限
PATIENCE = 20            # 下游早停耐心
LR = 1e-2
MLP_HIDDEN = 64
MLP_DROPOUT = 0.2
N_REPEATS = 5
VAL_FRACTION = 0.1

CLFS = ["fc", "mlp"]
METRIC_KEYS = ["macro", "weighted", "acc", "prec_macro", "rec_macro",
               "toxic_f1", "benign_f1", "auc"]

ARMS = ["direct_bert", "student_flat_fc", "student_flat_mlp"]


# ---------------- 数据 ----------------

class TextDataset(Dataset):
    """content 列表 + 可选目标（direct=0/1 标签；student=教师 p_yes 向量）。"""

    def __init__(self, texts, targets=None):
        self.texts = list(texts)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = self.targets[i] if self.targets is not None else 0.0
        return self.texts[i], t


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts = [b[0] for b in batch]
        tg = torch.from_numpy(np.stack(
            [np.asarray(b[1], dtype=np.float32) for b in batch]))
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        return enc, tg
    return collate


# ---------------- 学生 / 直接模型 ----------------

HEAD_HIDDEN = 256  # 概念头隐层宽度；main 中 --linear_head 时置 0（纯线性头）


class StudentModel(nn.Module):
    """RoBERTa 编码器 + 概念头。hidden>0: CLS→hidden→ReLU→Dropout→k（非线性头，
    Koh et al. 2020 标准做法）；hidden=0: CLS→k 纯线性头。下游只见 k 维输出。"""

    def __init__(self, model_name, k_out, head_hidden):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(h, head_hidden), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(head_hidden, k_out))
        else:
            self.head = nn.Linear(h, k_out)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]  # [CLS] 768 维
        return self.head(cls)


def finetune(model_name, texts_tr, y_tr, texts_va, y_va, k_out, seed, device,
             n_steps_hint=0):
    """微调 RoBERTa。k_out=1（direct，y=0/1）或 k=概念维（student，y=教师 p_yes）。
    早停=验证损失 patience=FT_PATIENCE，恢复最优权重。返回 (model, best_loss, hist)。"""
    torch.manual_seed(seed)
    model = StudentModel(model_name, k_out, HEAD_HIDDEN).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    collate = make_collate(tok, FT_MAX_LEN)

    dl_tr = DataLoader(TextDataset(texts_tr, y_tr), batch_size=FT_BATCH,
                       shuffle=True, collate_fn=collate)
    dl_va = DataLoader(TextDataset(texts_va, y_va), batch_size=FT_BATCH * 2,
                       shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    total_steps = max(1, len(dl_tr) * FT_MAX_EPOCHS)
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * FT_WARMUP_RATIO), total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    # bf16 优先，回退 fp16（配 GradScaler），再回退 fp32
    if device.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dt = None
    scaler = torch.amp.GradScaler(device.type, enabled=(dt == torch.float16))

    def run_epoch(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for enc, y in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                y = y.to(device)
                with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
                    logits = model(**enc).squeeze(-1)
                    loss = loss_fn(logits.float(), y)
                if train:
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), FT_GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                bs = y.shape[0]
                tot += loss.item() * bs
                n += bs
        return tot / max(n, 1)

    best_loss, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(1, FT_MAX_EPOCHS + 1):
        tr_loss = run_epoch(dl_tr, True)
        va_loss = run_epoch(dl_va, False)
        hist.append({"epoch": ep, "train_loss": round(tr_loss, 6),
                     "val_loss": round(va_loss, 6)})
        global_bar.update(1)
        global_bar.set_postfix_str(
            f"{train_tag} ep{ep} val_loss={va_loss:.4f}")
        if va_loss < best_loss - 1e-4:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss, hist


@torch.no_grad()
def predict(model_name, model, texts, device, batch_size=64):
    """返回 probs。direct: shape [n]；student: [n, k]。逐 batch 更新总进度条。"""
    tok = AutoTokenizer.from_pretrained(model_name)
    collate = make_collate(tok, FT_MAX_LEN)
    dl = DataLoader(TextDataset(texts), batch_size=batch_size,
                    shuffle=False, collate_fn=collate)
    model.eval()
    if device.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dt = None
    outs = []
    for enc, _ in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            logits = model(**enc)
        outs.append(torch.sigmoid(logits.float()).cpu().numpy())
        global_bar.update(1)
        global_bar.set_postfix_str(f"{train_tag} 推理 {sum(len(o) for o in outs)}/{len(texts)}")
    return np.concatenate(outs, axis=0)


# ---------------- 下游分类器（与 flat3 完全同款） ----------------

class FC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x)

    def l1(self):
        return self.lin.weight.abs().sum()

    def nnz(self):
        return int((self.lin.weight.abs() > 1e-6).sum())


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, MLP_HIDDEN), nn.ReLU(),
                                 nn.Dropout(MLP_DROPOUT), nn.Linear(MLP_HIDDEN, 1))

    def forward(self, x):
        return self.net(x)

    def l1(self):
        return sum(m.weight.abs().sum() for m in self.net if isinstance(m, nn.Linear))

    def nnz(self):
        return sum(int((m.weight.abs() > 1e-6).sum())
                   for m in self.net if isinstance(m, nn.Linear))


def macro_f1_acc(y_true, prob, tau):
    pred = (prob >= tau).astype(int)
    return f1_score(y_true, pred, average="macro", zero_division=0), \
        accuracy_score(y_true, pred)


def train_downstream(clf, Xtr, ytr, Xva, yva, alpha, seed, device):
    torch.manual_seed(seed)
    model = (FC(Xtr.shape[1]) if clf == "fc" else MLP(Xtr.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lam = 1.0 / (alpha * len(ytr))
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss()
    best_f1, best_ep, best_state, bad = -1.0, -1, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xt).squeeze(-1), yt) + lam * model.l1()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(Xv).squeeze(-1)).cpu().numpy()
        f1, acc = macro_f1_acc(yva, pva, 0.5)
        global_bar.update(1)
        global_bar.set_postfix_str(f"{train_tag} ep{ep} val宏F1={f1:.4f} valAcc={acc:.4f}")
        if f1 > best_f1:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_state, best_ep


def load_downstream(clf, d, seed, device):
    torch.manual_seed(seed)
    model = (FC(d) if clf == "fc" else MLP(d)).to(device)
    return model


def eval_all_metrics(y_true, prob, tau):
    pred = (prob >= tau).astype(int)
    return {"macro": f1_score(y_true, pred, average="macro", zero_division=0),
            "weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
            "acc": accuracy_score(y_true, pred),
            "prec_macro": precision_score(y_true, pred, average="macro", zero_division=0),
            "rec_macro": recall_score(y_true, pred, average="macro", zero_division=0),
            "toxic_f1": f1_score(y_true, pred, zero_division=0),
            "benign_f1": f1_score(y_true, pred, pos_label=0, zero_division=0),
            "auc": roc_auc_score(y_true, prob)}


# ---------------- 主流程 ----------------

global_bar = None
train_tag = ""


def main():
    global global_bar, train_tag
    ap = argparse.ArgumentParser(description="v14 学生 CBM（RoBERTa 学 v7 教师）")
    ap.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext",
                    help="RoBERTa 模型名或本地路径")
    ap.add_argument("--output_root", default=str(PROJECT_ROOT / "results/student_cbm_v14"))
    ap.add_argument("--n_repeats", type=int, default=N_REPEATS)
    ap.add_argument("--smoke", type=int, default=0,
                    help=">0 时截取前 N 样本、1 epoch 微调、3 下游 epoch，仅验链路")
    ap.add_argument("--linear_head", action="store_true",
                    help="概念头退化为纯线性 CLS→k（默认非线性 CLS→256→ReLU→Dropout→k）")
    args = ap.parse_args()
    global HEAD_HIDDEN
    if args.linear_head:
        HEAD_HIDDEN = 0
    if args.smoke > 0:
        global FT_MAX_EPOCHS, MAX_EPOCHS
        FT_MAX_EPOCHS, MAX_EPOCHS = 1, 3
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  torch = {torch.__version__}  model = {args.model_name}")

    # ---- fail-fast 前置断言 ----
    for p in (TRAIN_CSV, TEST_CSV):
        assert p.exists(), f"[FATAL] 缺文件 {p}"
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    assert {"id", "toxic", "content"}.issubset(tr.columns), "[FATAL] v7 矩阵缺 id/toxic/content"
    assert set(tr["toxic"].astype(int).unique()) <= {0, 1}, "[FATAL] v7 标签异常"
    feat_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    k_dim = len(feat_cols)
    assert k_dim == 180, f"[FATAL] v7 概念维应为 180，实际 {k_dim}"
    if args.smoke > 0:
        # smoke 截断保两类（否则 stratify/roc_auc 崩溃）
        n_half = args.smoke // 2
        tr = pd.concat([tr[tr["toxic"] == 1].head(n_half),
                        tr[tr["toxic"] == 0].head(n_half)])
        te = pd.concat([te[te["toxic"] == 1].head(10),
                        te[te["toxic"] == 0].head(10)])
    yte = te["toxic"].astype(int).values
    texts_te = te["content"].astype(str).tolist()
    print(f"v7 教师: train {len(tr)} / test {len(te)}，概念维 k={k_dim}")

    # 总进度条（flat3 同款：固定 total=预期事件数上界，早停导致实际略少于 total）
    infer_batch = math.ceil
    n_va = int(math.ceil(VAL_FRACTION * len(tr)))
    n_tr = len(tr) - n_va
    bs_inf = 64
    per_repeat = (4 * FT_MAX_EPOCHS                                  # direct+student 微调 epoch 事件
                  + 2 * len(CLFS) * len(ALPHA_GRID) * MAX_EPOCHS     # 下游 2clf×5α×epoch
                  + infer_batch(n_va / bs_inf) + infer_batch(len(te) / bs_inf)      # direct 推理 va/te
                  + infer_batch(n_tr / bs_inf) + infer_batch(n_va / bs_inf)
                  + infer_batch(len(te) / bs_inf))                   # student 推理 tr/va/te
    global_bar = tqdm(total=per_repeat * args.n_repeats,
                      desc="student_cbm_v14", dynamic_ncols=True)

    summary_rows = []
    for rep in range(args.n_repeats):
        seed = secrets.randbits(31)  # 每轮随机生成种子，direct 与 student 共用同一划分
        tr_sub, va_sub = train_test_split(
            tr, test_size=VAL_FRACTION, stratify=tr["toxic"].astype(int),
            random_state=seed)
        texts_tr = tr_sub["content"].astype(str).tolist()
        texts_va = va_sub["content"].astype(str).tolist()
        ytr = tr_sub["toxic"].astype(int).values
        yva = va_sub["toxic"].astype(int).values

        # ================= 臂 1：direct_bert =================
        train_tag = f"direct r{rep}"
        model_d, _, hist_d = finetune(
            args.model_name, texts_tr, ytr.astype(np.float32),
            texts_va, yva.astype(np.float32), k_out=1, seed=seed, device=device)
        pva_d = predict(args.model_name, model_d, texts_va, device).reshape(-1)
        pte_d = predict(args.model_name, model_d, texts_te, device).reshape(-1)
        taus = list(TAU_GRID)
        tau_star = float(taus[int(np.argmax([macro_f1_acc(yva, pva_d, t)[0] for t in taus]))])
        row = {"repeat": rep, "seed": seed, "arm": "direct_bert", "clf": "roberta",
               "tau": tau_star,
               **{k: round(v, 4) for k, v in eval_all_metrics(yte, pte_d, tau_star).items()}}
        summary_rows.append(row)
        _write_arm(out_root / "direct_bert", rep, seed, tau_star,
                   [{k: row[k] for k in ("repeat", "seed", "tau", *METRIC_KEYS)}],
                   hist_d, "direct_bert (RoBERTa 直接判毒)")
        del model_d
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ================= 臂 2/3：student → 下游 =================
        # 学生目标 = 教师 v7 p_yes（仅训练子集行；val 用于早停损失）
        teacher_tr = tr_sub[feat_cols].values.astype(np.float32)
        teacher_va = va_sub[feat_cols].values.astype(np.float32)
        # NaN 目标行过滤（若教师矩阵存在缺失，对应样本不参与学生训练）
        m_tr = np.isfinite(teacher_tr).all(axis=1)
        m_va = np.isfinite(teacher_va).all(axis=1)
        assert m_tr.mean() > 0.95, f"[FATAL] 教师目标缺失率过高 train {(1-m_tr.mean()):.3f}"

        train_tag = f"student r{rep}"
        model_s, _, hist_s = finetune(
            args.model_name,
            [t for t, m in zip(texts_tr, m_tr) if m], teacher_tr[m_tr],
            [t for t, m in zip(texts_va, m_va) if m], teacher_va[m_va],
            k_out=k_dim, seed=seed, device=device)

        # 学生概念预测（train/val/test 全部由学生前向给出）
        S_tr = predict(args.model_name, model_s, texts_tr, device)
        S_va = predict(args.model_name, model_s, texts_va, device)
        S_te = predict(args.model_name, model_s, texts_te, device)
        assert S_tr.shape == (len(texts_tr), k_dim), "[FATAL] 学生输出维度异常"
        # 学生概念保真度诊断（val：与教师读数的逐概念 Pearson r 均值）
        if m_va.sum() > 10:
            r_per_c = []
            for j in range(k_dim):
                a, b = teacher_va[m_va, j], S_va[m_va, j]
                if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    r_per_c.append(np.corrcoef(a, b)[0, 1])
            fidelity_r = float(np.mean(r_per_c))
        else:
            fidelity_r = float("nan")
        del model_s
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for clf in CLFS:
            tag = f"student_flat_{clf}"
            out_dir = out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            sel_rows, test_rows = [], []
            best = None  # (val_f1@tau, alpha, tau, state, best_ep, nnz)
            for alpha in ALPHA_GRID:
                train_tag = f"{tag} r{rep} a{alpha}"
                state, best_ep = train_downstream(
                    clf, S_tr, ytr, S_va, yva, alpha, seed, device)
                model = load_downstream(clf, S_tr.shape[1], seed, device)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    pva = torch.sigmoid(model(
                        torch.tensor(S_va, dtype=torch.float32, device=device)
                    ).squeeze(-1)).cpu().numpy()
                taus = list(TAU_GRID)
                f1s = [macro_f1_acc(yva, pva, t)[0] for t in taus]
                tau = float(taus[int(np.argmax(f1s))])
                val_f1, val_acc = macro_f1_acc(yva, pva, tau)
                nnz = model.nnz()
                sel_rows.append({"repeat": rep, "seed": seed, "alpha": alpha,
                                 "tau": tau, "val_macro_f1": round(val_f1, 4),
                                 "val_acc": round(val_acc, 4),
                                 "best_epoch": best_ep, "nnz": nnz})
                if best is None or val_f1 > best[0]:
                    best = (val_f1, alpha, tau, state, best_ep, nnz)
            _, alpha_star, tau_star, state, best_ep, nnz = best

            # ---- test 只评一次 ----
            model = load_downstream(clf, S_tr.shape[1], seed, device)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                pte = torch.sigmoid(model(
                    torch.tensor(S_te, dtype=torch.float32, device=device)
                ).squeeze(-1)).cpu().numpy()
            row = {"repeat": rep, "seed": seed, "arm": tag, "clf": clf,
                   "alpha": alpha_star, "tau": tau_star, "best_epoch": best_ep,
                   "nnz": nnz, "fidelity_r": round(fidelity_r, 4),
                   **{k: round(v, 4) for k, v in eval_all_metrics(yte, pte, tau_star).items()}}
            summary_rows.append(row)
            test_rows.append({k: row[k] for k in
                              ("repeat", "seed", "alpha", "tau", "best_epoch", "nnz",
                               *METRIC_KEYS)})
            pd.DataFrame(sel_rows).to_csv(out_dir / "selection.csv", index=False)
            tdf = pd.DataFrame(test_rows)
            tdf.to_csv(out_dir / "test_eval.csv", index=False)
            _write_arm_metrics(out_dir / "metrics.txt", tag, rep, test_rows, hist_s,
                               clf, fidelity_r)
        print(f"\n[完成] repeat {rep}")

    global_bar.close()

    # ---- 汇总 ----
    all_df = pd.DataFrame(summary_rows)
    all_df.to_csv(out_root / "summary_all.csv", index=False)
    labels = ["宏F1", "加权F1", "ACC", "精确宏", "召回宏", "有毒F1", "无毒F1", "AUC"]
    lines = [
        "=" * 122,
        f"v14 学生 CBM 汇总（test，{args.n_repeats} 轮随机种子，mean±std；model={args.model_name}）",
        "臂: direct_bert=RoBERTa直接判毒 | student_flat_{fc,mlp}=RoBERTa学v7教师180维概念→下游",
        "控制变量: 同backbone/同划分(同seed)/同微调预算(max_len128,batch64,lr3e-5,≤6ep,早停patience=2)",
        f"概念头: {'线性 CLS→k' if args.linear_head else '非线性 CLS→256→ReLU→Dropout(0.2)→k'}（direct/student 两臂同构）",
        "下游协议v2: α+τ@[0.30,0.70]验证子集筛选(宏F1) / 早停patience=20 / test仅评一次",
        "教师自身读出对照(GRM v7原值→FC/MLP)见 results/flat3/v7/summary_v7.txt",
        "=" * 122,
        "臂               分类器   " + "  ".join(f"{l:<16}" for l in labels),
    ]
    for arm in ARMS:
        rows = [r for r in summary_rows if r["arm"] == arm]
        assert len(rows) == args.n_repeats, f"[FATAL] {arm} 轮数不足 {len(rows)}"
        cells = "  ".join(
            f"{np.mean([r[k] for r in rows]):.4f}±{np.std([r[k] for r in rows]):.4f}"
            for k in METRIC_KEYS)
        clf_name = "roberta" if arm == "direct_bert" else arm.split("_")[-1]
        lines.append(f"{arm:<17}{clf_name:<9}{cells}")
    lines.append("-" * 122)
    lines.append("解读: student_flat vs direct_bert = 概念瓶颈的代价/收益；"
                 "student vs flat3教师读出 = 学生蒸馏损失（概念瓶颈两端各扣一次）")
    (out_root / "summary_v14.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n全部完成 → {out_root}/summary_all.csv 与 summary_v14.txt")


def _write_arm(out_dir, rep, seed, tau, rows, hist, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "test_eval.csv", index=False)
    lines = ["=" * 76, f"{title} | repeat {rep} seed {seed} tau {tau}",
             "epoch历史: " + "; ".join(
                 f"ep{h['epoch']}:tr{h['train_loss']:.4f}/va{h['val_loss']:.4f}"
                 for h in hist)]
    (out_dir / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_arm_metrics(path, tag, rep, test_rows, hist, clf, fidelity_r):
    lines = [
        "=" * 76,
        f"student_cbm_v14 | {tag} | repeat {rep} | "
        + ("FC Linear(d→1)+L1" if clf == "fc"
           else f"MLP {MLP_HIDDEN}→ReLU→Dropout({MLP_DROPOUT})→1 + L1"),
        f"学生概念保真度(val逐概念Pearson r均值): {fidelity_r:.4f}",
        "微调epoch历史: " + "; ".join(
            f"ep{h['epoch']}:tr{h['train_loss']:.4f}/va{h['val_loss']:.4f}" for h in hist),
        "-" * 76,
        "  repeat seed      α*     τ*   宏F1   加权F1   ACC   精确宏  召回宏  有毒F1  无毒F1   AUC",
    ]
    for r in test_rows:
        lines.append(f"  {r['repeat']:>4} {r['seed']:>9} {r['alpha']:>7} {r['tau']:>6} "
                     + " ".join(f"{r[k]:.4f}" for k in METRIC_KEYS))
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
