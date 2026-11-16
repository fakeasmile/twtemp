# -*- coding: utf-8 -*-
"""flat3-v7：TOXICN v7 概念向量单线读出（平铺 / v81秩+meta / 构念聚合）× {FC, MLP}。

严格单线：只用 v7 概念向量（180 条目，items_v7_glm-4-9b-chat 矩阵），
不混用 v3/v6。矩阵自带 content 列。

三臂处理（所有拟合统计量只在 9:1 划分后的训练子集上计算，val/test 仅投影）：
  arm1_flat       概念向量原值平铺（180 条目 p_yes）；缺失值 = 该列训练子集中位数
  arm2_v81        v8.1 M4 单线复刻（装配逻辑逐行参照 run_readout_v7_v81.py
                  的 rank_fit/meta_base，去掉 freeze 依赖）：
                  条目列 101 分位秩变换（train 边界冻结投影）
                  + meta 6 特征（log1p长度/emoji/感叹/问号/重复标点/规避写法，
                  train 标准化）
                  + len×top5 交互（top5 = train 子集 |点二列相关| 最高的 5 条目）
  arm3_construct  心理测量学构念聚合（MMPI 条目→量表架构）：
                  构念映射来自 cbm_concept_set_v7.csv（条目id, 构念id）；
                  经验极性对齐（train 裁决：corr<0.5 的条目取 1-x）
                  → 点二列相关 CITC 加权（w=max(corr,0)，全≤0 时等权）
                  → 构念分 → train z 标准化。

分类器（torch，本机 GPU 优先，无 GPU 自动回退 CPU）：
  FC  Linear(d→1)，权重 L1（λ=1/(α·n_train)，偏置不罚），AdamW lr=1e-2，全批量
  MLP Linear(d→64)→ReLU→Dropout(0.2)→Linear(64→1)，权重 L1 同上

协议 v2：每轮（arm×clf×repeat）用 secrets 随机生成种子并写入结果文件；
一次 9:1 分层划分；α∈{0.01,0.03,0.1,0.3,1.0} 与 τ∈[0.30,0.70]（步长0.005）
全部在验证子集筛选（指标=宏F1）；早停 patience=20，恢复最优 epoch 权重；
test 仅训练完成后评估一次。全程只有一个总进度条，postfix 实时报告
当前训练每轮的验证宏F1/Acc。

产物：results/flat3/v7/<arm>_<clf>/{selection.csv, test_eval.csv, test_predictions.csv,
      metrics.txt}，results/flat3/v7/summary_all.csv
  test_predictions.csv = 逐样本预测（每臂×分类器 5 轮 × 2411 test 样本），列含
  repeat/seed/alpha/tau/id/y_true/p_model/y_pred/toxic_type/expression/platform/
  topic/content，供后续按 TOXICN 标注体系（toxic_type、expression 等）做分组
  正确率分析。id 后缀号 = data/raw/TOXICN/test.json 行号（已验证 content 全等），
  toxic_type/expression 直接从 raw json 对齐写入，无需二次 join。

运行（本机，toxicnew 环境）：D:/miniconda3/envs/toxicnew/python.exe scripts/readout_flat3_v7.py
"""

import argparse
import json
import re
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # fail-fast：本任务必须依赖 torch
    raise SystemExit(f"[FATAL] 缺少 torch：{e}")

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
ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
TAU_GRID = np.round(np.arange(0.30, 0.7005, 0.005), 3)
MAX_EPOCHS = 150
PATIENCE = 20
LR = 1e-2
MLP_HIDDEN = 64
MLP_DROPOUT = 0.2
N_QUANT = 101
N_TOP_INTERACT = 5
N_REPEATS = 5
VAL_FRACTION = 0.1

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
EVASION_LEX = ("朲", "亽", "㐺", "亻", "怹")

TRAIN_CSV = PROJECT_ROOT / "data/processed/TOXICN/v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_train_matrix.csv"
TEST_CSV = PROJECT_ROOT / "data/processed/TOXICN/v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_test_matrix.csv"
ITEMS_CSV = PROJECT_ROOT / "data/concepts/cbm_concept_set_v7.csv"
RAW_TEST_JSON = PROJECT_ROOT / "data/raw/TOXICN/test.json"  # toxic_type/expression 等标注体系

ARMS = ["arm1_flat", "arm2_v81", "arm3_construct"]
CLFS = ["fc", "mlp"]
SET_TAG = "v7"

METRIC_KEYS = ["macro", "weighted", "acc", "prec_macro", "rec_macro",
               "toxic_f1", "benign_f1", "auc"]


# ---------------- 特征处理 ----------------

def impute_median(Xtr, *others):
    """缺失值 = 训练子集列中位数（train 统计冻结投影）。"""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.5, med)  # 整列全缺时退回中性 0.5

    def fill(X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = med[idx[1]]
        return X

    return fill(Xtr), *[fill(X) for X in others]


def rank_fit(raw_tr, raw_va, raw_te, n_quant=N_QUANT):
    """101 分位秩变换（train 边界冻结，参照 run_readout_v7_v81.rank_fit）。"""
    qs = np.linspace(0.0, 1.0, n_quant)
    out = []
    for raw in (raw_tr, raw_va, raw_te):
        R = raw.copy()
        for j in range(raw.shape[1]):
            edges = np.quantile(raw_tr[:, j], qs)
            R[:, j] = np.interp(raw[:, j], edges, qs)
        out.append(R)
    return out


def meta_base(texts_ref):
    """meta 6 特征（参照 run_readout_v7_v81.meta_base）。"""

    def base(texts):
        len_c = np.array([len(str(t).replace(" ", "")) for t in texts], dtype=float)
        emoji = np.array([len(EMOJI_RE.findall(str(t))) for t in texts], dtype=float)
        exclaim = np.array([str(t).count("！") + str(t).count("!") for t in texts], dtype=float)
        question = np.array([str(t).count("？") + str(t).count("?") for t in texts], dtype=float)
        repeat = np.array([len(re.findall(r"(？{2,}|！{2,}|\.{3,}|…{2,})", str(t)))
                           for t in texts], dtype=float)
        evade = np.array([sum(str(t).count(k) for k in EVASION_LEX) for t in texts], dtype=float)
        return np.column_stack([np.log1p(len_c), emoji, exclaim, question, repeat, evade])

    Mtr, Mva, Mte = base(texts_ref[0]), base(texts_ref[1]), base(texts_ref[2])
    mu, sd = Mtr.mean(axis=0), Mtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (Mtr - mu) / sd, (Mva - mu) / sd, (Mte - mu) / sd


def z_fit(*mats):
    """train z 标准化（train 统计冻结投影）。"""
    mu, sd = mats[0].mean(axis=0), mats[0].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return [(m - mu) / sd for m in mats]


def build_mapping():
    """构念id → 条目id 列表映射，来自概念集 CSV（条目id, 构念id 两列）。"""
    df = pd.read_csv(ITEMS_CSV)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    mapping = {}
    for cid, sub in df.groupby("构念id"):
        mapping[str(cid)] = sub["条目id"].astype(str).tolist()
    return mapping


def build_features(arm, tr_df, va_df, te_df, mapping):
    """按臂构造特征。tr_df=训练子集（拟合），va_df/te_df=仅投影。
    输入 DataFrame 均含 id/toxic/content(+特征列)，返回 (Xtr, Xva, Xte, names)。"""
    feat_cols = [c for c in tr_df.columns if c not in ("id", "toxic", "content")]
    texts = (tr_df["content"].tolist(), va_df["content"].tolist(), te_df["content"].tolist())
    raw = [tr_df[feat_cols].values.astype(float),
           va_df[feat_cols].values.astype(float),
           te_df[feat_cols].values.astype(float)]
    ysub = tr_df["toxic"].astype(int).values

    if arm == "arm1_flat":
        filled = impute_median(*raw)
        return *filled, feat_cols

    if arm == "arm2_v81":
        filled = impute_median(*raw)
        Rtr, Rva, Rte = rank_fit(*filled)
        Mtr, Mva, Mte = meta_base(texts)
        pbr = np.array([abs(np.corrcoef(filled[0][:, j], ysub)[0, 1])
                        if np.std(filled[0][:, j]) > 1e-9 else 0.0
                        for j in range(len(feat_cols))])
        top = np.argsort(-pbr)[:N_TOP_INTERACT]
        inter = [(M[:, 0:1] * R[:, top]) for M, R in zip((Mtr, Mva, Mte), (Rtr, Rva, Rte))]
        Xtr = np.hstack([Rtr, Mtr, inter[0]])
        Xva = np.hstack([Rva, Mva, inter[1]])
        Xte = np.hstack([Rte, Mte, inter[2]])
        names = (["rank_" + c for c in feat_cols]
                 + ["meta_len", "meta_emoji", "meta_exclaim", "meta_question",
                    "meta_repeat", "meta_evade"]
                 + [f"inter_len_{feat_cols[j]}" for j in top])
        return Xtr, Xva, Xte, names

    if arm == "arm3_construct":
        filled = impute_median(*raw)
        tr_m, va_m, te_m = filled
        cols, names = [], []
        for gname, gcols in mapping.items():
            gc = [c for c in gcols if c in feat_cols]
            assert gc, f"[FATAL] 构念 {gname} 的条目没有命中矩阵列"
            Atr = np.column_stack([tr_m[:, feat_cols.index(c)] for c in gc])
            Ava = np.column_stack([va_m[:, feat_cols.index(c)] for c in gc])
            Ate = np.column_stack([te_m[:, feat_cols.index(c)] for c in gc])
            # 经验极性对齐（train 裁决）
            for j, c in enumerate(gc):
                if np.std(Atr[:, j]) > 1e-9 and np.corrcoef(Atr[:, j], ysub)[0, 1] < 0.5:
                    Atr[:, j], Ava[:, j], Ate[:, j] = 1 - Atr[:, j], 1 - Ava[:, j], 1 - Ate[:, j]
            # CITC 加权（train 裁决）
            w = np.array([np.corrcoef(Atr[:, j], ysub)[0, 1]
                          if np.std(Atr[:, j]) > 1e-9 else 0.0
                          for j in range(len(gc))])
            w = np.clip(w, 0.0, None)
            if w.sum() <= 1e-9:
                w = np.full(len(gc), 1.0 / len(gc))
            else:
                w = w / w.sum()
            cols.append((Atr @ w, Ava @ w, Ate @ w))
            names.append(f"{gname}_score")
        Gtr = np.column_stack([c[0] for c in cols])
        Gva = np.column_stack([c[1] for c in cols])
        Gte = np.column_stack([c[2] for c in cols])
        Ztr, Zva, Zte = z_fit(Gtr, Gva, Gte)
        return Ztr, Zva, Zte, names

    raise AssertionError(f"[FATAL] 未知臂 {arm}")


# ---------------- 模型 ----------------

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


def train_one(clf, Xtr, ytr, Xva, yva, alpha, seed, device):
    """单次训练：全批量 AdamW + L1，早停 patience=20（指标=验证宏F1），
    返回 (best_state, best_epoch)。"""
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


# ---------------- 主流程 ----------------

global_bar = None
train_tag = ""


def main():
    global global_bar, train_tag
    ap = argparse.ArgumentParser(description="flat3-v7：v7 概念向量单线三臂读出")
    ap.add_argument("--output_root", default=str(PROJECT_ROOT / "results/flat3/v7"))
    ap.add_argument("--n_repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  torch = {torch.__version__}")

    # ---- fail-fast 前置断言 ----
    for p in (TRAIN_CSV, TEST_CSV, ITEMS_CSV):
        assert p.exists(), f"[FATAL] 缺文件 {p}"
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    assert {"id", "toxic", "content"}.issubset(tr.columns), "[FATAL] v7 矩阵缺 id/toxic/content"
    assert set(tr["toxic"].astype(int).unique()) <= {0, 1}, "[FATAL] v7 标签异常"
    feat_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    mapping = build_mapping()
    covered = set(c for gc in mapping.values() for c in gc)
    assert covered == set(feat_cols), \
        f"[FATAL] 构念映射与矩阵列不一致：映射缺 {sorted(set(feat_cols) - covered)[:5]}，" \
        f"矩阵缺 {sorted(covered - set(feat_cols))[:5]}"
    yte = te["toxic"].astype(int).values
    print(f"v7: train {len(tr)} / test {len(te)}，条目 {len(feat_cols)}，构念 {len(mapping)} 个")

    # ---- TOXICN raw 标注体系对齐（id 后缀号 = raw test.json 行号）----
    raw_te = json.loads(RAW_TEST_JSON.read_text(encoding="utf-8"))
    assert len(raw_te) == len(te), \
        f"[FATAL] raw test.json {len(raw_te)} 条 != 矩阵 test {len(te)} 条"
    te_ids = te["id"].tolist()
    te_contents = te["content"].tolist()
    for i, c in zip(te_ids, te_contents):
        assert raw_te[int(str(i).rsplit("_", 1)[-1])]["content"] == str(c), \
            f"[FATAL] 矩阵 id {i} 与 raw test.json 行 content 不一致"
    te_meta = {
        "toxic_type": [raw_te[int(str(i).rsplit("_", 1)[-1])]["toxic_type"] for i in te_ids],
        "expression": [raw_te[int(str(i).rsplit("_", 1)[-1])]["expression"] for i in te_ids],
        "platform": [raw_te[int(str(i).rsplit("_", 1)[-1])]["platform"] for i in te_ids],
        "topic": [raw_te[int(str(i).rsplit("_", 1)[-1])]["topic"] for i in te_ids],
    }

    n_trainings = len(ARMS) * len(CLFS) * args.n_repeats * len(ALPHA_GRID)
    global_bar = tqdm(total=n_trainings * MAX_EPOCHS, desc="flat3-v7", dynamic_ncols=True)

    summary_rows = []
    for arm in ARMS:
        for clf in CLFS:
            tag = f"{arm}_{clf}"
            out_dir = out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            sel_rows, test_rows, pred_rows = [], [], []
            for rep in range(args.n_repeats):
                seed = secrets.randbits(31)  # 每轮随机生成种子，写入结果
                tr_sub, va_sub = train_test_split(
                    tr, test_size=VAL_FRACTION, stratify=tr["toxic"].astype(int),
                    random_state=seed)
                Xtr, Xva, Xte, names = build_features(arm, tr_sub, va_sub, te, mapping)
                ytr = tr_sub["toxic"].astype(int).values
                yva = va_sub["toxic"].astype(int).values

                # ---- 验证子集筛选 α（τ 在最优 α 的验证概率上选）----
                best = None  # (val_f1@tau, alpha, tau, state, best_ep, nnz)
                for alpha in ALPHA_GRID:
                    train_tag = f"{tag} r{rep} a{alpha}"
                    state, best_ep = train_one(clf, Xtr, ytr, Xva, yva, alpha, seed, device)
                    torch.manual_seed(seed)
                    model = (FC(Xtr.shape[1]) if clf == "fc" else MLP(Xtr.shape[1])).to(device)
                    model.load_state_dict(state)
                    model.eval()
                    with torch.no_grad():
                        pva = torch.sigmoid(model(
                            torch.tensor(Xva, dtype=torch.float32, device=device)
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
                torch.manual_seed(seed)
                model = (FC(Xtr.shape[1]) if clf == "fc" else MLP(Xtr.shape[1])).to(device)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    pte = torch.sigmoid(model(
                        torch.tensor(Xte, dtype=torch.float32, device=device)
                    ).squeeze(-1)).cpu().numpy()
                pred = (pte >= tau_star).astype(int)
                row = {"repeat": rep, "seed": seed, "alpha": alpha_star,
                       "tau": tau_star, "best_epoch": best_ep, "nnz": nnz,
                       "macro": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
                       "weighted": round(f1_score(yte, pred, average="weighted", zero_division=0), 4),
                       "acc": round(accuracy_score(yte, pred), 4),
                       "prec_macro": round(precision_score(yte, pred, average="macro", zero_division=0), 4),
                       "rec_macro": round(recall_score(yte, pred, average="macro", zero_division=0), 4),
                       "toxic_f1": round(f1_score(yte, pred, zero_division=0), 4),
                       "benign_f1": round(f1_score(yte, pred, pos_label=0, zero_division=0), 4),
                       "auc": round(roc_auc_score(yte, pte), 4)}
                test_rows.append(row)
                summary_rows.append({"set": "v7", "arm": arm, "clf": clf, **row})

                # ---- 逐样本预测（含 raw 标注体系列，供分组正确率分析）----
                for k in range(len(te)):
                    pred_rows.append({
                        "repeat": rep, "seed": seed, "alpha": alpha_star,
                        "tau": tau_star, "id": te_ids[k],
                        "y_true": int(yte[k]),
                        "p_model": round(float(pte[k]), 4),
                        "y_pred": int(pred[k]),
                        "toxic_type": te_meta["toxic_type"][k],
                        "expression": te_meta["expression"][k],
                        "platform": te_meta["platform"][k],
                        "topic": te_meta["topic"][k],
                        "content": te_contents[k]})

            pd.DataFrame(sel_rows).to_csv(out_dir / "selection.csv", index=False)
            tdf = pd.DataFrame(test_rows)
            tdf.to_csv(out_dir / "test_eval.csv", index=False)
            pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

            # ---- metrics.txt 汇总 ----
            lines = [
                "=" * 76,
                f"flat3-v7 | {tag} | {args.n_repeats} 轮 | device={device}",
                f"协议v2: 9:1分层划分(每轮随机种子) / α{ALPHA_GRID}+τ@[0.30,0.70]验证子集筛选",
                f"早停 patience={PATIENCE} max_epochs={MAX_EPOCHS} AdamW lr={LR} | "
                + ("FC Linear(d→1)+L1" if clf == "fc"
                   else f"MLP {MLP_HIDDEN}→ReLU→Dropout({MLP_DROPOUT})→1 + L1"),
                "=" * 76,
                "test 评估（每轮种子见 selection/test_eval.csv）：",
                "  repeat seed      α*     τ*   宏F1   加权F1   ACC   精确宏  召回宏  有毒F1  无毒F1   AUC",
            ]
            for r in test_rows:
                lines.append(f"  {r['repeat']:>4} {r['seed']:>9} {r['alpha']:>7} {r['tau']:>6} "
                             + " ".join(f"{r[k]:.4f}" for k in METRIC_KEYS))
            lines.append("-" * 76)
            for k in METRIC_KEYS:
                vals = np.array([r[k] for r in test_rows])
                lines.append(f"  {k:<12} mean±std: {vals.mean():.4f} ± {vals.std():.4f}")
            (out_dir / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"\n[完成] v7/{tag}")

    global_bar.close()
    pd.DataFrame(summary_rows).to_csv(out_root / "summary_all.csv", index=False)

    # ---- 单线汇总文件：三处理臂 × {FC, MLP}，全部 test 指标（mean±std）----
    labels = ["宏F1", "加权F1", "ACC", "精确宏", "召回宏", "有毒F1", "无毒F1", "AUC"]
    lines = [
        "=" * 122,
        f"flat3-{SET_TAG} 单线汇总（test，{args.n_repeats} 轮随机种子，mean±std）",
        "协议v2: 9:1分层划分(每轮种子见 summary_all.csv) / α+τ@[0.30,0.70]验证子集筛选(宏F1) / "
        "早停patience=20 / test仅评一次",
        "FC = Linear(d→1)+L1(AdamW lr=1e-2)  |  MLP = Linear(d→64)→ReLU→Dropout(0.2)→Linear(64→1)+L1",
        "=" * 122,
        "处理臂          分类器  " + "  ".join(f"{l:<16}" for l in labels),
    ]
    for arm in ARMS:
        for clf in CLFS:
            rows = [r for r in summary_rows if r["arm"] == arm and r["clf"] == clf]
            assert len(rows) == args.n_repeats, f"[FATAL] {arm}/{clf} 轮数不足"
            cells = "  ".join(
                f"{np.mean([r[k] for r in rows]):.4f}±{np.std([r[k] for r in rows]):.4f}"
                for k in METRIC_KEYS)
            lines.append(f"{arm:<16}{clf:<8}{cells}")
    best = max(((arm, clf) for arm in ARMS for clf in CLFS),
               key=lambda ac: np.mean([r["macro"] for r in summary_rows
                                       if r["arm"] == ac[0] and r["clf"] == ac[1]]))
    lines.append("-" * 122)
    lines.append(f"宏F1均值最高组合: {best[0]} × {'FC' if best[1] == 'fc' else 'MLP'}"
                 "（描述性标注，非预登记裁决）")
    (out_root / f"summary_{SET_TAG}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n全部完成 → {out_root}/summary_all.csv 与 summary_{SET_TAG}.txt")


if __name__ == "__main__":
    main()
