# -*- coding: utf-8 -*-
"""v15 概念信息上限审查（ceiling audit）：训练集 9600 × 180 概念向量，
重复分层交叉验证（5 折 × 3 重复），嵌套协议——所有超参数与分类阈值都在
训练折内部选择（内层 3 折，τ = 内层 OOF 分数分位数网格 0.30~0.70 宏F1 最大）。

审查对象 = 纯 v7 概念向量（不含任何 meta/文本特征），回答"测量层信息上限"。

模型臂（每臂内层网格独立选参）：
  LR                线性可分性          C ∈ {0.03,0.1,0.3,1,3}
  RBF-SVM           非线性判别          C ∈ {0.5,2} × gamma ∈ {scale,0.05}
  LightGBM          高阶非线性          num_leaves ∈ {15,63}（lr=0.05, ≤600, 早停）
  CatBoost          高阶非线性          depth ∈ {6,10}（≤600, 早停）
  MLP-256           表示学习            lr ∈ {1e-3,3e-3} × wd ∈ {1e-4,1e-2}
  SparseInteract    概念交互            top40 主效应两两乘积 → f_classif top300
                                        → L1 LR, C ∈ {0.03,0.3}

输出（results/ceiling_audit_v7/）：
  ceiling_report.txt   各模型 OOF 宏F1 / OOF AUC / toxic_type 分组 / expression
                       分组 / benign FP 与 toxic FN，全部 3 重复 mean±std
  oof_predictions.csv  逐样本 OOF（repeat,fold,model,id,y_true,toxic_type,
                       expression,score,y_pred,tau）供后续误差分析
  selection.csv        每外层折每模型选中的超参与 τ

运行：D:\\miniconda3\\envs\\toxicnew\\python.exe scripts/ceiling_audit_v7.py [--smoke]
"""
import argparse
import json
import secrets
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated.*")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(r"D:\twtemp")
TRAIN_CSV = ROOT / "data/processed/TOXICN/v7_glm-4-9b-chat/items_v7_glm-4-9b-chat_train_matrix.csv"
RAW_TRAIN_JSON = ROOT / "data/raw/TOXICN/train.json"
OUT_DIR = ROOT / "results/ceiling_audit_v7"
TAU_Q = np.linspace(0.30, 0.70, 81)   # 折内 τ 候选 = 内层 OOF 分数分位数
SEED0 = 42

# ---------------------------------------------------------------- 工具


def macro_f1_at_tau(y, score, tau):
    return f1_score(y, (score >= tau).astype(int), average="macro")


def pick_tau(scores, y):
    """τ 候选 = 分数分位数（尺度无关，适用任意打分器），返回 (tau, macroF1)。"""
    taus = np.quantile(scores, TAU_Q)
    best_t, best_f = 0.5, -1.0
    for t in taus:
        f = macro_f1_at_tau(y, scores, t)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def standardize(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# ---------------------------------------------------------------- 模型臂


def fit_lr(Xtr, ytr, Xte, cfg, seed):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=cfg["C"], solver="liblinear", max_iter=3000,
                           random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def fit_svm(Xtr, ytr, Xte, cfg, seed):
    m = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", cache_size=600,
            random_state=seed)
    m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def fit_lgbm(Xtr, ytr, Xte, cfg, seed, Xev=None, yev=None):
    m = LGBMClassifier(n_estimators=600, learning_rate=0.05,
                       num_leaves=cfg["num_leaves"], min_child_samples=20,
                       subsample=0.9, colsample_bytree=0.9, random_state=seed,
                       n_jobs=-1, verbosity=-1)
    if Xev is not None:
        m.fit(Xtr, ytr, eval_set=[(Xev, yev)], eval_metric="auc",
              callbacks=[early_stopping(50, verbose=False),
                         log_evaluation(0)])
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def fit_cat(Xtr, ytr, Xte, cfg, seed, Xev=None, yev=None):
    m = CatBoostClassifier(iterations=600, learning_rate=0.05,
                           depth=cfg["depth"], loss_function="Logloss",
                           eval_metric="AUC", random_seed=seed, verbose=0,
                           allow_writing_files=False)
    if Xev is not None:
        m.fit(Xtr, ytr, eval_set=(Xev, yev), early_stopping_rounds=50)
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


class _MLP(nn.Module):
    def __init__(self, d, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                 nn.Dropout(p_drop), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xte, cfg, seed, Xev=None, yev=None):
    """GPU 全批量；早停 patience=8 / ≤40 epoch；eval 集只做早停与选参（折内）。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    model = _MLP(Xtr.shape[1]).to(dev)
    lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    has_ev = Xev is not None
    if has_ev:
        Xev_t = torch.tensor(Xev, dtype=torch.float32, device=dev)
        yev_t = torch.tensor(yev, dtype=torch.float32, device=dev)
    best_loss, best_state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(40):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if has_ev:
            model.eval()
            with torch.no_grad():
                ev_loss = lossf(model(Xev_t), yev_t).item()
            if ev_loss < best_loss - 1e-5:
                best_loss, best_ep, bad = ev_loss, ep, 0
                best_state = {k: v.detach().clone() for k, v in
                              model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
    if has_ev and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32, device=dev))
    return torch.sigmoid(out).cpu().numpy()


def fit_sparse(Xtr, ytr, Xte, cfg, seed):
    """Sparse Interaction：折内 top-m 主效应 → 两两乘积 → f_classif top-k → L1 LR。"""
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    m = cfg["m"]
    r = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1]) for j in range(Xtr.shape[1])])
    top = np.argsort(-r)[:m]
    idx = np.triu_indices(m, 1)
    P_tr = Xtr[:, idx[0]] * Xtr[:, idx[1]]
    k = min(cfg["k"], P_tr.shape[1])
    F, _ = f_classif(P_tr, ytr)
    F = np.nan_to_num(F)
    sel = np.argsort(-F)[:k]
    Xtr2 = np.hstack([Xtr[:, top], P_tr[:, sel]])
    P_te = Xte[:, idx[0]] * Xte[:, idx[1]]
    Xte2 = np.hstack([Xte[:, top], P_te[:, sel]])
    lr = LogisticRegression(C=cfg["C"], penalty="l1", solver="liblinear",
                            max_iter=3000, random_state=seed)
    lr.fit(Xtr2, ytr)
    return lr.predict_proba(Xte2)[:, 1]


MODELS = {
    "LR": {"fit": fit_lr, "cfgs": [{"C": c} for c in (0.03, 0.1, 0.3, 1, 3)]},
    "RBF-SVM": {"fit": fit_svm,
                "cfgs": [{"C": c, "gamma": g} for c in (0.5, 2)
                         for g in ("scale", 0.05)]},
    "LightGBM": {"fit": fit_lgbm, "cfgs": [{"num_leaves": n} for n in (15, 63)]},
    "CatBoost": {"fit": fit_cat, "cfgs": [{"depth": d} for d in (6, 10)]},
    "MLP-256": {"fit": fit_mlp,
                "cfgs": [{"lr": lr, "wd": wd} for lr in (1e-3, 3e-3)
                         for wd in (1e-4, 1e-2)]},
    "SparseInteract": {"fit": fit_sparse,
                       "cfgs": [{"m": 40, "k": 300, "C": c} for c in (0.03, 0.3)]},
}
# MLP / LightGBM / CatBoost 的 fit 支持 eval 早停；其余臂传 Xev=None

# ---------------------------------------------------------------- 主流程


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟：2000 样本×1 重复×3 折")
    args = ap.parse_args()
    n_repeats, n_splits = (1, 3) if args.smoke else (3, 5)

    tr = pd.read_csv(TRAIN_CSV)
    tr["id"] = tr["id"].astype(str)
    feat_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    y_all = tr["toxic"].astype(int).values

    # raw 标注体系对齐（id 后缀号 = raw train.json 行号）
    raw = json.loads(RAW_TRAIN_JSON.read_text(encoding="utf-8"))
    assert len(raw) == len(tr), f"[FATAL] raw {len(raw)} != 矩阵 {len(tr)}"
    tt, ex = [], []
    for i, c in zip(tr["id"], tr["content"]):
        rec = raw[int(str(i).rsplit("_", 1)[-1])]
        assert rec["content"] == str(c), f"[FATAL] id {i} 与 raw 行不一致"
        tt.append(rec["toxic_type"])
        ex.append(rec["expression"])
    tt, ex = np.array(tt), np.array(ex)

    X_all_raw = tr[feat_cols].values.astype(float)
    if args.smoke:
        rng = np.random.RandomState(0)
        keep = rng.permutation(len(tr))[:2000]
        X_all_raw, y_all, tt, ex = (X_all_raw[keep], y_all[keep], tt[keep], ex[keep])
        tr = tr.iloc[keep].reset_index(drop=True)
    print(f"数据：{len(tr)} 样本 × {len(feat_cols)} 概念  "
          f"(smoke={args.smoke}, {n_repeats}重复×{n_splits}折)")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=SEED0)
    oof_rows, sel_rows = [], []
    t0 = time.time()
    total = n_repeats * n_splits
    for k, (tri, tei) in enumerate(rskf.split(X_all_raw, y_all)):
        rep, fold = k // n_splits, k % n_splits
        seed = SEED0 + rep * 100 + fold
        Xtr_raw, Xte_raw = X_all_raw[tri], X_all_raw[tei]
        ytr, yte = y_all[tri], y_all[tei]
        med = np.nanmedian(Xtr_raw, axis=0)
        med = np.where(np.isnan(med), 0.5, med)
        ntr = np.where(np.isnan(Xtr_raw), med, Xtr_raw)
        nte = np.where(np.isnan(Xte_raw), med, Xte_raw)
        Xtr, Xte = standardize(ntr, nte)
        id_te = tr["id"].values[tei]

        for name, spec in MODELS.items():
            use_ev = name in ("LightGBM", "CatBoost", "MLP-256")
            inner = StratifiedKFold(3, shuffle=True, random_state=seed)
            tr_idx = list(inner.split(Xtr, ytr))
            best = None
            for ci, cfg in enumerate(spec["cfgs"]):
                scores = np.zeros(len(ytr))
                for a, b in tr_idx:
                    if use_ev:   # 早停 eval = 内层验证折本身（折内）
                        s = spec["fit"](Xtr[a], ytr[a], Xtr[b], cfg, seed,
                                        Xev=Xtr[b], yev=ytr[b])
                    else:
                        s = spec["fit"](Xtr[a], ytr[a], Xtr[b], cfg, seed)
                    scores[b] = s
                tau, f = pick_tau(scores, ytr)
                if best is None or f > best[2]:
                    best = (ci, cfg, f, tau)
            ci, cfg, inner_f, tau = best
            if use_ev:           # 最终重拟合：eval 用外层训练折内部 90/10 切分
                rs = np.random.RandomState(seed)
                a = rs.permutation(len(ytr))
                cut = int(0.9 * len(ytr))
                s = spec["fit"](Xtr[a[:cut]], ytr[a[:cut]], Xte, cfg, seed,
                                Xev=Xtr[a[cut:]], yev=ytr[a[cut:]])
            else:
                s = spec["fit"](Xtr, ytr, Xte, cfg, seed)
            pred = (s >= tau).astype(int)
            sel_rows.append({"repeat": rep, "fold": fold, "model": name,
                             "cfg_idx": ci, "cfg": json.dumps(cfg),
                             "inner_macroF1": round(inner_f, 4),
                             "tau": round(float(tau), 4)})
            for j in range(len(tei)):
                oof_rows.append({"repeat": rep, "fold": fold, "model": name,
                                 "id": id_te[j], "y_true": int(yte[j]),
                                 "toxic_type": int(tt[tei[j]]),
                                 "expression": int(ex[tei[j]]),
                                 "score": round(float(s[j]), 5),
                                 "y_pred": int(pred[j]), "tau": round(float(tau), 4)})
        print(f"[{k + 1}/{total}] rep{rep} fold{fold} 完成  "
              f"耗时 {time.time() - t0:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT_DIR / "selection.csv", index=False)

    # ---- 汇总报告 ----
    L = ["=" * 100,
         "v15 概念信息上限审查（纯 v7 概念向量 180 维，无 meta/文本特征）",
         f"协议：重复分层 {n_splits} 折 × {n_repeats} 重复；嵌套内层 3 折选超参；"
         "τ = 内层 OOF 分数分位数(0.30~0.70) 宏F1 最大 —— 全部折内选择",
         "=" * 100]

    def ms(vals):
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

    lines = []
    for name in MODELS:
        sub = oof[oof["model"] == name]
        rep_stats = {"macroF1": [], "auc": [], "fp": [], "fn": [],
                     "type0": [], "type1": [], "type2": [],
                     "e0": [], "e1": [], "e2": [], "e3": []}
        for rep in sorted(sub["repeat"].unique()):
            r = sub[sub["repeat"] == rep]
            rep_stats["macroF1"].append(
                f1_score(r["y_true"], r["y_pred"], average="macro"))
            rep_stats["auc"].append(roc_auc_score(r["y_true"], r["score"]))
            rep_stats["fp"].append((r.loc[r.y_true == 0, "y_pred"] == 1).mean())
            rep_stats["fn"].append((r.loc[r.y_true == 1, "y_pred"] == 0).mean())
            for key, col, vals in (("type0", "toxic_type", (0,)), ("type1", "toxic_type", (1,)),
                                   ("type2", "toxic_type", (2,)),
                                   ("e0", "expression", (0,)), ("e1", "expression", (1,)),
                                   ("e2", "expression", (2,)), ("e3", "expression", (3,))):
                msk = r[col].isin(vals)
                rep_stats[key].append((r.loc[msk, "y_pred"] == r.loc[msk, "y_true"]).mean())
        lines.append((name, rep_stats))

    L.append(f"\n{'模型':<16}{'OOF宏F1':>18}{'OOF AUC':>18}{'benign FP率':>16}"
             f"{'toxic FN率':>16}")
    for name, st in lines:
        L.append(f"{name:<16}{ms(st['macroF1']):>18}{ms(st['auc']):>18}"
                 f"{ms(st['fp']):>16}{ms(st['fn']):>16}")
    L.append(f"\n{'模型':<16}{'type=0 acc':>18}{'type=1 acc':>18}{'type=2 acc':>18}")
    for name, st in lines:
        L.append(f"{name:<16}{ms(st['type0']):>18}{ms(st['type1']):>18}{ms(st['type2']):>18}")
    L.append(f"\n{'模型':<16}{'expr=0 acc':>18}{'expr=1 acc':>18}"
             f"{'expr=2 acc':>18}{'expr=3 acc':>18}")
    for name, st in lines:
        L.append(f"{name:<16}{ms(st['e0']):>18}{ms(st['e1']):>18}"
                 f"{ms(st['e2']):>18}{ms(st['e3']):>18}")

    best_name, best_st = max(lines, key=lambda x: np.mean(x[1]["macroF1"]))
    L.append(f"\n信息上限结论：OOF宏F1 最高 = {best_name} "
             f"{ms(best_st['macroF1'])}（AUC {ms(best_st['auc'])}）。")
    L.append("解读：该值即 180 维概念矩阵在外层折外样本上的可达上限（含全部"
             "超参与阈值折内选择的代价）；与线性读出现值对比，差值 = 概念信息中"
             "未被线性读出利用的部分。")
    report = "\n".join(L)
    print(report)
    (OUT_DIR / "ceiling_report.txt").write_text(report, encoding="utf-8")
    print(f"\n[已写出] {OUT_DIR}\\ceiling_report.txt / oof_predictions.csv / selection.csv")


if __name__ == "__main__":
    main()
