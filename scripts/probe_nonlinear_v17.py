# -*- coding: utf-8 -*-
"""v17 臂①：隐状态非线性探针——"h 的信息是否以非线性形态存在"的最后裁决。

背景（v15/v15b/v16 证据链）：h 的所有线性读出封顶 0.863（bare_mean），
显著低于口头集成矩阵 M4 fresh 0.8906。但 h 的非线性读出从未测试（v14b
封死的是口头矩阵的非线性门缝，不是 h 的）。本脚本是 h 路线关闭前的
最后检验：GBDT（sklearn HistGradientBoosting）与 MLP（torch，CUDA 优先）
两种非线性探针 × 三臂 h，与同折线性孪生（L2-LR）配对比较。

【预登记决策规则（train 全量，test 零参与）】
  参照 = M4 fresh OOF AUC（3 种子，本脚本内重算）
  阶段1：seed42 扫描 3 臂 h × {线性孪生, GBDT, MLP}
  阶段2：最佳非线性组合补 seed43/44，同臂线性孪生同步补齐 → 逐种子配对
  判读：
    best_nonlinear ≥ M4_ref − 0.002 → h 信息存在但为非线性形态，
        线性读出浪费了它，SAE 臂值得投入；
    否则 → h 路线最终关闭（非线性也无更多可读信息）。
  MLP 早停用折内 10% 分层验证集（只用折内标签）；GBDT 内部早停同理。
  两条非线性路径无跨求解器配对问题——配对只在"同 h 同折的非线性 vs
  线性孪生"之间进行。

产物：results/v17/nonlinear/{nonlinear_report.txt, nonlinear_cv.csv}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import readout_v83 as r83  # noqa: F401 —— 基础工具，保持与 v16 同构
from readout_v12 import assemble_m4

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

T0 = time.time()
SEEDS = (42, 43, 44)
GATE = -0.002  # 相对 M4 参照的可接受差距


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def linear_probe(Xtr, ytr, Xva):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1], \
        int(np.ravel(m.n_iter_)[0])


def gbdt_probe(Xtr, ytr, Xva, seed):
    m = HistGradientBoostingClassifier(
        max_iter=400, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], int(np.ravel(m.n_iter_)[0])


def _mlp_fit(Xtr_s, ytr, Xva_s, seed, device):
    import torch
    torch.manual_seed(seed)
    idx = np.arange(len(ytr))
    itr, iva = train_test_split(idx, test_size=0.1, stratify=ytr,
                                random_state=seed)
    dev = device if device is not None else torch.device("cpu")
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                                device=dev)
    Xt, yt = to(Xtr_s), to(ytr.astype(np.float32))
    Xit, yit = Xt[itr], yt[itr]
    Xival, yival = Xt[iva], yt[iva]
    net = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], 512), torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 128), torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best_auc, best_state, patience = -1.0, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(yit)
    for _epoch in range(150):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            loss = bce(net(Xit[b]).squeeze(-1), yit[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xival)).squeeze(-1).cpu().numpy()
        auc = roc_auc_score(ytr[iva], pv)
        if auc > best_auc + 1e-4:
            best_auc, patience = auc, 0
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(to(Xva_s))).squeeze(-1).cpu().numpy()
    return pv, best_auc  # fit_stat = 内部验证最优 AUC


def mlp_probe(Xtr, ytr, Xva, seed, device):
    sc = StandardScaler().fit(Xtr)
    return _mlp_fit(sc.transform(Xtr), ytr, sc.transform(Xva), seed, device)


PROBES = {"linear": lambda X, y, Xv, s, d: linear_probe(X, y, Xv),
          "gbdt": lambda X, y, Xv, s, d: gbdt_probe(X, y, Xv, s),
          "mlp": lambda X, y, Xv, s, d: mlp_probe(X, y, Xv, s, d)}


def run_folds(X, pk, seed, folds, y, device):
    oof = np.zeros(len(y))
    stats = []
    for trn, va in folds:
        p, stat = PROBES[pk](X[trn], y[trn], X[va], seed, device)
        oof[va] = p
        stats.append(stat)
    return oof, roc_auc_score(y, oof), float(np.mean(stats))


def main():
    ap = argparse.ArgumentParser(description="v17 臂① 隐状态非线性探针")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--hidden_dir", default="results/v15/hidden")
    ap.add_argument("--output_dir", default="results/v17/nonlinear")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp("[加载] 矩阵...")
    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    y = tr["toxic"].astype(int).values
    ids = np.array([str(i) for i in tr["id"].values])

    z = np.load(Path(args.hidden_dir) / "hidden_train.npz")
    assert (z["ids"] == ids).all(), "hidden npz id 不对齐"
    ARMS = {k: z[k].astype(np.float32) for k in
            ("bare_mean", "bare_last", "chat_last")}
    for k, X in ARMS.items():
        assert np.isfinite(X).all()
        stamp(f"[臂] {k} dim={X.shape[1]}")

    stamp("[参照] M4 fresh 重算（3 种子线性探针）...")
    Xtr_m4, _ = assemble_m4(tr, te, v3tr, v3te, freeze, y)
    assert len(Xtr_m4) == len(y) and np.isfinite(Xtr_m4).all()
    folds_by_seed = {s: list(StratifiedKFold(5, shuffle=True,
                                             random_state=s).split(Xtr_m4, y))
                     for s in SEEDS}
    ref = []
    for s in SEEDS:
        _, a, _ = run_folds(Xtr_m4, "linear", s, folds_by_seed[s], y, None)
        ref.append(a)
    m4_ref = float(np.mean(ref))
    stamp(f"[参照] M4 OOF AUC = {m4_ref:.4f} ± {np.std(ref):.4f}")

    device = None
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
    except Exception:  # noqa: BLE001
        pass
    stamp(f"[执行] MLP: {'GPU' if device is not None else 'CPU torch'}；"
          f"GBDT/线性: CPU sklearn")

    # ---- 阶段1：seed42 扫描 3 臂 × 3 探针 ----
    rows = []
    nl_aucs = {}
    twin_aucs = {}
    for name, pk in tqdm([(n, p) for n in ARMS
                          for p in ("linear", "gbdt", "mlp")],
                         desc="阶段1[seed42扫描]", dynamic_ncols=True):
        _, auc, stat = run_folds(ARMS[name], pk, 42, folds_by_seed[42], y,
                                 device)
        rows.append({"stage": 1, "arm": name, "probe": pk, "seed": 42,
                     "oof_auc": auc, "fit_stat": stat})
        (twin_aucs if pk == "linear" else nl_aucs)[(name, pk)] = auc
        stamp(f"  {name:<10} {pk:<7} AUC={auc:.4f}")

    s1 = pd.DataFrame(rows)
    best_key = max(nl_aucs, key=nl_aucs.get)
    best_arm, best_probe = best_key

    # ---- 阶段2：最佳非线性组合 + 同臂线性孪生 补 seed43/44 ----
    stamp(f"[阶段2] {best_arm}/{best_probe} 及线性孪生补种子 43/44...")
    nl_by_seed = {42: nl_aucs[best_key]}
    tw_by_seed = {42: twin_aucs[(best_arm, "linear")]}
    for seed in (43, 44):
        _, a_nl, stat = run_folds(ARMS[best_arm], best_probe, seed,
                                  folds_by_seed[seed], y, device)
        _, a_tw, _ = run_folds(ARMS[best_arm], "linear", seed,
                               folds_by_seed[seed], y, None)
        nl_by_seed[seed] = a_nl
        tw_by_seed[seed] = a_tw
        rows.append({"stage": 2, "arm": best_arm, "probe": best_probe,
                     "seed": seed, "oof_auc": a_nl, "fit_stat": stat})
        rows.append({"stage": 2, "arm": best_arm, "probe": "linear",
                     "seed": seed, "oof_auc": a_tw, "fit_stat": np.nan})
        stamp(f"  seed{seed} 非线性={a_nl:.4f} 线性孪生={a_tw:.4f}")

    nl_mean = float(np.mean([nl_by_seed[s] for s in SEEDS]))
    tw_mean = float(np.mean([tw_by_seed[s] for s in SEEDS]))
    paired = [nl_by_seed[s] - tw_by_seed[s] for s in SEEDS]

    cv = pd.DataFrame(rows)
    cv.to_csv(out / "nonlinear_cv.csv", index=False, encoding="utf-8-sig")

    if nl_mean >= m4_ref + GATE:
        verdict = (f"h 信息以非线性形态存在（{nl_mean:.4f} ≥ M4 参照"
                   f"{m4_ref:.4f}{GATE}），线性读出浪费了它——SAE 臂值得投入")
    elif nl_mean > tw_mean + 0.002:
        verdict = ("非线性高于线性孪生但仍低于 M4 参照：h 有少量非线性信息，"
                   "不足以打开天花板，h 路线关闭")
    else:
        verdict = "h 路线最终关闭：非线性读出也无更多可读信息"

    lines = ["=" * 72, "v17 臂① 隐状态非线性探针（GBDT / MLP vs 线性孪生）",
             "=" * 72,
             f"参照 = M4 fresh OOF AUC = {m4_ref:.4f} ± {np.std(ref):.4f}"
             f"（3 种子，本脚本重算）；test 零参与",
             f"MLP: {'GPU' if device is not None else 'CPU'}；"
             f"早停=折内 10% 分层验证；", "",
             "阶段1（seed42）:", "-" * 72]
    for _, r in s1.iterrows():
        lines.append(f"  {r['arm']:<10} {r['probe']:<7} AUC={r['oof_auc']:.4f}")
    lines += ["-" * 72,
              f"最佳非线性: {best_arm}/{best_probe}（3 种子 "
              f"{nl_mean:.4f} ± {np.std([nl_by_seed[s] for s in SEEDS]):.4f}）",
              f"线性孪生（同折配对）: {tw_mean:.4f}",
              "逐种子配对差（非线性−线性）: "
              + ", ".join(f"s{s}={paired[i]:+.4f}"
                          for i, s in enumerate(SEEDS)),
              f"相对 M4 参照: {nl_mean - m4_ref:+.4f}（门槛 {GATE}）", "",
              f"判读: {verdict}"]
    report = "\n".join(lines)
    print(report)
    (out / "nonlinear_report.txt").write_text(report + "\n",
                                              encoding="utf-8")
    stamp("[完成] nonlinear_report.txt / nonlinear_cv.csv 已写出")


if __name__ == "__main__":
    main()
