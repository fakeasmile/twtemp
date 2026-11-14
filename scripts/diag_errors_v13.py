# -*- coding: utf-8 -*-
"""v13 错误深度分析（GLM 单教师）：M4 锚点预测错误样本的多层剖析。

背景：v12 裁决 Qwen 句级通道未过门（+0.0014），用户决定冻结 Qwen、
约束不放宽，回到 GLM 单教师对错误样本做最后一次深度分析。本脚本
只消费已有测量数据 + 确定性文本统计，零 GPU、零新假设，全部结论可复核。

分析层（六层）：
【1】复算校验：5 种子重算锚点 train OOF / test 预测，逐种子宏 F1
     与 v12 报告（0.8159±0.0018）对照（OpenBLAS 抖动容差 ±0.002）。
【2】错误分类：5 种子投票 → TP/TN/FN稳定/FN翻转/FP稳定/FP翻转
     （稳定 = ≥4/5 种子同错；翻转 = τ 平点边界带）。
【3】确定性文本标记：引号/冒号/转述动词/？/！/长度分箱，
     FP稳定 vs TP vs TN 对照——use/mention 结构的直接文本证据。
【4】v11 GLM 句级签名：assert 11 特征 + target 11 特征（复用
     analyze_assert_v11/analyze_target_v11 的 aggregate，口径零漂移）；
     "嵌入毒性高+断言毒性低" 签名占比。
【5】特征空间归因：块均值（条目均值/v3 概念均值/宏 T 均值/长度）+
     FP−TN 差值 top10 条目与 top10 v3 概念（泄漏面识别）。
【6】FP/FN 典型例句导出（人工复核）。

产物：results/v13/error_analysis/{report.txt, predictions_train.csv,
      predictions_test.csv, fp_examples.csv}
"""

import argparse
import faulthandler
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 看门狗：每 90s 转储全部线程调用栈——若卡住，可精确定位到行。
# 主进程写文件（终端可能不显示 stderr）；并行工作进程写 stderr。
if __name__ == "__main__":
    try:
        _wd = open("results/v13/error_analysis/watchdog.txt", "w", encoding="utf-8")
        faulthandler.dump_traceback_later(90, repeat=True, file=_wd)
    except OSError:
        faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
else:  # ProcessPoolExecutor spawn 子进程（__mp_main__）
    faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import readout_v83 as r83
from readout_v12 import assemble_m4
from analyze_assert_v11 import FEATS as AS_FEATS, aggregate as assert_aggregate
from analyze_target_v11 import FEATS as TG_FEATS, aggregate as target_aggregate

GROUPS = ["TN", "FP翻转", "FP稳定", "TP", "FN翻转", "FN稳定"]
MARKER_PAT = re.compile(
    r"(说|称|表示|指出|认为|回应|写道|评论|转发|爆料|传闻|据说|听说|有人|"
    r"网友|媒体|报道|引用|所谓|原文|神评论)")
QUOTE_CHARS = ('“', '”', '"', '「', '」', '『', '』', '‘', '’')
LEN_BINS = [(0, 20), (20, 40), (40, 100), (100, 10 ** 9)]


def text_markers(texts):
    """确定性文本统计（零测量依赖）。"""
    out = {"len": [], "quote": [], "colon": [], "report": [],
           "question": [], "exclaim": []}
    for s in texts:
        s = str(s)
        out["len"].append(len(s))
        out["quote"].append(int(any(c in s for c in QUOTE_CHARS)))
        out["colon"].append(int(("：" in s) or (":" in s)))
        out["report"].append(int(bool(MARKER_PAT.search(s))))
        out["question"].append(int("？" in s))
        out["exclaim"].append(int("！" in s))
    return pd.DataFrame(out)


def categorize(y, preds):
    """按 5 种子投票分类。preds: (n, S) 0/1。"""
    fp_votes = ((preds == 1) & (y == 0)).sum(1)
    fn_votes = ((preds == 0) & (y == 1)).sum(1)
    S = preds.shape[1]
    cats = []
    for i in range(len(y)):
        if y[i] == 1:
            if fn_votes[i] == 0:
                cats.append("TP")
            elif fn_votes[i] >= S - 1:
                cats.append("FN稳定")
            else:
                cats.append("FN翻转")
        else:
            if fp_votes[i] == 0:
                cats.append("TN")
            elif fp_votes[i] >= S - 1:
                cats.append("FP稳定")
            else:
                cats.append("FP翻转")
    return np.array(cats), fp_votes, fn_votes


def jaccard(a, b):
    i = np.intersect1d(a, b).size
    u = np.union1d(a, b).size
    return i / u if u else np.nan


def _diag_seed_worker(seed, Xtr, ytr, Xte, yte):
    """单种子复算（并行工作进程；路径与串行版逐行同构，数值零漂移）：
    cv_select（alpha 网格×5 折）→ 最优 alpha 全量重拟合 → test 概率。"""
    a, tau, o = r83.cv_select(Xtr, ytr, seed, f"diag[seed{seed}]")
    m = Pipeline([("m", LogisticRegression(penalty="l1", solver="liblinear",
                                           C=1.0 / a, max_iter=2000))]).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    f1_oof = f1_score(ytr, (o >= tau).astype(int), average="macro")
    return seed, a, float(tau), o, p, float(f1_oof)


def group_rows(F, cats, feats):
    rows = []
    for g in GROUPS:
        m = cats == g
        if m.sum() == 0:
            continue
        row = {"group": g, "n": int(m.sum())}
        for f in feats:
            row[f] = float(pd.to_numeric(F.loc[m, f], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="v13 错误深度分析")
    ap.add_argument("--train_matrix", required=True)
    ap.add_argument("--test_matrix", required=True)
    ap.add_argument("--v3_train", required=True)
    ap.add_argument("--v3_test", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--rings_train_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_train.json")
    ap.add_argument("--rings_test_json",
                    default="data/processed/TOXICN/glm-4-9b-chat_v3/rings_glm-4-9b-chat_test.json")
    ap.add_argument("--assert_train", default="results/v11/assert/assert_flat_train.csv")
    ap.add_argument("--assert_test", default="results/v11/assert/assert_flat_test.csv")
    ap.add_argument("--target_train", default="results/v11/target/target_flat_train.csv")
    ap.add_argument("--target_test", default="results/v11/target/target_flat_test.csv")
    ap.add_argument("--output_dir", default="results/v13/error_analysis")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))

    tr = pd.read_csv(args.train_matrix)
    te = pd.read_csv(args.test_matrix)
    v3tr = pd.read_csv(args.v3_train)
    v3te = pd.read_csv(args.v3_test)
    print(f"[{time.time() - T0:7.1f}s] [加载] train n={len(tr)}  test n={len(te)}  "
          f"item列={len([c for c in tr.columns if c not in ('id', 'toxic', 'content')])}",
          flush=True)
    assert (tr["id"].values == v3tr["id"].values).all()
    assert (te["id"].values == v3te["id"].values).all()
    ytr = tr["toxic"].astype(int).values
    yte = te["toxic"].astype(int).values
    ids_tr, ids_te = tr["id"].values, te["id"].values

    def expr_map(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {r["id"]: (-1 if r.get("expression") is None
                          else int(r["expression"])) for r in d["results"]}

    expr_tr = np.array([expr_map(args.rings_train_json)[i] for i in ids_tr])
    expr_te = np.array([expr_map(args.rings_test_json)[i] for i in ids_te])
    stamp("[加载] rings 表情组标签完成")

    # ---- 【1】5 种子复算 ----
    stamp("[装配] M4 特征装配中...")
    Xtr, Xte = assemble_m4(tr, te, v3tr, v3te, freeze, ytr)
    stamp(f"[装配] 完成 train {Xtr.shape} / test {Xte.shape}")
    S = len(r83.FINAL_SEEDS)
    oof = np.zeros((len(ytr), S))
    pte = np.zeros((len(yte), S))
    taus = np.zeros(S)
    n_jobs = min(S, os.cpu_count() or S)
    stamp(f"[复算] 并行启动 {n_jobs} 个工作进程（每种子独立，数值与串行版一致）")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_diag_seed_worker, s, Xtr, ytr, Xte, yte): s
                for s in r83.FINAL_SEEDS}
        done = 0
        for fut in tqdm(as_completed(futs), total=S, desc="v13 复算(并行)",
                        dynamic_ncols=True):
            seed, a, tau, o, p, f1_oof = fut.result()
            k = r83.FINAL_SEEDS.index(seed)
            oof[:, k] = o
            pte[:, k] = p
            taus[k] = tau
            done += 1
            stamp(f"[复算] seed={seed} 完成（{done}/{S}）  alpha={a}  τ={tau:.2f}  "
                  f"OOF宏F1={f1_oof:.4f}")
    pred_tr = (oof >= taus[None, :]).astype(int)
    pred_te = (pte >= taus[None, :]).astype(int)
    f1_tr = [f1_score(ytr, pred_tr[:, k], average="macro") for k in range(S)]
    f1_te = [f1_score(yte, pred_te[:, k], average="macro") for k in range(S)]

    L = ["=" * 78, "v13 错误深度分析（GLM 单教师 M4 锚点，5 种子复算）", "=" * 78,
         "【1】复算校验（对照 v12 报告 test 0.8159±0.0018，抖动容差 ±0.002）",
         f"  train OOF 宏F1/seed: " + ", ".join(f"s{s}={v:.4f}" for s, v in
                                                  zip(r83.FINAL_SEEDS, f1_tr))
         + f"  mean={np.mean(f1_tr):.4f}",
         f"  test    宏F1/seed: " + ", ".join(f"s{s}={v:.4f}" for s, v in
                                                zip(r83.FINAL_SEEDS, f1_te))
         + f"  mean={np.mean(f1_te):.4f}±{np.std(f1_te):.4f}",
         f"  τ/seed: {np.round(taus, 2).tolist()}", ""]

    # ---- 【2】错误分类 ----
    stamp("[分析] 【2】错误分类...")
    cat_tr, fpv_tr, fnv_tr = categorize(ytr, pred_tr)
    cat_te, fpv_te, fnv_te = categorize(yte, pred_te)
    L += ["【2】错误分类（5 种子投票；稳定=≥4/5 同错）",
          "  split/group: " + ", ".join(
              f"{g}={int((cat_tr == g).sum())}" for g in GROUPS) + "   (train)",
          "               " + ", ".join(
              f"{g}={int((cat_te == g).sum())}" for g in GROUPS) + "   (test)"]
    fp_sets_tr = [set(ids_tr[np.where((pred_tr[:, j] == 1) & (ytr == 0))[0]])
                  for j in range(S)]
    fp_sets_te = [set(ids_te[np.where((pred_te[:, j] == 1) & (yte == 0))[0]])
                  for j in range(S)]
    L += ["  FP 集合 5 种子 Jaccard（相邻种子对）: train "
          + ", ".join(f"{jaccard(fp_sets_tr[j], fp_sets_tr[j + 1]):.3f}"
                      for j in range(S - 1))
          + " | test " + ", ".join(f"{jaccard(fp_sets_te[j], fp_sets_te[j + 1]):.3f}"
                                   for j in range(S - 1)), ""]

    # ---- 【3】确定性文本标记 ----
    M_tr, M_te = text_markers(tr["content"].values), text_markers(te["content"].values)
    L += ["【3】确定性文本标记（均值率；FP稳定 vs TP vs TN）"]
    for split, M, cats in (("train", M_tr, cat_tr), ("test", M_te, cat_te)):
        L.append(f"  [{split}]")
        mk = ["quote", "colon", "report", "question", "exclaim", "len"]
        hdr = "    group      n   " + "  ".join(f"{c:>7}" for c in mk)
        L.append(hdr)
        for g in ["TN", "FP稳定", "FP翻转", "TP"]:
            m = cats == g
            if m.sum() == 0:
                continue
            L.append(f"    {g:<9}{int(m.sum()):<5} "
                     + "  ".join(f"{M.loc[m, c].mean():>7.3f}" for c in mk))
        L.append("    长度分箱无毒FP率（bin: <20 / 20-40 / 40-100 / >100）:")
        lens = M["len"].values
        for lo, hi in LEN_BINS:
            mlen = (lens >= lo) & (lens < hi)
            benign = (ytr if split == "train" else yte) == 0
            mm = mlen & benign
            fpb = (mm & (cats == "FP稳定")) | (mm & (cats == "FP翻转"))
            n_bin = int(mm.sum())
            L.append(f"      [{lo:>3},{hi if hi < 10**8 else 'inf'}): n={n_bin:<5} "
                     f"FP率={fpb.sum() / max(n_bin, 1):.3f}")
        L.append("")

    # ---- 【4】v11 GLM 句级签名 ----
    stamp("[分析] 【4】句级签名聚合...")
    AS_tr = assert_aggregate(pd.read_csv(args.assert_train), ids_tr)
    AS_te = assert_aggregate(pd.read_csv(args.assert_test), ids_te)
    TG_tr = target_aggregate(pd.read_csv(args.target_train), ids_tr)[TG_FEATS]
    TG_te = target_aggregate(pd.read_csv(args.target_test), ids_te)[TG_FEATS]
    L += ["【4】v11 GLM 句级签名（组均值；断言=教师句级测量口径）"]
    for split, F, cats in (("train", AS_tr, cat_tr), ("test", AS_te, cat_te)):
        tb = group_rows(F.reset_index(drop=True), cats, AS_FEATS)
        L.append(f"  [{split}] assert 特征:")
        for f in AS_FEATS:
            L.append(f"    {f:<22}" + "  ".join(
                f"{r['group']}={r[f]:.3f}" for _, r in tb.iterrows()))
        sig = ((F["p_tox_max_embedded"] >= 0.5)
               & (F["p_tox_max_asserted"] < 0.3)).astype(float)
        L.append("    嵌入高+断言低签名占比: " + "  ".join(
            f"{g}={sig[cats == g].mean():.3f}"
            for g in ["TN", "FP稳定", "FP翻转", "TP"] if (cats == g).any()))
    for split, F, cats in (("train", TG_tr, cat_tr), ("test", TG_te, cat_te)):
        tb = group_rows(F.reset_index(drop=True), cats, TG_FEATS)
        L.append(f"  [{split}] target 特征:")
        for f in TG_FEATS:
            L.append(f"    {f:<22}" + "  ".join(
                f"{r['group']}={r[f]:.3f}" for _, r in tb.iterrows()))
    L.append("")

    # ---- 【5】特征空间归因 ----
    stamp("[分析] 【5】特征空间归因...")
    item_cols = [c for c in tr.columns if c not in ("id", "toxic", "content")]
    Atr = tr[item_cols].copy()
    Ate = te[item_cols].copy()
    for c in item_cols:
        if freeze["item_polarity"].get(c, 1) == -1:
            Atr[c] = 1.0 - Atr[c]
            Ate[c] = 1.0 - Ate[c]
    kept_items = [c for lst in freeze["items_per_construct"].values() for c in lst
                  if c in Atr.columns]
    v3_concept_cols = [c for c in v3tr.columns
                       if c.startswith(("v3p_", "v3pd_", "v3d_"))]
    mac_tr = r83.construct_raw(tr, freeze, Atr, tr[item_cols])
    mac_te = r83.construct_raw(te, freeze, Ate, te[item_cols])
    mac_cols = [c for c in mac_tr.columns if c != "id"]
    L += ["【5】特征空间归因（FP稳定 vs TN 均值差 top10）"]
    for split, A, Av3, cats, mac, lab in (
            ("train", Atr, v3tr, cat_tr, mac_tr, "FP稳定"),
            ("test", Ate, v3te, cat_te, mac_te, "FP稳定")):
        fp = cats == lab
        tn = cats == "TN"
        d_item = A.loc[fp, kept_items].mean() - A.loc[tn, kept_items].mean()
        d_v3 = Av3.loc[fp, v3_concept_cols].mean() - Av3.loc[tn, v3_concept_cols].mean()
        d_mac = mac.loc[fp, mac_cols].mean() - mac.loc[tn, mac_cols].mean()
        top_i = d_item.sort_values(ascending=False).head(10)
        top_v = d_v3.sort_values(ascending=False).head(10)
        top_m = d_mac.sort_values(ascending=False).head(10)
        L.append(f"  [{split}] 条目 top10（差值）：")
        L += [f"    {k}  Δ={v:+.3f}" for k, v in top_i.items()]
        L.append(f"  [{split}] v3 概念 top10（差值）：")
        L += [f"    {k}  Δ={v:+.3f}" for k, v in top_v.items()]
        L.append(f"  [{split}] 宏 T 块 top10（差值）：")
        L += [f"    {k}  Δ={v:+.3f}" for k, v in top_m.items()]
        L.append(f"  [{split}] 块均值: 条目均值 FP={A.loc[fp, kept_items].mean(1).mean():.3f}"
                 f" TN={A.loc[tn, kept_items].mean(1).mean():.3f} | "
                 f"v3均值 FP={Av3.loc[fp, v3_concept_cols].mean(1).mean():.3f}"
                 f" TN={Av3.loc[tn, v3_concept_cols].mean(1).mean():.3f} | "
                 f"宏T均值 FP={mac.loc[fp, mac_cols].mean(1).mean():.3f}"
                 f" TN={mac.loc[tn, mac_cols].mean(1).mean():.3f}")
    L.append("")

    # ---- 【6】典型例句导出 ----
    stamp("[导出] 生成例句与预测明细...")
    flat_te = pd.read_csv(args.assert_test)
    flat_te["p_tox"] = pd.to_numeric(flat_te["p_tox"], errors="coerce")
    fp_ids = set(ids_te[cat_te == "FP稳定"])
    emb = flat_te[(flat_te["id"].isin(fp_ids))
                  & (flat_te["status_argmax"] != "断言")].copy()
    emb = emb.sort_values("p_tox", ascending=False).groupby("id").head(1)
    content_te = dict(zip(ids_te, te["content"].astype(str)))
    pmean_te = dict(zip(ids_te, pte.mean(1)))
    ex = emb.head(20).copy()
    ex["content"] = ex["id"].map(content_te)
    ex["p_model"] = ex["id"].map(pmean_te)
    ex["expr"] = ex["id"].map(dict(zip(ids_te, expr_te)))
    ex = ex[["id", "p_model", "expr", "p_tox", "status_argmax", "sent", "content"]]
    ex.to_csv(out / "fp_examples.csv", index=False, encoding="utf-8-sig")

    fn_ids = set(ids_te[cat_te == "FN稳定"])
    fn_sents = flat_te[flat_te["id"].isin(fn_ids)]
    L += ["【6】FP 典型例句：top20（按嵌入句毒性降序）→ fp_examples.csv"
          f"（覆盖 {ex['id'].nunique()} 个稳定 FP）",
          f"  稳定 FN（test n={len(fn_ids)}）句级状态分布: "
          + ", ".join(f"{k}={fn_sents['status_argmax'].eq(k).mean():.3f}"
                      for k in ("断言", "引用", "假设", "反问")),
          f"  稳定 FN 文本 expression 分布: "
          + ", ".join(f"e{k}={int((expr_te[np.isin(ids_te, list(fn_ids))] == k).sum())}"
                      for k in (0, 1, 2, 3)),
          "",
          f"预测明细已保存: predictions_train.csv / predictions_test.csv"]

    pd.DataFrame({"id": ids_tr, "toxic": ytr, "expr": expr_tr,
                  "cat": cat_tr, "fp_votes": fpv_tr, "fn_votes": fnv_tr,
                  **{f"oof_s{s}": oof[:, j] for j, s in enumerate(r83.FINAL_SEEDS)},
                  **{f"pred_s{s}": pred_tr[:, j] for j, s in enumerate(r83.FINAL_SEEDS)}}).to_csv(
        out / "predictions_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": ids_te, "toxic": yte, "expr": expr_te,
                  "cat": cat_te, "fp_votes": fpv_te, "fn_votes": fnv_te,
                  "p_mean": pte.mean(1), "content": te["content"].astype(str).values,
                  **{f"p_s{s}": pte[:, j] for j, s in enumerate(r83.FINAL_SEEDS)}}).to_csv(
        out / "predictions_test.csv", index=False, encoding="utf-8-sig")

    report = "\n".join(L)
    faulthandler.cancel_dump_traceback_later()
    stamp("[完成] 全部六层分析结束，报告如下。")
    print(report)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
