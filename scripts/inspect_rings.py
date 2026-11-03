"""rings v1.0 过触发样本抽查（只读结果 JSON，不加载模型）。

目的：在出 v1.1 措辞修正前，用实际文本确认三个疑似过触发的形态，
避免凭印象改提示词：
  GLM   C5b（假设/虚构 guard，命中 58.5%）
  GLM   C5c（元讨论 guard，命中 47.5%）
  GLM   C6a（驳斥方向 guard，命中 58.5%）
  GLM   U4a（委婉包装，命中 96.0%）
  GLM   C1 且 toxic=0（对应偏差：toxic0 也命中 93.5%，看这些文本里是否真有贬损）
  E6a   两模型的 p(是) top10（Qwen E6a p均 0.008 死维度，对照 GLM 0.522 看缺的是什么）

用法：
  python scripts/inspect_rings.py data/processed/CCC/glm-4-9b-chat data/processed/CCC/Qwen3.5-9B
每项输出 = p(是) 最高的若干条（模型最有把握的，最能暴露误判形态）
         + 随机若干条（seed=7，可复现，看典型情况）。
"""

import argparse
import json
import random
from pathlib import Path

SEED = 7
N_TOP = 6
N_RAND = 6
N_TOP_E6A = 10
N_C1_TOXIC0 = 8


def find_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(x for x in p.glob("rings_*.json") if x.is_file())
        elif p.is_file():
            files.append(p)
        else:
            print(f"[跳过] 路径不存在: {p}")
    if not files:
        raise SystemExit("未找到 rings_*.json 结果文件")
    return files


def show(x, cid=None):
    p = f" p={x['concepts'][cid]['stage1']['p_yes']:.3f}" if cid else ""
    v = f" value={x['concepts'][cid]['value']}" if cid else ""
    print(f"  id={x['id']} toxic={x['toxic']} expr={x.get('expression')}{p}{v}")
    print(f"    {x['content']}")


def dump_hits(d, cid, tag):
    """某概念命中（value>=1）样本：p(是) top-N + 随机 N。"""
    results = d["results"]
    hits = [x for x in results if x["concepts"].get(cid, {}).get("value", 0) >= 1]
    print(f"\n--- [{tag}] {cid} 命中 {len(hits)}/{len(results)} ---")
    if not hits:
        print("  无命中")
        return
    print(f"  [p(是) top{N_TOP}]")
    for x in sorted(hits, key=lambda x: x["concepts"][cid]["stage1"]["p_yes"], reverse=True)[:N_TOP]:
        show(x, cid)
    rng = random.Random(SEED)
    print(f"  [随机 {N_RAND} 条，seed={SEED}]")
    for x in rng.sample(hits, min(N_RAND, len(hits))):
        show(x, cid)


def dump_e6a_top(d, tag):
    """E6a 全部文本按 p(是) 排序取 top-N（不分命中与否，看两模型差距到底在哪）。"""
    results = sorted(d["results"],
                     key=lambda x: x["concepts"]["E6a"]["stage1"]["p_yes"], reverse=True)
    print(f"\n--- [{tag}] E6a p(是) top{N_TOP_E6A}（命中口径见分析报告）---")
    for x in results[:N_TOP_E6A]:
        show(x, "E6a")


def dump_c1_toxic0(d, tag):
    """C1 命中且 toxic=0：对应偏差的直接证据（这些文本里是否真有贬损断言）。"""
    results = d["results"]
    hits = [x for x in results
            if x["toxic"] == 0 and x["concepts"].get("C1", {}).get("value", 0) >= 1]
    print(f"\n--- [{tag}] C1 命中且 toxic=0 共 {len(hits)} 条 ---")
    print(f"  [p(是) top{N_TOP}]")
    for x in sorted(hits, key=lambda x: x["concepts"]["C1"]["stage1"]["p_yes"], reverse=True)[:N_TOP]:
        show(x, "C1")
    rng = random.Random(SEED)
    print(f"  [随机 {N_C1_TOXIC0} 条，seed={SEED}]")
    for x in rng.sample(hits, min(N_C1_TOXIC0, len(hits))):
        show(x, "C1")


def main():
    parser = argparse.ArgumentParser(description="rings v1.0 过触发样本抽查（只读）")
    parser.add_argument("paths", nargs="+", help="结果目录或 rings_*.json 文件")
    args = parser.parse_args()

    for p in find_files(args.paths):
        d = json.load(open(p, "r", encoding="utf-8"))
        tag = d["meta"]["model"]
        print("\n" + "=" * 72)
        print(f"模型 {tag} | 样本 {len(d['results'])} | 版本 {d['meta'].get('prompt_version')}")
        print("=" * 72)
        dump_hits(d, "C5b", "guard 过触发疑似")
        dump_hits(d, "C5c", "guard 过触发疑似")
        dump_hits(d, "C6a", "guard 过触发疑似")
        dump_hits(d, "U4a", "evidence 过触发疑似")
        dump_c1_toxic0(d, "对应偏差检查")
        dump_e6a_top(d, "E6a 死维度检查")


if __name__ == "__main__":
    main()
