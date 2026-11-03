"""构建 CCC 验证数据集：从 TOXICN 与 COLD 的 test 划分各随机抽取 50 条，共 100 条。

输出：data/raw/CCC/ccc.json
每条记录字段：
  id        来源_原始下标（如 TOXICN_123 / COLD_456）
  source    TOXICN | COLD
  content   文本
  toxic     0/1 标签
  topic     主题（两数据集均有）
  expression / toxic_type / target  仅 TOXICN 记录携带

使用示例：
  python scripts/sample_ccc.py
  python scripts/sample_ccc.py --seed 7 --per_source 50
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="抽样构建 CCC 验证数据集")
    parser.add_argument("--per_source", type=int, default=50, help="每个数据集抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    records = []
    stats = {}

    # TOXICN
    toxicn = load_json(project_root / "data" / "raw" / "TOXICN" / "test.json")
    idxs = rng.sample(range(len(toxicn)), args.per_source)
    for i in idxs:
        s = toxicn[i]
        records.append({
            "id": f"TOXICN_{i}",
            "source": "TOXICN",
            "content": s["content"],
            "toxic": s["toxic"],
            "topic": s.get("topic"),
            "expression": s.get("expression"),
            "toxic_type": s.get("toxic_type"),
            "target": s.get("target"),
        })
    stats["TOXICN"] = {
        "toxic": dict(Counter(r["toxic"] for r in records if r["source"] == "TOXICN")),
        "expression": dict(Counter(r["expression"] for r in records if r["source"] == "TOXICN")),
    }

    # COLD
    cold = load_json(project_root / "data" / "raw" / "COLD" / "test.json")
    idxs = rng.sample(range(len(cold)), args.per_source)
    for i in idxs:
        s = cold[i]
        records.append({
            "id": f"COLD_{i}",
            "source": "COLD",
            "content": s["content"],
            "toxic": s["toxic"],
            "topic": s.get("topic"),
        })
    stats["COLD"] = {
        "toxic": dict(Counter(r["toxic"] for r in records if r["source"] == "COLD")),
    }

    out_dir = project_root / "data" / "raw" / "CCC"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ccc.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"已抽样 {len(records)} 条（seed={args.seed}），保存到: {out_path}")
    for src, st in stats.items():
        print(f"  {src}: toxic={st['toxic']}" + (f", expression={st.get('expression')}" if "expression" in st else ""))


if __name__ == "__main__":
    main()
