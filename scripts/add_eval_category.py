import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/eval/eval_dev.jsonl"
OUT_PATH = ROOT / "data/eval/eval_dev_with_category.jsonl"


def infer_category(question: str) -> str:
    q = question.lower()

    if "发票" in question or "字段" in question or "invoice" in q:
        return "document_validation"

    if "图表" in question or "增长" in question or "同比" in question or "百分比" in question:
        return "chart_calculation"

    if "商品" in question or "一致" in question or "描述" in question:
        return "product_consistency"

    if "品牌" in question or "logo" in q:
        return "uncertain_refusal"

    return "general_vqa"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"找不到输入文件: {IN_PATH}")

    count = 0

    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            question = row.get("question", "")
            row["category"] = row.get("category") or infer_category(question)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()