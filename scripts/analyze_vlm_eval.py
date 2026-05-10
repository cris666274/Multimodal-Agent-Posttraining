import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "outputs/vlm_agent_eval_results.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_subsequence(expected, actual):
    if not expected:
        return True

    cursor = 0
    for item in actual:
        if item == expected[cursor]:
            cursor += 1
            if cursor == len(expected):
                return True
    return False


def is_refusal_answer(answer):
    refuse_words = [
        "无法可靠",
        "无法判断",
        "看不清",
        "信息不足",
        "不能确定",
        "无法确定",
        "无法识别",
    ]
    return any(word in (answer or "") for word in refuse_words)


def has_answer_keywords(answer, keywords):
    if not keywords:
        return True
    text = answer or ""
    return all(keyword in text for keyword in keywords)


def get_metric(row, name):
    if name in row:
        return bool(row.get(name))

    if name == "format_valid":
        return row.get("status") != "parse_error"

    if name == "tool_hit":
        return set(row.get("gold_tools", [])).issubset(set(row.get("pred_tools", [])))

    if name == "tool_order_hit":
        return is_subsequence(row.get("gold_tools", []), row.get("pred_tools", []))

    if name == "answer_keyword_hit":
        return has_answer_keywords(
            row.get("answer", ""),
            row.get("gold_answer_keywords", []),
        )

    if name == "refusal_correct":
        return is_refusal_answer(row.get("answer", "")) == row.get("should_refuse", False)

    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default=str(RESULT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result_path)
    if not result_path.is_absolute():
        result_path = ROOT / result_path

    if not result_path.exists():
        raise FileNotFoundError(f"找不到结果文件: {result_path}")

    rows = list(read_jsonl(result_path))

    total = len(rows)
    status_counter = Counter()
    category_counter = Counter()
    metric_names = [
        "format_valid",
        "tool_hit",
        "tool_order_hit",
        "answer_keyword_hit",
        "refusal_correct",
    ]
    metric_counter = Counter()
    pred_tool_counter = Counter()
    first_tool_counter = Counter()
    gold_tool_counter = Counter()
    error_cases = []

    for row in rows:
        status_counter[row.get("status", "unknown")] += 1
        category_counter[row.get("category", "unknown")] += 1

        for name in metric_names:
            if get_metric(row, name):
                metric_counter[name] += 1

        if not get_metric(row, "tool_hit"):
            error_cases.append(row)

        pred_tools = row.get("pred_tools", [])
        if pred_tools:
            first_tool_counter[pred_tools[0]] += 1
        else:
            first_tool_counter["<none>"] += 1

        for t in pred_tools:
            pred_tool_counter[t] += 1

        for t in row.get("gold_tools", []):
            gold_tool_counter[t] += 1

    print("========== Overall ==========")
    print("Total:", total)
    for name in metric_names:
        hit = metric_counter[name]
        if total:
            print(f"{name}: {hit}/{total} = {hit / total:.2%}")
        else:
            print(f"{name}: 0")

    print("\n========== Status ==========")
    for k, v in status_counter.most_common():
        print(k, v)

    print("\n========== Category ==========")
    for k, v in category_counter.most_common():
        print(k, v)

    print("\n========== Gold Tools ==========")
    for k, v in gold_tool_counter.most_common():
        print(k, v)

    print("\n========== Pred Tools ==========")
    for k, v in pred_tool_counter.most_common():
        print(k, v)

    print("\n========== First Pred Tool ==========")
    for k, v in first_tool_counter.most_common():
        print(k, v)

    print("\n========== First 10 Error Cases ==========")
    for row in error_cases[:10]:
        print("\nID:", row.get("id"))
        print("Category:", row.get("category"))
        print("Q:", row.get("question"))
        print("Gold:", row.get("gold_tools"))
        print("Pred:", row.get("pred_tools"))
        print("Status:", row.get("status"))
        print("Answer:", row.get("answer"))


if __name__ == "__main__":
    main()
