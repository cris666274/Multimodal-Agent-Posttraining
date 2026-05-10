import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "outputs/vlm_agent_eval_results.jsonl"
OUT_PATH = ROOT / "outputs/vlm_agent_eval_tagged.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tag_error(row):
    status = row.get("status")
    gold_tools = set(row.get("gold_tools", []))
    pred_tools = set(row.get("pred_tools", []))
    answer = row.get("answer", "") or ""

    if status == "parse_error":
        return "parse_error"

    if status == "empty_final_answer":
        return "empty_final_answer"

    if status == "bad_tool_args":
        return "bad_tool_args"

    if status == "bad_tool_name":
        return "bad_tool_name"

    if status == "repeated_tool_call":
        return "repeated_tool_call"

    if status == "max_steps_exceeded":
        return "max_steps_exceeded"

    if status == "unknown_parsed_type":
        return "unknown_parsed_type"

    # 该调用工具，但是一个工具都没调用
    if gold_tools and not pred_tools:
        return "missing_tool_call"

    # 不该调用工具，却调用了
    if not gold_tools and pred_tools:
        return "redundant_tool_call"

    # 调用了工具，但没有覆盖 gold tools
    if not gold_tools.issubset(pred_tools):
        return "wrong_tool_selection"

    if row.get("tool_order_hit") is False:
        return "wrong_tool_order"

    # 拒答判断
    refuse_words = [
        "无法可靠",
        "无法判断",
        "看不清",
        "信息不足",
        "不能确定",
        "无法确定",
        "无法识别",
    ]

    should_refuse = row.get("should_refuse", False)
    is_refusal = any(w in answer for w in refuse_words)

    if should_refuse and not is_refusal:
        return "should_refuse_but_answered"

    if not should_refuse and is_refusal:
        return "over_refusal"

    if row.get("answer_keyword_hit") is False:
        return "answer_keyword_miss"

    if row.get("tool_hit"):
        return "ok"

    return "unknown"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default=str(RESULT_PATH))
    parser.add_argument("--out_path", default=str(OUT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result_path)
    out_path = Path(args.out_path)

    if not result_path.is_absolute():
        result_path = ROOT / result_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if not result_path.exists():
        raise FileNotFoundError(f"找不到结果文件: {result_path}")

    rows = list(read_jsonl(result_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            row["error_tag"] = tag_error(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
