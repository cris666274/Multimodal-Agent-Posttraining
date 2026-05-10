import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data/sft/sft_agent_train_v3.jsonl"
DEFAULT_CHARTQA_INPUT = ROOT / "data/sft/sft_chartqa.jsonl"
DEFAULT_OUT = ROOT / "data/sft/sft_agent_train_real_v2.jsonl"


def read_jsonl(path):
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {line_no} JSON parse failed: {e}") from e
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row, source):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"{source}: invalid messages"
    if messages[0].get("role") != "user":
        return False, f"{source}: first message is not user"
    if not any(message.get("role") == "assistant" for message in messages):
        return False, f"{source}: missing assistant message"
    return True, ""


def uses_tool(row, tool_name):
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if f'"name":"{tool_name}"' in content.replace(" ", ""):
            return True
    return False


def is_chartqa_python_exec_row(row):
    metadata = row.get("metadata", {})
    if metadata.get("used_python_exec") is True:
        return True
    return uses_tool(row, "python_exec")


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input", default=str(DEFAULT_BASE_INPUT))
    parser.add_argument("--chartqa_input", default=str(DEFAULT_CHARTQA_INPUT))
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--chartqa_repeat",
        type=int,
        default=1,
        help="Repeat ChartQA python_exec rows. Keep this small to avoid washing out local tool policy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_input = resolve_path(args.base_input)
    chartqa_input = resolve_path(args.chartqa_input)
    out_path = resolve_path(args.out_path)

    base_rows = read_jsonl(base_input)
    chartqa_rows = [
        row
        for row in read_jsonl(chartqa_input)
        if is_chartqa_python_exec_row(row)
    ]

    merged = []
    skipped = []

    for source, rows, repeat in [
        (base_input, base_rows, 1),
        (chartqa_input, chartqa_rows, args.chartqa_repeat),
    ]:
        print(f"Input: {source} ({len(rows)} selected rows, repeat={repeat})")
        for repeat_index in range(repeat):
            for row in rows:
                ok, reason = validate_row(row, source)
                if not ok:
                    skipped.append(reason)
                    continue

                if repeat_index:
                    row = dict(row)
                    row["id"] = f"{row.get('id')}_repeat_{repeat_index}"

                merged.append(row)

    write_jsonl(out_path, merged)

    print(f"Output: {out_path} ({len(merged)} rows)")
    print(f"Skipped: {len(skipped)}")
    for item in skipped[:10]:
        print("  ", item)


if __name__ == "__main__":
    main()
