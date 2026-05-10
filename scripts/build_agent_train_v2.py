import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "data/sft/sft_first_tool_train.jsonl",
    ROOT / "data/sft/sft_full_trace_train.jsonl",
    ROOT / "data/sft/sft_second_tool_retrieve_docs.jsonl",
]
DEFAULT_OUT = ROOT / "data/sft/sft_agent_train_v2.jsonl"


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[str(path) for path in DEFAULT_INPUTS],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = []
    for item in args.inputs:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        input_paths.append(path)

    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    merged = []
    skipped = []
    for path in input_paths:
        rows = read_jsonl(path)
        print(f"Input: {path} ({len(rows)} rows)")
        for row in rows:
            ok, reason = validate_row(row, path)
            if not ok:
                skipped.append(reason)
                continue
            merged.append(row)

    write_jsonl(out_path, merged)

    print(f"Output: {out_path} ({len(merged)} rows)")
    print(f"Skipped: {len(skipped)}")
    for item in skipped[:10]:
        print("  ", item)


if __name__ == "__main__":
    main()
