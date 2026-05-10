import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIRST_TOOL_INPUTS = [
    ROOT / "data/sft/sft_first_tool_seed.jsonl",
    ROOT / "data/sft/sft_hard_first_tool.jsonl",
]
DEFAULT_FULL_TRACE_INPUTS = [
    ROOT / "data/sft/sft_seed.jsonl",
    ROOT / "data/sft/sft_hard_full_trace.jsonl",
    ROOT / "data/sft/sft_hard_format.jsonl",
]

DEFAULT_FIRST_TOOL_OUT = ROOT / "data/sft/sft_first_tool_train.jsonl"
DEFAULT_FULL_TRACE_OUT = ROOT / "data/sft/sft_full_trace_train.jsonl"


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


def validate_row(row, source_path):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"{source_path}: missing messages"

    if messages[0].get("role") != "user":
        return False, f"{source_path}: first message is not user"

    if not any(msg.get("role") == "assistant" for msg in messages):
        return False, f"{source_path}: missing assistant message"

    return True, ""


def merge_jsonl(paths):
    merged = []
    seen_ids = set()
    skipped = []

    for path in paths:
        rows = read_jsonl(path)
        for row in rows:
            ok, reason = validate_row(row, path)
            if not ok:
                skipped.append(reason)
                continue

            row_id = row.get("id")
            if row_id and row_id in seen_ids:
                skipped.append(f"{path}: duplicate id {row_id}")
                continue

            if row_id:
                seen_ids.add(row_id)
            merged.append(row)

    return merged, skipped


def build_split(name, input_paths, output_path):
    rows, skipped = merge_jsonl(input_paths)
    write_jsonl(output_path, rows)

    print(f"\n========== {name} ==========")
    for path in input_paths:
        print(f"Input: {path} ({len(read_jsonl(path))} rows)")
    print(f"Output: {output_path} ({len(rows)} rows)")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for item in skipped[:10]:
            print("  ", item)
    else:
        print("Skipped: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_tool_out", default=str(DEFAULT_FIRST_TOOL_OUT))
    parser.add_argument("--full_trace_out", default=str(DEFAULT_FULL_TRACE_OUT))
    args = parser.parse_args()

    build_split(
        "First-tool SFT",
        DEFAULT_FIRST_TOOL_INPUTS,
        Path(args.first_tool_out),
    )
    build_split(
        "Full-trace SFT",
        DEFAULT_FULL_TRACE_INPUTS,
        Path(args.full_trace_out),
    )


if __name__ == "__main__":
    main()
