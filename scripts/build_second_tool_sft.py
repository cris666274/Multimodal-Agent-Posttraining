import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "data/sft/sft_seed.jsonl",
    ROOT / "data/sft/sft_hard_full_trace.jsonl",
]
DEFAULT_OUT = ROOT / "data/sft/sft_second_tool_retrieve_docs.jsonl"


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


def assistant_calls_tool(message, tool_name):
    if message.get("role") != "assistant":
        return False
    content = message.get("content", "")
    if not isinstance(content, str):
        return False
    return (
        "<tool_call>" in content
        and f'"name":"{tool_name}"' in content.replace(" ", "")
    )


def has_previous_tool_result(messages, tool_name):
    return any(
        message.get("role") == "tool" and message.get("name") == tool_name
        for message in messages
    )


def build_second_tool_samples(rows, tool_name, repeat):
    samples = []

    for row in rows:
        messages = row.get("messages", [])
        for index, message in enumerate(messages):
            if not assistant_calls_tool(message, tool_name):
                continue

            prefix = messages[: index + 1]
            if not has_previous_tool_result(prefix, "vision_parse"):
                continue

            for repeat_index in range(repeat):
                sample = dict(row)
                sample["id"] = f"second_tool_{tool_name}_{row.get('id')}_{index}_{repeat_index}"
                sample["source"] = f"second_tool_{tool_name}"
                sample["target_tool"] = tool_name
                sample["messages"] = prefix
                samples.append(sample)

    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--tool_name", default="retrieve_docs")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat each extracted transition to upweight this action.",
    )
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

    rows = []
    for path in input_paths:
        part = read_jsonl(path)
        rows.extend(part)
        print(f"Input: {path} ({len(part)} rows)")

    samples = build_second_tool_samples(
        rows,
        tool_name=args.tool_name,
        repeat=args.repeat,
    )
    write_jsonl(out_path, samples)

    print(f"Generated {args.tool_name} second-tool samples: {len(samples)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
