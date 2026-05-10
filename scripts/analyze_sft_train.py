import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data/sft/sft_agent_train_real_v2.jsonl"


TOOLS = ["vision_parse", "retrieve_docs", "python_exec"]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def assistant_tool_name(message):
    if message.get("role") != "assistant":
        return ""
    content = message.get("content", "")
    if not isinstance(content, str) or "<tool_call>" not in content:
        return ""

    compact = content.replace(" ", "")
    for tool in TOOLS:
        if f'"name":"{tool}"' in compact:
            return tool
    return "<unknown_tool>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path

    rows = list(read_jsonl(path))
    source_counter = Counter()
    category_counter = Counter()
    tool_counter = Counter()
    first_assistant_counter = Counter()
    assistant_turns = 0

    for row in rows:
        source_counter[row.get("source", "<none>")] += 1
        category_counter[row.get("category", "<none>")] += 1

        first_assistant_seen = False
        for message in row.get("messages", []):
            if message.get("role") != "assistant":
                continue
            assistant_turns += 1
            tool = assistant_tool_name(message)
            if tool:
                tool_counter[tool] += 1
            else:
                tool_counter["final_answer"] += 1

            if not first_assistant_seen:
                first_assistant_counter[tool or "final_answer"] += 1
                first_assistant_seen = True

    print("========== SFT Train ==========")
    print("Rows:", len(rows))
    print("Assistant turns:", assistant_turns)

    print("\n========== Sources ==========")
    for key, value in source_counter.most_common():
        print(key, value)

    print("\n========== Categories ==========")
    for key, value in category_counter.most_common():
        print(key, value)

    print("\n========== Assistant Targets ==========")
    for key, value in tool_counter.most_common():
        print(key, value)

    print("\n========== First Assistant Target ==========")
    for key, value in first_assistant_counter.most_common():
        print(key, value)


if __name__ == "__main__":
    main()
