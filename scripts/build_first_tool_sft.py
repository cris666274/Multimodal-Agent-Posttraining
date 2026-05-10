import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/sft/sft_seed.jsonl"
OUT_PATH = ROOT / "data/sft/sft_first_tool_seed.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_first_user(messages):
    for msg in messages:
        if msg.get("role") == "user":
            return msg
    return None


def find_first_tool_call(messages):
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "<tool_call>" in content:
                return msg
    return None


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"找不到 SFT seed 文件: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for row in read_jsonl(IN_PATH):
            messages = row.get("messages", [])

            user_msg = find_first_user(messages)
            tool_msg = find_first_tool_call(messages)

            if user_msg is None or tool_msg is None:
                skipped += 1
                continue

            sample = {
                "id": row.get("id"),
                "source": "seed_first_tool",
                "messages": [
                    user_msg,
                    tool_msg,
                ],
            }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved: {count}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()