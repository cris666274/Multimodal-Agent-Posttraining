import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs/vlm_agent_eval_tagged.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"找不到文件: {PATH}")

    rows = list(read_jsonl(PATH))
    counter = Counter(row.get("error_tag", "unknown") for row in rows)

    print("========== Error Tags ==========")
    for k, v in counter.most_common():
        print(k, v)

    print("\n========== Non-OK Examples ==========")
    for row in rows:
        if row.get("error_tag") != "ok":
            print("\nID:", row.get("id"))
            print("Category:", row.get("category"))
            print("Tag:", row.get("error_tag"))
            print("Q:", row.get("question"))
            print("Gold:", row.get("gold_tools"))
            print("Pred:", row.get("pred_tools"))
            print("Status:", row.get("status"))
            print("Answer:", str(row.get("answer", ""))[:300])


if __name__ == "__main__":
    main()