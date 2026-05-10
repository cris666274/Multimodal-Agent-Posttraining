import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rule_agent import run_rule_agent

EVAL_PATH = ROOT / "data/eval/eval_dev.jsonl"
OUT_PATH = ROOT / "outputs/rule_agent_eval_results.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_tools_from_trace(trace):
    return [step["name"] for step in trace if step.get("type") == "tool_call"]

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    tool_hit = 0

    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_PATH}")

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for sample in read_jsonl(EVAL_PATH):
            total += 1

            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")

            result = run_rule_agent(image=image, question=question)

            pred_tools = get_tools_from_trace(result["trace"])
            gold_tools = sample.get("gold_tools", [])

            is_tool_hit = set(gold_tools).issubset(set(pred_tools))
            tool_hit += int(is_tool_hit)

            row = {
                "id": sample.get("id"),
                "image": image,
                "question": question,
                "gold_tools": gold_tools,
                "pred_tools": pred_tools,
                "tool_hit": is_tool_hit,
                "answer": result["answer"],
                "trace": result["trace"]
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    hit_rate = tool_hit / total if total else 0.0
    print(f"Total: {total}")
    print(f"Tool hit: {tool_hit}/{total} = {hit_rate:.2%}")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
