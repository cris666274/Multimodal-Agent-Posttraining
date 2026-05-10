import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.runtime import MultimodalAgent

# 注意：这里读取带 category 的 eval 文件
EVAL_PATH = ROOT / "data/eval/eval_dev_with_category.jsonl"
OUT_PATH = ROOT / "outputs/vlm_agent_eval_results.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_tools_from_trace(trace):
    tools = []
    for step in trace:
        if "tool_name" in step:
            tools.append(step["tool_name"])
    return tools


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--adapter_name", default=None)
    parser.add_argument("--eval_path", default=str(EVAL_PATH))
    parser.add_argument("--out_path", default=str(OUT_PATH))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--use_mock", action="store_true")
    parser.add_argument(
        "--enforce_required_tools",
        type=bool,
        default=True,
        help="Hard-block final answers until task-required second tools are called. Defaults to True.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_path = Path(args.eval_path)
    out_path = Path(args.out_path)
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        raise FileNotFoundError(
            f"找不到评测文件: {eval_path}\n"
            f"请先运行: python scripts/add_eval_category.py"
        )

    model = QwenVLModel(
        model_name=args.model_name,
        adapter_name=args.adapter_name,
        max_new_tokens=args.max_new_tokens,
        use_mock=args.use_mock,
    )
    agent = MultimodalAgent(
        model=model,
        max_steps=args.max_steps,
        enforce_required_tools=args.enforce_required_tools,
    )

    total = 0
    tool_hit = 0
    tool_order_hit = 0
    finished = 0
    format_valid = 0
    answer_keyword_hit = 0
    refusal_correct = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in read_jsonl(eval_path):
            total += 1

            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")
            gold_tools = sample.get("gold_tools", [])

            print(f"\n[{total}] {sample.get('id')}")
            print("Category:", sample.get("category", "unknown"))
            print("Q:", question)

            result = agent.run(
                image=image,
                question=question,
            )

            pred_tools = get_tools_from_trace(result["trace"])
            is_tool_hit = set(gold_tools).issubset(set(pred_tools))
            is_tool_order_hit = is_subsequence(gold_tools, pred_tools)
            is_format_valid = result["status"] != "parse_error"
            is_answer_keyword_hit = has_answer_keywords(
                result["answer"],
                sample.get("gold_answer_keywords", []),
            )
            is_refusal_correct = (
                is_refusal_answer(result["answer"])
                == sample.get("should_refuse", False)
            )

            tool_hit += int(is_tool_hit)
            tool_order_hit += int(is_tool_order_hit)
            finished += int(result["status"] == "finished")
            format_valid += int(is_format_valid)
            answer_keyword_hit += int(is_answer_keyword_hit)
            refusal_correct += int(is_refusal_correct)

            row = {
                "id": sample.get("id"),
                "category": sample.get("category", "unknown"),
                "image": image,
                "question": question,
                "gold_tools": gold_tools,
                "pred_tools": pred_tools,
                "tool_hit": is_tool_hit,
                "tool_order_hit": is_tool_order_hit,
                "format_valid": is_format_valid,
                "should_refuse": sample.get("should_refuse", False),
                "refusal_correct": is_refusal_correct,
                "gold_answer_keywords": sample.get("gold_answer_keywords", []),
                "answer_keyword_hit": is_answer_keyword_hit,
                "evidence_required": sample.get("evidence_required", False),
                "status": result["status"],
                "answer": result["answer"],
                "trace": result["trace"],
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            print("Gold tools:", gold_tools)
            print("Pred tools:", pred_tools)
            print("Tool hit:", is_tool_hit)
            print("Tool order hit:", is_tool_order_hit)
            print("Format valid:", is_format_valid)
            print("Answer keyword hit:", is_answer_keyword_hit)
            print("Refusal correct:", is_refusal_correct)
            print("Status:", result["status"])
            print("Answer:", str(result["answer"])[:200])

    print("\n========== Summary ==========")
    print(f"Total: {total}")
    print(f"Finished: {finished}/{total} = {finished / total:.2%}" if total else "Finished: 0")
    print(f"Format valid: {format_valid}/{total} = {format_valid / total:.2%}" if total else "Format valid: 0")
    print(f"Tool hit: {tool_hit}/{total} = {tool_hit / total:.2%}" if total else "Tool hit: 0")
    print(f"Tool order hit: {tool_order_hit}/{total} = {tool_order_hit / total:.2%}" if total else "Tool order hit: 0")
    print(f"Answer keyword hit: {answer_keyword_hit}/{total} = {answer_keyword_hit / total:.2%}" if total else "Answer keyword hit: 0")
    print(f"Refusal correct: {refusal_correct}/{total} = {refusal_correct / total:.2%}" if total else "Refusal correct: 0")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
