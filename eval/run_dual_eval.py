"""
Eval script for dual-model agent: V4 planner + V7B answer writer.

Usage:
  python eval/run_dual_eval.py \
    --planner_model .../Qwen2.5-VL-3B-Instruct --planner_adapter .../lora_agent_real_v4 \
    --answer_model .../Qwen2.5-VL-7B-Instruct --answer_adapter .../lora_agent_v7b_clean \
    --eval_path data/eval/eval_dev_v3.jsonl --out_path outputs/dual_dev_eval.jsonl
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.dual_agent import DualModelAgent


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


def get_enforcement_events(trace):
    return [step["event"] for step in trace if "event" in step]


def is_refusal_answer(answer):
    refuse_words = [
        "无法可靠", "无法判断", "看不清", "信息不足",
        "不能确定", "无法确定", "无法识别",
    ]
    return any(word in (answer or "") for word in refuse_words)


def has_answer_keywords(answer, keywords):
    if not keywords:
        return True
    text = answer or ""
    for keyword in keywords:
        if keyword in text:
            continue
        try:
            val = float(keyword)
            patterns = [f"{val:.2f}", f"{val:.1f}", f"{val:.0f}", str(round(val))]
            if not any(p in text for p in set(patterns)):
                return False
        except ValueError:
            return False
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--planner_model", required=True)
    p.add_argument("--planner_adapter", required=True)
    p.add_argument("--answer_model", required=True)
    p.add_argument("--answer_adapter", required=True)
    p.add_argument("--eval_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    eval_path = Path(args.eval_path)
    out_path = Path(args.out_path)
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Planner: {args.planner_model.split('/')[-1]} + {Path(args.planner_adapter).name}")
    print(f"Answer:  {args.answer_model.split('/')[-1]} + {Path(args.answer_adapter).name}")

    planner = QwenVLModel(model_name=args.planner_model, adapter_name=args.planner_adapter, max_new_tokens=args.max_new_tokens)
    answer = QwenVLModel(model_name=args.answer_model, adapter_name=args.answer_adapter, max_new_tokens=args.max_new_tokens)
    agent = DualModelAgent(planner_model=planner, answer_model=answer, max_steps=args.max_steps, enforce_required_tools=True)

    samples = list(read_jsonl(eval_path))
    metrics = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(samples):
            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")
            gold_tools = sample.get("gold_tools", [])

            print(f"[{i+1}/{len(samples)}] {sample.get('id')} [{sample.get('category','?')}]")

            result = agent.run(image=image, question=question)
            pred_tools = get_tools_from_trace(result["trace"])
            answer_text = result.get("answer", "") or ""

            gold_set = set(gold_tools)
            pred_set = set(pred_tools)

            row = {
                "id": sample.get("id"),
                "category": sample.get("category", "unknown"),
                "difficulty": sample.get("difficulty", "?"),
                "eval_focus": sample.get("eval_focus", "?"),
                "image": image,
                "question": question,
                "gold_tools": gold_tools,
                "pred_tools": pred_tools,
                "tool_hit": gold_set.issubset(pred_set),
                "tool_exact": gold_set == pred_set,
                "over_call": bool(pred_set - gold_set) and not bool(gold_set - pred_set),
                "miss_call": bool(gold_set - pred_set),
                "format_valid": result["status"] != "parse_error",
                "should_refuse": sample.get("should_refuse", False),
                "refusal_correct": is_refusal_answer(answer_text) == sample.get("should_refuse", False),
                "gold_answer_keywords": sample.get("gold_answer_keywords", []),
                "answer_keyword_hit": has_answer_keywords(answer_text, sample.get("gold_answer_keywords", [])),
                "tool_autonomy": 0 if get_enforcement_events(result["trace"]) else 1,
                "forbidden_violation": any(t in sample.get("forbidden_tools", []) for t in pred_tools),
                "answer_writer": result.get("answer_writer", "?"),
                "status": result["status"],
                "answer": answer_text,
                "trace": result["trace"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            metrics.append(row)

    # Summary
    n = len(metrics)
    tool_hit = sum(m["tool_hit"] for m in metrics)
    tool_exact = sum(m["tool_exact"] for m in metrics)
    miss = sum(m["miss_call"] for m in metrics)
    over = sum(m["over_call"] for m in metrics)
    kw = sum(m["answer_keyword_hit"] for m in metrics)
    ref = sum(m["refusal_correct"] for m in metrics)
    auto = sum(m["tool_autonomy"] for m in metrics)

    print(f"\n{'='*60}")
    print(f"  Dual Model (V4 planner + V7B answer)")
    print(f"{'='*60}")
    print(f"  Total: {n}")
    print(f"  Tool Hit:   {tool_hit}/{n} ({tool_hit/n:.0%})")
    print(f"  Tool Exact: {tool_exact}/{n} ({tool_exact/n:.0%})")
    print(f"  Miss-call:  {miss}/{n} ({miss/n:.0%})")
    print(f"  Over-call:  {over}/{n} ({over/n:.0%})")
    print(f"  Answer KW:  {kw}/{n} ({kw/n:.0%})")
    print(f"  Refusal:    {ref}/{n} ({ref/n:.0%})")
    print(f"  Autonomy:   {auto}/{n} ({auto/n:.0%})")

    cats = defaultdict(lambda: {"n": 0, "exact": 0, "miss": 0, "kw": 0})
    for m in metrics:
        c = m["category"]
        cats[c]["n"] += 1
        cats[c]["exact"] += m["tool_exact"]
        cats[c]["miss"] += m["miss_call"]
        cats[c]["kw"] += m["answer_keyword_hit"]
    print(f"\n  {'Category':<28} {'n':>3} {'Exact':>5} {'Miss':>5} {'KW':>5}")
    for c in sorted(cats):
        cc = cats[c]
        print(f"  {c:<28} {cc['n']:>3} {cc['exact']:>3}/{cc['n']}  {cc['miss']:>3}   {cc['kw']:>3}/{cc['n']}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
