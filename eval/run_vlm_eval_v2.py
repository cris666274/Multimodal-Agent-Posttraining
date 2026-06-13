"""
Eval script v2: adds Tool Exact Match, Over-call Rate, Miss-call Rate, Tool Autonomy.
Supports new eval v3 fields: allowed_tools, forbidden_tools, difficulty, eval_focus.

Usage:
  python eval/run_vlm_eval_v2.py \
    --model_name ... --adapter_name ... \
    --eval_path data/eval/eval_dev_v3.jsonl \
    --out_path outputs/vlm_agent_dev_eval.jsonl
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.runtime import MultimodalAgent


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
    """Count enforcement interventions from runtime."""
    events = []
    for step in trace:
        if "event" in step:
            events.append(step["event"])
    return events


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


def compute_metrics(sample, result):
    """Compute all eval metrics for a single sample."""
    trace = result.get("trace", [])
    pred_tools = get_tools_from_trace(trace)
    gold_tools = sample.get("gold_tools", [])
    answer = result.get("answer", "") or ""

    gold_set = set(gold_tools)
    pred_set = set(pred_tools)

    # Tool hit: gold subset of pred
    tool_hit = gold_set.issubset(pred_set)

    # Tool order: gold is subsequence of pred
    tool_order_hit = is_subsequence(gold_tools, pred_tools)

    # Tool exact match: pred == gold
    tool_exact = (gold_set == pred_set)

    # Over-call: extra tools, none missing
    over_call = bool(pred_set - gold_set) and not bool(gold_set - pred_set)

    # Miss-call: missing tools
    miss_call = bool(gold_set - pred_set)

    # Format valid
    format_valid = result.get("status") != "parse_error"

    # Answer keyword
    kw_hit = has_answer_keywords(answer, sample.get("gold_answer_keywords", []))

    # Refusal
    refusal_correct = (
        is_refusal_answer(answer) == sample.get("should_refuse", False)
    )

    # Tool autonomy: number of enforcement events (lower = more autonomous)
    enforce_events = get_enforcement_events(trace)
    autonomy = 0 if enforce_events else 1  # 1 = fully autonomous, 0 = needed help

    # Forbidden tools check
    forbidden = sample.get("forbidden_tools", [])
    used_forbidden = [t for t in pred_tools if t in forbidden]
    forbidden_violation = len(used_forbidden) > 0

    return {
        "tool_hit": tool_hit,
        "tool_order_hit": tool_order_hit,
        "tool_exact": tool_exact,
        "over_call": over_call,
        "miss_call": miss_call,
        "format_valid": format_valid,
        "kw_hit": kw_hit,
        "refusal_correct": refusal_correct,
        "autonomy": autonomy,
        "forbidden_violation": forbidden_violation,
        "pred_tools": pred_tools,
        "gold_tools": gold_tools,
        "enforce_events": enforce_events,
        "answer": answer,
    }


def print_summary(metrics_list, samples, label):
    """Print formatted eval summary with per-category breakdown."""
    n = len(metrics_list)
    if n == 0:
        return

    # Overall counts
    tool_hit = sum(m["tool_hit"] for m in metrics_list)
    tool_exact = sum(m["tool_exact"] for m in metrics_list)
    over_call = sum(m["over_call"] for m in metrics_list)
    miss_call = sum(m["miss_call"] for m in metrics_list)
    kw_hit = sum(m["kw_hit"] for m in metrics_list)
    fmt_ok = sum(m["format_valid"] for m in metrics_list)
    refusal_ok = sum(m["refusal_correct"] for m in metrics_list)
    autonomy_ok = sum(m["autonomy"] for m in metrics_list)
    forbidden_bad = sum(m["forbidden_violation"] for m in metrics_list)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total: {n}")
    print(f"  Tool Hit:        {tool_hit:>3}/{n} ({tool_hit/n:.1%})")
    print(f"  Tool Exact:      {tool_exact:>3}/{n} ({tool_exact/n:.1%})")
    print(f"  Over-call Rate:  {over_call:>3}/{n} ({over_call/n:.1%})")
    print(f"  Miss-call Rate:  {miss_call:>3}/{n} ({miss_call/n:.1%})")
    print(f"  Format Valid:    {fmt_ok:>3}/{n} ({fmt_ok/n:.1%})")
    print(f"  Answer KW:       {kw_hit:>3}/{n} ({kw_hit/n:.1%})")
    print(f"  Refusal OK:      {refusal_ok:>3}/{n} ({refusal_ok/n:.1%})")
    print(f"  Tool Autonomy:   {autonomy_ok:>3}/{n} ({autonomy_ok/n:.1%})")
    print(f"  Forbidden Tools: {forbidden_bad:>3}/{n} violations")

    # Per category
    cats = defaultdict(lambda: {"n": 0, "tool_exact": 0, "kw": 0, "over": 0, "miss": 0, "auto": 0})
    for m, s in zip(metrics_list, samples):
        c = s.get("category", "?")
        cats[c]["n"] += 1
        cats[c]["tool_exact"] += m["tool_exact"]
        cats[c]["kw"] += m["kw_hit"]
        cats[c]["over"] += m["over_call"]
        cats[c]["miss"] += m["miss_call"]
        cats[c]["auto"] += m["autonomy"]

    print(f"\n  {'Category':<28} {'n':>3} {'Exact':>5} {'KW':>5} {'Over':>5} {'Miss':>5} {'Auto':>5}")
    print(f"  {'-'*28} {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for c in sorted(cats):
        cc = cats[c]
        print(f"  {c:<28} {cc['n']:>3} {cc['tool_exact']:>3}/{cc['n']}  {cc['kw']:>3}/{cc['n']}  {cc['over']:>3}   {cc['miss']:>3}   {cc['auto']:>3}/{cc['n']}")

    # Per difficulty
    diffs = defaultdict(lambda: {"n": 0, "tool_exact": 0, "kw": 0, "auto": 0})
    for m, s in zip(metrics_list, samples):
        d = s.get("difficulty", "?")
        diffs[d]["n"] += 1
        diffs[d]["tool_exact"] += m["tool_exact"]
        diffs[d]["kw"] += m["kw_hit"]
        diffs[d]["auto"] += m["autonomy"]
    print(f"\n  {'Difficulty':<12} {'n':>3} {'Exact':>8} {'KW':>8} {'Auto':>8}")
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            dd = diffs[d]
            print(f"  {d:<12} {dd['n']:>3} {dd['tool_exact']:>3}/{dd['n']} ({dd['tool_exact']/max(dd['n'],1):.0%})  {dd['kw']:>3}/{dd['n']} ({dd['kw']/max(dd['n'],1):.0%})  {dd['auto']:>3}/{dd['n']} ({dd['auto']/max(dd['n'],1):.0%})")

    # Per eval_focus
    focuses = defaultdict(lambda: {"n": 0, "kw": 0, "miss": 0})
    for m, s in zip(metrics_list, samples):
        f = s.get("eval_focus", "?")
        focuses[f]["n"] += 1
        focuses[f]["kw"] += m["kw_hit"]
        focuses[f]["miss"] += m["miss_call"]
    print(f"\n  {'Eval Focus':<20} {'n':>3} {'KW':>8} {'Miss':>8}")
    for f in sorted(focuses):
        ff = focuses[f]
        print(f"  {f:<20} {ff['n']:>3} {ff['kw']:>3}/{ff['n']} ({ff['kw']/max(ff['n'],1):.0%})  {ff['miss']:>3}/{ff['n']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_name", default=None)
    p.add_argument("--eval_path", default=str(ROOT / "data/eval/eval_dev_v3.jsonl"))
    p.add_argument("--out_path", default=str(ROOT / "outputs/vlm_agent_eval_v3.jsonl"))
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

    if not eval_path.exists():
        raise FileNotFoundError(f"找不到评测文件: {eval_path}")

    model = QwenVLModel(
        model_name=args.model_name,
        adapter_name=args.adapter_name,
        max_new_tokens=args.max_new_tokens,
    )
    agent = MultimodalAgent(model=model, max_steps=args.max_steps, enforce_required_tools=True)

    samples = list(read_jsonl(eval_path))
    metrics_list = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(samples):
            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")

            print(f"\n[{i+1}/{len(samples)}] {sample.get('id')}  [{sample.get('category','?')}] [{sample.get('difficulty','?')}]")
            print(f"  Q: {question[:100]}")

            result = agent.run(image=image, question=question)
            m = compute_metrics(sample, result)

            print(f"  tools: gold={m['gold_tools']} pred={m['pred_tools']}")
            print(f"  exact={m['tool_exact']} over={m['over_call']} miss={m['miss_call']} "
                  f"kw={m['kw_hit']} ref={m['refusal_correct']} auto={m['autonomy']}")

            row = {
                "id": sample.get("id"),
                "category": sample.get("category", "unknown"),
                "difficulty": sample.get("difficulty", "?"),
                "eval_focus": sample.get("eval_focus", "?"),
                "image": image,
                "question": question,
                "gold_tools": m["gold_tools"],
                "pred_tools": m["pred_tools"],
                "tool_hit": m["tool_hit"],
                "tool_order_hit": m["tool_order_hit"],
                "tool_exact": m["tool_exact"],
                "over_call": m["over_call"],
                "miss_call": m["miss_call"],
                "format_valid": m["format_valid"],
                "should_refuse": sample.get("should_refuse", False),
                "refusal_correct": m["refusal_correct"],
                "gold_answer_keywords": sample.get("gold_answer_keywords", []),
                "answer_keyword_hit": m["kw_hit"],
                "tool_autonomy": m["autonomy"],
                "forbidden_violation": m["forbidden_violation"],
                "enforce_events": m["enforce_events"],
                "status": result.get("status"),
                "answer": m["answer"],
                "trace": result.get("trace", []),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            metrics_list.append(m)

    print_summary(metrics_list, samples, f"{args.model_name.split('/')[-1]} + {Path(args.adapter_name).name if args.adapter_name else 'base'}  on  {eval_path.name}")

    # Quick fail analysis
    fails = [(m, s) for m, s in zip(metrics_list, samples) if not m["kw_hit"]]
    if fails:
        print(f"\n  Top KW failures ({len(fails)}):")
        for m, s in fails[:10]:
            print(f"    [{s.get('category','?')}] {s['id']}  ans={m['answer'][:80]}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
