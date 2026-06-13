"""
Rejection sampling pipeline for multimodal agent.

Replaces high-variance RL/DPO with:
1. Run agent N times per training sample
2. Keep only rollouts with: exact tool match + correct answer + correct refusal
3. Output high-quality multi-turn SFT traces

Usage:
  python scripts/rejection_sample.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4 \
    --train_file data/eval/rl_train.jsonl \
    --out_file data/sft/sft_rejection_sampled.jsonl \
    --num_rollouts 8 --max_steps 6
"""
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_rollout(result: Dict, gold: Dict) -> Dict:
    """
    Score a rollout on multiple dimensions.
    Returns dict with pass/fail per dimension and overall.
    """
    trace = result.get("trace", [])
    pred_tools = []
    for step in trace:
        if "tool_name" in step:
            pred_tools.append(step["tool_name"])

    gold_tools = gold.get("gold_tools", [])
    answer = result.get("answer", "") or ""
    gold_kw = gold.get("gold_answer_keywords", [])
    should_refuse = gold.get("should_refuse", False)

    # 1. Exact tool match
    exact_tool = (set(pred_tools) == set(gold_tools))

    # 2. Answer keyword match
    kw_ok = True
    if gold_kw:
        for kw in gold_kw:
            if kw in answer:
                continue
            # Fuzzy numeric
            try:
                val = float(kw)
                patterns = [f"{val:.2f}", f"{val:.1f}", f"{val:.0f}", str(round(val))]
                if not any(p in answer for p in set(patterns)):
                    kw_ok = False
                    break
            except ValueError:
                kw_ok = False
                break

    # 3. Refusal correctness
    refuse_words = ["无法可靠", "无法判断", "看不清", "信息不足", "不能确定", "无法确定", "无法识别"]
    is_refusal = any(w in answer for w in refuse_words)
    refusal_ok = (is_refusal == should_refuse)

    # 4. Format valid
    format_ok = (result.get("status") != "parse_error")

    overall = exact_tool and kw_ok and refusal_ok and format_ok

    return {
        "exact_tool": exact_tool,
        "kw_ok": kw_ok,
        "refusal_ok": refusal_ok,
        "format_ok": format_ok,
        "overall": overall,
        "pred_tools": pred_tools,
        "answer": answer,
    }


def rollout_to_sft_messages(result: Dict, image: str, question: str) -> List[Dict]:
    """Convert agent trace to SFT training messages format."""
    from agent.prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    for step in result.get("trace", []):
        if "model_output" in step:
            out = step["model_output"]
            if out and out.strip():
                messages.append({"role": "assistant", "content": out})
        if "tool_name" in step:
            obs = json.dumps(step.get("observation", {}), ensure_ascii=False)
            messages.append({"role": "tool", "name": step["tool_name"], "content": obs})
            messages.append({"role": "user", "content": (
                f"工具 {step['tool_name']} 返回结果如下：\n{obs}\n\n"
                "请根据工具结果继续完成原始任务。只能输出一个 <tool_call> 或一个非空 <final_answer>。"
            )})

    return messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_name", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--num_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=6)
    args = p.parse_args()

    train_file = Path(args.train_file)
    if not train_file.is_absolute():
        train_file = ROOT / train_file
    out_file = Path(args.out_file)
    if not out_file.is_absolute():
        out_file = ROOT / out_file

    # Load model
    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent

    print(f"Loading model: {args.model_name}")
    print(f"Adapter: {args.adapter_name}")
    model = QwenVLModel(
        model_name=args.model_name,
        adapter_name=args.adapter_name,
        max_new_tokens=args.max_new_tokens,
    )
    agent = MultimodalAgent(model=model, max_steps=args.max_steps, enforce_required_tools=True)

    # Load training data
    train_data = load_jsonl(train_file)
    train_data = [s for s in train_data if s.get("image") or s.get("image_path")]
    print(f"Training samples: {len(train_data)}")
    print(f"Rollouts per sample: {args.num_rollouts}")

    # Rejection sampling
    total_rollouts = 0
    accepted = 0
    sft_samples = []
    stats = {"exact_tool": 0, "kw_ok": 0, "refusal_ok": 0, "format_ok": 0}

    for si, sample in enumerate(train_data):
        image = sample.get("image") or sample.get("image_path")
        question = sample.get("question", "")
        sample_ok = 0

        for ri in range(args.num_rollouts):
            total_rollouts += 1
            result = agent.run(image=image, question=question)
            scores = score_rollout(result, sample)

            for k in stats:
                stats[k] += int(scores[k])

            if scores["overall"]:
                accepted += 1
                sample_ok += 1
                messages = rollout_to_sft_messages(result, image, question)
                sft_samples.append({
                    "id": f"{sample.get('id', 'sample')}_rs{ri}",
                    "category": sample.get("category", "unknown"),
                    "image": image,
                    "question": question,
                    "messages": messages,
                })

        if (si + 1) % 5 == 0:
            acc_rate = accepted / total_rollouts * 100 if total_rollouts > 0 else 0
            print(f"  [{si+1}/{len(train_data)}] accepted={accepted}/{total_rollouts} ({acc_rate:.1f}%)"
                  f"  tools={stats['exact_tool']}/{total_rollouts}"
                  f"  kw={stats['kw_ok']}/{total_rollouts}")

    # Summary
    acc_rate = accepted / total_rollouts * 100 if total_rollouts > 0 else 0
    print(f"\n{'='*60}")
    print(f"Rejection sampling complete:")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Accepted: {accepted} ({acc_rate:.1f}%)")
    print(f"  Rejected: {total_rollouts - accepted}")
    print(f"  Per-dimension pass rates:")
    for k, v in stats.items():
        print(f"    {k}: {v}/{total_rollouts} ({v/total_rollouts*100:.1f}%)")

    # Write output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for s in sft_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(sft_samples)} SFT samples to {out_file}")


if __name__ == "__main__":
    main()
