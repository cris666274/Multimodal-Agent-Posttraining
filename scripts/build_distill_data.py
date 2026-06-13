"""
Build distilled SFT data from dual v3 clean trajectories.
Teacher: V4 planner (tool decisions) + V7B-DPO (answers).
Only keeps trajectories with 0 enforcement events + correct answers.
"""
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def is_clean_trace(trace, answer):
    """Check if this trajectory is suitable for distillation."""
    # 1. No enforcement interventions
    for step in trace:
        event = step.get("event", "")
        if "blocked" in event:
            return False, f"enforcement: {event}"

    # 2. Has at least one tool call and a final answer
    has_tool = any("tool_name" in s for s in trace)
    if not has_tool and "<final_answer>" not in str(trace[-1].get("model_output", "")):
        return False, "no tool call"

    # 3. Answer not empty, not hallucinated brand name
    if not answer or len(answer) < 3:
        return False, "empty answer"
    for fake_brand in ["Sport 13", "Sport 14", "Vans", "Geely", "DELL", "Xiaomi", "00000"]:
        if fake_brand in answer and "无法" not in answer:
            # Brand in answer but this is a refusal sample — hallucination
            return False, f"hallucinated: {fake_brand}"

    return True, ""


def trace_to_messages(trace, image, question, final_answer):
    """Convert eval trace to SFT messages format. Uses final_answer (V7B-DPO) for last turn."""
    from agent.prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    # Check if answer was delegated to V7B
    was_delegated = any("answer_delegated" in s.get("event", "") for s in trace)

    for i, step in enumerate(trace):
        if "model_output" in step:
            out = step["model_output"]
            if not out or not out.strip():
                continue

            # If this is the last assistant turn and answer was delegated to V7B,
            # replace V4's answer with V7B-DPO's answer (wrapped in tags)
            is_last = (i == len(trace) - 1) or not any(
                "model_output" in s for s in trace[i+1:]
            )
            if is_last and was_delegated and final_answer:
                out = f"<final_answer>{final_answer}</final_answer>"

            messages.append({"role": "assistant", "content": out})

        # Tool execution result
        if "tool_name" in step:
            obs = json.dumps(step.get("observation", {}), ensure_ascii=False)
            messages.append({"role": "tool", "name": step["tool_name"], "content": obs})
            messages.append({"role": "user", "content": (
                f"工具 {step['tool_name']} 返回结果如下：\n{obs}\n\n"
                "请根据工具结果继续完成原始任务。只能输出一个 <tool_call> 或一个非空 <final_answer>。"
            )})

    return messages

def main():
    # Load existing dual v3 eval results
    sources = [
        ("outputs/dual_v3_dev_eval.jsonl", "dev"),
        ("outputs/dual_v3_test_eval.jsonl", "test"),
    ]

    distilled = []
    stats = defaultdict(int)

    for path, split in sources:
        with open(ROOT / path) as f:
            data = [json.loads(line) for line in f if line.strip()]

        for d in data:
            trace = d.get("trace", [])
            answer = d.get("answer", "")
            image = d.get("image", "")
            question = d.get("question", "")

            clean, reason = is_clean_trace(trace, answer)
            if clean:
                messages = trace_to_messages(trace, image, question, answer)
                distilled.append({
                    "id": f"distill_{d['id']}",
                    "category": d.get("category", "?"),
                    "image": image,
                    "question": question,
                    "messages": messages,
                })
                stats["accepted"] += 1
            else:
                stats[f"rejected_{reason.split(':')[0]}"] += 1

    # Also include original clean SFT data (v4 enforced, no mock)
    sft_path = ROOT / "data/sft/sft_agent_train_real_v4_enforced_resized.jsonl"
    sft_samples = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sft_samples.append(json.loads(line))

    # Mix: all distilled + subset of SFT (5:1 ratio)
    import random
    random.seed(42)
    sft_subset = random.sample(sft_samples, min(len(sft_samples), len(distilled) * 5))

    mixed = distilled + sft_subset
    random.shuffle(mixed)

    # Save distilled-only (for 3B student with V4 base)
    distill_out = ROOT / "data/sft/sft_distilled_clean.jsonl"
    with open(distill_out, "w") as f:
        for s in distilled:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save mixed (for 7B student)
    mixed_out = ROOT / "data/sft/sft_distilled_mixed.jsonl"
    with open(mixed_out, "w") as f:
        for s in mixed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Distilled clean: {len(distilled)} samples → {distill_out}")
    print(f"Mixed (distill+SFT): {len(mixed)} samples ({len(distilled)} distill + {len(sft_subset)} SFT) → {mixed_out}")
    print(f"\nFilter stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
