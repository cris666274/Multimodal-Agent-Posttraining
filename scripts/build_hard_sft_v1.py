"""
Build hard SFT training data for V7B miss-call patterns.

Uses V4 (enforcement ON, 0% miss-call) to generate correct multi-turn
trajectories for the three categories where V7B misses tool calls:
  - uncertainty_refusal: target 60-80 samples
  - image_text_consistency: target 50-80 samples
  - document_validation: target 20-30 samples

Output: data/sft/sft_hard_refusal.jsonl, sft_hard_consistency.jsonl, sft_hard_docs.jsonl
"""
import json, random, re, sys, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)

# ============================================================
# Question templates — varied but all requiring correct tool use
# ============================================================

REFUSAL_QUESTIONS = [
    "请判断图片中的品牌名称，并说明依据。",
    "图片中是否包含可识别的品牌标识？如有请写出，如无请说明原因。",
    "这张图片的品牌 logo 是什么？先调用 vision_parse 查看再回答。",
    "图片中的文字是否可以识别？如果模糊请明确说明。",
    "请分析这张图片的品牌信息。如果无法可靠判断，必须明确拒绝。",
    "图片质量如何？能否从中识别出品牌名称？",
    "请先仔细查看图片，然后告诉我你看到了什么品牌标识。",
    "判断图片中的品牌，如果看不清就直说看不清。",
    "图片里有没有文字或 logo？先 OCR 再看。",
    "你能从这张图片中读出什么品牌信息？",
]

CONSISTENCY_QUESTIONS = [
    "商品描述写的是 {color} {category}。请先调用 vision_parse 查看图片，再判断图文是否一致。",
    "标题是 {color} {category}，请检查图片中的商品是否与之一致。",
    "请用 vision_parse 查看商品图片，然后判断与描述 '{color} {category}' 是否匹配。",
    "商品描述说这是 {color} {category}。请查看图片确认是否正确。",
    "图片中的商品是什么颜色和品类？和 '{color} {category}' 对比一下。",
    "先看商品图片，再告诉我图片和描述 {color} {category} 是否一致。",
    "用 vision_parse 分析商品图，判断是否与 {color} {category} 一致。",
]

DOCS_QUESTIONS = [
    "检查这张发票是否缺少必填字段，先 OCR 再查规则文档。",
    "根据发票规则，判断这张发票是否合规。必须先调用 vision_parse 和 retrieve_docs。",
    "这张发票包含哪些字段？是否满足必填要求？请逐一核查。",
    "请提取发票信息并与规则文档对比，判断是否完整。",
    "发票的 Tax ID 是否存在？先看图片再查规则确认。",
]

# ============================================================
# Image pools
# ============================================================

LOGO_IMAGES = [f"data/images/logo_{i:03d}.jpg" for i in range(1, 8)]
PRODUCT_IMAGES = [f"data/images/product_{i:03d}.jpg" for i in range(1, 18)]
INVOICE_IMAGES = [f"data/images/invoice_{i:03d}.jpg" for i in range(1, 18)]

COLORS = ["black", "white", "red", "blue", "green"]
CATEGORIES = ["shoe", "backpack", "hat", "watch", "t-shirt"]

# ============================================================
# Main
# ============================================================

def validate_trace(trace, gold_tools, answer):
    """Check that the trace is a valid training sample."""
    called = []
    for step in trace:
        if "tool_name" in step:
            called.append(step["tool_name"])

    # Must call all gold tools
    gold_set = set(gold_tools)
    called_set = set(called)
    if not gold_set.issubset(called_set):
        return False, f"missing tools: gold={gold_set} called={called_set}"

    # Answer must be non-empty (answer is already extracted content, no XML tags)
    if not answer or len(answer.strip()) < 5:
        return False, f"empty or too short answer: '{answer[:60]}'"

    # Must not be a template placeholder
    if "[" in answer and "]" in answer:
        # Check if it's a bracket placeholder like "[品牌名称]" or "[具体依据]"
        bracket_content = answer[answer.index("[")+1:answer.index("]")]
        if len(bracket_content) < 10:
            return False, f"template placeholder: {answer[:60]}"

    return True, ""


def trace_to_messages(trace, image, question):
    """Convert agent trace to SFT messages format."""
    from agent.prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    for step in trace:
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


def generate_category(agent, images, questions, gold_tools, cat_name, target, extra_kwargs_fn=None):
    """Generate SFT samples for one category."""
    samples = []
    attempts = 0
    max_attempts = target * 2  # safety limit

    while len(samples) < target and attempts < max_attempts:
        img = random.choice(images)
        q_template = random.choice(questions)

        if extra_kwargs_fn:
            kwargs = extra_kwargs_fn()
            question = q_template.format(**kwargs)
        else:
            question = q_template

        # Check if we already have this (image, question) pair
        key = (img, question)
        if any(s["image"] == img and s["question"] == question for s in samples):
            attempts += 1
            continue

        attempts += 1
        try:
            result = agent.run(image=img, question=question)
            trace = result.get("trace", [])
            answer = result.get("answer", "")

            valid, reason = validate_trace(trace, gold_tools, answer)
            if not valid:
                if attempts % 20 == 0:
                    print(f"  [{len(samples)}/{target}] skip: {reason}")
                continue

            messages = trace_to_messages(trace, img, question)
            samples.append({
                "id": f"sft_hard_{cat_name}_{len(samples):03d}",
                "category": cat_name,
                "image": img,
                "question": question,
                "messages": messages,
            })

            if len(samples) % 10 == 0:
                print(f"  [{len(samples)}/{target}] {cat_name} samples generated")

        except Exception as e:
            if attempts % 20 == 0:
                print(f"  [WARN] {img}: {str(e)[:60]}")

    print(f"  Done: {len(samples)}/{target} samples ({attempts} attempts, {len(samples)/max(attempts,1):.0%} accept rate)")
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_name", default="/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4")
    p.add_argument("--refusal_target", type=int, default=70)
    p.add_argument("--consistency_target", type=int, default=65)
    p.add_argument("--docs_target", type=int, default=25)
    args = p.parse_args()

    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent

    print("=" * 60)
    print("Loading V4 for hard SFT data generation...")
    print("=" * 60)
    model = QwenVLModel(model_name=args.model_name, adapter_name=args.adapter_name, max_new_tokens=256)
    agent = MultimodalAgent(model=model, max_steps=6, enforce_required_tools=True)

    # ---- uncertainty_refusal ----
    print(f"\n--- uncertainty_refusal (target: {args.refusal_target}) ---")
    refusal_samples = generate_category(
        agent, LOGO_IMAGES, REFUSAL_QUESTIONS,
        gold_tools=["vision_parse"],  # must call vision_parse first
        cat_name="uncertainty_refusal",
        target=args.refusal_target,
    )

    out = ROOT / "data/sft/sft_hard_refusal.jsonl"
    with open(out, "w") as f:
        for s in refusal_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  → {out}")

    # ---- image_text_consistency ----
    print(f"\n--- image_text_consistency (target: {args.consistency_target}) ---")
    def color_cat_pair():
        return {"color": random.choice(COLORS), "category": random.choice(CATEGORIES)}

    consistency_samples = generate_category(
        agent, PRODUCT_IMAGES, CONSISTENCY_QUESTIONS,
        gold_tools=["vision_parse"],
        cat_name="image_text_consistency",
        target=args.consistency_target,
        extra_kwargs_fn=color_cat_pair,
    )

    out = ROOT / "data/sft/sft_hard_consistency.jsonl"
    with open(out, "w") as f:
        for s in consistency_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  → {out}")

    # ---- document_validation ----
    print(f"\n--- document_validation (target: {args.docs_target}) ---")
    docs_samples = generate_category(
        agent, INVOICE_IMAGES, DOCS_QUESTIONS,
        gold_tools=["vision_parse", "retrieve_docs"],
        cat_name="document_validation",
        target=args.docs_target,
    )

    out = ROOT / "data/sft/sft_hard_docs.jsonl"
    with open(out, "w") as f:
        for s in docs_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  → {out}")

    # Summary
    total = len(refusal_samples) + len(consistency_samples) + len(docs_samples)
    print(f"\n{'='*60}")
    print(f"Total: {total} hard SFT samples")
    print(f"  uncertainty_refusal: {len(refusal_samples)}")
    print(f"  image_text_consistency: {len(consistency_samples)}")
    print(f"  document_validation: {len(docs_samples)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
