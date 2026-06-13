"""
Step 1: Generate soft labels from teacher model (Dual v3).

For each training sample:
  1. Run dual v3 to get correct tool trajectory
  2. For the final_answer step, extract V7B-DPO's token-level logits
  3. Apply temperature scaling → save softmax probabilities as soft labels

Output: data/soft_labels/teacher_probs.pt
"""
import json, sys, torch
from pathlib import Path
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T = 3.0  # temperature for softening

def extract_answer_logits(answer_model, messages):
    """
    Run answer model in generation mode and extract token-level logits.
    Uses generate() with output_scores=True.
    """
    from agent.vlm import process_vision_info
    processor = answer_model.processor

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision_info(messages)
    inputs = processor(text=[text], images=img, videos=vid, padding=True, return_tensors="pt")
    inputs = {k: v.to(answer_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = answer_model.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # outputs.sequences[0] = full token ids
    # outputs.scores[i] = logits at step i (before softmax), shape [1, vocab_size]
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_len:]

    scores = []
    for i, logit in enumerate(outputs.scores):
        # Temperature scaling + softmax
        prob = F.softmax(logit[0] / T, dim=-1).cpu()
        token_id = generated_ids[i].item()
        scores.append({"token_id": token_id, "prob": prob})

    return scores


def main():
    from agent.vlm import QwenVLModel
    from agent.dual_agent import DualModelAgent
    from agent.prompts import SYSTEM_PROMPT

    print("Loading dual v3 teacher...")
    planner = QwenVLModel(
        "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
        "/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4", 256)
    answerer = QwenVLModel(
        "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct",
        "/root/autodl-tmp/multimodal-agent-lora/lora_dpo_v3", 256)
    agent = DualModelAgent(planner, answerer, max_steps=6)

    # Load SFT data
    sft_paths = [
        ROOT / "data/sft/sft_agent_train_v7b_final.jsonl",
        ROOT / "data/eval/eval_dev_v3.jsonl",
        ROOT / "data/eval/eval_test_v3.jsonl",
    ]

    all_samples = []
    for sp in sft_paths:
        if sp.exists():
            with open(sp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_samples.append(json.loads(line))

    print(f"Total samples: {len(all_samples)}")
    # Use subset for speed — 300 samples
    import random
    random.seed(42)
    samples = random.sample(all_samples, min(300, len(all_samples)))
    print(f"Using {len(samples)} samples for soft labels")

    soft_labels = []
    success = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] success={success}")

        # Get image and question
        image = sample.get("image", "")
        question = ""
        # Extract question from messages or use direct field
        msgs = sample.get("messages", [])
        if msgs:
            for m in msgs:
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                question = item.get("text", "")
                    elif isinstance(content, str):
                        question = content
                    break
        if not question:
            question = sample.get("question", "")

        if not image or not question:
            continue

        try:
            # Run dual agent to get correct trajectory
            result = agent.run(image=image, question=question)

            # Build teacher's answer context (messages up to final answer)
            raw_messages = result.get("messages", [])
            if not raw_messages:
                continue

            # Remove the last user prompt (post-tool instruction) to isolate answer generation
            answer_context = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in raw_messages:
                if msg["role"] == "system":
                    continue  # replaced
                answer_context.append(msg)

            # Remove the final assistant answer from context (we want teacher to regenerate it)
            while answer_context and answer_context[-1]["role"] == "assistant":
                answer_context.pop()

            if len(answer_context) < 2:
                continue

            # Generate teacher logits for the answer
            scores = extract_answer_logits(answerer, answer_context)

            if scores:
                soft_labels.append({
                    "sample_id": sample.get("id", f"sample_{i}"),
                    "image": image,
                    "question": question,
                    "context_messages": answer_context,
                    "teacher_scores": scores,  # [{token_id, prob}, ...]
                })
                success += 1

        except Exception as e:
            if (i + 1) % 50 == 0:
                print(f"  [WARN] {e}")

    # Save
    out_dir = ROOT / "data/soft_labels"
    out_dir.mkdir(exist_ok=True)
    torch.save(soft_labels, out_dir / "teacher_probs.pt")

    print(f"\nSoft labels generated: {success}/{len(samples)}")
    print(f"Saved to {out_dir / 'teacher_probs.pt'}")


if __name__ == "__main__":
    main()
