"""
Generate teacher logits for soft KD — correct version.

Key fixes vs previous attempts:
1. Teacher uses SYSTEM_PROMPT (same as student training)
2. Saves top-K logits for ALL assistant turns (not just final_answer)
3. V4 solo (3B, 0% miss-call) generates tool_calls + final_answer

Output: data/soft_labels_v4/teacher_logits.pt
  [{sample_id, turn_index, teacher_logits_topk, teacher_token_ids}, ...]
"""
import json, sys, torch, random
from pathlib import Path
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP_K = 100  # save top-100 logits per position (34MB total)


def main():
    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent
    from qwen_vl_utils import process_vision_info

    # Load SFT data
    sft_path = ROOT / "data/sft/sft_agent_train_v7b_final.jsonl"
    samples = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                msgs = d.get("messages", [])
                q, img = "", ""
                for m in msgs:
                    if m["role"] == "user" and isinstance(m.get("content"), list):
                        for item in m["content"]:
                            if isinstance(item, dict):
                                if item.get("type") == "image": img = item.get("image", "")
                                if item.get("type") == "text": q = item.get("text", "")
                if img and q:
                    samples.append({"id": d.get("id", ""), "image": img, "question": q})

    random.seed(42)
    samples = random.sample(samples, min(300, len(samples)))
    print(f"Samples: {len(samples)}")

    # Load V4 teacher
    print("Loading V4 teacher...")
    v4 = QwenVLModel(
        "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
        "/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4", 256)
    agent = MultimodalAgent(model=v4, max_steps=6, enforce_required_tools=True)
    processor = v4.processor

    teacher_data = []
    success = 0

    for i, s in enumerate(samples):
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(samples)}] success={success}")

        try:
            result = agent.run(image=s["image"], question=s["question"])
            trace = result.get("trace", [])
            msgs = result.get("messages", [])

            # Skip if enforcement was needed
            if any("blocked" in step.get("event", "") for step in trace):
                continue

            # For each assistant turn, extract teacher logits
            turn_count = 0
            for step_idx, step in enumerate(trace):
                if "model_output" not in step:
                    continue
                out = step["model_output"]
                if not out or not out.strip():
                    continue

                # Build context up to this turn
                context = msgs[:2]  # system + first user
                for prev_step in trace[:step_idx + 1]:
                    if "model_output" in prev_step and prev_step != step:
                        context.append({"role": "assistant", "content": prev_step["model_output"]})
                    if "tool_name" in prev_step:
                        obs = json.dumps(prev_step.get("observation", {}), ensure_ascii=False)
                        context.append({"role": "tool", "name": prev_step["tool_name"], "content": obs})

                # Tokenize context + teacher output
                full_msgs = context + [{"role": "assistant", "content": out}]
                full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
                full_img, full_vid = process_vision_info(full_msgs)
                full_inputs = processor(text=[full_text], images=full_img, videos=full_vid,
                                        padding=False, return_tensors="pt")
                full_inputs = {k: v.to(v4.device) for k, v in full_inputs.items()}

                # Prefix length
                prefix_msgs = context
                prefix_text = processor.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
                prefix_img, prefix_vid = process_vision_info(prefix_msgs)
                prefix_inputs = processor(text=[prefix_text], images=prefix_img, videos=prefix_vid,
                                         padding=False, return_tensors="pt")
                prefix_len = prefix_inputs["input_ids"].shape[1]
                full_ids = full_inputs["input_ids"][0]

                # Forward pass to get teacher logits
                with torch.no_grad():
                    outputs = v4.model(**full_inputs)
                    logits = outputs.logits[0]  # [seq_len, vocab_size]

                # Extract teacher logits for the assistant turn tokens
                teacher_logits_topk = []
                teacher_token_ids = []
                for t in range(prefix_len, full_ids.shape[0]):
                    token_id = full_ids[t].item()
                    if t > 0 and t - 1 < logits.shape[0]:
                        logit_t = logits[t - 1, :]  # logits[t-1] predicts token[t]
                        topk_vals, topk_idx = torch.topk(logit_t, k=min(TOP_K, logit_t.shape[0]))
                        teacher_logits_topk.append({
                            "indices": topk_idx.cpu(),
                            "values": topk_vals.cpu()
                        })
                        teacher_token_ids.append(token_id)

                if teacher_logits_topk:
                    teacher_data.append({
                        "sample_id": s["id"],
                        "turn_index": turn_count,
                        "teacher_logits_topk": teacher_logits_topk,  # [{indices, values}, ...]
                        "teacher_token_ids": teacher_token_ids,       # [token_id, ...]
                    })
                    turn_count += 1

            success += 1

        except Exception as e:
            pass

    # Save
    out_dir = ROOT / "data/soft_labels_v4"
    out_dir.mkdir(exist_ok=True)
    torch.save(teacher_data, out_dir / "teacher_logits.pt")
    print(f"\nTeacher logits: {len(teacher_data)} turns from {success} samples")
    print(f"Saved to {out_dir / 'teacher_logits.pt'}")


if __name__ == "__main__":
    main()
