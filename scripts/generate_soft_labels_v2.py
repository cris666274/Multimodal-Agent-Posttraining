"""
Step 1a: Run V4 solo to generate correct trajectories for each sample. Save message contexts.
Step 1b: Load V7B-DPO solo to generate soft labels (logits) for final_answer step.

Two phases, only one model in GPU at a time.
"""
import json, sys, torch, random
from pathlib import Path
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T = 3.0

def main():
    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent

    # Load training samples
    sft_path = ROOT / "data/sft/sft_agent_train_v7b_final.jsonl"
    eval_paths = [
        ROOT / "data/eval/eval_dev_v3.jsonl",
        ROOT / "data/eval/eval_test_v3.jsonl",
    ]

    samples = []
    if sft_path.exists():
        with open(sft_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    # Extract question and image from messages
                    msgs = d.get("messages", [])
                    q, img = "", ""
                    for m in msgs:
                        if m["role"] == "user" and isinstance(m.get("content"), list):
                            for item in m["content"]:
                                if isinstance(item, dict):
                                    if item.get("type") == "image":
                                        img = item.get("image", "")
                                    if item.get("type") == "text":
                                        q = item.get("text", "")
                    if img and q:
                        samples.append({"id": d.get("id", ""), "image": img, "question": q})

    for ep in eval_paths:
        if ep.exists():
            with open(ep) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        samples.append(d)

    random.seed(42)
    samples = random.sample(samples, min(300, len(samples)))
    print(f"Total samples for soft labels: {len(samples)}")

    # ============================================================
    # Phase 1: V4 solo → correct trajectories
    # ============================================================
    print("\n=== Phase 1: V4 solo generating trajectories ===")
    v4 = QwenVLModel(
        "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
        "/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4",
        256,
    )
    agent = MultimodalAgent(model=v4, max_steps=6, enforce_required_tools=True)

    contexts = []
    success = 0
    for i, s in enumerate(samples):
        img = s.get("image", "") or s.get("image_path", "")
        q = s.get("question", "")
        if not img or not q:
            continue

        try:
            result = agent.run(image=img, question=q)
            msgs = result.get("messages", [])
            trace = result.get("trace", [])

            # Skip if enforcement was needed
            if any("blocked" in step.get("event", "") for step in trace):
                continue

            # Remove final assistant answer from messages (we'll regenerate)
            while msgs and msgs[-1].get("role") == "assistant":
                msgs.pop()

            if len(msgs) >= 2:
                contexts.append({
                    "sample_id": s.get("id", f"s_{i}"),
                    "image": img,
                    "question": q,
                    "context_messages": msgs,
                })
                success += 1
        except:
            pass

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] success={success}")

    print(f"Phase 1 done: {success} clean contexts")

    # Free V4 from GPU
    del v4, agent
    torch.cuda.empty_cache()

    # ============================================================
    # Phase 2: V7B-DPO solo → soft labels
    # ============================================================
    print("\n=== Phase 2: V7B-DPO generating soft labels ===")
    answerer = QwenVLModel(
        "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct",
        "/root/autodl-tmp/multimodal-agent-lora/lora_dpo_v3",
        256,
    )

    from qwen_vl_utils import process_vision_info
    processor = answerer.processor
    soft_labels = []
    label_success = 0

    for i, ctx in enumerate(contexts):
        try:
            msgs = ctx["context_messages"]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img, vid = process_vision_info(msgs)
            inputs = processor(text=[text], images=img, videos=vid, padding=True, return_tensors="pt")
            inputs = {k: v.to(answerer.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = answerer.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0, input_len:]
            scores = []
            for j, logit in enumerate(outputs.scores):
                prob = F.softmax(logit[0] / T, dim=-1).cpu()
                token_id = generated_ids[j].item()
                scores.append({"token_id": token_id, "prob": prob})

            if scores:
                ctx["teacher_scores"] = scores
                soft_labels.append(ctx)
                label_success += 1

        except Exception as e:
            pass

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(contexts)}] labels={label_success}")

    # Save
    out_dir = ROOT / "data/soft_labels"
    out_dir.mkdir(exist_ok=True)
    torch.save(soft_labels, out_dir / "teacher_probs.pt")
    print(f"\nPhase 2 done: {label_success} soft labels saved to {out_dir / 'teacher_probs.pt'}")


if __name__ == "__main__":
    main()
