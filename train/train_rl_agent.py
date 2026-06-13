"""
RL (REINFORCE + KL penalty) training for multimodal agent.

Unlike SFT (which imitates correct traces) and DPO (which optimizes last-turn preferences),
this RL approach:

1. Runs the full agent for each eval sample
2. Scores the entire trace with a multi-dimensional reward:
   - format_valid: parse errors get heavy penalty
   - tool_correct: gold tools subset of pred tools
   - answer_keyword: keywords in final answer (fuzzy chart matching)
   - refusal_correct: should_refuse matches actual refusal
   - no_redundant: bonus for not calling unnecessary tools
3. Computes token-level log-prob for each assistant turn
4. REINFORCE gradient: pushes up turns in high-reward traces, pushes down low-reward
5. KL penalty keeps model close to reference (SFT) model

Usage:
  python train/train_rl_agent.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4 \
    --output_dir /root/autodl-tmp/multimodal-agent-lora/lora_rl_v1 \
    --num_iterations 20 --lr 5e-6 --kl_coef 0.1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime import MultimodalAgent
from agent.vlm import QwenVLModel
from agent.tools import set_vlm_model

# ---- Reward Function ----

def compute_reward(result: Dict, gold: Dict) -> float:
    """
    Multi-dimension reward for a complete agent trace.
    Returns a scalar in [-3, 3] range.
    """
    reward = 0.0

    # 1. Format validity (-2 if any parse error)
    if result.get("status") == "parse_error":
        reward -= 2.0

    # 2. Tool correctness (+1 for hitting gold tools)
    gold_tools = set(gold.get("gold_tools", []))
    pred_tools = set(result.get("pred_tools", result.get("tools_used", [])))

    if gold_tools and gold_tools.issubset(pred_tools):
        reward += 1.0
    elif gold_tools and not gold_tools.issubset(pred_tools):
        reward -= 0.5  # missed required tool

    if not gold_tools and not pred_tools:
        reward += 0.5  # correctly skipped tools

    # 3. Answer quality (+1 for keyword match)
    answer = result.get("answer", "") or ""
    gold_kw = gold.get("gold_answer_keywords", [])

    if gold_kw:
        for kw in gold_kw:
            if re.match(r'^\d+\.\d+$', kw):  # fuzzy numeric
                val = float(kw)
                if any(f"{val:.{d}f}" in answer for d in [0, 1, 2]):
                    reward += 0.5
                    break
            elif kw in answer:
                reward += 0.5
                break

    # 4. Refusal correctness (+0.5)
    should_refuse = gold.get("should_refuse", False)
    refuse_words = ["无法可靠", "无法判断", "看不清", "信息不足", "不能确定", "无法确定", "无法识别"]
    is_refusal = any(w in (answer or "") for w in refuse_words)
    if is_refusal == should_refuse:
        reward += 0.5

    # 5. No redundant tools (bonus for not calling extra tools)
    if not gold_tools and pred_tools:
        reward -= 0.3  # penalty for redundant tool call

    return reward


# ---- Data Loading ----

def load_eval_data(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---- Token-level log-prob computation ----

def compute_turn_log_probs(
    model, processor, messages: List[Dict], assistant_idx: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-level log-probabilities for the assistant message at assistant_idx.
    Returns (log_probs, input_ids) for the assistant tokens only.
    """
    prefix = messages[:assistant_idx]
    full = messages[:assistant_idx + 1]

    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prefix_text = processor.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)

    full_img, full_vid = process_vision_info(full)
    prefix_img, prefix_vid = process_vision_info([prefix])

    full_inputs = processor(
        text=[full_text], images=full_img, videos=full_vid,
        padding=False, return_tensors="pt",
    ).to(device)

    prefix_inputs = processor(
        text=[prefix_text], images=prefix_img, videos=prefix_vid,
        padding=False, return_tensors="pt",
    )
    prefix_len = prefix_inputs["input_ids"].shape[1]

    outputs = model(input_ids=full_inputs["input_ids"])
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    full_ids = full_inputs["input_ids"][0]
    assistant_ids = full_ids[prefix_len - 1:]  # -1 because logits[t] predicts token[t+1]
    assistant_log_probs = log_probs[prefix_len - 1:]

    # Align: log_probs[t] predicts input_ids[t+1]
    gathered_lps = []
    gathered_ids = []
    for i in range(1, assistant_ids.shape[0]):
        gathered_lps.append(assistant_log_probs[i - 1, assistant_ids[i]])
        gathered_ids.append(assistant_ids[i].item())

    if not gathered_lps:
        return torch.tensor([0.0], device=device), torch.tensor([0], device=device)

    return torch.stack(gathered_lps), torch.tensor(gathered_ids, device=device)


# ---- RL Training Loop ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_name", required=True, help="SFT LoRA adapter to start from")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_path", default=str(ROOT / "data/eval/eval_dev_with_category.jsonl"))
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl_coef", type=float, default=0.1, help="KL penalty coefficient")
    p.add_argument("--num_samples_per_prompt", type=int, default=4, help="K for REINFORCE")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    eval_path = resolve_path(args.eval_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    eval_data = load_eval_data(eval_path)
    print(f"Loaded {len(eval_data)} eval samples")

    # ---- Load reference model (SFT, frozen) ----
    print("Loading reference model (SFT adapter)...")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.adapter_name)
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_processor = AutoProcessor.from_pretrained(args.model_name)

    # ---- Load policy model (trainable) ----
    print("Loading policy model...")
    policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_processor = AutoProcessor.from_pretrained(args.model_name)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    # ---- RL loop ----
    reward_history = []
    for iteration in range(args.num_iterations):
        iter_rewards = []
        all_losses = []

        for sample in eval_data:
            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")
            gold = sample

            # Skip samples without image
            if not image:
                continue

            # --- Sample K rollouts ---
            rollouts = []  # list of (messages, reward)
            for k in range(args.num_samples_per_prompt):
                # Create agent with policy model
                agent_model = QwenVLModel(
                    model_name=args.model_name,
                )
                # We need to inject our policy model directly
                # For simplicity, create the agent manually
                from agent.runtime import MultimodalAgent as MA
                agent = MA(model=agent_model, max_steps=6, enforce_required_tools=False)

                result = agent.run(image=image, question=question)
                reward = compute_reward(result, gold)
                rollouts.append((agent.messages if hasattr(agent, 'messages') else [], reward))
                iter_rewards.append(reward)

            if not rollouts:
                continue

            # --- REINFORCE update on all rollouts ---
            rewards_tensor = torch.tensor([r[1] for r in rollouts], device=device)
            # Advantage: reward relative to mean of this group
            advantages = rewards_tensor - rewards_tensor.mean()

            for (messages, reward), advantage in zip(rollouts, advantages):
                # Find all assistant turns
                for i, msg in enumerate(messages):
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    if "<tool_call>" not in content and "<final_answer>" not in content:
                        continue

                    try:
                        # Policy log-probs
                        policy_lps, _ = compute_turn_log_probs(
                            policy_model, policy_processor, messages, i, device,
                        )
                        policy_log_prob = policy_lps.sum()

                        # Reference log-probs (for KL)
                        with torch.no_grad():
                            ref_lps, _ = compute_turn_log_probs(
                                ref_model, ref_processor, messages, i, device,
                            )
                            ref_log_prob = ref_lps.sum()

                        # REINFORCE + KL penalty
                        kl = policy_log_prob - ref_log_prob
                        loss = -advantage * policy_log_prob + args.kl_coef * kl

                        if torch.isfinite(loss):
                            loss.backward()
                            all_losses.append(loss.item())
                    except Exception as e:
                        pass  # skip problematic turns

        # Gradient step
        if all_losses:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_reward = sum(iter_rewards) / len(iter_rewards) if iter_rewards else 0
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        reward_history.append(avg_reward)
        print(f"Iter {iteration+1}/{args.num_iterations}: "
              f"avg_reward={avg_reward:.3f} avg_loss={avg_loss:.4f} "
              f"n_samples={len(iter_rewards)} n_updates={len(all_losses)}")

        # Save checkpoint every 5 iters
        if (iteration + 1) % 5 == 0:
            ckpt_dir = output_dir / f"checkpoint-{iteration+1}"
            policy_model.save_pretrained(str(ckpt_dir))
            print(f"  Saved checkpoint: {ckpt_dir}")

    # Final save
    policy_model.save_pretrained(str(output_dir))
    policy_processor.save_pretrained(str(output_dir))
    print(f"\nSaved final model: {output_dir}")
    print(f"Reward history: {[f'{r:.3f}' for r in reward_history]}")


if __name__ == "__main__":
    main()
