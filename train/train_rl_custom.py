"""
GRPO-style RL training for multimodal agent (self-contained, no TRL).

Key design:
- Reference model: frozen SFT (merged). Anchors format across all iterations.
- Policy model: SFT adapter loaded as trainable LoRA via PeftModel(is_trainable=True).
- Rollouts: policy_model.generate() directly with do_sample=True, temp=0.7, top_p=0.9.
- Agent: QwenVLModel + MultimodalAgent created once, reused. Tool calls delegate to policy.
- Data split: rl_train.jsonl (40) / eval_dev.jsonl (50, unchanged) / eval_test.jsonl (0).

Usage:
  python train/train_rl_custom.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4 \
    --output_dir /root/autodl-tmp/multimodal-agent-lora/lora_grpo_v1 \
    --num_iterations 20 --lr 1e-5 --kl_coef 0.05
"""

import argparse, json, math, os, random, re, sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import TOOLS, set_vlm_model
from agent.parser import parse_model_output
from agent.runtime import MultimodalAgent
from agent.vlm import QwenVLModel

GRPO_K = 4
GRPO_CLIP = 0.2
MAX_STEPS = 6

random.seed(42)

# =====================================================================
# Agent wrapper: injects policy_model into a QwenVLModel-like interface
# =====================================================================

class PolicyModelWrapper:
    """
    Mimics QwenVLModel interface but delegates generate() to an external
    policy_model + processor, so rollout tokens come from the RL policy.
    """
    def __init__(self, policy_model, processor, max_new_tokens=256):
        self._policy = policy_model
        self._processor = processor
        self.max_new_tokens = max_new_tokens
        self.device = next(policy_model.parameters()).device

    def generate(self, messages: List[Dict]) -> str:
        # Prepend system prompt so the model knows the output format
        from agent.prompts import SYSTEM_PROMPT
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        text = self._processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True,
        )
        imgs, vids = process_vision_info(full_messages)
        inputs = self._processor(
            text=[text], images=imgs, videos=vids,
            padding=False, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._policy.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_ids, skip_special_tokens=True).strip()


# =====================================================================
# Reward function
# =====================================================================

def compute_reward(result: Dict, gold: Dict) -> float:
    reward = 0.0
    pred_tools = list(result.get("pred_tools", []))
    gold_tools = set(gold.get("gold_tools", []))
    pred_set = set(pred_tools)
    answer = result.get("answer", "") or ""

    if result.get("status") == "parse_error":
        reward -= 2.0
    if gold_tools and gold_tools.issubset(pred_set):
        reward += 1.0
    elif gold_tools:
        reward -= 0.5
    if not gold_tools and not pred_set:
        reward += 0.5

    gold_kw = gold.get("gold_answer_keywords", [])
    if gold_kw:
        for kw in gold_kw:
            if re.match(r'^\d+\.\d+$', kw):
                val = float(kw)
                if any(f"{val:.{d}f}" in answer for d in [0, 1, 2]):
                    reward += 0.5; break
            elif kw in answer:
                reward += 0.5; break

    should_refuse = gold.get("should_refuse", False)
    refuse_words = ["无法可靠","无法判断","看不清","信息不足","不能确定","无法确定","无法识别"]
    if should_refuse == any(w in answer for w in refuse_words):
        reward += 0.5
    if not gold_tools and pred_set:
        reward -= 0.3
    return reward


# =====================================================================
# Log-prob of assistant turn (with full visual inputs)
# =====================================================================

def compute_turn_log_prob(model, processor, messages: List[Dict], turn_idx: int) -> torch.Tensor:
    prefix = messages[:turn_idx]
    full = messages[:turn_idx + 1]

    full = _resolve_images(full)
    prefix = _resolve_images(prefix)

    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prefix_text = processor.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)

    full_img, full_vid = process_vision_info(full)
    prefix_img, prefix_vid = process_vision_info(prefix)

    full_inputs = processor(
        text=[full_text], images=full_img, videos=full_vid,
        padding=False, return_tensors="pt",
    ).to(model.device)

    prefix_inputs = processor(
        text=[prefix_text], images=prefix_img, videos=prefix_vid,
        padding=False, return_tensors="pt",
    )
    prefix_len = prefix_inputs["input_ids"].shape[1]

    outputs = model(**{k: v for k, v in full_inputs.items()})
    logits = outputs.logits[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    full_ids = full_inputs["input_ids"][0]
    gathered = []
    for t in range(prefix_len, min(full_ids.shape[0], log_probs.shape[0] + 1)):
        if t > 0:
            gathered.append(log_probs[t - 1, full_ids[t]])
    if not gathered:
        return torch.tensor(0.0, device=model.device)
    return torch.stack(gathered).sum()


def _resolve_images(messages: List[Dict]) -> List[Dict]:
    resolved = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image", "")
                    p = Path(img)
                    if not p.is_absolute():
                        p = ROOT / img
                    new_item = dict(item)
                    new_item["image"] = str(p) if p.exists() else img
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            new_msg = dict(msg)
            new_msg["content"] = new_content
            resolved.append(new_msg)
        else:
            resolved.append(msg)
    return resolved


# =====================================================================
# GRPO loss
# =====================================================================

def grpo_loss(
    new_lps: torch.Tensor,    # \pi_\theta (current policy)
    old_lps: torch.Tensor,    # \pi_old (previous-iteration policy, detached)
    ref_lps: torch.Tensor,    # \pi_ref (SFT, frozen)
    advantages: torch.Tensor,
    kl_coef: float = 0.05,
) -> torch.Tensor:
    # Ratio: \pi_\theta / \pi_old (standard GRPO)
    log_ratio = (new_lps - old_lps).clamp(-5.0, 5.0)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - GRPO_CLIP, 1 + GRPO_CLIP)
    # GRPO objective
    grpo = -torch.min(ratio * advantages, clipped * advantages).mean()
    # KL: \pi_\theta relative to \pi_ref (keep policy near SFT)
    kl = (new_lps - ref_lps).mean()
    return grpo + kl_coef * kl


# =====================================================================
# Data
# =====================================================================

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_rl_train():
    """Create rl_train.jsonl (40 samples) from eval data, keeping eval_dev unchanged."""
    rl_path = ROOT / "data/eval/rl_train.jsonl"
    if rl_path.exists():
        print(f"Using existing: {rl_path}")
        return

    dev_path = ROOT / "data/eval/eval_dev_with_category.jsonl"
    all_data = load_jsonl(dev_path)
    random.shuffle(all_data)
    rl_train = all_data[:40]
    write_jsonl(rl_path, rl_train)
    print(f"Created {rl_path} ({len(rl_train)} samples)")

    # Keep 10 as holdout (optional, not used during RL)
    holdout_path = ROOT / "data/eval/rl_holdout.jsonl"
    write_jsonl(holdout_path, all_data[40:])
    print(f"Created {holdout_path} ({len(all_data) - 40} samples for holdout)")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rl_train_path", default=str(ROOT/"data/eval/rl_train.jsonl"))
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--ref_on_cpu", action="store_true", help="Put ref model on CPU (for 7B on 96GB)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data split ----
    split_rl_train()
    train_data = load_jsonl(Path(args.rl_train_path))
    train_data = [s for s in train_data if s.get("image") or s.get("image_path")]
    print(f"RL train: {len(train_data)} samples")

    # ---- Processor ----
    processor = AutoProcessor.from_pretrained(args.model_name)

    # ---- Reference model: SFT merged, frozen ----
    ref_device = "cpu" if args.ref_on_cpu else "auto"
    ref_dtype = torch.float32 if args.ref_on_cpu else dtype
    print(f"Reference model (SFT -> merge -> freeze, device={ref_device})...")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=ref_dtype, device_map=ref_device,
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.adapter_name)
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---- Policy model: base + SFT LoRA (trainable) ----
    print("Policy model (base + SFT adapter, trainable)...")
    policy_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
    )
    policy_model = PeftModel.from_pretrained(
        policy_base, args.adapter_name, is_trainable=True,
    )
    policy_model.train()

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad], lr=args.lr,
    )

    # ---- Agent (policy_model injected, reused across rollouts) ----
    wrapper = PolicyModelWrapper(policy_model, processor)
    set_vlm_model(wrapper)
    agent = MultimodalAgent(model=wrapper, max_steps=MAX_STEPS, enforce_required_tools=False)

    reward_history = []

    # ---- GRPO Loop (REINFORCE + KL; no cross-iteration ratio) ----
    for iteration in range(args.num_iterations):
        print(f"\n--- Iter {iteration+1} start ---", flush=True)
        all_rewards = []
        all_losses = []
        grad_count = 0

        for si, sample in enumerate(train_data):
            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")

            # ===== Step 1: K rollouts from policy =====
            rollouts = []
            for ki in range(GRPO_K):
                try:
                    result = agent.run(image=image, question=question)
                except Exception as e:
                    print(f"  [CRASH] sample {si} rollout {ki}: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    continue
                reward = compute_reward(result, sample)
                messages = _reconstruct_messages(result, image, question)
                rollouts.append({
                    "messages": messages,
                    "reward": reward,
                })
                all_rewards.append(reward)

            if not rollouts:
                continue

            if (si + 1) % 10 == 0:
                avg_r = sum(r["reward"] for r in rollouts) / len(rollouts)
                print(f"  sample {si+1}/{len(train_data)} done, avg_reward={avg_r:.2f}", flush=True)

            # ===== Step 2: Group-relative advantage =====
            r_t = torch.tensor([r["reward"] for r in rollouts], device=device)
            std = r_t.std()
            advantages = (r_t - r_t.mean()) / (std + 1e-8) if std > 1e-8 else torch.zeros_like(r_t)

            # ===== Step 3: REINFORCE + KL loss (ratio=1, no cross-iteration replay) =====
            for rollout, advantage in zip(rollouts, advantages):
                msgs = rollout["messages"]
                if not msgs:
                    continue
                turn_plps, turn_rlps = [], []
                for i, msg in enumerate(msgs):
                    if msg["role"] != "assistant":
                        continue
                    c = msg.get("content", "")
                    if not isinstance(c, str) or ("<tool_call>" not in c and "<final_answer>" not in c):
                        continue
                    try:
                        # Policy log-prob (with gradient)
                        plp = compute_turn_log_prob(policy_model, processor, msgs, i)
                        turn_plps.append(plp)
                        # Reference log-prob (SFT, no gradient)
                        with torch.no_grad():
                            rlp = compute_turn_log_prob(ref_model, processor, msgs, i)
                            turn_rlps.append(rlp.to(device) if rlp.device != device else rlp)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            raise
                        print(f"  [WARN] {sample.get('id','?')} turn {i}: {str(e)[:80]}")
                        continue
                if not turn_plps:
                    continue
                plps = torch.stack(turn_plps)
                rlps = torch.stack(turn_rlps)
                # REINFORCE: push up high-advantage, push down low-advantage
                reinforce = -advantage * plps.mean()
                # KL: stay close to SFT
                kl = (plps - rlps).mean()
                loss = reinforce + args.kl_coef * kl
                if torch.isfinite(loss):
                    loss.backward()
                    all_losses.append(loss.item())
                    grad_count += 1

        # ===== Step 4: Update =====
        if grad_count > 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_r = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        avg_l = sum(all_losses) / len(all_losses) if all_losses else 0
        reward_history.append(avg_r)
        print(f"Iter {iteration+1}/{args.num_iterations}: "
              f"reward={avg_r:.3f} loss={avg_l:.4f} grads={grad_count}")

        if (iteration + 1) % 5 == 0:
            ckpt = output_dir / f"ckpt_{iteration+1}"
            policy_model.save_pretrained(str(ckpt))
            processor.save_pretrained(str(ckpt))
            print(f"  -> {ckpt}")

    # ---- Final ----
    final = output_dir / "final"
    policy_model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"\n{final}")
    print(f"Rewards: {[f'{r:.3f}' for r in reward_history]}")


def _reconstruct_messages(result: Dict, image: str, question: str) -> List[Dict]:
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
            obs = str(step.get("observation", ""))
            messages.append({"role": "tool", "name": step["tool_name"], "content": obs})
            messages.append({"role": "user", "content": f"工具 {step['tool_name']} 返回结果。请继续。"})
    return messages


if __name__ == "__main__":
    main()
