"""
Iterative SFT with rejection sampling --- a practical form of RL for agent tasks.

Unlike REINFORCE (which needs token-level gradients through multi-turn rollouts),
this approach:

1. Runs the full agent on each eval sample with the CURRENT policy (K rollouts)
2. Scores each rollout with the multi-dimension reward function
3. Keeps HIGH-reward traces as positive SFT data
4. Mixes with original SFT data to prevent forgetting
5. Trains one SFT epoch on the combined data
6. Repeats for N iterations

This combines:
- Exploration: K rollouts per sample discover diverse strategies
- Exploitation: only high-reward traces become training data
- Stability: SFT updates preserve format discipline

Usage:
  python train/train_iterative_sft.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4 \
    --output_dir /root/autodl-tmp/multimodal-agent-lora/lora_rl_iter_v1 \
    --num_iterations 10 --lr 5e-5
"""

import argparse, json, re, sys, copy, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    Trainer, TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.runtime import MultimodalAgent
from agent.tools import set_vlm_model

# ---- Reward Function ----
# Same as before: multi-dimension scoring

def compute_reward(result: Dict, gold: Dict) -> float:
    reward = 0.0

    if result.get("status") == "parse_error":
        reward -= 2.0

    gold_tools = set(gold.get("gold_tools", []))
    pred_tools = set(result.get("pred_tools", []))

    if gold_tools and gold_tools.issubset(pred_tools):
        reward += 1.0
    elif gold_tools:
        reward -= 0.5

    if not gold_tools and not pred_tools:
        reward += 0.5

    answer = result.get("answer", "") or ""
    gold_kw = gold.get("gold_answer_keywords", [])

    if gold_kw:
        for kw in gold_kw:
            if re.match(r'^\d+\.\d+$', kw):
                val = float(kw)
                if any(f"{val:.{d}f}" in answer for d in [0, 1, 2]):
                    reward += 0.5
                    break
            elif kw in answer:
                reward += 0.5
                break

    should_refuse = gold.get("should_refuse", False)
    refuse_words = ["无法可靠", "无法判断", "看不清", "信息不足", "不能确定", "无法确定", "无法识别"]
    is_refusal = any(w in (answer or "") for w in refuse_words)
    if is_refusal == should_refuse:
        reward += 0.5

    if not gold_tools and pred_tools:
        reward -= 0.3

    return reward


# ---- Utils ----

def read_jsonl(path: Path) -> List[Dict]:
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


def agent_trace_to_sft_messages(result: Dict, image: str, question: str) -> Optional[List[Dict]]:
    """Convert agent trace back to trainable SFT messages."""
    trace = result.get("trace", [])
    if not trace:
        return None

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    for step in trace:
        if "model_output" in step:
            output = step["model_output"]
            if output and output.strip():
                messages.append({"role": "assistant", "content": output})
        if "tool_name" in step:
            tool_result = step.get("observation", "")
            messages.append({"role": "tool", "name": step["tool_name"], "content": str(tool_result)})
        if "event" in step and "correction" in str(step.get("event", "")).lower():
            event = step.get("event", "")
            if "blocked" in str(event) or "correction" in str(event):
                reason = step.get("reason", "")
                messages.append({"role": "user", "content": str(reason)})

    # Ensure at least last message is from assistant
    if messages and messages[-1]["role"] != "assistant":
        return None

    return messages


# ---- SFT Dataset (same as train_sft_lora.py) ----

class AgentSFTDataset(Dataset):
    def __init__(self, rows: List[Dict], expand_assistant_turns: bool = True):
        self.rows = self._expand(rows) if expand_assistant_turns else rows
        if not self.rows:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    @staticmethod
    def _expand(rows):
        expanded = []
        for row in rows:
            messages = row.get("messages", [])
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    sample = dict(row)
                    sample["id"] = f"{row.get('id','s')}_t{i}"
                    sample["messages"] = messages[:i+1]
                    sample["target_turn_index"] = i
                    expanded.append(sample)
        return expanded


class QwenVLSFTCollator:
    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        full_texts, prefix_texts = [], []
        for ex in examples:
            messages = self._resolve_paths(ex["messages"])
            target = self._last_assistant_idx(messages)
            prefix = messages[:target]
            full = messages[:target+1]
            prefix_texts.append(self.processor.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(
                full, tokenize=False, add_generation_prompt=False))

        full_img, full_vid = process_vision_info(
            [self._resolve_paths(ex["messages"]) for ex in examples])
        inputs = self.processor(
            text=full_texts, images=full_img, videos=full_vid,
            padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for i, ex in enumerate(examples):
            prefix = self._resolve_paths(ex["messages"])[:self._last_assistant_idx(ex["messages"])]
            p_img, p_vid = process_vision_info([prefix])
            p_inputs = self.processor(
                text=[prefix_texts[i]], images=p_img, videos=p_vid,
                padding=False, return_tensors="pt")
            labels[i, :p_inputs["input_ids"].shape[1]] = -100

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs

    @staticmethod
    def _resolve_paths(messages):
        resolved = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                resolved.append(msg)
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image")
                    if isinstance(image, str) and not Path(image).is_absolute():
                        new_item = dict(item)
                        new_item["image"] = str(ROOT / image)
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            new_msg = dict(msg)
            new_msg["content"] = new_content
            resolved.append(new_msg)
        return resolved

    @staticmethod
    def _last_assistant_idx(messages):
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "assistant":
                return i
        raise ValueError("no assistant message")


# ---- Main ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_path", default=str(ROOT/"data/eval/eval_dev_with_category.jsonl"))
    p.add_argument("--base_sft_path", default=str(ROOT/"data/sft/sft_agent_train_real_v4_enforced_resized.jsonl"))
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_rollouts", type=int, default=4, help="K rollouts per sample")
    p.add_argument("--reward_threshold", type=float, default=0.5, help="Min reward to keep trace")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_length", type=int, default=1024)
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    eval_path = resolve_path(args.eval_path)
    base_sft_path = resolve_path(args.base_sft_path)
    output_dir = resolve_path(args.output_dir)

    eval_data = read_jsonl(eval_path)
    base_sft_data = read_jsonl(base_sft_path)
    print(f"Eval: {len(eval_data)}, Base SFT: {len(base_sft_data)}")

    processor = AutoProcessor.from_pretrained(args.model_name)
    reward_history = []
    current_adapter = args.adapter_name  # start from SFT adapter

    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration+1}/{args.num_iterations}")
        print(f"{'='*60}")

        # Load model with current adapter
        model = QwenVLModel(
            model_name=args.model_name,
            adapter_name=current_adapter,
            max_new_tokens=256,
        )
        set_vlm_model(model)
        agent = MultimodalAgent(model=model, max_steps=6, enforce_required_tools=False)

        # --- Rollout phase: sample K traces per eval sample ---
        rollout_traces = []
        all_rewards = []
        for sample in eval_data:
            image = sample.get("image") or sample.get("image_path")
            question = sample.get("question", "")
            if not image:
                continue

            for k in range(args.num_rollouts):
                result = agent.run(image=image, question=question)
                reward = compute_reward(result, sample)
                all_rewards.append(reward)

                if reward >= args.reward_threshold:
                    messages = agent_trace_to_sft_messages(result, image, question)
                    if messages:
                        rollout_traces.append({
                            "id": f"rl_{iteration}_{sample['id']}_{k}",
                            "source": f"iterative_rl_iter{iteration}",
                            "category": sample.get("category", "?"),
                            "reward": reward,
                            "messages": messages,
                        })

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        reward_history.append(avg_reward)

        # Stats
        high_reward = sum(1 for r in all_rewards if r >= args.reward_threshold)
        print(f"Avg reward: {avg_reward:.3f} | "
              f"High-reward traces: {high_reward}/{len(all_rewards)} ({high_reward/len(all_rewards):.1%}) | "
              f"Kept: {len(rollout_traces)}")

        if not rollout_traces:
            print("No high-reward traces. Skipping SFT step.")
            continue

        # --- SFT phase: train on base data + high-reward traces ---
        train_rows = list(base_sft_data) + rollout_traces
        random.shuffle(train_rows)

        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )

        # Load fresh base model each iteration
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto",
        )
        # Start from the current adapter
        base_model = PeftModel.from_pretrained(base_model, current_adapter)
        base_model = base_model.merge_and_unload()

        peft_model = get_peft_model(base_model, lora_config)

        dataset = AgentSFTDataset(train_rows, expand_assistant_turns=True)
        collator = QwenVLSFTCollator(processor, max_length=args.max_length)

        training_args = TrainingArguments(
            output_dir=str(output_dir / f"iter_{iteration}"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            num_train_epochs=1,
            logging_steps=5,
            save_steps=999999,
            bf16=args.bf16,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=peft_model, args=training_args,
            train_dataset=dataset, data_collator=collator,
        )
        trainer.train()

        # Save and update adapter path for next iteration
        iter_dir = output_dir / f"iter_{iteration}"
        peft_model.save_pretrained(str(iter_dir))
        processor.save_pretrained(str(iter_dir))
        current_adapter = str(iter_dir)
        print(f"Saved iteration model: {iter_dir}")

        # Clean up GPU
        del base_model, peft_model, trainer
        torch.cuda.empty_cache()

    # Final save
    peft_model.save_pretrained(str(output_dir / "final"))
    print(f"\nFinal model: {output_dir}/final")
    print(f"Reward history: {[f'{r:.3f}' for r in reward_history]}")
    print(f"Reward trend: {reward_history[0]:.3f} -> {reward_history[-1]:.3f}")


if __name__ == "__main__":
    main()
