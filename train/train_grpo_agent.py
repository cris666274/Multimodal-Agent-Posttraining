"""
GRPO training for multimodal agent.

Strategy: Decompose multi-turn tool calling into single-turn decisions.
For each assistant turn in a trace:
  - prompt = all messages BEFORE this turn
  - the model generates ONE action (<tool_call> or <final_answer>)
  - reward scores the action based on format + correctness

Uses TRL GRPOTrainer which handles on-policy generation + reward scoring.

Usage:
  python train/train_grpo_agent.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4 \
    --output_dir /root/autodl-tmp/multimodal-agent-lora/lora_grpo_v1 \
    --num_epochs 3 --lr 5e-6
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.runtime import MultimodalAgent
from agent.tools import set_vlm_model


# ---- Step 1: Build single-turn GRPO prompts from agent traces ----

def build_grpo_dataset(
    eval_path: Path,
    model_name: str,
    adapter_name: str,
    num_rollouts: int = 2,
) -> List[Dict]:
    """
    Run the SFT agent on each eval sample, decompose the trace into
    single-turn GRPO training examples.
    """
    eval_data = _read_jsonl(eval_path)
    model = QwenVLModel(model_name=model_name, adapter_name=adapter_name, max_new_tokens=256)
    set_vlm_model(model)
    agent = MultimodalAgent(model=model, max_steps=6, enforce_required_tools=False)

    grpo_examples = []
    for sample in eval_data:
        image = sample.get("image") or sample.get("image_path")
        question = sample.get("question", "")
        if not image:
            continue

        for _ in range(num_rollouts):
            result = agent.run(image=image, question=question)
            trace = result.get("trace", [])
            if not trace:
                continue

            # Reconstruct messages from trace
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]},
            ]
            turn_num = 0
            for step in trace:
                if "model_output" in step:
                    output = step["model_output"]
                    if output and output.strip():
                        messages.append({"role": "assistant", "content": output})
                        turn_num += 1
                if "tool_name" in step:
                    obs = str(step.get("observation", ""))
                    messages.append({"role": "tool", "name": step["tool_name"], "content": obs})
                    # Add user continue prompt (matching runtime behavior)
                    messages.append({"role": "user", "content": _continue_prompt(step["tool_name"])})

            # For each assistant turn, extract (prompt=context, reference=output)
            for i, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                prefix = messages[:i]
                if not prefix:
                    continue
                grpo_examples.append({
                    "prompt": prefix,
                    "reference": content,
                    "gold_tools": sample.get("gold_tools", []),
                    "gold_kw": sample.get("gold_answer_keywords", []),
                    "should_refuse": sample.get("should_refuse", False),
                })

    print(f"Built {len(grpo_examples)} GRPO examples from {len(eval_data)} eval samples")
    return grpo_examples


def _continue_prompt(tool_name: str) -> str:
    return (f"工具 {tool_name} 返回结果如下。"
            "请继续，只输出一个 <tool_call> 或 <final_answer>。")


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---- Step 2: Reward function ----

def reward_format(completions: List[str], **kwargs) -> List[float]:
    """Score format: +1 for valid tool_call or final_answer, -2 for parse error."""
    rewards = []
    for text in completions:
        has_tool = "<tool_call>" in text
        has_final = "<final_answer>" in text
        if has_tool or has_final:
            # Bonus for including reasoning prefix (teaches decision awareness)
            has_reasoning = len(text.split("<tool_call>")[0].strip()) > 10 if has_tool else False
            has_reasoning = has_reasoning or (len(text.split("<final_answer>")[0].strip()) > 10 if has_final else False)
            rewards.append(1.5 if has_reasoning else 1.0)
        else:
            rewards.append(-2.0)
    return rewards


def _parse_metadata(kwargs, key, default=None):
    """Extract field from metadata JSON string passed by GRPOTrainer."""
    meta_str = kwargs.get("metadata", "{}")
    if isinstance(meta_str, str):
        try:
            meta = json.loads(meta_str if isinstance(meta_str, str) else meta_str[0])
            return meta.get(key, default)
        except:
            pass
    return default


def reward_tool_choice(completions: List[str], metadata=None, **kwargs) -> List[float]:
    """Score tool selection: +1 if correct tool called, 0 otherwise."""
    rewards = []
    # metadata is a list of JSON strings, one per completion in the batch
    if isinstance(metadata, (list, tuple)):
        meta_items = [json.loads(m) if isinstance(m, str) else m for m in metadata]
    else:
        meta_items = [{}] * len(completions)
    for i, text in enumerate(completions):
        gold_tools = meta_items[i].get("gold_tools", []) if i < len(meta_items) else []
        # Extract tool name from text
        tool_name = None
        if "<tool_call>" in text:
            try:
                json_str = text.split("<tool_call>")[1].split("</tool_call>")[0]
                tool_name = json.loads(json_str).get("name", "")
            except:
                pass
        if "<final_answer>" in text:
            tool_name = "final_answer"

        if gold_tools and tool_name in gold_tools:
            rewards.append(1.0)
        elif not gold_tools and tool_name == "final_answer":
            rewards.append(0.5)  # correctly didn't call tools
        elif tool_name is None:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


def reward_answer_quality(completions: List[str], metadata=None, **kwargs) -> List[float]:
    """Score answer quality: keyword matching with fuzzy chart support."""
    rewards = []
    if isinstance(metadata, (list, tuple)):
        meta_items = [json.loads(m) if isinstance(m, str) else m for m in metadata]
    else:
        meta_items = [{}] * len(completions)
    for i, text in enumerate(completions):
        gold_kw = meta_items[i].get("gold_kw", []) if i < len(meta_items) else []
        if not gold_kw or "<final_answer>" not in text:
            rewards.append(0.0)
            continue

        answer = text.split("<final_answer>")[1].split("</final_answer>")[0] if "</final_answer>" in text else text
        score = 0.0
        for kw in gold_kw:
            if re.match(r'^\d+\.\d+$', kw):
                val = float(kw)
                if any(f"{val:.{d}f}" in answer for d in [0, 1, 2]):
                    score += 0.5
                    break
            elif kw in answer:
                score += 0.5
        rewards.append(score)
    return rewards


# ---- Step 3: Training ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_path", default=str(ROOT / "data/eval/eval_dev_with_category.jsonl"))
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--num_rollouts", type=int, default=2, help="rollouts per eval sample for data building")
    p.add_argument("--num_generations", type=int, default=4, help="K: generations per prompt in GRPO")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()


def resolve_path(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def main():
    args = parse_args()
    eval_path = resolve_path(args.eval_path)
    output_dir = resolve_path(args.output_dir)
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ---- Build GRPO dataset from SFT agent rollouts ----
    print("Building GRPO dataset from SFT agent rollouts...")
    grpo_data = build_grpo_dataset(
        eval_path=eval_path,
        model_name=args.model_name,
        adapter_name=args.adapter_name,
        num_rollouts=args.num_rollouts,
    )
    # Build a simple list-based dataset that GRPOTrainer can use.
    # Each item: {"prompt": list_of_messages, "metadata": json_string}
    class AgentGRPODataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    dataset = AgentGRPODataset([
        {
            "prompt": ex["prompt"],  # list of message dicts
            "metadata": json.dumps({
                "gold_tools": ex["gold_tools"],
                "gold_kw": ex["gold_kw"],
                "should_refuse": ex["should_refuse"],
            }, ensure_ascii=False),
        }
        for ex in grpo_data
    ])
    print(f"Dataset: {len(dataset)} examples")

    # ---- Load model with LoRA ----
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto",
    )
    # Load SFT adapter and merge
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_name)
    model = model.merge_and_unload()

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # ---- GRPO config ----
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=5,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        temperature=0.8,
    )

    # ---- GRPO Trainer ----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_tool_choice, reward_answer_quality],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
