"""
DPO (Direct Preference Optimization) LoRA trainer for Qwen2.5-VL.

Loads an SFT LoRA adapter as the reference model, then trains a policy
LoRA to prefer chosen responses over rejected ones.

Usage:
  python train/train_dpo_lora.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter_name /root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v2 \
    --dpo_file data/preference/dpo_seed.jsonl \
    --output_dir /root/autodl-tmp/multimodal-agent-lora/lora_dpo_v1 \
    --num_train_epochs 2 --learning_rate 5e-5 --dpo_beta 0.1
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    def __init__(self, path: Path):
        self.samples = self._load(path)
        if not self.samples:
            raise ValueError(f"empty DPO file: {path}")
        print(f"Loaded {len(self.samples)} DPO samples.")

    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{path} line {line_no} JSON parse failed: {e}"
                    ) from e
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def resolve_image_paths(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Log-probability helpers
# ---------------------------------------------------------------------------

def compute_log_prob(
    model,
    processor,
    messages: List[Dict[str, Any]],
    max_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute the total log-probability of the last assistant turn in `messages`.

    Returns a scalar tensor (sum of token-level log-probs) that preserves
    gradients when called on a trainable model.
    """
    device = model.device

    # Find the last assistant message index
    assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            assistant_idx = i
            break
    if assistant_idx is None:
        return torch.tensor(0.0, device=device)

    prefix = messages[:assistant_idx]
    full = messages[: assistant_idx + 1]

    full_text = processor.apply_chat_template(
        full, tokenize=False, add_generation_prompt=False,
    )
    prefix_text = processor.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True,
    )

    full_img, full_vid = process_vision_info(full)
    prefix_img, prefix_vid = process_vision_info([prefix])

    full_inputs = processor(
        text=[full_text],
        images=full_img,
        videos=full_vid,
        padding=False,
        return_tensors="pt",
    ).to(device)

    prefix_inputs = processor(
        text=[prefix_text],
        images=prefix_img,
        videos=prefix_vid,
        padding=False,
        return_tensors="pt",
    )

    prefix_len = prefix_inputs["input_ids"].shape[1]
    full_ids = full_inputs["input_ids"][0]

    outputs = model(input_ids=full_inputs["input_ids"])
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather log-probs for assistant tokens
    gathered = []
    for t in range(prefix_len, full_ids.shape[0]):
        token_id = full_ids[t].item()
        if t > 0 and t - 1 < log_probs.shape[0]:
            gathered.append(log_probs[t - 1, token_id])

    if not gathered:
        return torch.tensor(0.0, device=device)

    return torch.stack(gathered).sum()


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

class DPOTrainer(Trainer):
    def __init__(
        self,
        ref_model: Qwen2_5_VLForConditionalGeneration,
        processor: Any,
        max_length: int,
        dpo_beta: float,
        dtype: torch.dtype,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.processor = processor
        self.max_length = max_length
        self.dpo_beta = dpo_beta
        self.compute_dtype = dtype

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute DPO loss for a batch.

        inputs is a dict from the collator; we decode the raw samples
        and compute log-probs on the fly.
        """
        samples = inputs.get("samples", [])
        if not samples:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        losses = []
        for sample in samples:
            prompt_msgs = resolve_image_paths(sample["prompt"])
            chosen_msgs = sample["chosen"]
            rejected_msgs = sample["rejected"]

            # Build full conversations
            full_chosen = list(prompt_msgs) + list(chosen_msgs)
            full_rejected = list(prompt_msgs) + list(rejected_msgs)

            # Policy log-probs (differentiable)
            chosen_lp = compute_log_prob(
                model, self.processor, full_chosen,
                self.max_length, self.compute_dtype,
            )
            rejected_lp = compute_log_prob(
                model, self.processor, full_rejected,
                self.max_length, self.compute_dtype,
            )

            # Reference log-probs (frozen, no gradients)
            with torch.no_grad():
                ref_chosen_lp = compute_log_prob(
                    self.ref_model, self.processor, full_chosen,
                    self.max_length, self.compute_dtype,
                )
                ref_rejected_lp = compute_log_prob(
                    self.ref_model, self.processor, full_rejected,
                    self.max_length, self.compute_dtype,
                )

            # DPO loss: all terms are tensors, policy terms carry gradients
            log_ratio = (
                self.dpo_beta
                * (chosen_lp - ref_chosen_lp - rejected_lp + ref_rejected_lp)
            )
            loss = -F.logsigmoid(log_ratio)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Collator: just passes through raw samples
# ---------------------------------------------------------------------------

@dataclass
class DPOCollator:
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"samples": samples}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--adapter_name",
        required=True,
        help="Path to SFT LoRA adapter (used as reference model).",
    )
    parser.add_argument("--dpo_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dpo_file = Path(args.dpo_file)
    output_dir = Path(args.output_dir)
    if not dpo_file.is_absolute():
        dpo_file = ROOT / dpo_file
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name)

    # --- Policy model (trainable) ---
    print("Loading base model for policy...")
    policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    # --- Reference model (frozen: base + SFT adapter) ---
    print("Loading reference model (SFT adapter)...")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.adapter_name)
    ref_model = ref_model.merge_and_unload()  # merge SFT LoRA into base weights
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference model frozen.")

    dataset = DPODataset(dpo_file)
    collator = DPOCollator()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = DPOTrainer(
        ref_model=ref_model,
        processor=processor,
        max_length=args.max_length,
        dpo_beta=args.dpo_beta,
        dtype=dtype,
        model=policy_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    policy_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved DPO LoRA adapter and processor to: {output_dir}")


if __name__ == "__main__":
    main()
