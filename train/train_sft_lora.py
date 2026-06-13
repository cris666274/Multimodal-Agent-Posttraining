import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {line_no} JSON parse failed: {e}") from e
    return rows


class AgentSFTDataset(Dataset):
    def __init__(self, path: Path, expand_assistant_turns: bool = True):
        raw_rows = read_jsonl(path)
        self.rows = (
            self._expand_assistant_turns(raw_rows)
            if expand_assistant_turns
            else raw_rows
        )
        if not self.rows:
            raise ValueError(f"empty train file: {path}")

        print(f"Loaded {len(raw_rows)} raw rows, {len(self.rows)} train rows.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def _expand_assistant_turns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded = []
        for row in rows:
            messages = row.get("messages", [])
            for index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue

                sample = dict(row)
                sample["id"] = f"{row.get('id', 'sample')}_turn_{index}"
                sample["messages"] = messages[: index + 1]
                sample["target_turn_index"] = index
                expanded.append(sample)

        return expanded


def assistant_target_index(messages: List[Dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("sample has no assistant message")


def resolve_image_paths(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resolved = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            resolved.append(message)
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

        new_message = dict(message)
        new_message["content"] = new_content
        resolved.append(new_message)

    return resolved


@dataclass
class QwenVLSFTCollator:
    processor: Any
    max_length: int

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prefix_texts = []
        full_messages = []
        prefix_messages = []

        for example in examples:
            messages = resolve_image_paths(example["messages"])
            target_index = assistant_target_index(messages)

            # Prefix includes the original context and a generation marker, but not the target answer.
            prefix = messages[:target_index]
            full = messages[: target_index + 1]

            full_messages.append(full)
            prefix_messages.append(prefix)
            full_texts.append(
                self.processor.apply_chat_template(
                    full,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prefix_texts.append(
                self.processor.apply_chat_template(
                    prefix,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        full_image_inputs, full_video_inputs = process_vision_info(full_messages)
        inputs = self.processor(
            text=full_texts,
            images=full_image_inputs,
            videos=full_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        for row_index, prefix in enumerate(prefix_messages):
            prefix_image_inputs, prefix_video_inputs = process_vision_info([prefix])
            prefix_inputs = self.processor(
                text=[prefix_texts[row_index]],
                images=prefix_image_inputs,
                videos=prefix_video_inputs,
                padding=False,
                return_tensors="pt",
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]
            labels[row_index, :prefix_len] = -100

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--no_expand_assistant_turns",
        action="store_true",
        help="Only train the last assistant turn in each sample.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--adapter_name",
        default=None,
        help="Path to existing LoRA adapter to fine-tune from (instead of training from scratch).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)

    if not train_file.is_absolute():
        train_file = ROOT / train_file
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    processor = AutoProcessor.from_pretrained(args.model_name)
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load existing adapter for fine-tuning
    if args.adapter_name:
        print(f"Loading existing LoRA adapter for fine-tuning: {args.adapter_name}")
        model = PeftModel.from_pretrained(model, args.adapter_name)
        # Merge and reapply LoRA for continued training with fresh adapters
        model = model.merge_and_unload()
        print("Merged existing adapter. Applying fresh LoRA on top.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = AgentSFTDataset(
        train_file,
        expand_assistant_turns=not args.no_expand_assistant_turns,
    )
    collator = QwenVLSFTCollator(
        processor=processor,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter and processor to: {output_dir}")


if __name__ == "__main__":
    main()
