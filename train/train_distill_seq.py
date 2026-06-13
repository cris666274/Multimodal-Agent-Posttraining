"""
Sequence-level KD: teacher's generated token sequence serves as extra CE targets.

Simple, reliable: no KL alignment, no temperature tricks.
For samples with soft labels, add: CE(student_logits, teacher_token_ids).
"""
import argparse, json, torch, sys, random
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model

ALPHA = 0.7  # weight for main CE loss, (1-alpha) for teacher CE loss


def resolve_image_paths(messages, root=ROOT):
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
                    new_item["image"] = str(root / image)
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            else:
                new_content.append(item)
        new_msg = dict(msg)
        new_msg["content"] = new_content
        resolved.append(new_msg)
    return resolved


class DistillSeqDataset(Dataset):
    """Dataset with teacher token targets for KD."""
    def __init__(self, sft_path: Path, soft_labels_path: Path):
        with open(sft_path) as f:
            self.sft = [json.loads(line) for line in f if line.strip()]
        soft = torch.load(soft_labels_path, weights_only=False)
        self.teacher_map = {}
        for s in soft:
            self.teacher_map[s["sample_id"]] = s
        self.rows = self._expand()
        print(f"SFT: {len(self.sft)} raw → {len(self.rows)} expanded")
        print(f"Teacher targets: {len(self.teacher_map)} samples")

    def _expand(self):
        rows = []
        for row in self.sft:
            messages = row.get("messages", [])
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                sample = dict(row)
                sample["id"] = f"{row.get('id', 'sample')}_turn_{i}"
                sample["messages"] = messages[:i+1]
                sample["target_turn_index"] = i
                rows.append(sample)
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class DistillSeqCollator:
    processor: Any
    max_length: int
    teacher_map: Dict
    alpha: float

    def __call__(self, examples):
        full_texts, prefix_texts = [], []
        full_msgs, prefix_msgs = [], []
        example_ids = []

        for ex in examples:
            messages = resolve_image_paths(ex["messages"])
            target = len(messages) - 1
            while target >= 0 and messages[target]["role"] != "assistant":
                target -= 1
            if target < 0:
                continue
            prefix = messages[:target]
            full = messages[:target+1]
            full_msgs.append(full)
            prefix_msgs.append(prefix)
            full_texts.append(self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False))
            prefix_texts.append(self.processor.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True))
            example_ids.append(ex.get("id", ""))

        full_img, full_vid = process_vision_info(full_msgs)
        inputs = self.processor(text=full_texts, images=full_img, videos=full_vid, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for i, prefix in enumerate(prefix_msgs):
            p_img, p_vid = process_vision_info([prefix])
            p_inputs = self.processor(text=[prefix_texts[i]], images=p_img, videos=p_vid, padding=False, return_tensors="pt")
            prefix_len = p_inputs["input_ids"].shape[1]
            labels[i, :prefix_len] = -100

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        inputs["example_ids"] = example_ids
        return inputs


class DistillSeqTrainer(Trainer):
    def __init__(self, teacher_map, processor, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_map = teacher_map
        self.processor = processor
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        example_ids = inputs.pop("example_ids", [])
        del inputs["labels"]

        # Main SFT loss
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss

        # Teacher CE loss for samples with soft labels
        kd_loss = torch.tensor(0.0, device=ce_loss.device)
        kd_count = 0

        for i, eid in enumerate(example_ids):
            if eid not in self.teacher_map:
                continue
            teacher = self.teacher_map[eid]
            teacher_scores = teacher.get("teacher_scores", [])
            if not teacher_scores:
                continue

            # Get teacher's context messages and encode
            ctx = teacher["context_messages"]
            ctx_resolved = resolve_image_paths(ctx)

            # Encode full sequence (context + teacher answer)
            ctx_text = self.processor.apply_chat_template(ctx_resolved, tokenize=False, add_generation_prompt=True)
            ctx_img, ctx_vid = process_vision_info(ctx_resolved)
            ctx_inputs = self.processor(text=[ctx_text], images=ctx_img, videos=ctx_vid, padding=False, return_tensors="pt")
            ctx_inputs = {k: v.to(model.device) for k, v in ctx_inputs.items()}

            # Teacher token IDs (the tokens teacher would generate)
            teacher_ids = torch.tensor([t["token_id"] for t in teacher_scores], device=model.device).unsqueeze(0)
            teacher_labels = teacher_ids.clone()

            # Get student logits for the teacher's answer tokens
            student_out = model(**ctx_inputs)
            student_logits = student_out.logits

            # Take the last N positions where N = len(teacher_ids)
            seq_len = student_logits.shape[1]
            t_len = teacher_ids.shape[1]
            if seq_len >= t_len:
                student_logits_slice = student_logits[0, -t_len:, :]
                kd_loss += torch.nn.functional.cross_entropy(
                    student_logits_slice, teacher_ids[0], reduction='mean')
                kd_count += 1

        if kd_count > 0:
            kd_loss = kd_loss / kd_count
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        else:
            total_loss = ce_loss

        return total_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_file", default=str(ROOT / "data/sft/sft_agent_train_v7b_final.jsonl"))
    p.add_argument("--soft_labels", default=str(ROOT / "data/soft_labels/teacher_probs.pt"))
    p.add_argument("--output_dir", default="/root/autodl-tmp/multimodal-agent-lora/lora_7b_distilled_seq")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    train_file = Path(args.train_file)
    soft_file = Path(args.soft_labels)
    if not train_file.is_absolute():
        train_file = ROOT / train_file
    if not soft_file.is_absolute():
        soft_file = ROOT / soft_file

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = DistillSeqDataset(train_file, soft_file)
    collator = DistillSeqCollator(processor=processor, max_length=args.max_length,
                                   teacher_map=dataset.teacher_map, alpha=ALPHA)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    trainer = DistillSeqTrainer(
        teacher_map=dataset.teacher_map, processor=processor, alpha=ALPHA,
        model=model, args=training_args, train_dataset=dataset, data_collator=collator,
    )

    ckpt_dir = Path(args.output_dir)
    resume = False
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
        if ckpts:
            resume = True
            print(f"Resuming from {ckpts[-1]}")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
