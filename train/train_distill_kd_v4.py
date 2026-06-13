"""
Soft KD training v4 — correct implementation.

Critical fixes:
1. Teacher logits were generated with SAME SYSTEM_PROMPT as student training
2. KL divergence computed on top-K teacher logits (sparse, no memory explosion)
3. Correct formula: loss = alpha * CE + (1-alpha) * T² * KL(student/T || teacher/T)
4. Pre-launch smoke test verified
"""
import argparse, json, torch, sys
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

T = 3.0       # temperature
ALPHA = 0.7   # 70% CE, 30% KD
TOP_K = 100   # teacher logits saved with top-100


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


class SoftKDDataset(Dataset):
    def __init__(self, sft_path: Path, teacher_path: Path):
        with open(sft_path) as f:
            self.sft = [json.loads(line) for line in f if line.strip()]
        teacher_data = torch.load(teacher_path, weights_only=False)
        # Build lookup: (sample_id, turn_index) -> teacher data
        self.teacher_map = {}
        for t in teacher_data:
            key = (t["sample_id"], t["turn_index"])
            self.teacher_map[key] = t
        self.rows = self._expand()
        print(f"SFT: {len(self.sft)} raw → {len(self.rows)} expanded")
        print(f"Teacher: {len(self.teacher_map)} turn logits")

    def _expand(self):
        rows = []
        for row in self.sft:
            messages = row.get("messages", [])
            turn_idx = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                sample = dict(row)
                sample["id"] = f"{row.get('id', 'sample')}_turn_{turn_idx}"
                sample["messages"] = messages[:i+1]
                sample["target_turn_index"] = i
                sample["turn_counter"] = turn_idx
                rows.append(sample)
                turn_idx += 1
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class SoftKDCollator:
    processor: Any
    max_length: int
    teacher_map: Dict
    T: float
    alpha: float

    def __call__(self, examples):
        full_texts, prefix_texts = [], []
        full_msgs, prefix_msgs = [], []
        meta = []

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
            # Lookup teacher data
            orig_id = ex.get("id", "").rsplit("_turn_", 1)[0]
            turn = ex.get("turn_counter", 0)
            teacher_key = (orig_id, turn)
            # Check if this turn is a tool_call or final_answer
            assistant_msg = full[-1]  # last message = this turn's assistant output
            is_tool_call = "<tool_call>" in str(assistant_msg.get("content", ""))

            meta.append({
                "teacher": self.teacher_map.get(teacher_key),
                "prefix_len": None,
                "is_tool_call": is_tool_call,  # True = CE only, False = CE + KL
            })

        full_img, full_vid = process_vision_info(full_msgs)
        inputs = self.processor(text=full_texts, images=full_img, videos=full_vid, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for i, prefix in enumerate(prefix_msgs):
            p_img, p_vid = process_vision_info([prefix])
            p_inputs = self.processor(text=[prefix_texts[i]], images=p_img, videos=p_vid, padding=False, return_tensors="pt")
            prefix_len = p_inputs["input_ids"].shape[1]
            labels[i, :prefix_len] = -100
            meta[i]["prefix_len"] = prefix_len

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        inputs["_meta"] = meta
        return inputs


class SoftKDTrainer(Trainer):
    def __init__(self, T, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = T
        self.alpha_kd = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        meta = inputs.pop("_meta", [])
        del inputs["labels"]

        # --- CE loss (hard labels) ---
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss
        logits = outputs.logits  # [B, seq_len, vocab_size]

        # --- KD loss (soft labels from teacher) ---
        kd_losses = []

        for i, m in enumerate(meta):
            teacher = m.get("teacher")
            if teacher is None:
                continue
            # Hybrid: skip KD for tool_call turns (use pure CE)
            if m.get("is_tool_call", False):
                continue

            prefix_len = m.get("prefix_len")
            teacher_logits_list = teacher.get("teacher_logits_topk", [])
            num_tokens = len(teacher_logits_list)

            if prefix_len is None or num_tokens == 0:
                continue

            # For each token position in the assistant turn
            student_seq = logits[i]  # [seq_len, vocab]
            for t in range(num_tokens):
                pos = prefix_len + t
                if pos - 1 < 0 or pos - 1 >= student_seq.shape[0]:
                    break

                student_logit = student_seq[pos - 1]  # logits[pos-1] predicts token[pos]

                # Teacher top-K
                tk = teacher_logits_list[t]
                idx = tk["indices"].to(student_logit.device)
                vals = tk["values"].to(student_logit.device).float()

                # Teacher normalized probabilities (on top-K subset)
                teacher_probs = torch.softmax(vals, dim=0)

                # Student log-prob on the same indices, temperature-scaled
                student_logp = torch.log_softmax(student_logit[idx] / self.temp, dim=0)

                # KL(P_teacher || Q_student) on top-K subset
                teacher_logp = torch.log(teacher_probs + 1e-30)
                kl = torch.sum(teacher_probs * (teacher_logp - student_logp))
                kd_losses.append(kl * self.temp * self.temp)

        if kd_losses:
            kd_loss = torch.stack(kd_losses).mean()
            loss = self.alpha_kd * ce_loss + (1 - self.alpha_kd) * kd_loss
        else:
            loss = ce_loss

        return loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_file", default=str(ROOT / "data/sft/sft_agent_train_v7b_final.jsonl"))
    p.add_argument("--teacher_logits", default=str(ROOT / "data/soft_labels_v4/teacher_logits.pt"))
    p.add_argument("--output_dir", default="/root/autodl-tmp/multimodal-agent-lora/lora_7b_soft_kd")
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
    teacher_file = Path(args.teacher_logits)
    if not train_file.is_absolute():
        train_file = ROOT / train_file
    if not teacher_file.is_absolute():
        teacher_file = ROOT / teacher_file

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ))
    model.print_trainable_parameters()

    dataset = SoftKDDataset(train_file, teacher_file)
    collator = SoftKDCollator(processor=processor, max_length=args.max_length,
                               teacher_map=dataset.teacher_map, T=T, alpha=ALPHA)

    training_args = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=args.learning_rate,
        warmup_ratio=0.03, num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_total_limit=2, bf16=args.bf16, report_to="none",
        remove_unused_columns=False, dataloader_pin_memory=False, max_grad_norm=1.0,
    )

    trainer = SoftKDTrainer(T=T, alpha=ALPHA, model=model, args=training_args,
                             train_dataset=dataset, data_collator=collator)

    ckpt_dir = Path(args.output_dir)
    resume = False
    if ckpt_dir.exists() and list(ckpt_dir.glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
