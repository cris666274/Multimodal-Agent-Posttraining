"""
用真实 Qwen2.5-VL-3B-Instruct 模型重建 sft_seed.jsonl 中的 vision_parse 结果。
将 mock OCR/chart_values/caption 结果替换为真实 VLM 输出。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_SEED = ROOT / "data/sft/sft_seed.jsonl"
DEFAULT_OUT = ROOT / "data/sft/sft_seed_real_vlm.jsonl"
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"

VISION_MODE_PROMPTS = {
    "ocr": "请对这张图片进行 OCR，提取所有可见的文字内容。直接输出文字，不要添加分析。",
    "caption": "请用一句话描述这张图片的主要内容。直接输出描述，不要添加分析。",
    "chart_values": "请读取这张图表中的数值数据。如果有多个年份/类别，请列出每个对应的数值。直接输出数据，不要添加分析。",
    "ocr+caption": "请先对这张图片进行 OCR 提取文字，再描述图片内容。按以下格式输出：OCR结果：...\n图片描述：...",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model: {model_name} (device={device}, dtype={dtype})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_vision_parse(
    model,
    processor,
    image_path: str,
    mode: str,
    max_new_tokens: int = 256,
) -> str:
    prompt = VISION_MODE_PROMPTS.get(mode, VISION_MODE_PROMPTS["ocr"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def extract_vision_parse_calls(messages: List[Dict]) -> List[Dict[str, Any]]:
    """找出所有 vision_parse 调用及其对应的 tool 结果位置。"""
    calls = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if "vision_parse" not in content:
            continue
        # 尝试提取 mode
        mode = "ocr"
        try:
            if "<tool_call>" in content:
                json_str = content.split("<tool_call>")[1].split("</tool_call>")[0]
                args = json.loads(json_str).get("args", {})
                mode = args.get("mode", "ocr")
        except Exception:
            pass

        # 找到紧随其后的 tool 消息
        if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
            calls.append({
                "assistant_idx": i,
                "tool_idx": i + 1,
                "mode": mode,
            })
    return calls


def extract_image_from_messages(messages: List[Dict]) -> Optional[str]:
    """从 messages 中找到图片路径。"""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image", "")
                    # 转为绝对路径
                    image_path = Path(image)
                    if not image_path.is_absolute():
                        image_path = ROOT / image_path
                    if image_path.exists():
                        return str(image_path)
                    # 如果相对路径不存在，尝试直接用原路径
                    if Path(image).exists():
                        return image
        elif isinstance(content, str):
            # 纯文本问题，无图片
            return None
    return None


def rebuild_seed(seed_path: Path, model, processor) -> List[Dict]:
    rows = read_jsonl(seed_path)
    total_replaced = 0
    total_errors = 0
    total_cached = 0
    result_cache: Dict[str, str] = {}  # (image_path, mode) -> result

    for row_idx, row in enumerate(rows):
        messages = row.get("messages", [])
        image_path = extract_image_from_messages(messages)
        calls = extract_vision_parse_calls(messages)

        if not calls:
            continue

        for call in calls:
            mode = call["mode"]
            tool_idx = call["tool_idx"]
            old_result = messages[tool_idx].get("content", "")

            if image_path:
                cache_key = f"{image_path}||{mode}"
                if cache_key in result_cache:
                    new_result = result_cache[cache_key]
                    total_cached += 1
                else:
                    try:
                        new_result = run_vision_parse(model, processor, image_path, mode)
                        result_cache[cache_key] = new_result
                    except Exception as e:
                        total_errors += 1
                        print(f"  [{row_idx}] ERROR {Path(image_path).name}: {e}")
                        continue

                messages[tool_idx]["name"] = "vision_parse"
                messages[tool_idx]["content"] = new_result
                total_replaced += 1
                print(f"  [{row_idx}] {Path(image_path).name} mode={mode}: "
                      f"'{old_result[:60]}...' -> '{new_result[:80]}...'")
            else:
                print(f"  [{row_idx}] SKIP: no image found for vision_parse call")

        row["source"] = row.get("source", "seed") + "_real_vlm"

    print(f"\nTotal replaced: {total_replaced}, cached: {total_cached}, errors: {total_errors}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_path", default=str(DEFAULT_SEED))
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_path = Path(args.seed_path)
    out_path = Path(args.out_path)

    if not seed_path.is_absolute():
        seed_path = ROOT / seed_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    print(f"Seed: {seed_path}")
    print(f"Output: {out_path}")

    model, processor = load_model(args.model_name)
    rows = rebuild_seed(seed_path, model, processor)
    write_jsonl(out_path, rows)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
