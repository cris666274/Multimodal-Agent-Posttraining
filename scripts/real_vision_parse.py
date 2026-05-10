"""
共享的真实 VLM vision_parse 模块。
供 convert_chartqa_to_sft.py / convert_docvqa_to_sft.py / convert_cord_to_sft.py 使用。
"""

import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"

VISION_MODE_PROMPTS = {
    "ocr": "请对这张图片进行 OCR，提取所有可见的文字内容。直接输出文字，不要添加分析。",
    "caption": "请用一句话描述这张图片的主要内容。直接输出描述，不要添加分析。",
    "chart_values": "请读取这张图表中的数值数据。如果有多个年份/类别，请列出每个对应的数值。直接输出数据，不要添加分析。",
    "ocr+caption": "请先对这张图片进行 OCR 提取文字，再描述图片内容。按以下格式输出：OCR结果：...\n图片描述：...",
}


class RealVisionParse:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading VLM for real vision_parse: {model_name} (device={device})")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def parse(self, image_path: str, mode: str = "ocr", max_new_tokens: int = 256) -> str:
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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()


# 全局单例，由外部脚本初始化
_vision_parser: Optional[RealVisionParse] = None


def get_vision_parser(model_name: str = DEFAULT_MODEL) -> RealVisionParse:
    global _vision_parser
    if _vision_parser is None:
        _vision_parser = RealVisionParse(model_name)
    return _vision_parser
