from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError:
    AutoProcessor = None
    PeftModel = None
    Qwen2_5_VLForConditionalGeneration = None
    process_vision_info = None


class QwenVLModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_name: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.use_mock = False  # 不再支持 mock 回退

        # 检查依赖可用性
        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None or process_vision_info is None:
            raise ImportError(
                "Qwen2.5-VL 依赖不可用（transformers/qwen_vl_utils）。"
                "请安装: pip install transformers qwen-vl-utils"
            )
        if adapter_name is not None and PeftModel is None:
            raise ImportError("LoRA adapter 需要 peft 库。请安装: pip install peft")

        print(f"Loading model: {model_name}")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        if adapter_name:
            print(f"Loading LoRA adapter: {adapter_name}")
            self.model = PeftModel.from_pretrained(self.model, adapter_name)

        self.processor = AutoProcessor.from_pretrained(model_name)

        print("Model loaded.")
        print("CUDA available:", torch.cuda.is_available())

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """输入 messages，输出模型生成文本。"""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    def generate_vision(self, messages: List[Dict[str, Any]]) -> str:
        """
        视觉任务推理（OCR / caption / chart_values）。
        临时禁用 LoRA adapter，使用基座模型的原始视觉能力。
        避免 agent 格式污染（<tool_call> / <final_answer>）。
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 检查是否有 LoRA adapter 可禁用
        has_adapter = hasattr(self.model, 'disable_adapter')

        with torch.no_grad():
            if has_adapter:
                with self.model.disable_adapter():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
