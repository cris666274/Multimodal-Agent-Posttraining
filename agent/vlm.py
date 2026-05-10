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
        use_mock: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        deps_available = (
            AutoProcessor is not None
            and Qwen2_5_VLForConditionalGeneration is not None
            and process_vision_info is not None
            and (adapter_name is None or PeftModel is not None)
        )
        self.use_mock = not deps_available if use_mock is None else use_mock

        if self.use_mock:
            reason = "requested by caller" if use_mock else "Qwen2.5-VL dependencies are unavailable"
            print(f"Using mock QwenVLModel: {reason}.")
            return

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
        """
        输入 messages，输出模型生成文本。
        如果当前环境缺少 Qwen2.5-VL 真实模型依赖，则使用 mock 输出跑通流程。
        """
        if self.use_mock:
            return self._mock_generate(messages)

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

    def _mock_generate(self, messages: List[Dict[str, Any]]) -> str:
        last_text = self._last_user_text(messages)
        all_user_text = self._all_user_text(messages)
        last_image = self._last_user_image(messages)
        called_tools = self._called_tools(messages)
        image_name = Path(last_image).name.lower() if last_image else ""

        if "工具" in last_text and "返回结果" in last_text:
            if ("发票" in all_user_text or "字段" in all_user_text or "invoice" in image_name) and "retrieve_docs" not in called_tools:
                return '<tool_call>{"name":"retrieve_docs","args":{"query":"发票必填字段规则"}}</tool_call>'

            if ("图表" in all_user_text or "同比" in all_user_text or "增长" in all_user_text or "chart" in image_name) and "python_exec" not in called_tools:
                return '<tool_call>{"name":"python_exec","args":{"code":"(90-80)/80*100"}}</tool_call>'

            return "<final_answer>根据工具返回结果，该样例已完成 mock VLM 流程验证。</final_answer>"

        if "发票" in last_text or "字段" in last_text or "invoice" in image_name:
            return '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>'

        if "图表" in last_text or "同比" in last_text or "增长" in last_text or "chart" in image_name:
            return '<tool_call>{"name":"vision_parse","args":{"mode":"chart_values"}}</tool_call>'

        if "商品" in last_text or "图文" in last_text or "一致" in last_text or "product" in image_name:
            return '<tool_call>{"name":"vision_parse","args":{"mode":"caption"}}</tool_call>'

        if "品牌" in last_text or "logo" in image_name:
            return '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'

        return "<final_answer>当前信息不足，无法可靠判断。</final_answer>"

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(parts)
        return ""

    @staticmethod
    def _last_user_image(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        return item.get("image", "")
        return ""

    @staticmethod
    def _all_user_text(messages: List[Dict[str, Any]]) -> str:
        text_parts = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                text_parts.extend(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        return "\n".join(text_parts)

    @staticmethod
    def _called_tools(messages: List[Dict[str, Any]]) -> set:
        tools = set()
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            for tool_name in ("vision_parse", "retrieve_docs", "python_exec"):
                if f'"name":"{tool_name}"' in content or f'"name": "{tool_name}"' in content:
                    tools.add(tool_name)
        return tools
