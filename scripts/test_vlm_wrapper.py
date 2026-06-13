import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompts import SYSTEM_PROMPT
from agent.vlm import QwenVLModel


def main() -> int:
    image = "data/images/invoice_001.jpg"
    image_path = ROOT / image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = QwenVLModel(
    model_name="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
    max_new_tokens=256,)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "检查这张发票是否缺少必填字段。",
                },
            ],
        },
    ]

    output = model.generate(messages)
    print("模型输出：")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
