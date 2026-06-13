import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime import MultimodalAgent
from agent.vlm import QwenVLModel


def main() -> int:
    image = "data/images/invoice_001.jpg"
    image_path = ROOT / image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = QwenVLModel(
    model_name="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
    max_new_tokens=256,)
    agent = MultimodalAgent(model=model, max_steps=4)

    result = agent.run(
        image=image,
        question="检查这张发票是否缺少必填字段，并给出依据。",
    )

    print("状态：", result["status"])
    print("最终答案：")
    print(result["answer"])

    print("\nTrace:")
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
