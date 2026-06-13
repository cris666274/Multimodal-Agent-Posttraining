"""
Generate refusal boundary SFT samples using real VLM outputs from logo images.
Creates samples teaching the model to refuse when OCR only shows numbers/noise.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.vlm import QwenVLModel
from agent.tools import vision_parse, set_vlm_model

random.seed(42)

LOGO_IMAGES = sorted((ROOT / "data/images").glob("logo_*.jpg"))

QUESTION_VARIANTS = [
    "请判断图片中的品牌名称，并说明依据。",
    "这是什么品牌的logo？",
    "请识别图中的品牌标识。",
    "这张图里有品牌信息吗？有的话是什么？",
    "图中的logo属于哪个品牌？",
]

CORRECT_REFUSAL = (
    "图片信息不足，无法可靠识别品牌名称。"
    "视觉解析结果中没有可辨认的品牌文字或logo，不能仅凭模糊线索猜测。"
)


def main():
    print(f"Loading VLM model...")
    model = QwenVLModel(
        model_name="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct",
    )
    set_vlm_model(model)
    print(f"Found {len(LOGO_IMAGES)} logo images")

    samples = []
    for img_path in LOGO_IMAGES:
        img_str = str(img_path)
        # Run real VLM vision_parse on this logo
        result = vision_parse(img_str, mode="ocr+caption")
        ocr_text = result.get("result", "")
        if not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)

        print(f"\n{img_path.name}: {ocr_text[:120]}...")

        # Determine if this should trigger refusal
        # Refuse if OCR contains only digits/symbols with no brand name
        has_alpha_word = any(
            word.lower() in ocr_text.lower()
            for word in ["acme", "zen", "luna", "volt", "nike", "adidas"]
        )

        if has_alpha_word:
            print(f"  -> SKIP (recognizable brand)")
            continue

        # This logo should trigger refusal
        for q in random.sample(QUESTION_VARIANTS, min(3, len(QUESTION_VARIANTS))):
            sample = {
                "id": f"v7_refusal_{img_path.stem}_{len(samples)}",
                "source": "v7_refusal_boundary",
                "category": "uncertainty_refusal",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_str},
                            {"type": "text", "text": q},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>',
                    },
                    {
                        "role": "tool",
                        "name": "vision_parse",
                        "content": ocr_text,
                    },
                    {
                        "role": "user",
                        "content": "工具 vision_parse 返回结果如下：请继续完成任务。",
                    },
                    {
                        "role": "assistant",
                        "content": f"<final_answer>{CORRECT_REFUSAL}</final_answer>",
                    },
                ],
            }
            samples.append(sample)

    # Save
    out_path = ROOT / "data/sft/sft_refusal_boundary_v7.jsonl"
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(samples)} refusal boundary samples -> {out_path}")


if __name__ == "__main__":
    main()
