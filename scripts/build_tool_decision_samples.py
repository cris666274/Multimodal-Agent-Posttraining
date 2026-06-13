"""
Build training samples targeting tool decision (not just tool execution):
1. Direct-answer samples: teach model when NOT to call tools
2. Refusal boundary samples: reinforce proper refusal with tool chain
"""

import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

random.seed(42)

PRODUCT_IMAGES = sorted((ROOT / "data/images").glob("product_*.jpg"))
INVOICE_IMAGES = sorted((ROOT / "data/images").glob("invoice_*.jpg"))

# ---- Direct answer samples (B-class): no tools needed ----
DIRECT_QUESTIONS = [
    "请简要描述图片中的商品类型。",
    "图中展示的是什么？",
    "这是什么商品？",
    "简短描述一下图片内容。",
    "图片里有什么？",
    "请用一句话描述图中商品。",
]

DIRECT_ANSWERS = {
    "black shoe": "根据图片内容，该商品为黑色鞋子，属于运动鞋类型。",
    "red shoe": "根据图片内容，该商品为红色鞋子，属于运动鞋类型。",
    "blue shoe": "根据图片内容，该商品为蓝色鞋子，属于运动鞋类型。",
    "green shoe": "根据图片内容，该商品为绿色鞋子，属于运动鞋类型。",
    "white shoe": "根据图片内容，该商品为白色鞋子，属于运动鞋类型。",
}

COLORS = ["black shoe", "red shoe", "blue shoe", "green shoe", "white shoe"]


def main():
    samples = []

    # ---- 1. Direct answer samples WITH reasoning prefix ----
    print("=== Building direct-answer samples with reasoning ===")
    for img in PRODUCT_IMAGES:
        for _ in range(3):
            color = random.choice(COLORS)
            q = random.choice(DIRECT_QUESTIONS)
            answer = DIRECT_ANSWERS[color]
            reasoning = "这是一个简单的商品描述问题，不需要调用任何工具，直接回答即可。"
            samples.append({
                "id": f"v9_direct_{img.stem}_{len(samples)}",
                "source": "v9_direct_with_reasoning",
                "category": "direct_vqa",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": str(img)},
                        {"type": "text", "text": q},
                    ]},
                    {"role": "assistant", "content": f"{reasoning}\n<final_answer>{answer}</final_answer>"},
                ],
            })

    print(f"  Direct-answer samples: {len(samples)}")

    # ---- 2. Refusal with tool chain + reasoning ----
    print("=== Building refusal+tool samples with reasoning ===")
    logo_images = sorted((ROOT / "data/images").glob("logo_*.jpg"))
    refusal_ocr = {
        "logo_002.jpg": "OCR结果：NETFLIX\n图片描述：这是一张显示'NETFLIX'文字的图片。",
        "logo_004.jpg": "OCR结果：ONCE\n图片描述：这是一张模糊的图片，显示了一个矩形框内写着'ONCE'。",
        "logo_006.jpg": "OCR结果：MELIA\n图片描述：这是一张模糊的图片，显示了一个矩形框内有'MELIA'字样。",
    }

    refusal_count = 0
    for img in logo_images:
        ocr = refusal_ocr.get(img.name)
        if not ocr:
            continue
        for _ in range(5):  # oversample refusal
            q = random.choice([
                "请判断图片中的品牌名称，并说明依据。",
                "这是什么品牌的logo？",
            ])
            samples.append({
                "id": f"v9_refusal_{img.stem}_{refusal_count}",
                "source": "v9_refusal_with_reasoning",
                "category": "uncertainty_refusal",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": str(img)},
                        {"type": "text", "text": q},
                    ]},
                    {"role": "assistant", "content": "这是一个品牌识别任务。需要先调用 vision_parse 提取图片中的文字和视觉信息，再判断是否为已知品牌。\n<tool_call>{\"name\":\"vision_parse\",\"args\":{\"mode\":\"ocr+caption\"}}</tool_call>"},
                    {"role": "tool", "name": "vision_parse", "content": ocr},
                    {"role": "user", "content": "工具 vision_parse 返回结果如下：请继续完成任务。"},
                    {"role": "assistant", "content": "<final_answer>图片中显示的文字无法确认为已知品牌标识。不能仅凭模糊或不确定的文字信息进行品牌判断，因此无法可靠识别。</final_answer>"},
                ],
            })
            refusal_count += 1

    print(f"  Refusal+tool samples: {refusal_count}")

    # Save
    out = ROOT / "data/sft/sft_tool_decision_v8.jsonl"
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(samples)} -> {out}")


if __name__ == "__main__":
    main()
