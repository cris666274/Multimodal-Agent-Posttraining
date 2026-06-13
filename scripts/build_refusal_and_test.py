"""
Build refusal training samples and eval_test.jsonl.

1. Refusal samples: multi-turn SFT traces showing correct refusal pipeline
   (vision_parse → blur result → "无法可靠判断")
   Uses real VLM to get actual vision_parse output for blurry logos.
2. eval_test.jsonl: balanced test set from expanded eval
"""
import json, sys, os, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)

# Refusal sample templates — vision_parse results filled by real VLM
REFUSAL_TEMPLATES = [
    # Blurry logos → should refuse
    {
        "id": "refusal_train_001",
        "image": "data/images/logo_002.jpg",
        "should_refuse": True,
        "refusal_reason": "图片模糊，vision_parse 未能提取到清晰的品牌标识文字。",
    },
    {
        "id": "refusal_train_002",
        "image": "data/images/logo_004.jpg",
        "should_refuse": True,
        "refusal_reason": "图片清晰度不足，vision_parse 无法可靠识别品牌名称。",
    },
    {
        "id": "refusal_train_003",
        "image": "data/images/logo_006.jpg",
        "should_refuse": True,
        "refusal_reason": "图片内容模糊，vision_parse 未能识别到清晰的品牌标识。",
    },
    # Readable logo → should NOT refuse (boundary sample)
    {
        "id": "refusal_train_004",
        "image": "data/images/logo_003.jpg",
        "should_refuse": False,
    },
]

SYSTEM_PROMPT = (
    "你是一个多模态 Agent，按以下流程工作：\n"
    "1. 对需要视觉判断的问题，必须先用 <tool_call> 调用 vision_parse\n"
    "2. 拿到工具结果后，如证据充分则输出包含具体结论的 <final_answer>\n"
    "3. 如图片模糊、vision_parse 无法可靠识别，必须明确拒绝，使用以下短语之一："
    "无法可靠判断、看不清、信息不足、无法确定\n\n"
    "工具：vision_parse(mode: ocr/caption/chart_values/ocr+caption)"
)


def build_refusal_samples(vlm_model):
    """Run vision_parse on logo images and construct SFT traces."""
    from agent.tools import vision_parse, set_vlm_model
    set_vlm_model(vlm_model)

    samples = []
    for tmpl in REFUSAL_TEMPLATES:
        print(f"\n--- {tmpl['id']}: {tmpl['image']} ---")

        # Get real vision_parse output
        vp_result = vision_parse(tmpl["image"], mode="ocr+caption")
        vp_text = vp_result["result"]
        print(f"  vision_parse: {vp_text[:120]}")

        if tmpl["should_refuse"]:
            final_answer = (
                f"<final_answer>"
                f"{tmpl['refusal_reason']}"
                f"</final_answer>"
            )
        else:
            final_answer = (
                f"<final_answer>"
                f"品牌名称为 {_extract_brand(vp_text)}。"
                f"依据：vision_parse 结果中显示了品牌标识。"
                f"</final_answer>"
            )

        sample = {
            "id": tmpl["id"],
            "category": "uncertainty_refusal",
            "image": tmpl["image"],
            "question": "请判断图片中的品牌名称，并说明依据。",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": tmpl["image"]},
                    {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
                ]},
                {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
                {"role": "tool", "name": "vision_parse", "content": json.dumps(vp_result, ensure_ascii=False)},
                {"role": "user", "content": (
                    f"工具 vision_parse 返回结果如下：\n"
                    f"{json.dumps(vp_result, ensure_ascii=False)}\n\n"
                    "请根据工具结果继续完成原始任务。"
                    "只能输出一个非空 <final_answer>。"
                )},
                {"role": "assistant", "content": final_answer},
            ],
        }
        samples.append(sample)

    return samples


def _extract_brand(vp_text: str) -> str:
    """Try to extract brand name from vision_parse output."""
    import re
    # OCR result typically has the clearest brand text
    m = re.search(r'OCR[结果]*[：:]\s*(\S+)', vp_text)
    if m and len(m.group(1)) >= 2 and len(m.group(1)) <= 20:
        return m.group(1)
    # Fallback: look for uppercase words
    for word in re.findall(r'\b([A-Z]{2,})\b', vp_text):
        return word
    return "[品牌名称]"


# ---- Eval test set ----
def build_test_set():
    """Split expanded eval into dev (40) + test (19)."""
    expanded_path = ROOT / "data/eval/expanded_eval.jsonl"
    if not expanded_path.exists():
        print(f"WARNING: {expanded_path} not found. Run build_expanded_eval.py first.")
        return

    with open(expanded_path) as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    # Stratified split by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for s in all_samples:
        by_cat[s["category"]].append(s)

    dev, test = [], []
    for cat, samples in sorted(by_cat.items()):
        random.shuffle(samples)
        # ~70/30 split per category, at least 1 in test
        n_test = max(1, len(samples) // 3)
        test.extend(samples[:n_test])
        dev.extend(samples[n_test:])

    random.shuffle(dev)
    random.shuffle(test)

    # Write dev
    dev_path = ROOT / "data/eval/eval_dev_v2.jsonl"
    with open(dev_path, "w") as f:
        for s in dev:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Dev: {len(dev)} samples → {dev_path}")

    # Write test
    test_path = ROOT / "data/eval/eval_test.jsonl"
    with open(test_path, "w") as f:
        for s in test:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Test: {len(test)} samples → {test_path}")

    # Show distribution
    for name, subset in [("dev", dev), ("test", test)]:
        cats = defaultdict(int)
        for s in subset:
            cats[s["category"]] += 1
        print(f"  {name}: {dict(cats)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_name", default="/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4")
    args = p.parse_args()

    from agent.vlm import QwenVLModel

    # Step 1: Load VLM and build refusal SFT samples
    print("=" * 60)
    print("Loading VLM for vision_parse on logo images...")
    print("=" * 60)
    model = QwenVLModel(model_name=args.model_name, adapter_name=args.adapter_name, max_new_tokens=256)

    refusal_samples = build_refusal_samples(model)

    refusal_path = ROOT / "data/sft/sft_refusal_train.jsonl"
    with open(refusal_path, "w") as f:
        for sample in refusal_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nRefusal SFT samples: {len(refusal_samples)} → {refusal_path}")
    for s in refusal_samples:
        n_turns = len(s["messages"])
        print(f"  {s['id']}: {n_turns} turns, image={s['image']}")

    # Step 2: Build eval dev/test split
    print()
    build_test_set()


if __name__ == "__main__":
    main()
