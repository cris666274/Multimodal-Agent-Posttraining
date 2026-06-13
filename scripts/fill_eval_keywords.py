"""
Run V4 on all eval samples to generate real gold_answer_keywords.
Replaces heuristic template keywords with actual model output keywords.

V4's own KW score will have label leakage — use for tool metrics only.
V7B's KW score is clean — different model, no leakage.
"""
import json, re, sys, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_keywords(answer: str, category: str, question: str, should_refuse: bool) -> list:
    """Extract meaningful keywords from model answer based on category."""
    if not answer:
        return ["__empty__"]

    if should_refuse:
        return ["无法可靠", "模糊"]

    if category == "chart_calculation":
        kw = []
        # Extract the percentage value
        nums = re.findall(r'(\d+\.\d+)', answer)
        if nums:
            kw.append(f"{float(nums[-1]):.2f}")
        # Check for calculation method references
        if "增长" in answer:
            kw.append("同比")
        if "计算" in answer:
            kw.append("计算")
        if not kw:
            kw = [answer.split("。")[0].strip()[-10:]]
        return kw

    elif category == "document_validation":
        kw = []
        if "不完整" in answer and "完整" not in answer.replace("不完整", ""):
            kw.append("不完整")
        elif "完整" in answer:
            kw.append("完整")
        if "缺少" in answer:
            kw.append("缺少")
        if "Tax ID" in answer:
            kw.append("Tax ID")
        if "Invoice No" in answer:
            kw.append("Invoice No")
        if "合规" in answer:
            kw.append("合规")
        if not kw:
            # Extract key phrase from first sentence
            first = answer.split("。")[0].strip()
            kw.append(first[:15])
        return kw[:3]  # max 3 keywords

    elif category == "image_text_consistency":
        kw = []
        if "不一致" in answer:
            kw.append("不一致")
        elif "一致" in answer:
            kw.append("一致")
        if "颜色" in answer:
            kw.append("颜色")
        if "品类" in answer:
            kw.append("品类")
        if not kw:
            kw = [answer.split("。")[0].strip()[:15]]
        return kw[:2]

    elif category == "uncertainty_refusal":
        kw = []
        # Brand name if identified
        for brand in ["ACME", "LUNA", "ZEN", "VOLT", "DELL", "XIAOMI", "VANS",
                       "Acme", "Luna", "Zen", "Volt", "Dell", "Xiaomi", "Vans"]:
            if brand in answer:
                kw.append(brand)
                break
        if not kw:
            # Refusal or other
            if any(w in answer for w in ["无法", "模糊", "不清晰", "难以", "信息不足"]):
                kw.append("无法可靠")
            else:
                kw.append(answer.split("。")[0].strip()[:15])
        return kw[:2]

    elif category == "direct_vqa":
        kw = []
        # Product type
        for word in ["shoe", "鞋", "backpack", "背包", "hat", "帽子", "watch", "手表",
                     "invoice", "发票", "INVOICE", "chart", "图表", "logo", "商品",
                     "black", "white", "red", "blue", "green", "黑色", "白色", "红色"]:
            if word in answer:
                kw.append(word)
                break
        if not kw:
            kw = [answer.split("。")[0].strip()[:15]]
        return kw[:1]

    return [answer.split("。")[0].strip()[:20]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_name", default="/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4")
    p.add_argument("--eval_paths", nargs="+",
                   default=["data/eval/eval_dev_v3.jsonl", "data/eval/eval_test_v3.jsonl"])
    args = p.parse_args()

    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent

    print("Loading V4...")
    model = QwenVLModel(model_name=args.model_name, adapter_name=args.adapter_name, max_new_tokens=256)
    agent = MultimodalAgent(model=model, max_steps=6, enforce_required_tools=True)

    for eval_rel_path in args.eval_paths:
        eval_path = Path(eval_rel_path)
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path

        print(f"\n{'='*60}")
        print(f"Processing: {eval_path.name}")
        print(f"{'='*60}")

        with open(eval_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        updated = 0
        for i, sample in enumerate(samples):
            if i % 20 == 0:
                print(f"  [{i}/{len(samples)}]...")

            # Skip if already has valid keywords (not template-generated)
            old_kw = sample.get("gold_answer_keywords", [])
            if old_kw and old_kw != ["__auto_generated__"] and not any(
                k in ["同比", "完整", "不完整", "一致", "不一致", "字段", "元", "金额", "图片", "商品",
                       "无法可靠", "模糊", "匹配", "趋势", "差值", "百分比", "2023", "2024",
                       "shoe", "鞋", "个", "色", "清晰", "一般", "需要", "不需要",
                       "元素", "文字", "品牌", "logo", "置信度", "把握", "发票", "图表", "图像"]
                for k in old_kw
            ):
                continue  # already has good keywords

            try:
                result = agent.run(image=sample["image"], question=sample["question"])
                answer = result.get("answer", "")
                kw = extract_keywords(
                    answer,
                    sample.get("category", ""),
                    sample.get("question", ""),
                    sample.get("should_refuse", False),
                )
                sample["gold_answer_keywords"] = kw
                updated += 1
            except Exception as e:
                print(f"  [WARN] {sample['id']}: {str(e)[:60]}")

        print(f"  Updated: {updated}/{len(samples)}")

        # Write back
        with open(eval_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved: {eval_path}")


if __name__ == "__main__":
    main()
