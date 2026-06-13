"""
Run dual v2 to fill gold_answer_keywords for answer quality evaluation.
Uses dual v2 (V4 planner + V7B answerer with answerer prompt) to generate
reference answers, then extracts keywords as gold standard.

Label leakage: V4's tool metrics unaffected, V7B KW metrics have leakage.
For cross-model comparison, compute KW against these dual-generated keywords.
"""
import json, re, sys, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_keywords(answer: str, category: str, should_refuse: bool) -> list:
    """Extract meaningful keywords from dual model answer."""
    if not answer:
        return ["__empty__"]

    if should_refuse:
        return ["无法可靠", "模糊"]

    if category == "chart_calculation":
        kw = []
        nums = re.findall(r'(\d+\.\d+)', answer)
        if nums:
            kw.append(f"{float(nums[-1]):.2f}")
        if "增长" in answer or "同比" in answer:
            kw.append("同比")
        if "计算" in answer:
            kw.append("计算")
        if not kw:
            kw = [answer[:20].strip()]
        return kw[:3]

    elif category == "document_validation":
        kw = []
        if "不完整" in answer and "完整" in answer.replace("不完整", ""):
            pass  # both words present, ambiguous
        elif "不完整" in answer or "缺少" in answer:
            kw.append("不完整")
        elif "完整" in answer:
            kw.append("完整")
        if "Tax ID" in answer:
            kw.append("Tax")
        if "合规" in answer:
            kw.append("合规")
        if not kw:
            kw = [answer.split("。")[0][:15].strip()]
        return kw[:3]

    elif category == "image_text_consistency":
        kw = []
        if "不一致" in answer:
            kw.append("不一致")
        elif "一致" in answer:
            kw.append("一致")
        if not kw:
            kw = [answer.split("。")[0][:15].strip()]
        return kw[:2]

    elif category == "uncertainty_refusal":
        kw = []
        for brand in ["ACME", "LUNA", "ZEN", "VOLT", "SPORT"]:
            if brand.upper() in answer.upper():
                kw.append(brand)
                break
        if not kw:
            if any(w in answer for w in ["无法", "模糊", "不清晰", "难以", "信息不足", "不能确定"]):
                kw.append("无法可靠")
            else:
                kw.append(answer.split("。")[0][:15].strip())
        return kw[:2]

    elif category == "direct_vqa":
        for word in ["shoe", "鞋", "backpack", "背包", "hat", "帽子", "watch", "手表",
                     "invoice", "发票", "INVOICE", "chart", "图表", "商品", "发票",
                     "black", "white", "red", "blue", "green", "图片", "一个", "清晰"]:
            if word in answer:
                return [word]
        return [answer[:15].strip()]

    return [answer[:20].strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_paths", nargs="+",
                   default=["data/eval/eval_dev_v3.jsonl", "data/eval/eval_test_v3.jsonl"])
    args = p.parse_args()

    from agent.vlm import QwenVLModel
    from agent.dual_agent import DualModelAgent

    print("Loading dual v2: V4 planner + V7B answerer...")
    planner = QwenVLModel(
        model_name="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
        adapter_name="/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4",
        max_new_tokens=256,
    )
    answerer = QwenVLModel(
        model_name="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct",
        adapter_name="/root/autodl-tmp/multimodal-agent-lora/lora_agent_v7b_clean",
        max_new_tokens=256,
    )
    agent = DualModelAgent(planner_model=planner, answer_model=answerer, max_steps=6, enforce_required_tools=True)

    for eval_rel_path in args.eval_paths:
        eval_path = Path(eval_rel_path)
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path

        print(f"\nProcessing: {eval_path.name}")
        with open(eval_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        updated = 0
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(samples)}]")

            try:
                result = agent.run(image=sample["image"], question=sample["question"])
                answer = result.get("answer", "")
                kw = extract_keywords(answer, sample.get("category", ""), sample.get("should_refuse", False))
                sample["gold_answer_keywords"] = kw
                updated += 1
            except Exception as e:
                print(f"  [WARN] {sample['id']}: {str(e)[:60]}")

        print(f"  Updated: {updated}/{len(samples)}")

        with open(eval_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved: {eval_path}")


if __name__ == "__main__":
    main()
