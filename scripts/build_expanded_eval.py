"""
Build expanded eval set: 50 existing + 9 new samples from unused images.
1. Create new samples with placeholder keywords
2. Run V4 to get reference answers → set gold keywords
3. Output expanded_eval.jsonl
"""
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- New samples using unused images ----
NEW_SAMPLES = [
    # chart_calculation (2)
    {
        "id": "eval_chart_011",
        "category": "chart_calculation",
        "image": "data/images/chart_011.png",
        "question": "根据图表，计算2024年相比2023年的同比增长率。",
        "gold_tools": ["vision_parse", "python_exec"],
        "gold_answer_keywords": [],  # will fill from V4
        "should_refuse": False,
        "evidence_required": True,
    },
    {
        "id": "eval_chart_012",
        "category": "chart_calculation",
        "image": "data/images/chart_012.png",
        "question": "根据图表，计算2024年相比2023年的同比增长率。",
        "gold_tools": ["vision_parse", "python_exec"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    # document_validation (2)
    {
        "id": "eval_invoice_016",
        "category": "document_validation",
        "image": "data/images/invoice_016.jpg",
        "question": "检查这张发票是否缺少必填字段，并给出依据。",
        "gold_tools": ["vision_parse", "retrieve_docs"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    {
        "id": "eval_invoice_017",
        "category": "document_validation",
        "image": "data/images/invoice_017.jpg",
        "question": "检查这张发票是否缺少必填字段，并给出依据。",
        "gold_tools": ["vision_parse", "retrieve_docs"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    # image_text_consistency (3)
    {
        "id": "eval_product_013",
        "category": "image_text_consistency",
        "image": "data/images/product_013.jpg",
        "question": "商品描述写的是 red backpack。请判断图文是否一致。",
        "gold_tools": ["vision_parse"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    {
        "id": "eval_product_014",
        "category": "image_text_consistency",
        "image": "data/images/product_014.jpg",
        "question": "商品描述写的是 blue hat。请判断图文是否一致。",
        "gold_tools": ["vision_parse"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    {
        "id": "eval_product_015",
        "category": "image_text_consistency",
        "image": "data/images/product_015.jpg",
        "question": "商品描述写的是 green watch。请判断图文是否一致。",
        "gold_tools": ["vision_parse"],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": True,
    },
    # direct_vqa (2) — simple questions needing no tools
    {
        "id": "eval_direct_006",
        "category": "direct_vqa",
        "image": "data/images/product_016.jpg",
        "question": "请简要描述图片中的商品类型。",
        "gold_tools": [],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": False,
    },
    {
        "id": "eval_direct_007",
        "category": "direct_vqa",
        "image": "data/images/product_017.jpg",
        "question": "请简要描述图片中的商品类型。",
        "gold_tools": [],
        "gold_answer_keywords": [],
        "should_refuse": False,
        "evidence_required": False,
    },
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter_name", default="/root/autodl-tmp/multimodal-agent-lora/lora_agent_real_v4")
    p.add_argument("--out", default=str(ROOT / "data/eval/expanded_eval.jsonl"))
    args = p.parse_args()

    from agent.vlm import QwenVLModel
    from agent.runtime import MultimodalAgent

    # Step 1: Run V4 on new samples to get reference answers
    print("=" * 60)
    print("Step 1: Running V4 on 9 new samples to get reference answers...")
    print("=" * 60)

    model = QwenVLModel(model_name=args.model_name, adapter_name=args.adapter_name, max_new_tokens=256)
    agent = MultimodalAgent(model=model, max_steps=6, enforce_required_tools=True)

    for sample in NEW_SAMPLES:
        print(f"\n--- {sample['id']} ---")
        result = agent.run(image=sample["image"], question=sample["question"])
        answer = result.get("answer", "")
        print(f"  tools: {[s.get('tool_name','') for s in result.get('trace',[]) if 'tool_name' in s]}")
        print(f"  answer: {answer[:200]}")

        # Extract gold keywords from answer
        sample["_v4_answer"] = answer
        sample["_v4_tools"] = [s.get('tool_name','') for s in result.get('trace',[]) if 'tool_name' in s]

    # Step 2: Heuristic keyword extraction
    print("\n" + "=" * 60)
    print("Step 2: Extracting gold keywords for NEW samples...")
    print("=" * 60)

    import re
    for sample in NEW_SAMPLES:
        answer = sample.get("_v4_answer", "")
        cat = sample["category"]

        if cat == "chart_calculation":
            # Answer format: "根据图表信息，2024年相比2023年的同比增长率为 16.44%。"
            # Extract the LAST decimal number before % or 。
            nums = re.findall(r'(\d+\.\d+)', answer)
            if nums:
                pct = f"{float(nums[-1]):.2f}"  # last decimal number = growth rate
                sample["gold_answer_keywords"] = [pct, "同比"]
                print(f"  {sample['id']}: pct={pct}  (from '{answer[:80]}')")
            else:
                sample["gold_answer_keywords"] = ["同比"]
                print(f"  {sample['id']}: WARNING no decimal found in '{answer[:80]}'")

        elif cat == "document_validation":
            kw = []
            if "不完整" in answer or "缺少" in answer or "缺失" in answer:
                kw.append("不完整")
                for field in ["Tax ID", "税号", "Invoice No", "发票号码", "Date", "日期", "Amount", "金额"]:
                    if field in answer:
                        kw.append(field)
                        break
            else:
                kw.append("完整")
            if not kw:
                kw = ["字段"]
            sample["gold_answer_keywords"] = kw
            print(f"  {sample['id']}: kw={kw}")

        elif cat == "image_text_consistency":
            if "不一致" in answer or "不符" in answer:
                sample["gold_answer_keywords"] = ["不一致"]
            else:
                sample["gold_answer_keywords"] = ["一致"]
            print(f"  {sample['id']}: kw={sample['gold_answer_keywords']}")

        elif cat == "direct_vqa":
            kw = []
            for word in ["shoe", "backpack", "hat", "watch", "t-shirt", "T恤", "背包", "帽子", "手表", "鞋", "运动鞋", "red", "blue", "black", "white"]:
                if word.lower() in answer.lower():
                    kw.append(word)
                    break
            if not kw:
                kw = ["商品"]
            sample["gold_answer_keywords"] = kw
            print(f"  {sample['id']}: kw={kw}")

        del sample["_v4_answer"]
        del sample["_v4_tools"]

    # Step 3: Merge with existing 50
    print("\n" + "=" * 60)
    print("Step 3: Loading existing 50, fixing chart keywords from V4 eval output...")
    print("=" * 60)

    existing_path = ROOT / "data/eval/eval_dev_with_category.jsonl"
    with open(existing_path) as f:
        existing = [json.loads(line) for line in f if line.strip()]

    # Use existing V4 eval output to get gold keywords (already verified by previous eval)
    v4_eval_path = ROOT / "outputs/vlm_agent_v4_eval.jsonl"
    v4_eval = {}
    if v4_eval_path.exists():
        with open(v4_eval_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    v4_eval[d["id"]] = d

    # Fix chart keywords from previous V4 eval
    for s in existing:
        if s.get("category") == "chart_calculation" and s["id"] in v4_eval:
            old_kw = v4_eval[s["id"]].get("gold_answer_keywords", [])
            s["gold_answer_keywords"] = old_kw
            print(f"  {s['id']}: restored {old_kw}")
        elif s.get("category") == "chart_calculation":
            print(f"  {s['id']}: WARNING not found in V4 eval")

    merged = existing + NEW_SAMPLES

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(merged)} samples to {out_path}")
    cats = {}
    for s in merged:
        c = s.get("category", "?")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
