"""
Build eval v3: 100 dev + 100 test samples with rich metadata.

Each sample gets:
  - allowed_tools: tools that are acceptable (superset of gold_tools)
  - forbidden_tools: tools that must NOT be used (anti-pattern)
  - difficulty: easy | medium | hard
  - eval_focus: tool_selection | answer_quality | refusal | tool_order | format | tool_autonomy

Generates diverse questions per image to reach 200 samples (40 per category).
"""
import json, random, sys, os, argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)

# ============================================================
# Question templates per category
# ============================================================

DOCUMENT_VALIDATION_TEMPLATES = [
    # (question, gold_tools, difficulty, eval_focus)
    ("检查这张发票是否缺少必填字段，并给出依据。",
     ["vision_parse", "retrieve_docs"], "easy", "tool_order"),
    ("提取这张发票的发票号码和日期。",
     ["vision_parse"], "easy", "answer_quality"),
    ("这张发票的金额是多少？请给出具体数值和依据。",
     ["vision_parse"], "easy", "answer_quality"),
    ("这张发票是否包含 Tax ID？如有，请写出 Tax ID 号码。",
     ["vision_parse"], "medium", "tool_selection"),
    ("根据发票规则，判断这张发票是否符合合规要求。",
     ["vision_parse", "retrieve_docs"], "medium", "tool_order"),
    ("发票中是否有任何字段缺失或不清晰？请逐字段检查。",
     ["vision_parse", "retrieve_docs"], "medium", "answer_quality"),
    ("对比这张发票的必填字段和实际字段，列出所有差异。",
     ["vision_parse", "retrieve_docs"], "hard", "answer_quality"),
    ("仅根据图片中的文字信息，列出你能读取到的所有字段名和值。",
     ["vision_parse"], "medium", "tool_autonomy"),
]

CHART_CALCULATION_TEMPLATES = [
    ("根据图表，计算2024年相比2023年的同比增长率。",
     ["vision_parse", "python_exec"], "medium", "answer_quality"),
    ("图表中2024年的数值是多少？",
     ["vision_parse"], "easy", "answer_quality"),
    ("图表中2023年和2024年哪一年的数值更高？高出多少？",
     ["vision_parse", "python_exec"], "medium", "answer_quality"),
    ("计算图表中两年数据的差值。",
     ["vision_parse", "python_exec"], "easy", "tool_selection"),
    ("根据图表数据，计算2024年数值占总和的百分比。",
     ["vision_parse", "python_exec"], "hard", "answer_quality"),
    ("如果2025年按照同样的增长率继续增长，预测2025年的数值。",
     ["vision_parse", "python_exec"], "hard", "answer_quality"),
    ("仅提取图表中的原始数值，不需要任何计算。",
     ["vision_parse"], "easy", "tool_autonomy"),
    ("判断图表数据是否呈现上升趋势，并引用具体数值。",
     ["vision_parse", "python_exec"], "medium", "answer_quality"),
]

IMAGE_CONSISTENCY_TEMPLATES = [
    ("商品描述写的是 {color} {category}。请判断图文是否一致。",
     ["vision_parse"], "easy", "tool_selection"),
    ("标题描述为 {color} {category}，图片中的商品是否与之匹配？",
     ["vision_parse"], "easy", "tool_selection"),
    ("图片中的商品颜色和品类是什么？与描述 '{color} {category}' 对比。",
     ["vision_parse"], "medium", "answer_quality"),
    ("这张商品图的标题和描述可能有冲突，请根据图文一致性规则判断。",
     ["vision_parse", "retrieve_docs"], "hard", "tool_order"),
    ("请仅描述图片中商品的颜色和品类，不要做任何判断。",
     ["vision_parse"], "easy", "tool_autonomy"),
    ("如果图片不够清晰，请说明原因并拒绝判断图文是否一致。",
     ["vision_parse"], "medium", "refusal"),
    ("商品实际是 {color} {category}，但描述写的是 {wrong_color} {wrong_category}，请判断一致性。",
     ["vision_parse"], "medium", "answer_quality"),
]

UNCERTAINTY_REFUSAL_TEMPLATES = [
    ("请判断图片中的品牌名称，并说明依据。",
     ["vision_parse"], "medium", "refusal"),
    ("图片中是否包含可识别的文字？如有，请写出。",
     ["vision_parse"], "easy", "tool_selection"),
    ("图片质量是否足够清晰以识别品牌？请先分析再回答。",
     ["vision_parse"], "medium", "tool_autonomy"),
    ("基于图片内容，你有多大把握确定品牌名称？请给出置信度评估。",
     ["vision_parse"], "hard", "refusal"),
    ("请描述图片中能看到的所有视觉元素，不限于文字。",
     ["vision_parse"], "easy", "answer_quality"),
    ("如果无法可靠识别，请明确拒绝。如果可以识别，请给出品牌名称。",
     ["vision_parse"], "medium", "refusal"),
    ("图片中logo区域是否清晰可读？先判断清晰度再回答品牌。",
     ["vision_parse"], "hard", "tool_autonomy"),
]

DIRECT_VQA_TEMPLATES = [
    ("请简要描述图片中的商品类型。",
     [], "easy", "tool_autonomy"),
    ("图片中有几个主要物体？",
     [], "easy", "tool_selection"),
    ("图片中的主色调是什么？",
     [], "easy", "tool_selection"),
    ("不需要调用任何工具，直接回答：图片内容是什么类型的图像（发票/图表/商品/logo）？",
     [], "easy", "tool_autonomy"),
    ("这张图片的整体质量如何（清晰/一般/模糊）？不要使用工具。",
     [], "easy", "tool_autonomy"),
    ("请用一句话总结图片内容，不需要技术分析。",
     [], "medium", "tool_autonomy"),
    ("这张图片是否需要调用工具才能回答？先判断再决定。",
     [], "medium", "tool_selection"),
]


# ============================================================
# Image inventory
# ============================================================

INVOICE_IMAGES = [f"data/images/invoice_{i:03d}.jpg" for i in range(1, 18)]  # 17
CHART_IMAGES = [f"data/images/chart_{i:03d}.png" for i in range(1, 13)]    # 12
PRODUCT_IMAGES = [f"data/images/product_{i:03d}.jpg" for i in range(1, 18)] # 17
LOGO_IMAGES = [f"data/images/logo_{i:03d}.jpg" for i in range(1, 8)]        # 7

# Color/category pairs for product consistency questions
PRODUCT_PAIRS = [
    ("black", "shoe"), ("red", "shoe"), ("blue", "shoe"), ("white", "shoe"),
    ("green", "shoe"), ("black", "backpack"), ("red", "backpack"), ("blue", "hat"),
    ("green", "watch"), ("white", "t-shirt"),
]

# Blurry logos (002, 004, 006) and readable logos (001, 003, 005, 007)
BLURRY_LOGOS = ["data/images/logo_002.jpg", "data/images/logo_004.jpg", "data/images/logo_006.jpg"]
READABLE_LOGOS = ["data/images/logo_001.jpg", "data/images/logo_003.jpg", "data/images/logo_005.jpg", "data/images/logo_007.jpg"]


# ============================================================
# Sample generator
# ============================================================

def generate_document_validation(target: int) -> List[Dict]:
    samples = []
    for img in INVOICE_IMAGES:
        templates = random.sample(DOCUMENT_VALIDATION_TEMPLATES, min(len(DOCUMENT_VALIDATION_TEMPLATES), 3))
        for tmpl in templates:
            question, gold_tools, difficulty, eval_focus = tmpl
            samples.append({
                "id": f"eval_doc_{len(samples):03d}",
                "category": "document_validation",
                "image": img,
                "question": question,
                "gold_tools": gold_tools,
                "allowed_tools": ["vision_parse", "retrieve_docs"],
                "forbidden_tools": ["python_exec"],
                "difficulty": difficulty,
                "eval_focus": eval_focus,
                "gold_answer_keywords": [],
                "should_refuse": False,
                "evidence_required": True,
            })
    random.shuffle(samples)
    return samples[:target]


def generate_chart_calculation(target: int) -> List[Dict]:
    samples = []
    for img in CHART_IMAGES:
        templates = random.sample(CHART_CALCULATION_TEMPLATES, min(len(CHART_CALCULATION_TEMPLATES), 4))
        for tmpl in templates:
            question, gold_tools, difficulty, eval_focus = tmpl
            samples.append({
                "id": f"eval_chart_{len(samples):03d}",
                "category": "chart_calculation",
                "image": img,
                "question": question,
                "gold_tools": gold_tools,
                "allowed_tools": ["vision_parse", "python_exec"],
                "forbidden_tools": ["retrieve_docs"],
                "difficulty": difficulty,
                "eval_focus": eval_focus,
                "gold_answer_keywords": [],
                "should_refuse": False,
                "evidence_required": True,
            })
    random.shuffle(samples)
    return samples[:target]


def generate_image_consistency(target: int) -> List[Dict]:
    samples = []
    for img in PRODUCT_IMAGES:
        pair = random.choice(PRODUCT_PAIRS)
        color, category = pair
        templates = random.sample(IMAGE_CONSISTENCY_TEMPLATES, min(len(IMAGE_CONSISTENCY_TEMPLATES), 3))
        for tmpl in templates:
            question, gold_tools, difficulty, eval_focus = tmpl
            # Fill template placeholders
            wrong_color = random.choice([c for c, _ in PRODUCT_PAIRS if c != color])
            wrong_category = random.choice([cat for _, cat in PRODUCT_PAIRS if cat != category])
            q = question.format(
                color=color, category=category,
                wrong_color=wrong_color, wrong_category=wrong_category,
            )
            samples.append({
                "id": f"eval_product_{len(samples):03d}",
                "category": "image_text_consistency",
                "image": img,
                "question": q,
                "gold_tools": gold_tools,
                "allowed_tools": ["vision_parse", "retrieve_docs"],
                "forbidden_tools": ["python_exec"],
                "difficulty": difficulty,
                "eval_focus": eval_focus,
                "gold_answer_keywords": [],
                "should_refuse": False,
                "evidence_required": True,
            })
    random.shuffle(samples)
    return samples[:target]


def generate_uncertainty_refusal(target: int) -> List[Dict]:
    samples = []
    # Blurry logos + some blur-simulating product images: should refuse
    refusal_images = BLURRY_LOGOS + random.sample(PRODUCT_IMAGES, 5)
    answer_images = READABLE_LOGOS + random.sample(PRODUCT_IMAGES, 5)

    for img in refusal_images:
        for tmpl in UNCERTAINTY_REFUSAL_TEMPLATES:
            if len(samples) >= target:
                break
            question, gold_tools, difficulty, eval_focus = tmpl
            samples.append({
                "id": f"eval_refusal_{len(samples):03d}",
                "category": "uncertainty_refusal",
                "image": img,
                "question": question,
                "gold_tools": gold_tools,
                "allowed_tools": ["vision_parse"],
                "forbidden_tools": [],
                "difficulty": difficulty,
                "eval_focus": eval_focus,
                "gold_answer_keywords": ["无法可靠", "模糊"],
                "should_refuse": True if eval_focus == "refusal" else False,
                "evidence_required": True,
            })

    for img in answer_images:
        for tmpl in UNCERTAINTY_REFUSAL_TEMPLATES:
            if len(samples) >= target:
                break
            question, gold_tools, difficulty, eval_focus = tmpl
            samples.append({
                "id": f"eval_refusal_{len(samples):03d}",
                "category": "uncertainty_refusal",
                "image": img,
                "question": question,
                "gold_tools": gold_tools,
                "allowed_tools": ["vision_parse"],
                "forbidden_tools": [],
                "difficulty": difficulty,
                "eval_focus": eval_focus if eval_focus != "refusal" else "answer_quality",
                "gold_answer_keywords": [],
                "should_refuse": False,
                "evidence_required": True,
            })
    random.shuffle(samples)
    return samples[:target]


def generate_direct_vqa(target: int) -> List[Dict]:
    samples = []
    all_imgs = PRODUCT_IMAGES + INVOICE_IMAGES[:5] + CHART_IMAGES[:5]
    for img in all_imgs:
        templates = random.sample(DIRECT_VQA_TEMPLATES, min(len(DIRECT_VQA_TEMPLATES), 2))
        for tmpl in templates:
            question, gold_tools, difficulty, eval_focus = tmpl
            samples.append({
                "id": f"eval_direct_{len(samples):03d}",
                "category": "direct_vqa",
                "image": img,
                "question": question,
                "gold_tools": gold_tools,
                "allowed_tools": [],  # direct_vqa should NOT use tools
                "forbidden_tools": ["vision_parse", "retrieve_docs", "python_exec"],
                "difficulty": difficulty,
                "eval_focus": eval_focus,
                "gold_answer_keywords": [],
                "should_refuse": False,
                "evidence_required": False,
            })
    random.shuffle(samples)
    return samples[:target]


# ============================================================
# Main
# ============================================================

def main():

    # Step 1: Generate all samples
    print("=" * 60)
    print("Generating 200 eval samples (100 dev + 100 test)...")
    print("=" * 60)

    targets = {
        "document_validation": 40,
        "chart_calculation": 40,
        "image_text_consistency": 40,
        "uncertainty_refusal": 40,
        "direct_vqa": 40,
    }

    generators = {
        "document_validation": generate_document_validation,
        "chart_calculation": generate_chart_calculation,
        "image_text_consistency": generate_image_consistency,
        "uncertainty_refusal": generate_uncertainty_refusal,
        "direct_vqa": generate_direct_vqa,
    }

    all_samples = []
    for cat, target in targets.items():
        cat_samples = generators[cat](target)
        all_samples.extend(cat_samples)
        # Summary
        diffs = defaultdict(int)
        focuses = defaultdict(int)
        for s in cat_samples:
            diffs[s["difficulty"]] += 1
            focuses[s["eval_focus"]] += 1
        print(f"  {cat}: {len(cat_samples)} samples")
        print(f"    difficulty: {dict(diffs)}")
        print(f"    eval_focus: {dict(focuses)}")

    # Assign unique IDs
    for i, s in enumerate(all_samples):
        idx = f"{i:04d}"
        s["id"] = f"eval_{s['category'][:4]}_{idx}"

    random.shuffle(all_samples)
    print(f"\nTotal: {len(all_samples)} samples across {len(targets)} categories")

    # Step 2: Set gold_answer_keywords heuristically from question templates
    print("\n" + "=" * 60)
    print("Setting gold_answer_keywords from question/template patterns...")
    print("=" * 60)

    import re

    for sample in all_samples:
        q = sample["question"]
        cat = sample["category"]
        should_refuse = sample.get("should_refuse", False)

        if should_refuse:
            sample["gold_answer_keywords"] = ["无法可靠", "模糊"]
            continue

        if cat == "document_validation":
            if "缺少" in q or "缺失" in q or "差异" in q:
                sample["gold_answer_keywords"] = ["不完整"]
            elif "提取" in q or "列出" in q or "读取" in q:
                sample["gold_answer_keywords"] = ["字段"]
            elif "金额" in q:
                sample["gold_answer_keywords"] = ["元", "金额"]
            elif "Tax ID" in q or "税号" in q:
                sample["gold_answer_keywords"] = ["Tax", "税号"]
            elif "合规" in q or "符合" in q:
                sample["gold_answer_keywords"] = ["合规", "完整"]
            else:
                sample["gold_answer_keywords"] = ["完整"]

        elif cat == "chart_calculation":
            if "增长率" in q or "增长" in q:
                sample["gold_answer_keywords"] = ["同比"]
            elif "预测" in q:
                sample["gold_answer_keywords"] = ["预测", "2025"]
            elif "提取" in q or "原始" in q:
                sample["gold_answer_keywords"] = ["2023", "2024"]
            elif "趋势" in q:
                sample["gold_answer_keywords"] = ["上升", "下降", "趋势"]
            elif "差值" in q:
                sample["gold_answer_keywords"] = ["差值"]
            elif "百分比" in q or "总和" in q:
                sample["gold_answer_keywords"] = ["百分比"]
            elif "数值" in q and "2024" in q:
                sample["gold_answer_keywords"] = ["2024"]
            else:
                sample["gold_answer_keywords"] = ["同比"]

        elif cat == "image_text_consistency":
            if "不一致" in q or "冲突" in q:
                sample["gold_answer_keywords"] = ["不一致"]
            elif "描述" in q:
                sample["gold_answer_keywords"] = ["一致", "不一致"]
            elif "匹配" in q:
                sample["gold_answer_keywords"] = ["匹配"]
            elif "拒绝" in q or "不够清晰" in q:
                sample["gold_answer_keywords"] = ["无法", "模糊"]
                sample["should_refuse"] = True
            else:
                sample["gold_answer_keywords"] = ["一致"]

        elif cat == "uncertainty_refusal":
            if "置信度" in q or "把握" in q:
                sample["gold_answer_keywords"] = ["置信度", "把握"]
            elif "拒绝" in q:
                sample["gold_answer_keywords"] = ["无法可靠", "无法判断"]
                sample["should_refuse"] = True
            elif "清晰度" in q:
                sample["gold_answer_keywords"] = ["清晰"]
            elif "文字" in q:
                sample["gold_answer_keywords"] = ["文字"]
            elif "视觉元素" in q:
                sample["gold_answer_keywords"] = ["元素"]
            else:
                sample["gold_answer_keywords"] = ["品牌", "logo"]

        elif cat == "direct_vqa":
            if "有几个" in q or "数量" in q:
                sample["gold_answer_keywords"] = ["个"]
            elif "色调" in q or "颜色" in q:
                sample["gold_answer_keywords"] = ["色"]
            elif "类型" in q or "是什么类型" in q:
                sample["gold_answer_keywords"] = ["发票", "图表", "商品", "logo", "图像"]
            elif "质量" in q:
                sample["gold_answer_keywords"] = ["清晰", "一般", "模糊"]
            elif "一句话" in q:
                sample["gold_answer_keywords"] = ["图片"]
            elif "是否需要调用工具" in q:
                sample["gold_answer_keywords"] = ["需要", "不需要"]
            elif "商品类型" in q:
                sample["gold_answer_keywords"] = ["shoe", "鞋", "商品"]
            else:
                sample["gold_answer_keywords"] = ["图片"]

    # Step 3: Split into dev (100) + test (100)
    print("\n" + "=" * 60)
    print("Splitting into dev (100) + test (100)...")
    print("=" * 60)

    by_cat = defaultdict(list)
    for s in all_samples:
        by_cat[s["category"]].append(s)

    dev, test = [], []
    for cat, cat_samples in sorted(by_cat.items()):
        random.shuffle(cat_samples)
        n_dev = len(cat_samples) // 2
        dev.extend(cat_samples[:n_dev])
        test.extend(cat_samples[n_dev:])

    random.shuffle(dev)
    random.shuffle(test)

    # Write
    for name, subset in [("eval_dev_v3", dev), ("eval_test_v3", test)]:
        path = ROOT / f"data/eval/{name}.jsonl"
        with open(path, "w") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(subset)} samples → {path}")
        cats = defaultdict(int)
        diffs = defaultdict(int)
        for s in subset:
            cats[s["category"]] += 1
            diffs[s["difficulty"]] += 1
        print(f"    categories: {dict(cats)}")
        print(f"    difficulty: {dict(diffs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
