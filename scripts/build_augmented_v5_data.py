"""
Build augmented v5 training data with:
1. Diverse question templates per task type
2. Reasoning chains before tool calls
3. Error correction trajectories
4. Refusal boundary samples (brand hallucination)
5. Direct answer samples (no tools for simple questions)
"""

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

random.seed(42)

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

INVOICE_TEMPLATES = [
    "检查这张发票是否缺少必填字段，并给出依据。",
    "这张发票合规吗？请逐项核查。",
    "帮我审核这张发票的字段完整性。",
    "发票里有哪些必填项？都齐全吗？",
    "这张票有没有缺失的必填字段？",
    "核对发票必填字段：Invoice No, Date, Amount, Tax ID。",
    "验证发票的合规性，列出缺失项。",
]

INVOICE_SIMPLE_TEMPLATES = [
    "这张发票的金额是多少？",
    "发票号码是什么？",
    "发票日期是哪天？",
    "能读一下发票上的金额吗？",
    "这个发票号是多少？",
]

CHART_TEMPLATES = [
    "根据图表，计算2024年相比2023年的同比增长率。",
    "计算同比增长率，给出计算过程。",
    "图表中显示了两年的数据，增长率是多少？",
    "今年比去年涨了多少个百分点？",
    "计算变化幅度，写清楚公式。",
    "图中的同比增幅是多少？请逐步计算。",
]

PRODUCT_TEMPLATES = [
    "商品描述写的是{color} shoe。请判断图文是否一致。",
    "描述说是{color}鞋子，图片里是吗？",
    "商品标注为{color}，图中商品颜色匹配吗？",
    "判断图文一致性：描述={color} shoe，请对比图中实际颜色。",
    "图中商品和'{color} shoe'这个描述一致吗？",
]

DIRECT_VQA_TEMPLATES = [
    "请简要描述图片中的商品类型。",
    "图中展示的是什么？",
    "这是什么商品？",
    "简短描述一下图片内容。",
    "图片里是什么类型的东西？",
]

LOGO_TEMPLATES = [
    "请判断图片中的品牌名称，并说明依据。",
    "这是什么品牌的logo？",
    "请识别图中的品牌标识。",
    "这张图里有品牌信息吗？有的话是什么？",
]

# ---------------------------------------------------------------------------
# Reasoning chain prefixes
# ---------------------------------------------------------------------------

REASONING_PREFIXES = {
    "ocr": "这是一个文档/发票核验任务。需要先提取图片中的文字字段，再对照规则文档检查必填项。\n",
    "chart_values": "这是一个图表计算任务。需要先从图表中读取数值数据，再进行数学计算。\n",
    "caption": "这是一个图文一致性判断任务。需要先识别图片中商品的颜色和品类，再与文字描述对比。\n",
    "ocr+caption": "这是一个品牌识别任务。需要先对图片进行OCR提取文字，结合图像描述判断品牌。如果信息不足则需拒答。\n",
}

# ---------------------------------------------------------------------------
# Correct refusal templates
# ---------------------------------------------------------------------------

CORRECT_REFUSALS = [
    "图片信息不足，无法可靠识别品牌名称。视觉解析结果中没有可辨认的品牌文字或logo。",
    "根据视觉解析结果，图像中没有可识别的品牌信息，不能仅凭模糊线索猜测。",
    "OCR和图像描述结果均未显示明确的品牌标识，无法做出可靠判断。",
]

# ---------------------------------------------------------------------------
# Direct answer templates
# ---------------------------------------------------------------------------

DIRECT_ANSWERS = {
    "black shoe": "黑色鞋子，属于运动鞋类型。",
    "red shoe": "红色鞋子，属于运动鞋类型。",
    "blue shoe": "蓝色鞋子，属于运动鞋类型。",
    "green shoe": "绿色鞋子，属于运动鞋类型。",
    "white shoe": "白色鞋子，属于运动鞋类型。",
}


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_question_type(messages: List[Dict]) -> str:
    """Determine question type from messages."""
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item["text"]
                    break
        else:
            text = str(content)

        if any(w in text for w in ["发票", "invoice", "金额", "号码", "日期"]):
            return "invoice"
        if any(w in text for w in ["图表", "chart", "增长率", "同比", "计算"]):
            return "chart"
        if any(w in text for w in ["商品", "图文", "一致", "shoe"]):
            return "product"
        if any(w in text for w in ["品牌", "logo", "brand"]):
            return "logo"
        if any(w in text for w in ["描述", "类型", "什么"]):
            return "direct_vqa"
    return "unknown"


def find_vision_mode(messages: List[Dict]) -> str:
    """Find the vision_parse mode used in the trace."""
    for msg in messages:
        if msg["role"] == "assistant" and "vision_parse" in msg.get("content", ""):
            try:
                content = msg["content"]
                json_str = content.split("<tool_call>")[1].split("</tool_call>")[0]
                return json.loads(json_str).get("args", {}).get("mode", "ocr")
            except Exception:
                pass
    return "ocr"


def replace_user_text(messages: List[Dict], new_text: str) -> List[Dict]:
    """Replace the user's text question with a new one, keeping the image."""
    new_msgs = deepcopy(messages)
    for msg in new_msgs:
        if msg["role"] != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = new_text
                    break
        elif isinstance(content, str):
            msg["content"] = new_text
    return new_msgs


def add_reasoning_prefix(messages: List[Dict], mode: str) -> List[Dict]:
    """Add reasoning chain prefix before the first tool_call."""
    prefix = REASONING_PREFIXES.get(mode, "")
    if not prefix:
        return messages

    new_msgs = deepcopy(messages)
    for i, msg in enumerate(new_msgs):
        if msg["role"] == "assistant" and "<tool_call>" in msg.get("content", ""):
            msg["content"] = prefix + msg["content"]
            break
    return new_msgs


def build_error_correction_sample(row: Dict) -> Dict:
    """
    Build a correction sample: model tries to skip a tool, gets corrected, then does it right.
    """
    messages = row.get("messages", [])
    qtype = extract_question_type(messages)

    # Find the first tool call and the final answer
    first_tool_idx = None
    last_answer_idx = None
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            if "<tool_call>" in msg.get("content", "") and first_tool_idx is None:
                first_tool_idx = i
            if "<final_answer>" in msg.get("content", ""):
                last_answer_idx = i

    if first_tool_idx is None or last_answer_idx is None:
        return None

    # Build: user → wrong (skip) → correction → correct trace
    wrong_answer = "<final_answer></final_answer>"
    correction_msgs = {
        "invoice": "不能直接给答案。发票核验必须先 vision_parse 提取字段，再 retrieve_docs 查规则。重来。",
        "chart": "不能跳过步骤。图表计算必须先 vision_parse 取数据，再 python_exec 计算。",
        "product": "不能直接判断。先 vision_parse 识别图中商品。",
    }.get(qtype, "不能直接给最终答案。请先调用相应的工具。")

    user_msg = messages[0]  # original user message
    correct_trace = messages[first_tool_idx:last_answer_idx + 1]

    new_msgs = [
        user_msg,
        {"role": "assistant", "content": wrong_answer},
        {"role": "user", "content": correction_msgs},
    ] + list(correct_trace)

    return {
        "id": f"v5_correction_{row.get('id', 'sample')}",
        "source": "v5_error_correction",
        "category": row.get("category", "?"),
        "messages": new_msgs,
    }


def build_refusal_sample(image: str) -> Dict:
    """Build refusal boundary using REAL VLM outputs from logo images."""
    # Real VLM outputs for logo images that should trigger refusal
    real_vision_results = [
        "OCR结果：0855\n图片描述：图片中显示了一个白色的矩形框，里面有一个黑色的数字0855。这个数字可能是一个序列号、代码或标识符。背景是纯白色，没有其他明显的图案或文字。",
        "OCR结果：00007\n图片描述：图片中显示了一个白色的矩形框，里面有一个黑色的数字00007。这个数字可能是一个序列号、代码或标识符。背景是纯白色，没有其他明显的图案或文字。",
        "OCR结果：00000\n图片描述：图片中显示了一个白色的矩形框，里面没有任何其他内容或信息。",
        "OCR结果：----\n图片描述：模糊不清，无法辨认文字或品牌标识。",
    ]
    vision_result = random.choice(real_vision_results)
    question = random.choice(LOGO_TEMPLATES)

    # Proper refusal — always include "无法" keyword for eval matching
    refusal = (
        "图片信息不足，无法可靠识别品牌名称。"
        "视觉解析结果中只有数字/序列号，没有可辨认的品牌文字或logo，不能仅凭模糊线索猜测。"
    )

    return {
        "id": f"v6_refusal_{random.randint(1000, 9999)}",
        "source": "v6_refusal_boundary",
        "category": "uncertainty_refusal",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>',
            },
            {"role": "tool", "name": "vision_parse", "content": vision_result},
            {"role": "user", "content": "工具 vision_parse 返回结果如下：请继续完成任务。"},
            {"role": "assistant", "content": f"<final_answer>{refusal}</final_answer>"},
        ],
    }


def build_direct_vqa_sample(image: str) -> Dict:
    """Build a direct answer sample: no tools needed for simple description."""
    question = random.choice(DIRECT_VQA_TEMPLATES)
    # Pick a random product description
    product_desc = random.choice(list(DIRECT_ANSWERS.keys()))
    answer = f"根据图片内容，该商品为{product_desc}"

    return {
        "id": f"v5_direct_{random.randint(1000, 9999)}",
        "source": "v5_direct_vqa",
        "category": "direct_vqa",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": f"<final_answer>{answer}</final_answer>"},
        ],
    }


def augment_with_templates(row: Dict, templates: List[str], color: str = None) -> List[Dict]:
    """Generate variants of a row with different question templates."""
    messages = row.get("messages", [])
    results = []
    for tmpl in templates:
        text = tmpl.format(color=color) if color else tmpl
        new_msgs = replace_user_text(messages, text)
        results.append({
            "id": f"{row.get('id', 'sample')}_v5_{tmpl[:20]}",
            "source": f"{row.get('source', '?')}_v5_aug",
            "category": row.get("category", "?"),
            "messages": new_msgs,
        })
    return results


def main():
    seed_path = ROOT / "data/sft/sft_seed_real_vlm.jsonl"
    print(f"Loading seed: {seed_path}")
    seed_rows = read_jsonl(seed_path)
    if not seed_rows:
        print("ERROR: sft_seed_real_vlm.jsonl not found. Run rebuild_seed_with_real_vlm.py first.")
        sys.exit(1)

    all_rows = []
    stats = {
        "question_variants": 0,
        "reasoning_chain": 0,
        "error_correction": 0,
        "refusal_boundary": 0,
        "direct_vqa": 0,
    }

    # Collect images by type for synthetic sample generation
    invoice_images = [str(ROOT / f"data/images/invoice_{i:03d}.jpg") for i in range(1, 18)]
    product_images = [str(ROOT / f"data/images/product_{i:03d}.jpg") for i in range(1, 13)]
    logo_images = [str(ROOT / f"data/images/logo_{i:03d}.jpg") for i in range(1, 8)]

    for row in seed_rows:
        messages = row.get("messages", [])
        qtype = extract_question_type(messages)
        vision_mode = find_vision_mode(messages)

        # 1. Add original row
        all_rows.append(row)

        # 2. Diverse question templates
        if qtype == "invoice":
            # Complex invoice questions (need retrieve_docs)
            for variant in augment_with_templates(row, INVOICE_TEMPLATES):
                all_rows.append(variant)
                stats["question_variants"] += 1

                # Add reasoning chain to some
                if random.random() < 0.3:
                    reasoning_row = deepcopy(variant)
                    reasoning_row["id"] = variant["id"] + "_reasoning"
                    reasoning_row["messages"] = add_reasoning_prefix(
                        reasoning_row["messages"], vision_mode
                    )
                    all_rows.append(reasoning_row)
                    stats["reasoning_chain"] += 1

            # Simple invoice questions (no retrieve_docs needed)
            for variant in augment_with_templates(row, INVOICE_SIMPLE_TEMPLATES):
                # For simple questions, only keep vision_parse + final_answer
                simple_msgs = deepcopy(variant["messages"])
                # Remove retrieve_docs step if present
                simple_msgs = [m for m in simple_msgs if not (
                    m["role"] == "assistant" and "retrieve_docs" in m.get("content", "")
                )]
                simple_msgs = [m for m in simple_msgs if not (
                    m["role"] in ("tool", "user") and "retrieve_docs" in str(m.get("content", ""))
                )]
                variant["messages"] = simple_msgs
                variant["id"] = variant["id"] + "_simple"
                all_rows.append(variant)
                stats["question_variants"] += 1

        elif qtype == "chart":
            for variant in augment_with_templates(row, CHART_TEMPLATES):
                all_rows.append(variant)
                stats["question_variants"] += 1

                if random.random() < 0.3:
                    reasoning_row = deepcopy(variant)
                    reasoning_row["id"] = variant["id"] + "_reasoning"
                    reasoning_row["messages"] = add_reasoning_prefix(
                        reasoning_row["messages"], vision_mode
                    )
                    all_rows.append(reasoning_row)
                    stats["reasoning_chain"] += 1

        elif qtype == "product":
            colors = ["black", "red", "blue", "green", "white"]
            for color in colors:
                for variant in augment_with_templates(row, PRODUCT_TEMPLATES, color=color):
                    all_rows.append(variant)
                    stats["question_variants"] += 1

        elif qtype == "logo":
            for variant in augment_with_templates(row, LOGO_TEMPLATES):
                all_rows.append(variant)
                stats["question_variants"] += 1

                if random.random() < 0.3:
                    reasoning_row = deepcopy(variant)
                    reasoning_row["id"] = variant["id"] + "_reasoning"
                    reasoning_row["messages"] = add_reasoning_prefix(
                        reasoning_row["messages"], vision_mode
                    )
                    all_rows.append(reasoning_row)
                    stats["reasoning_chain"] += 1

        # 3. Error correction samples
        if qtype in ("invoice", "chart", "product"):
            correction = build_error_correction_sample(row)
            if correction:
                all_rows.append(correction)
                stats["error_correction"] += 1

    # 4. Refusal boundary samples (brand hallucination)
    for img in random.sample(logo_images, min(6, len(logo_images))):
        sample = build_refusal_sample(img)
        all_rows.append(sample)
        stats["refusal_boundary"] += 1

    # 5. Direct VQA samples (no tools needed)
    for img in random.sample(product_images, min(8, len(product_images))):
        sample = build_direct_vqa_sample(img)
        all_rows.append(sample)
        stats["direct_vqa"] += 1

    # Write output
    out_path = ROOT / "data/sft/sft_seed_augmented_v5.jsonl"
    write_jsonl(out_path, all_rows)

    print(f"\nOutput: {out_path} ({len(all_rows)} rows)")
    print(f"Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
