"""
Convert CORD (naver-clova-ix/cord-v2) to the agent SFT format.

CORD (Consolidated Receipt Dataset) contains receipt images with parsed
JSON ground truth (menu items, prices, totals, store info).

We convert each sample to a tool-calling trace:
  vision_parse(ocr) → retrieve_docs → final_answer

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/convert_cord_to_sft.py --split train --max_samples 500
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_OUT = ROOT / "data/sft/sft_cord.jsonl"
DEFAULT_IMAGE_DIR = Path("/root/autodl-tmp/multimodal-agent-data/cord/images")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def slug_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value)
    value = value.strip("_")
    return value[:80] or "sample"


def save_image(image_value: Any, image_dir: Path, sample_id: str) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(image_value, "save"):
        dst = image_dir / f"{sample_id}.png"
        if not dst.exists():
            image_value.save(dst)
        try:
            return str(dst.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(dst).replace("\\", "/")
    raise ValueError(f"Unsupported image value type: {type(image_value)}")


PLACEHOLDER_RESULTS = {
    "ocr": "收据已解析；请根据视觉证据回答问题。",
    "chart_values": "图表已解析；请结合图表视觉证据和问题进行回答。",
}


def _get_vision_result(vision_parser, image_path: str, mode: str) -> str:
    if vision_parser is not None:
        try:
            return vision_parser.parse(image_path, mode)
        except Exception as e:
            print(f"  WARNING: real VLM failed for {image_path}: {e}, using placeholder")
    return PLACEHOLDER_RESULTS.get(mode, PLACEHOLDER_RESULTS["ocr"])


def tool_call(name: str, args: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": f"<tool_call>{json.dumps({'name': name, 'args': args}, ensure_ascii=False)}</tool_call>",
    }


def tool_result(name: str, content: str) -> Dict[str, str]:
    return {"role": "tool", "name": name, "content": content}


def final_answer(content: str) -> Dict[str, str]:
    return {"role": "assistant", "content": f"<final_answer>{content}</final_answer>"}


def extract_receipt_info(gt: dict) -> dict:
    """Extract summary from CORD ground truth JSON."""
    info = {}
    gt_parse = gt.get("gt_parse", {}) if isinstance(gt, dict) else {}
    menu = gt_parse.get("menu", [])
    if isinstance(menu, list):
        info["items"] = []
        info["total_items"] = len(menu)
        for item in menu:
            if isinstance(item, dict):
                info["items"].append({
                    "name": item.get("nm", ""),
                    "count": item.get("cnt", ""),
                    "price": item.get("price", ""),
                })
            elif isinstance(item, str):
                info["items"].append({"name": item, "count": "", "price": ""})
    total = gt_parse.get("total", {}) if isinstance(gt_parse, dict) else {}
    if isinstance(total, dict):
        info["total_price"] = total.get("total_price", "")
        info["cashprice"] = total.get("cashprice", "")
        info["changeprice"] = total.get("changeprice", "")
    return info


def generate_questions(info: dict) -> List[str]:
    """Generate multiple QA pairs from one receipt."""
    questions = []

    # 1. How many items?
    if info.get("total_items"):
        questions.append(f"这张收据中有几件商品？答案：{info['total_items']}件。")

    # 2. Total amount
    if info.get("total_price"):
        questions.append(f"这张收据的总金额是多少？答案：{info['total_price']}。")

    # 3. Item list
    items = info.get("items", [])
    if items:
        item_names = "、".join(it["name"] for it in items[:5])
        questions.append(f"这张收据中包含哪些商品？答案：{item_names}。")

    # 4. First/Most expensive item question (for field checking)
    if items:
        first = items[0]
        questions.append(
            f"收据中第一件商品是什么，价格是多少？"
            f"答案：{first['name']}，价格{first['price']}。"
        )

    return questions


def convert_example(
    example: Dict[str, Any],
    index: int,
    split: str,
    image_dir: Path,
    vision_parser: Any = None,
) -> Optional[List[Dict[str, Any]]]:
    image = example.get("image")
    gt = example.get("ground_truth")
    if image is None or not gt:
        return None

    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return None
    if not isinstance(gt, dict):
        return None

    info = extract_receipt_info(gt)
    if not info.get("items"):
        return None

    sample_id = f"cord_{split}_{index:06d}"
    image_path = save_image(image, image_dir=image_dir, sample_id=sample_id)

    questions = generate_questions(info)
    rows = []
    for qi, qa_text in enumerate(questions):
        question, answer = qa_text.split("答案：", 1) if "答案：" in qa_text else (qa_text, "")
        question = question.strip()
        answer = answer.strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
            tool_call("vision_parse", {"mode": "ocr"}),
            tool_result(
                "vision_parse",
                _get_vision_result(vision_parser, image_path, "ocr"),
            ),
            tool_call("retrieve_docs", {"query": "invoice receipt required fields"}),
            tool_result(
                "retrieve_docs",
                "规则文档：收据/发票应包含商品明细、数量、单价、总金额等字段。",
            ),
            final_answer(f"根据收据信息，{answer}"),
        ]

        rows.append({
            "id": f"{sample_id}_q{qi}",
            "source": "cord",
            "category": "document_validation",
            "messages": messages,
            "metadata": {
                "split": split,
                "question_index": qi,
            },
        })

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="naver-clova-ix/cord-v2")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument(
        "--use_real_vlm",
        action="store_true",
        help="Use real Qwen2.5-VL model for vision_parse instead of placeholder text.",
    )
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
        help="Path to VLM model for --use_real_vlm.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if not path_obj.is_absolute() else path_obj


def main():
    args = parse_args()
    out_path = resolve_path(args.out_path)
    image_dir = resolve_path(args.image_dir)

    vision_parser = None
    if args.use_real_vlm:
        from scripts.real_vision_parse import RealVisionParse
        vision_parser = RealVisionParse(args.model_name)

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        if hasattr(dataset, "keys"):
            dataset = dataset[args.split]
    else:
        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            streaming=True,
        )

    rows = []
    skipped = 0
    for index, example in enumerate(dataset):
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
        sample_rows = convert_example(example, index, args.split, image_dir,
                                       vision_parser=vision_parser)
        if sample_rows is None:
            skipped += 1
            continue
        rows.extend(sample_rows)

    count = write_jsonl(out_path, rows)

    print("========== CORD Conversion ==========")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Receipts processed: {len(rows) // 4 if rows else 0} (~4 QA per receipt)")
    print(f"Output QA rows: {count}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out_path}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
