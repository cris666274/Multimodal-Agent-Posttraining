"""
Convert DocVQA (lmms-lab/DocVQA) to the agent SFT format.

DocVQA contains document images with questions and answers.
We convert each sample to a tool-calling trace:
  vision_parse(ocr) → [retrieve_docs if compliance-related] → final_answer

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/convert_docvqa_to_sft.py --split validation --max_samples 500
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
DEFAULT_OUT = ROOT / "data/sft/sft_docvqa.jsonl"
DEFAULT_IMAGE_DIR = Path("/root/autodl-tmp/multimodal-agent-data/docvqa/images")

COMPLIANCE_KEYWORDS = [
    "required", "mandatory", "must", "should", "comply", "compliance",
    "missing", "field", "rule", "regulation", "standard",
    "必填", "必须", "合规", "字段", "规则", "缺少", "规范",
]


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
    "chart_values": "图表已解析；请结合图表视觉证据和问题进行回答。此样本来自 ChartQA，标准答案用于监督最终回答。",
    "ocr": "文档已解析；请根据视觉证据回答问题。",
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


def is_compliance_question(question: str) -> bool:
    q = question.lower()
    return any(kw.lower() in q for kw in COMPLIANCE_KEYWORDS)


def convert_example(
    example: Dict[str, Any],
    index: int,
    split: str,
    image_dir: Path,
    vision_parser: Any = None,
) -> Optional[Dict[str, Any]]:
    question = (example.get("question") or "").strip()
    answers = example.get("answers") or []
    answer = answers[0] if answers else ""
    image = example.get("image")

    if not question or not answer or image is None:
        return None

    sample_id = f"docvqa_{split}_{index:06d}_{slug_id(question)}"
    image_path = save_image(image, image_dir=image_dir, sample_id=sample_id)

    needs_rules = is_compliance_question(question)

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
    ]

    if needs_rules:
        messages.extend([
            tool_call("retrieve_docs", {"query": "document field rules"}),
            tool_result("retrieve_docs", "规则文档：请根据文档内容判断是否满足相关字段要求。"),
        ])

    messages.append(final_answer(f"根据文档信息，{answer}。"))

    return {
        "id": sample_id,
        "source": "docvqa",
        "category": "document_validation" if needs_rules else "ocr_qa",
        "messages": messages,
        "metadata": {
            "split": split,
            "original_answers": answers,
            "used_retrieve_docs": needs_rules,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lmms-lab/DocVQA")
    parser.add_argument("--dataset_config", default="DocVQA")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--max_samples", type=int, default=1000)
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
            args.dataset_config,
            split=args.split,
            streaming=True,
        )

    rows = []
    skipped = 0
    for index, example in enumerate(dataset):
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
        row = convert_example(example, index, args.split, image_dir,
                              vision_parser=vision_parser)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    count = write_jsonl(out_path, rows)
    with_retrieve = sum(r["metadata"]["used_retrieve_docs"] for r in rows)

    print("========== DocVQA Conversion ==========")
    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Split: {args.split}")
    print(f"Output rows: {count}")
    print(f"Skipped rows: {skipped}")
    print(f"Rows with retrieve_docs: {with_retrieve}")
    print(f"Output: {out_path}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
