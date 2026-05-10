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
DEFAULT_OUT = ROOT / "data/sft/sft_chartqa.jsonl"
DEFAULT_IMAGE_DIR = ROOT / "data/external/chartqa/images"


QUESTION_FIELDS = ["query", "question", "Question", "input", "prompt"]
ANSWER_FIELDS = ["label", "answer", "answers", "Answer", "output"]
IMAGE_FIELDS = ["image", "img", "chart", "png", "image_path"]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def first_existing_field(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for field in candidates:
        if field in example and example[field] is not None:
            return field
    return None


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def normalize_question(value: Any) -> str:
    return str(value).strip()


def slug_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value)
    value = value.strip("_")
    return value[:80] or "sample"


def save_image(image_value: Any, image_dir: Path, sample_id: str) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, str):
        src = Path(image_value)
        if src.exists():
            suffix = src.suffix or ".png"
            dst = image_dir / f"{sample_id}{suffix}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            return path_for_json(dst)
        return image_value

    if hasattr(image_value, "save"):
        dst = image_dir / f"{sample_id}.png"
        image_value.save(dst)
        return path_for_json(dst)

    raise ValueError(f"Unsupported image value type: {type(image_value)}")


def path_for_json(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def is_numeric_answer(answer: str) -> bool:
    text = answer.strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?%?", text))


def numeric_code(answer: str) -> str:
    text = answer.strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    return text


def should_use_python_exec(question: str, answer: str, mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return is_numeric_answer(answer)

    keywords = [
        "calculate",
        "computed",
        "difference",
        "sum",
        "total",
        "average",
        "percent",
        "percentage",
        "ratio",
        "growth",
        "increase",
        "decrease",
        "how many more",
        "how much more",
        "计算",
        "增长",
        "同比",
        "百分比",
        "总和",
        "平均",
    ]
    q = question.lower()
    return is_numeric_answer(answer) and any(keyword in q for keyword in keywords)


PLACEHOLDER_RESULTS = {
    "chart_values": "图表已解析；请结合图表视觉证据和问题进行回答。此样本来自 ChartQA，标准答案用于监督最终回答。",
    "ocr": "文档已解析；请结合文档视觉证据和问题进行回答。此样本来自外部数据集，标准答案用于监督最终回答。",
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
    return {
        "role": "tool",
        "name": name,
        "content": content,
    }


def final_answer(content: str) -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": f"<final_answer>{content}</final_answer>",
    }


def convert_example(
    example: Dict[str, Any],
    index: int,
    split: str,
    image_dir: Path,
    python_exec_mode: str,
    vision_parser: Any = None,
) -> Optional[Dict[str, Any]]:
    question_field = first_existing_field(example, QUESTION_FIELDS)
    answer_field = first_existing_field(example, ANSWER_FIELDS)
    image_field = first_existing_field(example, IMAGE_FIELDS)

    if question_field is None or answer_field is None or image_field is None:
        return None

    question = normalize_question(example[question_field])
    answer = normalize_answer(example[answer_field]).strip()
    if not question or not answer:
        return None

    sample_id = f"chartqa_{split}_{index:06d}_{slug_id(question)}"
    image_path = save_image(example[image_field], image_dir=image_dir, sample_id=sample_id)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        tool_call("vision_parse", {"mode": "chart_values"}),
        tool_result(
            "vision_parse",
            _get_vision_result(vision_parser, image_path, "chart_values"),
        ),
    ]

    if should_use_python_exec(question, answer, python_exec_mode):
        code = numeric_code(answer)
        messages.extend(
            [
                tool_call("python_exec", {"code": code}),
                tool_result("python_exec", code),
            ]
        )

    messages.append(final_answer(f"根据图表信息，答案是 {answer}。"))

    return {
        "id": sample_id,
        "source": "chartqa",
        "category": "chart_calculation",
        "messages": messages,
        "metadata": {
            "split": split,
            "question_field": question_field,
            "answer_field": answer_field,
            "image_field": image_field,
            "used_python_exec": should_use_python_exec(question, answer, python_exec_mode),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HuggingFaceM4/ChartQA")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Local dataset path saved by datasets.save_to_disk(). Overrides --dataset_name.",
    )
    parser.add_argument(
        "--data_files",
        nargs="*",
        default=None,
        help="Local json/jsonl/parquet/csv files. Example: --data_files train.jsonl",
    )
    parser.add_argument(
        "--data_format",
        choices=["json", "parquet", "csv"],
        default=None,
        help="Required when --data_files is used unless file suffix is obvious.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--python_exec_mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="When to add a weak python_exec step for numeric answers.",
    )
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
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    return path_obj


def infer_data_format(data_files: List[str]) -> str:
    suffixes = {Path(item).suffix.lower() for item in data_files}
    if suffixes & {".json", ".jsonl"}:
        return "json"
    if ".parquet" in suffixes:
        return "parquet"
    if ".csv" in suffixes:
        return "csv"
    raise ValueError(
        "无法根据文件后缀推断数据格式，请显式传入 --data_format json/parquet/csv"
    )


def load_chartqa_dataset(args):
    if args.dataset_path:
        dataset_path = resolve_path(args.dataset_path)
        dataset = load_from_disk(str(dataset_path))
        if hasattr(dataset, "keys"):
            return dataset[args.split]
        return dataset

    if args.data_files:
        data_files = [str(resolve_path(item)) for item in args.data_files]
        data_format = args.data_format or infer_data_format(data_files)
        dataset = load_dataset(
            data_format,
            data_files={args.split: data_files},
            split=args.split,
        )
        return dataset

    try:
        return load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        raise RuntimeError(
            "无法在线加载 HuggingFace 数据集。当前环境可能无法访问网络。\n"
            "可选解决方案：\n"
            "1. 在有网机器下载后用 dataset.save_to_disk() 保存，再传到服务器，使用 --dataset_path。\n"
            "2. 准备本地 json/jsonl/parquet 文件，使用 --data_files 和 --data_format。\n"
            "3. 如果你有 HuggingFace 镜像，可先设置 HF_ENDPOINT 后重试。\n"
            f"原始错误：{e}"
        ) from e


def main():
    args = parse_args()
    out_path = resolve_path(args.out_path)
    image_dir = resolve_path(args.image_dir)

    vision_parser = None
    if args.use_real_vlm:
        from real_vision_parse import RealVisionParse
        vision_parser = RealVisionParse(args.model_name)

    dataset = load_chartqa_dataset(args)
    rows = []
    skipped = 0

    for index, example in enumerate(dataset):
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

        row = convert_example(
            example=example,
            index=index,
            split=args.split.replace("/", "_"),
            image_dir=image_dir,
            python_exec_mode=args.python_exec_mode,
            vision_parser=vision_parser,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    count = write_jsonl(out_path, rows)
    used_python_exec = sum(row["metadata"]["used_python_exec"] for row in rows)

    print("========== ChartQA Conversion ==========")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Output rows: {count}")
    print(f"Skipped rows: {skipped}")
    print(f"Rows with python_exec: {used_python_exec}")
    print(f"Output: {out_path}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
