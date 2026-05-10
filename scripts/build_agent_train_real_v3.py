"""
Build sft_agent_train_real_v3_enforced.jsonl by merging:
  - sft_agent_train_v3.jsonl (base: seed + hard cases + second_tool)
  - sft_chartqa.jsonl (ChartQA, python_exec rows)
  - sft_docvqa.jsonl (DocVQA, document understanding)
  - sft_cord.jsonl (CORD, receipt/invoice understanding)

Focus: upsample rows that use retrieve_docs or python_exec to
strengthen the model's second-tool-calling behavior.
Add recovery patterns to teach the model enforcement flow.
"""

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data/sft/sft_agent_train_v3.jsonl"
DEFAULT_CHARTQA_INPUT = ROOT / "data/sft/sft_chartqa.jsonl"
DEFAULT_DOCVQA_INPUT = ROOT / "data/sft/sft_docvqa.jsonl"
DEFAULT_CORD_INPUT = ROOT / "data/sft/sft_cord.jsonl"
DEFAULT_OUT = ROOT / "data/sft/sft_agent_train_real_v3_enforced.jsonl"

# Recovery prompt templates — aligned with runtime.py _missing_required_tool_prompt
_RECOVERY_PROMPTS = {
    "retrieve_docs": (
        "你已经读取了图片信息，但这个问题涉及发票字段、必填项、合规或规则判断。\n"
        "在给出最终答案之前，必须调用 retrieve_docs 获取规则依据。\n"
        "请输出：<tool_call>{\"name\":\"retrieve_docs\",\"args\":{\"query\":\"发票必填字段规则\"}}</tool_call>"
    ),
    "python_exec": (
        "你已经读取了图片信息，但这个问题涉及图表、同比、增长率或计算。\n"
        "在给出最终答案之前，必须调用 python_exec 完成数值计算。\n"
        "请输出：<tool_call>{\"name\":\"python_exec\",\"args\":{\"code\":\"(90-80)/80*100\"}}</tool_call>\n"
        "如果工具返回的图表数值不是 90 和 80，请根据 vision_parse 的结果替换表达式中的数字。"
    ),
}


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path} line {line_no} JSON parse failed: {e}"
                ) from e
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row, source):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, f"{source}: invalid messages"
    if messages[0].get("role") != "user":
        return False, f"{source}: first message is not user"
    if not any(msg.get("role") == "assistant" for msg in messages):
        return False, f"{source}: missing assistant message"
    return True, ""


def uses_tool(row, tool_name):
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if f'"name":"{tool_name}"' in content.replace(" ", ""):
            return True
    return False


def count_tool_calls(row):
    count = 0
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for t in ("vision_parse", "retrieve_docs", "python_exec"):
            if f'"name":"{t}"' in content.replace(" ", ""):
                count += 1
    return count


def is_chartqa_python_exec_row(row):
    metadata = row.get("metadata", {})
    if metadata.get("used_python_exec"):
        return True
    return uses_tool(row, "python_exec")


def find_second_tool_name(row):
    """Return 'retrieve_docs' or 'python_exec' if the row uses that tool."""
    for tool in ("retrieve_docs", "python_exec"):
        if uses_tool(row, tool):
            return tool
    return None


def _is_tool_result(msg):
    """Tool result messages may use role 'tool' or 'user'."""
    return msg.get("role") in ("tool", "user")


def build_recovery_variant(row, second_tool_name):
    """
    Given a row that has a full multi-step flow ending with final_answer,
    generate a recovery variant where the model tries to skip the second tool
    and gets corrected.

    Insert a wrong final_answer + correction prompt before the real second tool call.
    """
    messages = row.get("messages", [])
    if len(messages) < 4:
        return None

    # Find where vision_parse was called and its result was returned
    vision_tool_index = None
    vision_result_index = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "vision_parse" in msg.get("content", ""):
            vision_tool_index = i
        if (
            _is_tool_result(msg)
            and vision_tool_index is not None
            and i > vision_tool_index
        ):
            vision_result_index = i
            break

    if vision_result_index is None:
        return None

    # Find the second tool call and final answer
    second_tool_index = None
    second_result_index = None
    final_answer_index = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and second_tool_name in msg.get("content", ""):
            second_tool_index = i
        if (
            _is_tool_result(msg)
            and second_tool_index is not None
            and i > second_tool_index
            and second_result_index is None
        ):
            second_result_index = i
        if msg.get("role") == "assistant" and "<final_answer>" in msg.get("content", ""):
            final_answer_index = i

    if second_tool_index is None or final_answer_index is None:
        return None

    # Build new messages: prefix up to vision_result, insert wrong answer + correction,
    # then continue with second tool and final answer
    new_messages = list(messages[: vision_result_index + 1])

    # Wrong premature final_answer
    wrong_answer = "<final_answer></final_answer>"
    new_messages.append({"role": "assistant", "content": wrong_answer})

    # Correction prompt
    correction_prompt = _RECOVERY_PROMPTS.get(second_tool_name, "")
    if not correction_prompt:
        return None
    new_messages.append({"role": "user", "content": correction_prompt})

    # Append the rest (second tool call, its result, final answer)
    if second_result_index is not None:
        new_messages.extend(messages[second_tool_index:])
    else:
        new_messages.extend(messages[second_tool_index : final_answer_index + 1])

    new_row = dict(row)
    new_row["id"] = f"{row.get('id', 'sample')}_recovery"
    new_row["source"] = row.get("source", "?") + "_recovery"
    new_row["messages"] = new_messages
    return new_row


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input", default=str(DEFAULT_BASE_INPUT))
    parser.add_argument("--chartqa_input", default=str(DEFAULT_CHARTQA_INPUT))
    parser.add_argument("--docvqa_input", default=str(DEFAULT_DOCVQA_INPUT))
    parser.add_argument("--cord_input", default=str(DEFAULT_CORD_INPUT))
    parser.add_argument("--out_path", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--python_exec_repeat",
        type=int,
        default=2,
        help="Repeat rows that use python_exec.",
    )
    parser.add_argument(
        "--retrieve_docs_repeat",
        type=int,
        default=2,
        help="Repeat rows that use retrieve_docs.",
    )
    parser.add_argument(
        "--external_sample_ratio",
        type=float,
        default=0.5,
        help="Fraction of external (ChartQA/CORD/DocVQA) rows to keep (0-1).",
    )
    parser.add_argument(
        "--recovery_repeat",
        type=int,
        default=1,
        help="Repeat recovery variants per qualifying row.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_input = resolve_path(args.base_input)
    chartqa_input = resolve_path(args.chartqa_input)
    docvqa_input = resolve_path(args.docvqa_input)
    cord_input = resolve_path(args.cord_input)
    out_path = resolve_path(args.out_path)

    # Load all sources
    base_rows = read_jsonl(base_input)
    chartqa_rows = [
        row for row in read_jsonl(chartqa_input) if is_chartqa_python_exec_row(row)
    ]
    docvqa_rows = read_jsonl(docvqa_input)
    cord_rows = read_jsonl(cord_input)

    # --- Sampling: reduce external single-step data ---
    import random
    random.seed(42)

    sample_n = max(1, int(len(chartqa_rows) * args.external_sample_ratio))
    if len(chartqa_rows) > sample_n:
        chartqa_rows = random.sample(chartqa_rows, sample_n)
        print(f"ChartQA downsampled to {len(chartqa_rows)} rows")

    sample_n = max(1, int(len(cord_rows) * args.external_sample_ratio))
    if len(cord_rows) > sample_n:
        cord_rows = random.sample(cord_rows, sample_n)
        print(f"CORD downsampled to {len(cord_rows)} rows")

    sample_n = max(1, int(len(docvqa_rows) * args.external_sample_ratio))
    if len(docvqa_rows) > sample_n:
        docvqa_rows = random.sample(docvqa_rows, sample_n)
        print(f"DocVQA downsampled to {len(docvqa_rows)} rows")

    # Source -> (rows, repeat_factor)
    sources = [
        (base_input, base_rows, 1),
        (chartqa_input, chartqa_rows, args.python_exec_repeat),
        (docvqa_input, docvqa_rows, 1),
        (cord_input, cord_rows, args.retrieve_docs_repeat),
    ]

    merged = []
    skipped = []

    for source_path, rows, repeat in sources:
        path_str = str(source_path)
        if not rows:
            print(f"Input: {path_str} — SKIP (no data)")
            continue

        effective_rows = 0
        for repeat_index in range(repeat):
            for row in rows:
                ok, reason = validate_row(row, path_str)
                if not ok:
                    skipped.append(reason)
                    continue
                if repeat_index:
                    row = dict(row)
                    row["id"] = f"{row.get('id')}_rep{repeat_index}"
                merged.append(row)
                effective_rows += 1

        print(
            f"Input: {path_str} "
            f"({len(rows)} selected rows, repeat={repeat}, "
            f"effective={effective_rows})"
        )

    # --- Generate recovery variants ---
    recovery_rows = []
    for row in merged:
        second_tool = find_second_tool_name(row)
        if second_tool is None:
            continue

        for rep_idx in range(args.recovery_repeat):
            variant = build_recovery_variant(row, second_tool)
            if variant:
                variant["id"] = f"{variant.get('id')}_rec{rep_idx}"
                recovery_rows.append(variant)

    if recovery_rows:
        print(f"Generated {len(recovery_rows)} recovery variants")
        merged.extend(recovery_rows)

    write_jsonl(out_path, merged)

    # Stats
    from collections import Counter
    cats = Counter(r.get("category", "?") for r in merged)
    sources_cnt = Counter(r.get("source", "?") for r in merged)
    with_python = sum(1 for r in merged if uses_tool(r, "python_exec"))
    with_retrieve = sum(1 for r in merged if uses_tool(r, "retrieve_docs"))
    with_vision = sum(1 for r in merged if uses_tool(r, "vision_parse"))
    tool_counts = Counter(count_tool_calls(r) for r in merged)
    multi_tool = sum(1 for r in merged if count_tool_calls(r) >= 2)

    print(f"\nOutput: {out_path} ({len(merged)} rows)")
    print(f"Skipped: {len(skipped)}")
    for item in skipped[:10]:
        print("  ", item)
    print(f"Tool calls per row: {dict(sorted(tool_counts.items()))}")
    print(f"Multi-tool rows (>=2): {multi_tool}/{len(merged)} = {multi_tool/len(merged):.1%}")
    print(f"Categories: {dict(cats)}")
    print(f"Sources: {dict(sources_cnt)}")
    print(f"Rows with vision_parse: {with_vision}")
    print(f"Rows with retrieve_docs: {with_retrieve}")
    print(f"Rows with python_exec: {with_python}")


if __name__ == "__main__":
    main()
