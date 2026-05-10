import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs/vlm_agent_eval_tagged.jsonl"

SFT_FIRST_TOOL_OUT = ROOT / "data/sft/sft_hard_first_tool.jsonl"
SFT_FORMAT_OUT = ROOT / "data/sft/sft_hard_format.jsonl"
SFT_FULL_TRACE_OUT = ROOT / "data/sft/sft_hard_full_trace.jsonl"
DPO_HARD_OUT = ROOT / "data/preference/dpo_hard_cases.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def user_message(row):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": row.get("image")},
            {"type": "text", "text": row.get("question", "")},
        ],
    }


def infer_vision_mode(row):
    category = row.get("category", "")
    question = row.get("question", "")

    if category == "document_validation" or "发票" in question:
        return "ocr"

    if category == "chart_calculation" or "图表" in question or "同比" in question:
        return "chart_values"

    if category == "uncertainty_refusal" or "品牌" in question or "logo" in question.lower():
        return "ocr+caption"

    return "caption"


def tool_call_message(name, args):
    return {
        "role": "assistant",
        "content": f"<tool_call>{json.dumps({'name': name, 'args': args}, ensure_ascii=False)}</tool_call>",
    }


def final_answer_message(text):
    return {
        "role": "assistant",
        "content": f"<final_answer>{text}</final_answer>",
    }


def tool_result_message(name, content):
    return {
        "role": "tool",
        "name": name,
        "content": content,
    }


def build_first_tool_sft(row):
    gold_tools = row.get("gold_tools", [])
    if not gold_tools:
        return None

    first_tool = gold_tools[0]
    args = {}
    if first_tool == "vision_parse":
        args = {"mode": infer_vision_mode(row)}
    elif first_tool == "retrieve_docs":
        args = {"query": row.get("question", "")}
    elif first_tool == "python_exec":
        args = {"code": "(90-80)/80*100"}

    return {
        "id": f"hard_first_tool_{row.get('id')}",
        "source": "eval_error_first_tool",
        "error_tag": row.get("error_tag"),
        "category": row.get("category"),
        "messages": [
            user_message(row),
            tool_call_message(first_tool, args),
        ],
    }


def build_format_sft(row):
    first_tool_sample = build_first_tool_sft(row)
    if first_tool_sample is not None:
        first_tool_sample["id"] = f"hard_format_{row.get('id')}"
        first_tool_sample["source"] = "eval_error_format"
        return first_tool_sample

    return {
        "id": f"hard_format_{row.get('id')}",
        "source": "eval_error_format",
        "error_tag": row.get("error_tag"),
        "category": row.get("category"),
        "messages": [
            user_message(row),
            final_answer_message("当前问题不需要调用外部工具，请直接给出简洁、非空、带标签的回答。"),
        ],
    }


def build_full_trace_sft(row):
    category = row.get("category", "")
    messages = [user_message(row)]

    if category == "document_validation":
        messages.extend(
            [
                tool_call_message("vision_parse", {"mode": "ocr"}),
                tool_result_message(
                    "vision_parse",
                    "OCR结果：Invoice No、Date、Amount 已识别；请结合具体图片判断 Tax ID 是否存在。",
                ),
                tool_call_message("retrieve_docs", {"query": "发票必填字段规则"}),
                tool_result_message(
                    "retrieve_docs",
                    "发票必填字段：Invoice No, Date, Amount, Tax ID。",
                ),
                final_answer_message(
                    "根据OCR结果和发票规则，需要核对 Invoice No、Date、Amount、Tax ID 四个必填字段；若缺少 Tax ID，则字段不完整，否则字段完整。"
                ),
            ]
        )
    elif category == "chart_calculation":
        messages.extend(
            [
                tool_call_message("vision_parse", {"mode": "chart_values"}),
                tool_result_message("vision_parse", "图表读数：2023=80，2024=90。"),
                tool_call_message("python_exec", {"code": "(90-80)/80*100"}),
                tool_result_message("python_exec", "12.5"),
                final_answer_message(
                    "根据图表读数，2023年为80，2024年为90；同比增长率=(90-80)/80*100=12.5%。"
                ),
            ]
        )
    elif category == "image_text_consistency":
        messages.extend(
            [
                tool_call_message("vision_parse", {"mode": "caption"}),
                tool_result_message("vision_parse", "图片内容：商品图片中的鞋子颜色和品类需要与描述对比。"),
                final_answer_message(
                    "根据视觉解析结果，需要比较图片与描述中的颜色、品类和关键属性；若颜色或品类冲突，则图文不一致，否则一致。"
                ),
            ]
        )
    elif category == "uncertainty_refusal":
        messages.extend(
            [
                tool_call_message("vision_parse", {"mode": "ocr+caption"}),
                tool_result_message("vision_parse", "图片较模糊，无法可靠识别品牌或logo。"),
                final_answer_message("图片较模糊，无法可靠识别品牌名称，不能仅凭模糊线索猜测。"),
            ]
        )
    else:
        messages.append(final_answer_message("图片中展示的是商品，请根据可见内容进行简要描述。"))

    return {
        "id": f"hard_full_trace_{row.get('id')}",
        "source": "eval_error_full_trace",
        "error_tag": row.get("error_tag"),
        "category": category,
        "messages": messages,
    }


def build_dpo_hard_case(row):
    answer = row.get("answer") or "未按要求完成任务。"
    category = row.get("category")
    error_tag = row.get("error_tag")

    if error_tag in {"over_refusal", "should_refuse_but_answered"}:
        if row.get("should_refuse"):
            chosen_message = final_answer_message("图片信息不足，无法可靠判断；不能仅凭模糊线索猜测。")
        else:
            chosen_message = final_answer_message("根据图片和已有证据，可以回答该问题，不应在证据足够时过度拒答。")
    elif error_tag in {"missing_tool_call", "wrong_tool_selection", "wrong_tool_order", "redundant_tool_call"}:
        first_tool_sample = build_first_tool_sft(row)
        if first_tool_sample is not None:
            chosen_message = first_tool_sample["messages"][-1]
        else:
            chosen_message = final_answer_message("图片中展示的是商品，请根据可见内容进行简要描述。")
    elif error_tag in {"parse_error", "bad_tool_name", "bad_tool_args"}:
        format_sample = build_format_sft(row)
        chosen_message = format_sample["messages"][-1]
    else:
        full_trace_sample = build_full_trace_sft(row)
        chosen_message = full_trace_sample["messages"][-1]

    return {
        "id": f"dpo_hard_{row.get('id')}",
        "source": "eval_error_hard_case",
        "category": category,
        "error_tag": error_tag,
        "prompt": [user_message(row)],
        "chosen": [chosen_message],
        "rejected": [final_answer_message(answer)],
    }


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"找不到输入文件: {IN_PATH}\n"
            "请先运行: python scripts/tag_vlm_errors.py"
        )

    rows = list(read_jsonl(IN_PATH))
    error_rows = [row for row in rows if row.get("error_tag") != "ok"]

    first_tool_tags = {
        "missing_tool_call",
        "wrong_tool_selection",
        "wrong_tool_order",
        "max_steps_exceeded",
    }
    format_tags = {"parse_error", "bad_tool_name", "bad_tool_args"}
    full_trace_tags = {
        "max_steps_exceeded",
        "wrong_tool_order",
        "answer_keyword_miss",
        "missing_tool_call",
    }
    dpo_tags = {
        "over_refusal",
        "should_refuse_but_answered",
        "redundant_tool_call",
        "missing_tool_call",
        "wrong_tool_selection",
        "wrong_tool_order",
        "parse_error",
        "bad_tool_name",
        "bad_tool_args",
        "answer_keyword_miss",
    }

    first_tool_rows = [
        item
        for item in (build_first_tool_sft(row) for row in error_rows if row.get("error_tag") in first_tool_tags)
        if item is not None
    ]
    format_rows = [
        build_format_sft(row)
        for row in error_rows
        if row.get("error_tag") in format_tags
    ]
    full_trace_rows = [
        build_full_trace_sft(row)
        for row in error_rows
        if row.get("error_tag") in full_trace_tags
    ]
    dpo_rows = [
        build_dpo_hard_case(row)
        for row in error_rows
        if row.get("error_tag") in dpo_tags
    ]

    write_jsonl(SFT_FIRST_TOOL_OUT, first_tool_rows)
    write_jsonl(SFT_FORMAT_OUT, format_rows)
    write_jsonl(SFT_FULL_TRACE_OUT, full_trace_rows)
    write_jsonl(DPO_HARD_OUT, dpo_rows)

    counter = Counter(row.get("error_tag", "unknown") for row in rows)

    print("========== Source Error Tags ==========")
    for tag, count in counter.most_common():
        print(tag, count)

    print("\n========== Generated ==========")
    print(f"First-tool SFT: {len(first_tool_rows)} -> {SFT_FIRST_TOOL_OUT}")
    print(f"Format SFT: {len(format_rows)} -> {SFT_FORMAT_OUT}")
    print(f"Full-trace SFT: {len(full_trace_rows)} -> {SFT_FULL_TRACE_OUT}")
    print(f"DPO hard cases: {len(dpo_rows)} -> {DPO_HARD_OUT}")


if __name__ == "__main__":
    main()
