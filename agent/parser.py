import json
import re
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FINAL_PATTERN = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return ""


def _parse_tool_json(raw: str, original_text: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "type": "parse_error",
            "raw": original_text,
            "error": f"tool_call JSON 解析失败：{e}",
        }

    return {
        "type": "tool_call",
        "name": obj.get("name"),
        "args": obj.get("args", {}),
        "raw": original_text,
    }


def parse_model_output(text: str) -> dict:
    """
    解析模型输出：
    1. 工具调用
    2. 最终答案
    3. 严格拒绝无标签普通文本
    """
    if not isinstance(text, str):
        return {
            "type": "parse_error",
            "raw": text,
            "error": "模型输出不是字符串。",
        }

    tool_match = TOOL_PATTERN.search(text)
    if tool_match:
        raw = tool_match.group(1).strip()
        return _parse_tool_json(raw, text)

    # 容错：模型有时会输出 <tool_call>{...} 但漏掉闭合标签。
    if "<tool_call>" in text:
        raw = _extract_json_object(text.split("<tool_call>", 1)[1])
        if raw:
            parsed = _parse_tool_json(raw, text)
            if parsed["type"] == "tool_call":
                parsed["recovered"] = True
            return parsed

    final_match = FINAL_PATTERN.search(text)
    if final_match:
        return {
            "type": "final_answer",
            "content": final_match.group(1).strip(),
            "raw": text,
        }

    # 容错：模型有时会输出 <final_answer>... 但漏掉闭合标签。
    if "<final_answer>" in text:
        content = text.split("<final_answer>", 1)[1].strip()
        if content:
            return {
                "type": "final_answer",
                "content": content,
                "raw": text,
                "recovered": True,
            }

    return {
        "type": "parse_error",
        "raw": text,
        "error": "模型输出缺少 <tool_call> 或 <final_answer> 标签。",
    }


def main() -> int:
    smoke_cases = [
        '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>',
        "<final_answer>根据 OCR 结果，该发票字段完整。</final_answer>",
        "这是一段没有标签的普通最终回答。",
        '<tool_call>{"name":"vision_parse","args":</tool_call>',
        '<tool_call>{"name":"python_exec","args":{"code":"(90-80)/80*100"}}',
        "<final_answer>缺少闭合标签但内容非空",
    ]

    for index, text in enumerate(smoke_cases, 1):
        print(f"\nCase {index}:")
        print(json.dumps(parse_model_output(text), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
