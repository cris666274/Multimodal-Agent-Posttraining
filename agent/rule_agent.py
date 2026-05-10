import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


try:
    from agent.tools import TOOLS
except ModuleNotFoundError:
    # Allow direct execution with: python agent/rule_agent.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.tools import TOOLS


def _call_tool(trace: list, name: str, **kwargs) -> dict:
    observation = TOOLS[name](**kwargs)
    trace.append(
        {
            "type": "tool_call",
            "name": name,
            "args": kwargs,
            "observation": observation,
        }
    )
    return observation


def run_rule_agent(image: str, question: str, max_steps: int = 4) -> dict:
    """
    规则版 agent。
    根据问题类型选择 mock 工具，并返回 answer + trace。
    """
    trace = []
    q = question.lower()

    if max_steps <= 0:
        return {
            "answer": "已达到最大推理步数，无法继续调用工具。",
            "trace": trace,
        }

    # 1. 发票 / 文档校验
    if "发票" in question or "字段" in question or "invoice" in q:
        _call_tool(trace, "vision_parse", image=image, mode="ocr")
        _call_tool(trace, "retrieve_docs", query="发票必填字段规则")

        return {
            "answer": "根据 OCR 结果和规则文档，发票必填字段包括发票号码、日期、金额、税号。若 OCR 未检测到税号字段，则该发票字段不完整。",
            "trace": trace,
        }

    # 2. 图表计算
    if "增长" in question or "同比" in question or "图表" in question or "chart" in q:
        _call_tool(trace, "vision_parse", image=image, mode="chart_values")
        calc = _call_tool(trace, "python_exec", code="(90-80)/80*100")

        return {
            "answer": f"根据图表读数，2023 年为 80，2024 年为 90，因此同比增长率为 {calc['result']}%。",
            "trace": trace,
        }

    # 3. 图文一致性
    if (
        "一致" in question
        or "商品" in question
        or "图文" in question
        or "product" in q
    ):
        _call_tool(trace, "vision_parse", image=image, mode="caption")
        _call_tool(trace, "retrieve_docs", query="商品图文一致性规则")

        return {
            "answer": "根据视觉结果和商品图文一致性规则，需要比较图片与描述中的品类、颜色和关键属性；若这些信息冲突，则判定为不一致。",
            "trace": trace,
        }

    # 4. 模糊 logo / 拒答
    if "品牌" in question or "logo" in q:
        _call_tool(trace, "vision_parse", image=image, mode="ocr+caption")

        return {
            "answer": "如果图像较模糊，无法可靠识别品牌，应说明无法可靠判断，并建议提供更清晰的 logo 区域图片。",
            "trace": trace,
        }

    # 5. 默认直接回答
    return {
        "answer": "当前信息不足，无法可靠判断。",
        "trace": trace,
    }


def main() -> int:
    smoke_cases = [
        ("data/images/invoice_001.jpg", "检查这张发票是否缺少必填字段，并给出依据。"),
        ("data/images/chart_001.png", "根据图表，计算2024年相比2023年的同比增长率。"),
        ("data/images/product_001.jpg", "商品描述写的是 black shoe。请判断图文是否一致。"),
        ("data/images/logo_002.jpg", "请判断图片中的品牌名称，并说明依据。"),
        ("data/images/product_001.jpg", "请简要描述图片。"),
    ]

    for index, (image, question) in enumerate(smoke_cases, 1):
        print(f"\nCase {index}: {question}")
        result = run_rule_agent(image=image, question=question)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
