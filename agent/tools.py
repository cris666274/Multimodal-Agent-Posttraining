"""
Multimodal Agent 工具实现。

vision_parse:  真实 Qwen-VL 视觉解析（OCR / 图像描述 / 图表读数）
               当 VLM 模型不可用时自动回退到 mock 模式。
retrieve_docs: 真实规则文档检索，从 data/docs/ 加载规则文件。
python_exec:   安全 Python 数学表达式执行。
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 全局 VLM 模型引用（可由外部注入）
# ---------------------------------------------------------------------------
_vlm_model: Any = None  # QwenVLModel 实例，用于真实视觉解析


def set_vlm_model(model: Any) -> None:
    """注入 VLM 模型，使 vision_parse 使用真实推理。"""
    global _vlm_model
    _vlm_model = model


def has_vlm_model() -> bool:
    return _vlm_model is not None


# ---------------------------------------------------------------------------
# 规则文档加载
# ---------------------------------------------------------------------------

# 从 data/docs/ 加载所有规则文档
_RULES_CACHE: Dict[str, str] = {}


def _load_rules() -> Dict[str, str]:
    """加载 data/docs/ 下的所有 .txt 规则文档到内存缓存。"""
    global _RULES_CACHE
    if _RULES_CACHE:
        return _RULES_CACHE

    if not DOCS_DIR.exists():
        return {}

    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            if content:
                _RULES_CACHE[txt_file.name] = content
        except Exception:
            pass

    return _RULES_CACHE


# ---------------------------------------------------------------------------
# retrieve_docs：真实规则文档检索
# ---------------------------------------------------------------------------

# 查询关键词 → 规则文件映射
_QUERY_RULES_MAP = {
    "发票": "invoice_rules.txt",
    "invoice": "invoice_rules.txt",
    "字段": "invoice_rules.txt",
    "必填": "invoice_rules.txt",
    "税号": "invoice_rules.txt",
    "合规": "invoice_rules.txt",
    "商品": "product_rules.txt",
    "一致": "product_rules.txt",
    "图文": "product_rules.txt",
    "颜色": "product_rules.txt",
    "品类": "product_rules.txt",
    "品牌": "product_rules.txt",
    "图表": "chart_rules.txt",
    "chart": "chart_rules.txt",
    "增长": "chart_rules.txt",
    "同比": "chart_rules.txt",
    "计算": "chart_rules.txt",
    "百分比": "chart_rules.txt",
}


def retrieve_docs(query: str) -> dict:
    """
    从 data/docs/ 规则文档中检索相关内容。
    根据查询关键词匹配对应的规则文件，返回完整规则文本。
    """
    rules = _load_rules()
    query_lower = query.lower()

    # 匹配规则文件
    matched_files: set = set()
    for keyword, filename in _QUERY_RULES_MAP.items():
        if keyword.lower() in query_lower:
            matched_files.add(filename)

    if not matched_files:
        return {
            "tool": "retrieve_docs",
            "query": query,
            "result": "未检索到与查询相关的规则文档。",
        }

    # 合并匹配到的规则文档内容
    results: list[str] = []
    for filename in sorted(matched_files):
        if filename in rules:
            results.append(f"[{filename}] {rules[filename]}")

    return {
        "tool": "retrieve_docs",
        "query": query,
        "result": "\n".join(results) if results else "规则文档为空。",
    }


# ---------------------------------------------------------------------------
# vision_parse：真实 VLM 视觉解析 + mock 回退
# ---------------------------------------------------------------------------

_VISION_MODE_PROMPTS = {
    "ocr": "请对这张图片进行 OCR，提取所有可见的文字内容。直接输出文字，不要添加分析。",
    "caption": "请用一句话描述这张图片的主要内容。直接输出描述，不要添加分析。",
    "chart_values": "请读取这张图表中的数值数据。如果有多个年份/类别，请列出每个对应的数值。直接输出数据，不要添加分析。",
    "ocr+caption": "请先对这张图片进行 OCR 提取文字，再描述图片内容。按以下格式输出：OCR结果：...\n图片描述：...",
}


def _extract_image_index(image: str) -> int:
    name = Path(image).stem
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _mock_vision_parse(image: str, mode: str) -> str:
    """当 VLM 模型不可用时的 mock 回退。返回多样化的模拟结果。"""
    image_name = Path(image).name.lower()
    idx = _extract_image_index(image)

    if "invoice" in image_name:
        base_amount = 100.0 + idx * 47.3
        has_tax_id = (idx % 3 != 0)
        tax_part = (
            f"税号 TAX-2026-{1000 + idx}"
            if has_tax_id
            else "未检测到税号字段"
        )
        return (
            f"OCR结果：发票号码 INV-2026-{idx:03d}，"
            f"日期 2026-{(idx % 12) + 1:02d}-{10 + (idx % 20):02d}，"
            f"金额 {base_amount:.2f} 元，{tax_part}。"
        )

    if "chart" in image_name:
        base_2023 = 70 + idx * 3
        base_2024 = base_2023 + 8 + idx * 2
        return f"图表读数：2023={base_2023}，2024={base_2024}。"

    if "product" in image_name:
        colors = ["黑色", "白色", "红色", "蓝色", "绿色"]
        categories = ["运动鞋", "T恤", "背包", "手表", "帽子"]
        color = colors[idx % len(colors)]
        cat = categories[(idx // len(colors)) % len(categories)]
        return f"图片内容：{color}{cat}。"

    if "blur" in image_name or "logo" in image_name:
        return "图片较模糊，无法可靠识别品牌或 logo。"

    return "图片内容无法确定。"


def vision_parse(image: str, mode: str = "ocr") -> dict:
    """
    视觉解析工具。支持 OCR / 图像描述 / 图表读数。
    当 VLM 模型已注入时使用真实推理，否则使用 mock 回退。

    mode 可选值：
        - "ocr": OCR 文字提取
        - "caption": 图像内容描述
        - "chart_values": 图表数值读取
        - "ocr+caption": OCR + 描述
    """
    mode = mode or "ocr"

    # 真实 VLM 推理
    if _vlm_model is not None:
        try:
            prompt = _VISION_MODE_PROMPTS.get(
                mode, _VISION_MODE_PROMPTS["ocr"]
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            result_text = _vlm_model.generate(messages)
            return {
                "tool": "vision_parse",
                "mode": mode,
                "result": result_text.strip(),
            }
        except Exception as e:
            # 真实推理失败时回退到 mock
            return {
                "tool": "vision_parse",
                "mode": mode,
                "result": f"[真实 VLM 推理失败，使用 mock 结果] {_mock_vision_parse(image, mode)}",
            }

    # Mock 回退
    return {
        "tool": "vision_parse",
        "mode": mode,
        "result": _mock_vision_parse(image, mode),
    }


# ---------------------------------------------------------------------------
# python_exec：安全数学表达式执行
# ---------------------------------------------------------------------------

_PYTHON_EXEC_SAFE_RE = re.compile(r"[0-9\.\+\-\*/\(\)\s]+")


def python_exec(code: str) -> dict:
    try:
        if not _PYTHON_EXEC_SAFE_RE.fullmatch(code):
            return {
                "tool": "python_exec",
                "code": code,
                "result": "拒绝执行：代码包含不安全字符。",
            }
        result = eval(code, {"__builtins__": {}}, {})
        return {"tool": "python_exec", "code": code, "result": str(result)}
    except Exception as e:
        return {"tool": "python_exec", "code": code, "result": f"执行失败：{e}"}


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------

TOOLS = {
    "vision_parse": vision_parse,
    "retrieve_docs": retrieve_docs,
    "python_exec": python_exec,
}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main() -> int:
    # 加载规则文档
    rules = _load_rules()
    print(f"Loaded {len(rules)} rule documents:")
    for name, content in rules.items():
        print(f"  {name}: {content[:80]}...")

    print()

    # 测试 retrieve_docs
    smoke_queries = [
        "发票缺少税号应该怎么判断？",
        "商品图片和描述是否一致？",
        "计算图表中2024年的同比增长率",
        "未知主题的查询",
    ]
    for q in smoke_queries:
        result = retrieve_docs(q)
        print(f"Q: {q}")
        print(f"   Result: {result['result'][:120]}")
        print()

    # 测试 vision_parse (mock mode)
    print("vision_parse (mock mode):")
    smoke_images = [
        ("data/images/invoice_001.jpg", "ocr"),
        ("data/images/chart_001.png", "chart_values"),
        ("data/images/product_001.jpg", "caption"),
        ("data/images/logo_001.jpg", "ocr+caption"),
    ]
    for img, mode in smoke_images:
        result = vision_parse(str(ROOT / img), mode)
        print(f"  {img} ({mode}): {result['result'][:120]}")

    # 测试 python_exec
    print()
    smoke_codes = ["128.50 * 2", "__import__('os').system('dir')", "1/0"]
    for code in smoke_codes:
        result = python_exec(code)
        print(f"  python_exec({code}): {result['result']}")

    print(f"\nTools loaded: {', '.join(sorted(TOOLS))}")
    print(f"VLM model available: {has_vlm_model()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
