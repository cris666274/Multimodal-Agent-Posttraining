"""
Multimodal Agent 工具实现。

vision_parse:  真实 Qwen-VL 视觉解析（OCR / 图像描述 / 图表读数）
               当 VLM 模型不可用时自动回退到 mock 模式。
retrieve_docs: 真实规则文档检索，从 data/docs/ 加载规则文件。
python_exec:   安全 Python 数学表达式执行。
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# ---- BM25 retrieval over rule documents ----

def _tokenize(text: str) -> List[str]:
    """Tokenizer: CJK unigrams + whole ASCII words. Avoids English bigram noise."""
    tokens = []
    # Split on whitespace to separate English words from CJK text
    for chunk in text.lower().split():
        # ASCII words: keep as-is
        if chunk.isascii():
            tokens.append(chunk)
        else:
            # CJK text: character unigrams
            tokens.extend(chunk)
    return tokens


class _BM25:
    """Minimal BM25 scorer for retrieval over small doc sets."""
    def __init__(self, docs: Dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.doc_ids = list(docs.keys())
        self.doc_texts = [docs[k] for k in self.doc_ids]
        self.k1 = k1
        self.b = b

        # Tokenize all docs
        self.doc_tokens = [_tokenize(t) for t in self.doc_texts]
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1)

        # DF (document frequency) for each token
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        self.N = len(self.doc_ids)

    def score(self, query: str) -> List[tuple]:
        """Return [(doc_id, score), ...] sorted by descending score."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [(did, 0.0) for did in self.doc_ids]

        scores = []
        for i, tokens in enumerate(self.doc_tokens):
            doc_len = len(tokens)
            score = 0.0
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in self.df:
                    continue
                # IDF
                idf = max(0, math.log((self.N - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1.0))
                # TF with saturation
                tf_q = tf.get(qt, 0)
                numerator = tf_q * (self.k1 + 1)
                denominator = tf_q + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / max(denominator, 1e-8)

            scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


_bm25_index: _BM25 = None


def _get_bm25() -> _BM25:
    global _bm25_index
    if _bm25_index is None:
        rules = _load_rules()
        if rules:
            _bm25_index = _BM25(rules)
    return _bm25_index


def retrieve_docs(query: str) -> dict:
    """
    从 data/docs/ 规则文档中检索相关内容。
    混合策略：关键词快速匹配 + BM25 排序。支持中英文混合查询。
    """
    rules = _load_rules()
    if not rules:
        return {
            "tool": "retrieve_docs",
            "query": query,
            "result": "未找到规则文档（data/docs/ 为空）。",
        }

    bm25 = _get_bm25()
    if bm25 is None:
        return {"tool": "retrieve_docs", "query": query, "result": "BM25 索引构建失败。"}

    scored = bm25.score(query)
    max_score = max(s for _, s in scored) if scored else 0

    # Include docs with score >= 30% of max; skip if max is zero (no match)
    results = []
    for doc_id, score in scored:
        if max_score > 0 and score >= max_score * 0.3:
            results.append(f"[{doc_id}] {rules[doc_id]}")

    if not results:
        return {
            "tool": "retrieve_docs",
            "query": query,
            "result": "未检索到与查询相关的规则文档。",
        }

    return {
        "tool": "retrieve_docs",
        "query": query,
        "result": "\n".join(results),
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


def vision_parse(image: str, mode: str = "ocr") -> dict:
    """
    视觉解析工具。支持 OCR / 图像描述 / 图表读数。
    必须注入真实 VLM 模型，不允许静默 mock 回退。

    mode 可选值：
        - "ocr": OCR 文字提取
        - "caption": 图像内容描述
        - "chart_values": 图表数值读取
        - "ocr+caption": OCR + 描述
    """
    mode = mode or "ocr"

    if _vlm_model is None:
        raise RuntimeError(
            "vision_parse 需要注入真实 VLM 模型，但当前 _vlm_model 为 None。"
            "请调用 set_vlm_model() 注入 QwenVLModel 实例。"
            "禁止在训练/数据构建中使用 mock 回退。"
        )

    prompt = _VISION_MODE_PROMPTS.get(mode, _VISION_MODE_PROMPTS["ocr"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # 使用 generate_vision（禁用 LoRA）避免 agent 格式污染
    if hasattr(_vlm_model, 'generate_vision'):
        result_text = _vlm_model.generate_vision(messages)
    else:
        result_text = _vlm_model.generate(messages)
    return {
        "tool": "vision_parse",
        "mode": mode,
        "result": result_text.strip(),
    }


# ---------------------------------------------------------------------------
# python_exec：安全数学表达式执行
# ---------------------------------------------------------------------------

def _safe_eval(code: str) -> float:
    """
    AST-whitelist safe expression evaluator.
    Only allows: numbers, +, -, *, /, (), abs, round, min, max, pow.
    """
    import ast
    import operator
    import math

    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
    }
    SAFE_FUNCS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt,
    }

    ALLOWED_NODES.add(ast.Call)

    tree = ast.parse(code.strip(), mode="eval")

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            if isinstance(node, ast.Name):
                if node.id not in SAFE_FUNCS:
                    raise ValueError(f"禁止使用变量或函数：{node.id}")
            elif isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS):
                    raise ValueError(f"禁止调用：{ast.dump(node.func)}")
            else:
                raise ValueError(f"禁止的语法节点：{type(node).__name__}")

    # Compile and eval in restricted namespace
    compiled = compile(tree, "<safe_eval>", "eval")
    result = eval(compiled, {"__builtins__": {}}, SAFE_FUNCS.copy())
    return result


def python_exec(code: str) -> dict:
    """安全数学表达式执行（AST 白名单）。"""
    try:
        result = _safe_eval(code)
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
    print("retrieve_docs smoke test:")
    smoke_queries = [
        "发票缺少税号应该怎么判断？",
        "商品图片和描述是否一致？",
        "计算图表中2024年的同比增长率",
        "未知主题的查询",
    ]
    for q in smoke_queries:
        result = retrieve_docs(q)
        print(f"  Q: {q}")
        print(f"    Result: {result['result'][:120]}")
        print()

    # 测试 python_exec
    print("python_exec smoke test:")
    smoke_codes = ["128.50 * 2", "__import__('os').system('dir')", "1/0"]
    for code in smoke_codes:
        result = python_exec(code)
        print(f"  {code}: {result['result']}")

    print(f"\nTools: {', '.join(sorted(TOOLS))}")
    print(f"vision_parse requires set_vlm_model() before use (no mock fallback).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
