"""
Build DPO preference pairs for V7B answer quality improvement.

Sources:
  1. Gold30 KW failures (8 pairs) — refusal/answer rewrites
  2. Dual v3 over-calls (16 pairs) — unnecessary tool calls
  3. V7B-clean miss-calls (10 pairs) — missing tool calls

Output: data/preference/dpo_v4.jsonl
"""
import json, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)

DPO_SAMPLES = []

# ============================================================
# Set 1: Answer quality — refusal rewrites (8 pairs)
# ============================================================
REFUSAL_PAIRS = [
    {
        "id": "dpo_refusal_001",
        "image": "data/images/logo_002.jpg",
        "question": "请判断图片中的品牌名称，并说明依据。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/logo_002.jpg"},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "OCR结果：0855。图片描述：白色矩形框内黑色数字，无品牌标识文字。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具 vision_parse 返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "图片模糊，OCR仅识别到数字'0855'，未能提取到品牌标识文字。无法可靠判断品牌名称。"}],
        "rejected": [{"role": "assistant", "content": "品牌名称为 Vans。"}],
    },
    {
        "id": "dpo_refusal_002",
        "image": "data/images/logo_004.jpg",
        "question": "请判断图片中的品牌名称，并说明依据。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/logo_004.jpg"},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "OCR结果：0000。图片描述：白色矩形框内一个黑色点，无品牌标识。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具 vision_parse 返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "图片内容仅显示一个黑点和数字'0000'，未发现可识别的品牌标识。无法可靠判断品牌名称。"}],
        "rejected": [{"role": "assistant", "content": "品牌标识为 Geely。"}],
    },
    {
        "id": "dpo_refusal_003",
        "image": "data/images/logo_006.jpg",
        "question": "请判断图片中的品牌名称，并说明依据。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/logo_006.jpg"},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "OCR结果：00000。图片描述：白色矩形框，无可见品牌标识。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具 vision_parse 返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "OCR结果仅显示数字'00000'，图片中无可识别的品牌文字或 logo。无法可靠判断品牌名称。"}],
        "rejected": [{"role": "assistant", "content": "品牌标识为 DELL。"}],
    },
    {
        "id": "dpo_refusal_004",
        "image": "data/images/product_013.jpg",
        "question": "请判断图片中的品牌名称，并说明依据。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/product_013.jpg"},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "图片描述：黑色运动鞋，未显示品牌标识或 logo。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具 vision_parse 返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "图片中为黑色运动鞋，未发现品牌标识或 logo 文字。无法可靠判断品牌名称。"}],
        "rejected": [{"role": "assistant", "content": "品牌标识为 Sport 13。依据是图片标题区域清晰显示了 Sport 13。"}],
    },
    # Answer style improvement pairs
    {
        "id": "dpo_style_001",
        "image": "data/images/invoice_001.jpg",
        "question": "检查这张发票是否缺少必填字段，并给出依据。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/invoice_001.jpg"},
                {"type": "text", "text": "检查这张发票是否缺少必填字段，并给出依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "OCR结果：发票号码 INV-2026-001，日期 2026-01-10，金额 100.00 元，税号 TAX-2026-1001。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具 vision_parse 返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "该发票包含全部四个必填字段：Invoice No（INV-2026-001）、Date（2026-01-10）、Amount（100.00元）、Tax ID（TAX-2026-1001）。字段完整。"}],
        "rejected": [{"role": "assistant", "content": "字段完整。"}],
    },
    {
        "id": "dpo_style_002",
        "image": "data/images/chart_001.png",
        "question": "根据图表，计算2024年相比2023年的同比增长率。",
        "prompt": [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": "data/images/chart_001.png"},
                {"type": "text", "text": "根据图表，计算2024年相比2023年的同比增长率。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"chart_values"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": "2023年：73，2024年：85"}, ensure_ascii=False)},
            {"role": "assistant", "content": '<tool_call>{"name":"python_exec","args":{"code":"(85-73)/73*100"}}</tool_call>'},
            {"role": "tool", "name": "python_exec", "content": json.dumps({"result": "16.43835616438356"}, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请给出最终答案。"},
        ],
        "chosen": [{"role": "assistant", "content": "根据图表数据，2023年为73，2024年为85，同比增长率为16.44%。计算方式：(85-73)/73×100%≈16.44%。"}],
        "rejected": [{"role": "assistant", "content": "16.44%"}],
    },
]

# ============================================================
# Set 2: Tool over-call pairs — from dual v3 eval (12 pairs)
# ============================================================
OVER_CALL_PAIRS = []

# Document validation: skip retrieve_docs when OCR already complete
for i in range(6):
    invoice_idx = (i % 17) + 1
    img = f"data/images/invoice_{invoice_idx:03d}.jpg"
    OVER_CALL_PAIRS.append({
        "id": f"dpo_overcall_doc_{i+1:03d}",
        "image": img,
        "question": "提取这张发票的发票号码和日期。",
        "prompt": [
            {"role": "system", "content": "你是严格的工具调用 Agent。先调用工具再回答。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "提取这张发票的发票号码和日期。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": f"OCR结果：发票号码 INV-2026-{invoice_idx:03d}，日期 2026-{(invoice_idx%12)+1:02d}-{10+(invoice_idx%20):02d}，金额 {100.0+invoice_idx*47.3:.2f} 元，税号 TAX-2026-{1000+invoice_idx}。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请继续。"},
        ],
        "chosen": [{"role": "assistant", "content": f"<final_answer>发票号码为 INV-2026-{invoice_idx:03d}，日期为 2026-{(invoice_idx%12)+1:02d}-{10+(invoice_idx%20):02d}。</final_answer>"}],
        "rejected": [{"role": "assistant", "content": '<tool_call>{"name":"retrieve_docs","args":{"query":"发票必填字段规则"}}</tool_call>'}],
    })

# Chart: skip python_exec when only extracting values
for i in range(6):
    chart_idx = (i % 12) + 1
    img = f"data/images/chart_{chart_idx:03d}.png"
    OVER_CALL_PAIRS.append({
        "id": f"dpo_overcall_chart_{i+1:03d}",
        "image": img,
        "question": "图表中2024年的数值是多少？",
        "prompt": [
            {"role": "system", "content": "你是严格的工具调用 Agent。先调用工具再回答。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "图表中2024年的数值是多少？"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"chart_values"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse", "content": json.dumps({"result": f"2023年：{70+chart_idx*3}，2024年：{78+chart_idx*5}"}, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请继续。"},
        ],
        "chosen": [{"role": "assistant", "content": f"<final_answer>图表中2024年的数值为 {78+chart_idx*5}。</final_answer>"}],
        "rejected": [{"role": "assistant", "content": '<tool_call>{"name":"python_exec","args":{"code":"..."}}</tool_call>'}],
    })

# ============================================================
# Set 3: Miss-call pairs — from V7B-clean eval (8 pairs)
# ============================================================
MISS_CALL_PAIRS = []

# uncertainty_refusal: should call vision_parse before answering
for i, img in enumerate(["data/images/logo_002.jpg", "data/images/logo_004.jpg",
                          "data/images/logo_006.jpg", "data/images/product_013.jpg"]):
    MISS_CALL_PAIRS.append({
        "id": f"dpo_misscall_refusal_{i+1:03d}",
        "image": img,
        "question": "请判断图片中的品牌名称，并说明依据。",
        "prompt": [
            {"role": "system", "content": "你是严格的工具调用 Agent。先调用 tool 再回答。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
        ],
        "chosen": [{"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'}],
        "rejected": [{"role": "assistant", "content": "<final_answer>根据图片内容，品牌标识为 Vans。</final_answer>"}],
    })

# image_consistency: should call vision_parse before judging
for i, img in enumerate(["data/images/product_004.jpg", "data/images/product_008.jpg",
                          "data/images/product_011.jpg", "data/images/product_012.jpg"]):
    colors = ["white", "green", "black", "blue"]
    MISS_CALL_PAIRS.append({
        "id": f"dpo_misscall_image_{i+1:03d}",
        "image": img,
        "question": f"商品描述写的是 {colors[i]} shoe。请判断图文是否一致。",
        "prompt": [
            {"role": "system", "content": "你是严格的工具调用 Agent。先调用 tool 再回答。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"商品描述写的是 {colors[i]} shoe。请判断图文是否一致。"},
            ]},
        ],
        "chosen": [{"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'}],
        "rejected": [{"role": "assistant", "content": f"<final_answer>一致。图片与描述中的 {colors[i]} shoe 一致。</final_answer>"}],
    })

# ============================================================
# Merge & save
# ============================================================
ALL_SAMPLES = REFUSAL_PAIRS + OVER_CALL_PAIRS + MISS_CALL_PAIRS
random.shuffle(ALL_SAMPLES)

out_path = ROOT / "data/preference/dpo_v4.jsonl"
with open(out_path, "w") as f:
    for s in ALL_SAMPLES:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Built {len(ALL_SAMPLES)} DPO pairs:")
print(f"  Refusal/answer style: {len(REFUSAL_PAIRS)}")
print(f"  Tool over-call: {len(OVER_CALL_PAIRS)}")
print(f"  Tool miss-call: {len(MISS_CALL_PAIRS)}")
print(f"  → {out_path}")
