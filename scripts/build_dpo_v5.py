"""
Build larger DPO dataset (~150 pairs) for stable training.

Uses V4 (0% miss-call) to generate correct chosen trajectories
for all categories and question types.
"""
import json, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)

# ---- Image inventory ----
INVOICES = [f"data/images/invoice_{i:03d}.jpg" for i in range(1, 18)]
CHARTS = [f"data/images/chart_{i:03d}.png" for i in range(1, 13)]
PRODUCTS = [f"data/images/product_{i:03d}.jpg" for i in range(1, 18)]
LOGOS = [f"data/images/logo_{i:03d}.jpg" for i in range(1, 8)]

BLURRY_LOGOS = ["data/images/logo_002.jpg", "data/images/logo_004.jpg", "data/images/logo_006.jpg"]
CLEAR_LOGOS = ["data/images/logo_001.jpg", "data/images/logo_003.jpg", "data/images/logo_005.jpg", "data/images/logo_007.jpg"]

def main():
    samples = []

    # ============================================================
    # Set A: Refusal pairs — chosen says "无法判断", rejected invents brands (40 pairs)
    # ============================================================
    brands = ["Vans", "Geely", "DELL", "Xiaomi", "Nike", "Adidas", "Puma", "Apple", "Samsung", "Honda"]
    refusal_imgs = BLURRY_LOGOS * 10 + random.sample(PRODUCTS, 5) + random.sample(CHARTS, 3)

    for i, img in enumerate(refusal_imgs):
        brand = brands[i % len(brands)]
        prompt = [
            {"role": "system", "content": "你是多模态回答专家。根据工具结果给出答案。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse",
             "content": json.dumps({"result": "图片模糊，OCR未能提取到清晰的品牌标识文字。"}, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请给出最终答案。"},
        ]
        chosen = [{"role": "assistant", "content": "图片模糊，vision_parse未能提取到清晰的品牌标识。无法可靠判断品牌名称。"}]
        rejected = [{"role": "assistant", "content": f"品牌名称为 {brand}。"}]
        samples.append({"id": f"dpo_refusal_{i+1:03d}", "image": img,
                        "question": "请判断图片中的品牌名称，并说明依据。",
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # ============================================================
    # Set B: Tool over-call — chosen skips unnecessary 2nd tool (40 pairs)
    # ============================================================
    for i, img in enumerate(INVOICES * 2):
        prompt = [
            {"role": "system", "content": "你是严格的工具调用 Agent。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "提取这张发票的发票号码和日期。"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse",
             "content": json.dumps({
                 "result": f"OCR结果：发票号码 INV-{1001+i}，日期 2026-{(i%12)+1:02d}-{10+(i%20):02d}，金额 {100.0+i*30:.2f}元，税号 TAX-{2000+i}。"
             }, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请继续。"},
        ]
        chosen = [{"role": "assistant",
                    "content": f"<final_answer>发票号码为 INV-{1001+i}，日期为 2026-{(i%12)+1:02d}-{10+(i%20):02d}。</final_answer>"}]
        rejected = [{"role": "assistant",
                      "content": '<tool_call>{"name":"retrieve_docs","args":{"query":"发票必填字段规则"}}</tool_call>'}]
        samples.append({"id": f"dpo_overcall_doc_{i+1:03d}", "image": img,
                        "question": "提取这张发票的发票号码和日期。",
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # Chart over-call
    for i, img in enumerate(CHARTS * 2 + CHARTS[:4]):
        v2023 = 70 + (i % 12) * 3
        v2024 = v2023 + 8 + (i % 12) * 2
        prompt = [
            {"role": "system", "content": "你是严格的工具调用 Agent。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "图表中2024年的数值是多少？"},
            ]},
            {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"chart_values"}}</tool_call>'},
            {"role": "tool", "name": "vision_parse",
             "content": json.dumps({"result": f"2023年：{v2023}，2024年：{v2024}"}, ensure_ascii=False)},
            {"role": "user", "content": "工具返回结果。请继续。"},
        ]
        chosen = [{"role": "assistant", "content": f"<final_answer>图表中2024年的数值为 {v2024}。</final_answer>"}]
        rejected = [{"role": "assistant", "content": '<tool_call>{"name":"python_exec","args":{"code":"..."}}</tool_call>'}]
        samples.append({"id": f"dpo_overcall_chart_{i+1:03d}", "image": img,
                        "question": "图表中2024年的数值是多少？",
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # ============================================================
    # Set C: Miss-call pairs — chosen calls vision_parse, rejected skips it (30 pairs)
    # ============================================================
    refusal_imgs_2 = BLURRY_LOGOS * 6 + random.sample(PRODUCTS, 6) + random.sample(CLEAR_LOGOS, 4)
    for i, img in enumerate(refusal_imgs_2):
        prompt = [
            {"role": "system", "content": "你是严格的工具调用 Agent。先调用 tool 再回答。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "请判断图片中的品牌名称，并说明依据。"},
            ]},
        ]
        chosen = [{"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'}]
        rejected = [{"role": "assistant", "content": "<final_answer>根据图片内容，品牌标识为 Vans。</final_answer>"}]
        samples.append({"id": f"dpo_misscall_{i+1:03d}", "image": img,
                        "question": "请判断图片中的品牌名称，并说明依据。",
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # Image consistency miss-call
    colors = ["black", "white", "red", "blue", "green"] * 4
    for i, img in enumerate(random.choices(PRODUCTS, k=15)):
        c = colors[i % len(colors)]
        prompt = [
            {"role": "system", "content": "你是严格的工具调用 Agent。"},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"商品描述写的是 {c} shoe。请判断图文是否一致。"},
            ]},
        ]
        chosen = [{"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr+caption"}}</tool_call>'}]
        rejected = [{"role": "assistant", "content": f"<final_answer>一致。图片中可见商品与描述中的 {c} shoe 一致。</final_answer>"}]
        samples.append({"id": f"dpo_misscall_img_{i+1:03d}", "image": img,
                        "question": f"商品描述写的是 {c} shoe。请判断图文是否一致。",
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # ============================================================
    # Set D: Answer style pairs (40 pairs)
    # ============================================================
    for i in range(30):
        if i < 20:
            img = INVOICES[i % 17]
            question = "检查这张发票是否缺少必填字段，并给出依据。"
            prompt = [
                {"role": "system", "content": "你是多模态回答专家。"},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"ocr"}}</tool_call>'},
                {"role": "tool", "name": "vision_parse",
                 "content": json.dumps({
                     "result": f"OCR结果：发票号码 INV-{1001+i}，日期 2026-{(i%12)+1:02d}-{10+(i%20):02d}，金额 {100.0+i*30:.2f}元，税号 TAX-{2000+i}。"
                 }, ensure_ascii=False)},
                {"role": "assistant", "content": '<tool_call>{"name":"retrieve_docs","args":{"query":"发票必填字段规则"}}</tool_call>'},
                {"role": "tool", "name": "retrieve_docs",
                 "content": json.dumps({"result": "发票必填字段：Invoice No, Date, Amount, Tax ID。缺少任一字段应判定为字段不完整。"}, ensure_ascii=False)},
                {"role": "user", "content": "工具返回结果。请给出最终答案。"},
            ]
            chosen = [{"role": "assistant",
                       "content": f"该发票包含全部四个必填字段：Invoice No（INV-{1001+i}）、Date（2026-{(i%12)+1:02d}-{10+(i%20):02d}）、Amount（{100.0+i*30:.2f}元）、Tax ID（TAX-{2000+i}）。字段完整，符合合规要求。"}]
            rejected = [{"role": "assistant", "content": "字段完整。"}]
        else:
            img = CHARTS[i % 12]
            v2023 = 70 + (i % 12) * 3
            v2024 = v2023 + 8 + (i % 12) * 2
            question = "根据图表，计算2024年相比2023年的同比增长率。"
            pct = (v2024 - v2023) / v2023 * 100
            prompt = [
                {"role": "system", "content": "你是多模态回答专家。"},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": '<tool_call>{"name":"vision_parse","args":{"mode":"chart_values"}}</tool_call>'},
                {"role": "tool", "name": "vision_parse",
                 "content": json.dumps({"result": f"2023年：{v2023}，2024年：{v2024}"}, ensure_ascii=False)},
                {"role": "assistant", "content": f'<tool_call>{{"name":"python_exec","args":{{"code":"({v2024}-{v2023})/{v2023}*100"}}}}</tool_call>'},
                {"role": "tool", "name": "python_exec",
                 "content": json.dumps({"result": str(pct)}, ensure_ascii=False)},
                {"role": "user", "content": "工具返回结果。请给出最终答案。"},
            ]
            chosen = [{"role": "assistant",
                       "content": f"根据图表数据，2023年为{v2023}，2024年为{v2024}，同比增长率为{pct:.2f}%。" +
                                 f"计算方式：({v2024}-{v2023})/{v2023}×100%≈{pct:.2f}%。"}]
            rejected = [{"role": "assistant", "content": f"增长率为{pct:.1f}%。"}]
        samples.append({"id": f"dpo_style_{i+1:03d}", "image": img,
                        "question": question,
                        "prompt": prompt, "chosen": chosen, "rejected": rejected})

    # ============================================================
    # Shuffle and save
    # ============================================================
    random.shuffle(samples)

    out = ROOT / "data/preference/dpo_v5.jsonl"
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    cats = {}
    for s in samples:
        typ = s["id"].split("_")[1]
        cats[typ] = cats.get(typ, 0) + 1

    print(f"Built {len(samples)} DPO pairs:")
    for typ, cnt in sorted(cats.items()):
        print(f"  {typ}: {cnt}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
