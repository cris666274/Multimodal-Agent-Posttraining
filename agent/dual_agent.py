"""
Dual-model agent: V4 (planner, tool decisions) + V7B (answer writer).

v2: + answerer prompt for V7B, + direct_vqa router to skip V4 for simple queries.

Usage:
  from agent.dual_agent import DualModelAgent
  agent = DualModelAgent(planner_model=v4, answer_model=v7b, max_steps=6)
  result = agent.run(image="...", question="...")
"""
import json
import re
from typing import Any, Dict, List

from agent.parser import parse_model_output
from agent.tools import TOOLS, set_vlm_model
from agent.prompts import SYSTEM_PROMPT
from agent.vlm import QwenVLModel
from agent.runtime import MultimodalAgent


# ---- Answerer prompt: V7B receives this when writing final answers ----
ANSWERER_PROMPT = """
你是一个专业的多模态回答专家。你的任务是根据以下工具执行结果，为用户提供准确、清晰、有依据的最终答案。

回答规则：
1. 直接回答用户的问题，不需要输出工具调用。
2. 引用工具返回的具体证据（OCR结果、规则文档、计算数值）。
3. 如果工具结果显示图片模糊或信息不足，必须明确说明"无法可靠判断"，不得编造。
4. 回答应简洁但完整，包含关键数值和判断依据。
5. 只输出回答内容，不要加标签、不要输出 <tool_call> 或 <final_answer>。

示例回答风格：
- "该发票包含 Invoice No、Date、Amount、Tax ID 四个必填字段，字段完整。"
- "图片模糊，OCR结果仅显示数字'0855'，无法可靠识别品牌名称。"
- "根据图表数据，2023年为80，2024年为100，同比增长率为25.0%。计算：(100-80)/80*100=25.0%。"
- "不一致。图片中商品为 black shoe，但描述写的是 red shoe，颜色不匹配。"
""".strip()


# ---- Direct VQA patterns: skip planner, go straight to answerer ----
_DIRECT_VQA_PATTERNS = [
    r'请简要描述',
    r'请描述图片',
    r'图片中有几个',
    r'主色调',
    r'是什么类型',
    r'整体质量',
    r'一句话总结',
    r'不需要调用.*工具',
    r'直接回答',
    r'图片质量',
    r'什么类型',
    r'包含.*物体',
]

_DIRECT_VQA_FORBIDDEN_PATTERNS = [
    r'发票', r'字段', r'合规', r'必填', r'缺少', r'是否完整',
    r'图表', r'增长率', r'同比', r'计算', r'百分比', r'数值',
    r'品牌', r'logo', r'识别', r'判断.*品牌',
    r'一致', r'匹配', r'对比', r'冲突',
    r'提取', r'列出', r'检查',
    r'视觉元素', r'所有.*元素', r'不限于文字',  # these need vision_parse
]


def _is_direct_vqa(question: str) -> bool:
    """Check if question is simple enough to skip the planner."""
    # Must match a direct_vqa pattern
    if not any(re.search(p, question) for p in _DIRECT_VQA_PATTERNS):
        return False
    # Must NOT match any tool-requiring pattern
    if any(re.search(p, question) for p in _DIRECT_VQA_FORBIDDEN_PATTERNS):
        return False
    return True


class DualModelAgent(MultimodalAgent):
    """
    V4 selects tools, V7B writes answers.

    v2 improvements:
    - Dedicated answerer prompt for V7B (clearer, more accurate answers)
    - Direct VQA router: skip V4 for simple questions that don't need tools
    """

    def __init__(
        self,
        planner_model: QwenVLModel,
        answer_model: QwenVLModel,
        max_steps: int = 6,
        enforce_required_tools: bool = True,
    ):
        self.planner_model = planner_model
        self.answer_model = answer_model

        super().__init__(
            model=planner_model,
            max_steps=max_steps,
            enforce_required_tools=enforce_required_tools,
        )
        set_vlm_model(planner_model)

    def _generate_answer(self, messages: List[Dict], v4_answer: str) -> tuple:
        """
        Have V7B write the final answer.
        Replaces the system prompt with the answerer prompt,
        appends context, and generates.
        """
        # Build answerer context: replace system prompt, keep tool results
        answer_messages = [{"role": "system", "content": ANSWERER_PROMPT}]

        # Copy user image+question, tool calls, and tool results
        for msg in messages:
            if msg["role"] == "system":
                continue  # replaced by answerer prompt
            answer_messages.append(msg)

        # Append a final instruction
        answer_messages.append({
            "role": "user",
            "content": "请根据以上工具结果和对话历史，给出最终答案。直接回答，不要输出工具调用或标签。"
        })

        output = self.answer_model.generate(answer_messages)

        # Strip any accidental tags
        for tag in ["<final_answer>", "</final_answer>", "<tool_call>", "</tool_call>"]:
            output = output.replace(tag, "")
        output = output.strip()

        if not output:
            output = v4_answer  # fallback to V4

        return output

    def _should_skip_second_tool(self, question: str, called_tools: List[str],
                                   vision_result: str) -> List[str]:
        """
        Check if the second required tool is really necessary.
        Returns list of tools that CAN be skipped.
        """
        skippable = []
        required = self._required_tool_after_vision(question, called_tools)

        if required == "retrieve_docs" and vision_result:
            # Doc validation: if OCR already shows enough fields, skip retrieve_docs
            fields_en = ["Invoice No", "Date", "Amount", "Tax ID"]
            fields_cn = ["发票号码", "日期", "金额", "税号"]
            found_en = sum(1 for f in fields_en if f in vision_result)
            found_cn = sum(1 for f in fields_cn if f in vision_result)
            # Also check common patterns like "INV-", "TAX-"
            has_inv = "INV-" in vision_result or "TAX-" in vision_result
            if (found_en >= 3 or found_cn >= 3 or has_inv) and \
               ("缺少" not in question and "缺失" not in question and
                "差异" not in question and "对比" not in question and
                "合规" not in question and "符合" not in question):
                skippable.append("retrieve_docs")

        if required == "python_exec" and "vision_parse" in called_tools:
            # Chart: if just extracting values (no computation keywords), skip python_exec
            calc_keywords = ["增长率", "同比", "计算", "预测", "百分比",
                           "差值", "增长", "总和", "趋势", "高出", "多少",
                           "哪一年", "哪个", "比较"]
            if not any(kw in question for kw in calc_keywords):
                skippable.append("python_exec")

        return skippable

    def run(self, image: str, question: str) -> Dict[str, Any]:
        messages = self.build_initial_messages(image=image, question=question)
        trace = []
        used_tool_calls = set()
        blocked_tools: set = set()
        last_vision_result = ""

        # ---- Direct VQA router ----
        if _is_direct_vqa(question):
            answer = self._generate_answer(messages, "")
            trace.append({
                "step": 0,
                "event": "direct_vqa_routed",
                "routed_to": "V7B",
            })
            return {
                "answer": answer,
                "trace": trace,
                "messages": messages,
                "status": "finished",
                "planner": "none (direct_vqa)",
                "answer_writer": "V7B",
            }

        # ---- Main planning loop ----
        for step in range(self.max_steps):
            model_output = self.planner_model.generate(messages)
            parsed = parse_model_output(model_output)

            trace.append({
                "step": step,
                "model_output": model_output,
                "parsed": parsed,
            })

            # --- Final answer: delegate to V7B ---
            if parsed["type"] == "final_answer":
                final_content = parsed.get("content", "").strip()

                if not final_content:
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": (
                        "你刚才输出了空的 <final_answer></final_answer>。这是无效回答。"
                        "请根据已有信息继续完成任务。只能输出一个 <tool_call> 或非空 <final_answer>。"
                    )})
                    continue

                called_tool_names = self._called_tool_names(trace)
                required_tool = self._required_tool_after_vision(
                    question=question, called_tool_names=called_tool_names,
                )
                # Allow skipping if vision_parse result is sufficient
                skippable = self._should_skip_second_tool(
                    question, called_tool_names, last_vision_result)
                if required_tool in skippable:
                    required_tool = ""
                if self.enforce_required_tools and required_tool:
                    trace.append({
                        "step": step,
                        "event": "final_answer_blocked_missing_required_tool",
                        "required_tool": required_tool,
                    })
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": self._missing_required_tool_prompt(required_tool)})
                    continue

                # V7B writes the answer
                answer = self._generate_answer(messages, final_content)

                trace.append({
                    "step": step,
                    "event": "answer_delegated_to_v7b",
                    "planner_answer": final_content,
                })

                return {
                    "answer": answer,
                    "trace": trace,
                    "messages": messages,
                    "status": "finished",
                    "planner": "V4",
                    "answer_writer": "V7B",
                }

            # --- Parse error ---
            if parsed["type"] == "parse_error":
                return {
                    "answer": f"模型输出格式解析失败：{parsed.get('error', 'unknown error')}",
                    "trace": trace,
                    "messages": messages,
                    "status": "parse_error",
                }

            # --- Tool call ---
            if parsed["type"] == "tool_call":
                tool_name = parsed.get("name")
                tool_args = parsed.get("args", {})

                if not isinstance(tool_name, str) or not tool_name:
                    return {
                        "answer": "工具名称格式错误。",
                        "trace": trace, "messages": messages,
                        "status": "bad_tool_name",
                    }
                if not isinstance(tool_args, dict):
                    return {
                        "answer": "工具参数格式错误。",
                        "trace": trace, "messages": messages,
                        "status": "bad_tool_args",
                    }

                called_tool_names = self._called_tool_names(trace)

                if (
                    self._needs_vision_first(question=question, image=image)
                    and "vision_parse" not in called_tool_names
                    and tool_name != "vision_parse"
                ):
                    blocked_tools.add(tool_name)
                    trace.append({
                        "step": step,
                        "event": "tool_order_blocked",
                        "blocked_tool_name": tool_name,
                    })
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": self._tool_order_correction(
                        tool_name=tool_name, question=question,
                    )})
                    continue

                required_tool = self._required_tool_after_vision(
                    question=question, called_tool_names=called_tool_names,
                )
                if self.enforce_required_tools and required_tool and tool_name != required_tool:
                    trace.append({
                        "step": step,
                        "event": "tool_call_blocked_missing_required_tool",
                        "required_tool": required_tool,
                    })
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": self._missing_required_tool_prompt(required_tool)})
                    continue

                if tool_name == "vision_parse" and "image" not in tool_args:
                    tool_args["image"] = image

                # Execute tool and capture result for over-call reduction
                if tool_name == "vision_parse":
                    if tool_name not in TOOLS:
                        observation = {"error": f"未知工具：{tool_name}"}
                    else:
                        try:
                            observation = TOOLS[tool_name](**tool_args)
                        except Exception as e:
                            observation = {"error": f"工具执行失败：{e}"}
                    last_vision_result = str(observation.get("result", ""))

                    trace.append({
                        "step": step,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": observation,
                    })
                    messages.append({"role": "assistant", "content": model_output})

                    # Check if we can skip the second tool
                    called_tool_names = self._called_tool_names(trace)
                    skippable = self._should_skip_second_tool(
                        question, called_tool_names, last_vision_result)

                    post_lines = [
                        f"工具 {tool_name} 返回结果如下：",
                        f"{json.dumps(observation, ensure_ascii=False)}",
                        "",
                    ]
                    if skippable:
                        post_lines.append(
                            f"重要：vision_parse 已经返回了足够的信息。"
                            f"不需要再调用 {', '.join(skippable)}。"
                            f"请直接输出 <final_answer>最终答案</final_answer>。"
                            f"不要再输出 <tool_call>。"
                        )
                    else:
                        post_lines += [
                            "请根据工具结果继续完成原始任务。",
                            "只能输出一个 <tool_call> 或一个非空 <final_answer>。",
                        ]

                    messages.append({"role": "user", "content": "\n".join(post_lines)})
                    continue

                # Non-vision_parse tool execution
                call_key = (tool_name, json.dumps(tool_args, ensure_ascii=False, sort_keys=True))
                if call_key in used_tool_calls:
                    return {
                        "answer": "模型重复调用相同工具，停止执行。",
                        "trace": trace, "messages": messages,
                        "status": "repeated_tool_call",
                    }
                used_tool_calls.add(call_key)

                if tool_name not in TOOLS:
                    observation = {"error": f"未知工具：{tool_name}"}
                else:
                    try:
                        observation = TOOLS[tool_name](**tool_args)
                    except Exception as e:
                        observation = {"error": f"工具执行失败：{e}"}

                trace.append({
                    "step": step,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "observation": observation,
                })

                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": (
                    f"工具 {tool_name} 返回结果如下：\n"
                    f"{json.dumps(observation, ensure_ascii=False)}\n\n"
                    "请根据工具结果继续完成原始任务。只能输出一个 <tool_call> 或一个非空 <final_answer>。"
                )})
                continue

            return {
                "answer": f"未知解析类型：{parsed.get('type')}",
                "trace": trace, "messages": messages,
                "status": "unknown_parsed_type",
            }

        return {
            "answer": "达到最大工具调用步数，仍未得到最终答案。",
            "trace": trace, "messages": messages,
            "status": "max_steps_exceeded",
        }
