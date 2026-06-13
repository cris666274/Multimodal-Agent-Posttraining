import json
from typing import Dict, Any, List

from agent.vlm import QwenVLModel
from agent.parser import parse_model_output
from agent.tools import TOOLS, set_vlm_model
from agent.prompts import SYSTEM_PROMPT


class MultimodalAgent:
    def __init__(
        self,
        model: QwenVLModel,
        max_steps: int = 6,
        enforce_required_tools: bool = False,
    ):
        self.model = model
        self.max_steps = max_steps
        self.enforce_required_tools = enforce_required_tools

        # 将真实 VLM 模型注入 tools，使 vision_parse 使用真实推理
        set_vlm_model(model)

    def build_initial_messages(self, image: str, question: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

    @staticmethod
    def _called_tool_names(trace: List[Dict[str, Any]]) -> List[str]:
        return [item["tool_name"] for item in trace if "tool_name" in item]

    @staticmethod
    def _should_skip_second_tool(question, called_tools, vision_result, required):
        """Check if 2nd tool can be skipped based on vision_parse results."""
        skippable = []
        if required == "retrieve_docs" and vision_result:
            fields_en = ["Invoice No", "Date", "Amount", "Tax ID"]
            fields_cn = ["发票号码", "日期", "金额", "税号"]
            found_en = sum(1 for f in fields_en if f in vision_result)
            found_cn = sum(1 for f in fields_cn if f in vision_result)
            has_inv = "INV-" in vision_result or "TAX-" in vision_result
            if (found_en >= 3 or found_cn >= 3 or has_inv) and \
               ("缺少" not in question and "缺失" not in question and
                "差异" not in question and "对比" not in question and
                "合规" not in question and "符合" not in question):
                skippable.append("retrieve_docs")
        if required == "python_exec" and "vision_parse" in called_tools:
            calc_keywords = ["增长率", "同比", "计算", "预测", "百分比",
                           "差值", "增长", "总和", "趋势", "高出", "多少",
                           "哪一年", "哪个", "比较"]
            if not any(kw in question for kw in calc_keywords):
                skippable.append("python_exec")
        return skippable

    @staticmethod
    def _needs_vision_first(question: str, image: str) -> bool:
        if image:
            return True

        keywords = [
            "图片",
            "图像",
            "发票",
            "表单",
            "商品图",
            "图表",
            "logo",
            "品牌",
            "ocr",
            "invoice",
            "chart",
            "product",
        ]
        q = question.lower()
        return any(keyword in question or keyword in q for keyword in keywords)

    @staticmethod
    def _tool_order_correction(tool_name: str, question: str) -> str:
        if tool_name == "retrieve_docs":
            return (
                "你还没有调用 vision_parse，不能先调用 retrieve_docs。\n"
                "对于包含图片、发票、商品图、图表或品牌识别的任务，必须先读取图片证据。\n"
                "请重新输出：<tool_call>{\"name\":\"vision_parse\",\"args\":{\"mode\":\"ocr\"}}</tool_call>\n"
                "如果是商品图文一致性，请使用 mode=\"caption\"；如果是图表计算，请使用 mode=\"chart_values\"；如果是品牌/logo，请使用 mode=\"ocr+caption\"。\n"
                "重要：调用 vision_parse 拿到结果后，你还必须调用 retrieve_docs 获取规则依据，才能给出最终答案。"
            )

        if tool_name == "python_exec":
            return (
                "你还没有调用 vision_parse，不能先调用 python_exec。\n"
                "图表计算必须先读取图表数值，再进行计算。\n"
                "请重新输出：<tool_call>{\"name\":\"vision_parse\",\"args\":{\"mode\":\"chart_values\"}}</tool_call>\n"
                "重要：调用 vision_parse 拿到图表数值后，你还必须调用 python_exec 完成计算，才能给出最终答案。"
            )

        return (
            f"当前不能先调用 {tool_name}。\n"
            "请先调用 vision_parse 获取图片证据。\n"
            f"重要：调用 vision_parse 拿到结果后，你还必须调用 {tool_name} 完成原始任务。"
        )

    @staticmethod
    def _needs_python_exec(question: str) -> bool:
        keywords = [
            "增长率",
            "同比",
            "百分比",
            "计算",
            "图表",
            "chart",
        ]
        q = question.lower()
        return any(keyword in question or keyword in q for keyword in keywords)

    @staticmethod
    def _needs_retrieve_docs(question: str) -> bool:
        keywords = [
            "发票",
            "字段",
            "必填",
            "合规",
            "规则",
            "核验",
            "invoice",
        ]
        q = question.lower()
        return any(keyword in question or keyword in q for keyword in keywords)

    @staticmethod
    def _missing_required_tool_prompt(tool_name: str) -> str:
        if tool_name == "python_exec":
            return (
                "你已经读取了图片信息，但这个问题涉及图表、同比、增长率或计算。\n"
                "在给出最终答案之前，必须调用 python_exec 完成数值计算。\n"
                "请输出：<tool_call>{\"name\":\"python_exec\",\"args\":{\"code\":\"(90-80)/80*100\"}}</tool_call>\n"
                "如果工具返回的图表数值不是 90 和 80，请根据 vision_parse 的结果替换表达式中的数字。"
            )

        if tool_name == "retrieve_docs":
            return (
                "你已经读取了图片信息，但这个问题涉及发票字段、必填项、合规或规则判断。\n"
                "在给出最终答案之前，必须调用 retrieve_docs 获取规则依据。\n"
                "请输出：<tool_call>{\"name\":\"retrieve_docs\",\"args\":{\"query\":\"发票必填字段规则\"}}</tool_call>"
            )

        return f"在最终回答前还必须调用 {tool_name}。"

    def _required_tool_after_vision(
        self,
        question: str,
        called_tool_names: List[str],
    ) -> str:
        if "vision_parse" not in called_tool_names:
            return ""

        if self._needs_python_exec(question) and "python_exec" not in called_tool_names:
            return "python_exec"

        if self._needs_retrieve_docs(question) and "retrieve_docs" not in called_tool_names:
            return "retrieve_docs"

        return ""

    def run(self, image: str, question: str) -> Dict[str, Any]:
        messages = self.build_initial_messages(image=image, question=question)
        trace = []
        used_tool_calls = set()
        blocked_tools: set = set()
        last_vision_result = ""

        for step in range(self.max_steps):
            model_output = self.model.generate(messages)
            parsed = parse_model_output(model_output)

            trace.append(
                {
                    "step": step,
                    "model_output": model_output,
                    "parsed": parsed,
                }
            )

            # 1. 最终答案
            if parsed["type"] == "final_answer":
                final_content = parsed.get("content", "").strip()
                called_tool_names = self._called_tool_names(trace)

                # 如果模型输出空答案，不立刻结束，而是追加提示让它继续
                if not final_content:
                    trace.append(
                        {
                            "step": step,
                            "event": "empty_final_answer_detected",
                        }
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": model_output,
                        }
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "你刚才输出了空的 <final_answer></final_answer>。\n"
                                "这是无效回答。\n"
                                "请根据已有图片信息和工具返回结果继续完成任务。\n"
                                "如果证据足够，请输出非空的 <final_answer>最终答案</final_answer>。\n"
                                "如果还缺少规则依据，请调用 retrieve_docs。\n"
                                "如果还需要计算，请调用 python_exec。\n"
                                "只能输出一个 <tool_call> 或一个非空 <final_answer>。"
                            ),
                        }
                    )

                    continue

                required_tool = self._required_tool_after_vision(
                    question=question,
                    called_tool_names=called_tool_names,
                )
                # Allow skip if vision_parse result is sufficient
                skippable = self._should_skip_second_tool(
                    question, called_tool_names, last_vision_result, required_tool)
                if required_tool in skippable:
                    required_tool = ""
                if self.enforce_required_tools and required_tool:
                    trace.append(
                        {
                            "step": step,
                            "event": "final_answer_blocked_missing_required_tool",
                            "required_tool": required_tool,
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": model_output,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._missing_required_tool_prompt(required_tool),
                        }
                    )
                    continue

                return {
                    "answer": final_content,
                    "trace": trace,
                    "messages": messages,
                    "status": "finished",
                }

            # 2. 解析失败
            if parsed["type"] == "parse_error":
                return {
                    "answer": f"模型输出格式解析失败：{parsed.get('error', 'unknown error')}",
                    "trace": trace,
                    "messages": messages,
                    "status": "parse_error",
                }

            # 3. 工具调用
            if parsed["type"] == "tool_call":
                tool_name = parsed.get("name")
                tool_args = parsed.get("args", {})

                if not isinstance(tool_name, str) or not tool_name:
                    return {
                        "answer": "工具名称格式错误。",
                        "trace": trace,
                        "messages": messages,
                        "status": "bad_tool_name",
                    }

                if not isinstance(tool_args, dict):
                    return {
                        "answer": "工具参数格式错误。",
                        "trace": trace,
                        "messages": messages,
                        "status": "bad_tool_args",
                    }

                called_tool_names = self._called_tool_names(trace)
                if (
                    self._needs_vision_first(question=question, image=image)
                    and "vision_parse" not in called_tool_names
                    and tool_name != "vision_parse"
                ):
                    blocked_tools.add(tool_name)
                    trace.append(
                        {
                            "step": step,
                            "event": "tool_order_blocked",
                            "blocked_tool_name": tool_name,
                            "reason": "vision_parse_required_before_other_tools",
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": model_output,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._tool_order_correction(
                                tool_name=tool_name,
                                question=question,
                            ),
                        }
                    )
                    continue

                required_tool = self._required_tool_after_vision(
                    question=question,
                    called_tool_names=called_tool_names,
                )
                if self.enforce_required_tools and required_tool and tool_name != required_tool:
                    trace.append(
                        {
                            "step": step,
                            "event": "tool_call_blocked_missing_required_tool",
                            "blocked_tool_name": tool_name,
                            "required_tool": required_tool,
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": model_output,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._missing_required_tool_prompt(required_tool),
                        }
                    )
                    continue

                # vision_parse 自动补 image 参数
                if tool_name == "vision_parse" and "image" not in tool_args:
                    tool_args["image"] = image

                # 防止重复调用同一个工具和同一组参数
                call_key = (
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False, sort_keys=True),
                )

                if call_key in used_tool_calls:
                    return {
                        "answer": "模型重复调用相同工具，停止执行。",
                        "trace": trace,
                        "messages": messages,
                        "status": "repeated_tool_call",
                    }

                used_tool_calls.add(call_key)

                # 执行工具
                if tool_name not in TOOLS:
                    observation = {
                        "error": f"未知工具：{tool_name}",
                        "available_tools": list(TOOLS.keys()),
                    }
                else:
                    try:
                        observation = TOOLS[tool_name](**tool_args)
                        if tool_name == "vision_parse":
                            last_vision_result = str(observation.get("result", ""))
                    except Exception as e:
                        observation = {
                            "error": f"工具执行失败：{e}",
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        }

                trace.append(
                    {
                        "step": step,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "observation": observation,
                    }
                )

                # 把模型刚才的工具调用写入历史
                messages.append(
                    {
                        "role": "assistant",
                        "content": model_output,
                    }
                )

                # 构建工具调用后的引导消息
                post_tool_lines = [
                    f"工具 {tool_name} 返回结果如下：",
                    f"{json.dumps(observation, ensure_ascii=False)}",
                    "",
                ]

                # Over-call reduction: check if 2nd tool can be skipped
                if tool_name == "vision_parse":
                    called_tool_names = self._called_tool_names(trace)
                    required = self._required_tool_after_vision(question, called_tool_names)
                    skippable = self._should_skip_second_tool(
                        question, called_tool_names,
                        str(observation.get("result", "")), required)

                    if skippable:
                        post_tool_lines.append(
                            f"重要：vision_parse 已经返回了足够的信息。"
                            f"不需要再调用 {', '.join(skippable)}。"
                            f"请直接输出 <final_answer>最终答案</final_answer>。"
                            f"不要再输出 <tool_call>。"
                        )
                    else:
                        post_tool_lines.append("请根据工具结果继续完成原始任务。")
                else:
                    post_tool_lines.append("请根据工具结果继续完成原始任务。")

                # 如果当前刚调完 vision_parse，且之前有工具被阻断过，显式提醒
                still_blocked = [
                    t for t in blocked_tools
                    if t not in self._called_tool_names(trace)
                ]
                if tool_name == "vision_parse" and still_blocked:
                    blocked_names = "、".join(still_blocked)
                    post_tool_lines.append(
                        f"你之前尝试调用 {blocked_names} 但被阻止了，因为必须先完成 vision_parse。"
                    )
                    post_tool_lines.append(
                        f"现在你已拿到图片证据，请立即调用 {blocked_names}。"
                    )

                post_tool_lines += [
                    "你必须遵守：",
                    "1. 不允许输出空的 <final_answer></final_answer>。",
                    "2. 如果原始问题是发票、字段、合规、规则核验类问题，并且还没有调用 retrieve_docs，请调用 retrieve_docs。",
                    "3. 如果原始问题是图表、增长率、同比、百分比、计算类问题，并且还没有调用 python_exec，请调用 python_exec。",
                    "4. 如果已有证据足够，请输出包含具体结论和依据的 <final_answer>...</final_answer>。",
                    "5. 只能输出一个 <tool_call> 或一个非空 <final_answer>，不要输出其他内容。",
                ]

                messages.append(
                    {
                        "role": "user",
                        "content": "\n".join(post_tool_lines),
                    }
                )

                continue

            # 4. 未知解析类型
            return {
                "answer": f"未知解析类型：{parsed.get('type')}",
                "trace": trace,
                "messages": messages,
                "status": "unknown_parsed_type",
            }

        return {
            "answer": "达到最大工具调用步数，仍未得到最终答案。",
            "trace": trace,
            "messages": messages,
            "status": "max_steps_exceeded",
        }
