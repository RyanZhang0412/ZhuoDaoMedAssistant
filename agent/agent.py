"""本地 Agent 主循环 —— 基于 LLM function calling，注册本地工具，无任何联网工具。

工具注册与 schema 绑定：@register_tool(schema={...}) 在工具同文件一次声明，
写入全局 TOOL_REGISTRY；Agent.get_tool_schemas() 直接收集。
（不学百聆把 schema 放独立 json，避免与代码漂移。）

ToolResult.action 语义：
  REQLLM   —— 结果需回灌 LLM 二次组织语言（读病历/推方案/列表类）
  RESPONSE —— 话术已就绪，可直接用（建档/排期确认类）
Agent.chat 内部处理 REQLLM 循环（max_tool_rounds 兜底防死循环）。

依赖统一的 core.llm.base.LLMBase（与 explainer 共用同一抽象）。
对外稳定名 Agent = LocalAgent（见文件末尾别名），供 core/robot 引用。
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from core.llm.base import LLMBase, Message, ToolCall

__all__ = [
    "LocalAgent",
    "Agent",
    "ToolResult",
    "ToolAction",
    "RegisteredTool",
    "register_tool",
    "TOOL_REGISTRY",
]


class ToolAction(enum.Enum):
    REQLLM = "reqllm"        # 结果回灌 LLM 二次组织语言
    RESPONSE = "response"    # 话术已就绪
    NOTFOUND = "notfound"    # 工具未注册
    ERROR = "error"          # 执行出错


@dataclass
class ToolResult:
    """工具执行后的结果（区别于 core.llm.ToolCall：那是模型"想调用"的意图）。"""

    action: ToolAction
    result: Any = None          # 结构化数据（回灌 LLM 用）
    response: str | None = None  # 直接话术（RESPONSE 时）


@dataclass
class RegisteredTool:
    name: str
    func: Callable[..., ToolResult]
    schema: dict


# 全局注册表：name -> RegisteredTool
TOOL_REGISTRY: dict[str, RegisteredTool] = {}


def register_tool(schema: dict) -> Callable[[Callable], Callable]:
    """装饰器：把工具函数与其 schema 一次性绑定并注册。

    schema 形如 {"name", "description", "input_schema": {...JSON Schema...}}
    （anthropic 原生格式；LLM 适配层会按 provider 转换）。
    """

    def deco(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        name = schema["name"]
        TOOL_REGISTRY[name] = RegisteredTool(name=name, func=func, schema=schema)
        return func

    return deco


class LocalAgent:
    """本地 agent。LLM 决定调用哪个工具，Agent 执行并循环到无工具调用。"""

    def __init__(
        self,
        llm: LLMBase,
        *,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds

    # ---- 工具 schema 收集 ----
    def get_tool_schemas(self) -> list[dict]:
        return [t.schema for t in TOOL_REGISTRY.values()]

    # ---- 依赖绑定（装配末期调用，把 repository/recommender/scheduler 注入工具上下文）----
    @staticmethod
    def bind_context(repository, recommender, scheduler) -> None:  # noqa: ANN001
        from agent.tools.context import ToolContext, set_context

        set_context(ToolContext(repository=repository, recommender=recommender, scheduler=scheduler))

    # ---- 主入口：吃文本、吐文本 ----
    def chat(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        patient_id: str | None = None,
    ) -> str:
        """处理一轮用户输入，返回最终文本回复。

        history: 形如 [{"role","content"}] 的历史（由 Robot/DialogueMemory 提供，
                 Agent 自身不持久化历史 —— 单一所有者是 Robot）。
        patient_id: 当前会话聚焦的患者（注入系统提示，便于 LLM 省略追问）。
        """
        messages: list[Message] = _to_messages(history)
        # 当前患者上下文注入（轻量）
        user_text = query
        if patient_id:
            user_text = f"[当前患者ID: {patient_id}]\n{query}"
        messages.append(Message(role="user", content=user_text))

        tools = self.get_tool_schemas()
        for _ in range(self.max_tool_rounds):
            resp = self.llm.chat(messages, system=self.system_prompt, tools=tools)

            # 无工具调用 -> 直接返回文本
            if not resp.tool_calls:
                return resp.text or ""

            # 记录 assistant 的工具调用意图
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )

            # 执行每个工具调用
            direct_response: str | None = None
            for call in resp.tool_calls:
                tr = self._dispatch_tool(call)
                if tr.action == ToolAction.RESPONSE and tr.response:
                    # 话术已就绪，但仍回灌让 LLM 串成连贯回复；同时记下兜底
                    direct_response = tr.response
                    payload = {"status": "ok", "message": tr.response}
                elif tr.action == ToolAction.REQLLM:
                    payload = tr.result if tr.result is not None else {}
                elif tr.action == ToolAction.NOTFOUND:
                    payload = {"error": f"工具未注册: {call.name}"}
                else:  # ERROR
                    payload = {"error": str(tr.result)}
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(payload, ensure_ascii=False),
                    )
                )
            # 继续下一轮，让 LLM 基于工具结果组织语言

        # 达到最大轮数仍未收敛：返回最近一次直接话术或提示
        return direct_response or "（已达到最大工具调用轮数，请补充信息后重试）"

    # ---- 工具分发 ----
    def _dispatch_tool(self, call: ToolCall) -> ToolResult:
        tool = TOOL_REGISTRY.get(call.name)
        if tool is None:
            return ToolResult(action=ToolAction.NOTFOUND, result=call.name)
        try:
            return tool.func(**call.arguments)
        except Exception as e:  # 工具内异常不应崩溃整个 agent
            return ToolResult(action=ToolAction.ERROR, result=f"{type(e).__name__}: {e}")


def _to_messages(history: list[dict] | None) -> list[Message]:
    """把 [{"role","content"}] 历史转为统一 Message 列表（边界转换点）。"""
    if not history:
        return []
    return [Message(role=h["role"], content=h.get("content", "")) for h in history]


# 对外稳定名（core/robot 等引用 Agent）
Agent = LocalAgent
