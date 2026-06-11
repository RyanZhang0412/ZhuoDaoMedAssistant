"""LLM 抽象层 —— 全系统唯一权威接口。

按交叉验证结论收口：整个项目只有这一个 LLM 抽象基类 ``LLMBase``，
medical/service.py 与 agent/agent 都从这里 import，禁止另立第二套命名或方法名。

统一接口（两个方法）：
  - chat(messages, *, system, tools, tool_choice, max_tokens) -> LLMResponse
      普通对话 + function calling 都走这个。tools 非空时支持工具调用。
  - stream_chat(messages, *, system, max_tokens) -> Iterator[str]
      流式纯文本（供语音 TTS 尽早播放）。

provider 无关的中间表示：
  - Message(role, content, tool_calls, tool_call_id)  —— 统一消息
  - ToolCall(id, name, arguments)                     —— 模型"想调用工具"的意图
  - LLMResponse(text, tool_calls, raw)                —— 一次响应（text 可能为 None）

provider 工厂在 core/llm/__init__.py（create_llm）。Agent/explainer 只依赖本基类，
完全不感知 provider；移植服务器换本地部署只改 config。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Message", "ToolCall", "LLMResponse", "LLMBase"]


@dataclass
class ToolCall:
    """模型请求调用某工具的意图（provider 无关）。

    注意区分 agent 层的 ToolResult（工具*执行后*的结果）：
    ToolCall 是"模型想调用工具"，由 LLMResponse 产出、Agent 用它去 dispatch。

    raw_arguments / raw: 兼容 Gemini OpenAI tool-calling 之类 provider 的额外字段回灌。
    某些 provider（如 Gemini）要求把 tool_call 的隐藏元数据原样带回下一轮，
    不能只保留 name/arguments 再自行重建。
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class Message:
    """统一消息表示（provider 无关）。

    role: "user" | "assistant" | "system" | "tool"
    content: 文本内容（assistant 发起纯工具调用时可为 None）
    tool_calls: 当 assistant 消息携带工具调用时
    tool_call_id: 当 role=="tool" 时，对应被回灌的 ToolCall.id
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    """一次 LLM 响应。

    text: 文本回复，可能为 None（模型只发起工具调用、未输出文本时）。
          下游取文本务必处理 None，例如 ``resp.text or ""``。
    tool_calls: 模型本轮请求的工具调用（无则空列表）。
    raw: provider 原始响应对象，调试用。
    """

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMBase(abc.ABC):
    """LLM provider 的统一抽象基类（全系统唯一权威）。

    所有 provider（openai_compatible / anthropic / ollama / local）实现此接口。
    构造时应调用 core.net_guard.assert_local_endpoint 校验端点（offline 早失败层）。
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 1024)

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """普通对话 + function calling 统一入口。

        tools: function-calling 工具 schema 列表，形如
               {"name", "description", "input_schema": {...JSON Schema...}}
               （anthropic 原生格式；openai 适配层内部转换为 OpenAI function 格式）。
        tool_choice: None=auto；"auto"/"any"/"none" 或具体工具名。
        返回 LLMResponse（可能含 text 和/或 tool_calls）。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式纯文本输出（不含工具调用），供语音 TTS 尽早播放。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放底层资源（如本地模型）。默认 no-op。"""
