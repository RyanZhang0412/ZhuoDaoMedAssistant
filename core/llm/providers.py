"""LLM provider 具体实现：openai_compatible / anthropic / ollama / local。

全部实现 core.llm.base.LLMBase 的统一接口。Agent 与 explainer 不感知这里，
移植服务器只需在 config.yaml 切换 llm.provider。

- openai_compatible / ollama：走 OpenAI 兼容 wire format（openai SDK）。
  function calling 用 OpenAI tools 格式；内部把统一的 input_schema 转成 OpenAI function。
- anthropic：用官方 anthropic SDK，model 默认 claude-opus-4-8，adaptive thinking，
  tools=[{name,description,input_schema}]，tool_use 块映射为统一 ToolCall。
- local：本地推理占位（HF transformers / llama.cpp），移植服务器时填实现。

每个 provider 构造时调用 net_guard.assert_local_endpoint 做 offline 早失败校验。
依赖按需 import（未装某 SDK 时不影响用其他 provider）。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from core.llm.base import LLMBase, LLMResponse, Message, ToolCall
from core.net_guard import assert_local_endpoint

__all__ = ["OpenAICompatibleLLM", "AnthropicLLM", "OllamaLLM", "LocalLLM"]


# --------------------------------------------------------------------------- #
# OpenAI 兼容（DeepSeek / Qwen API / 智谱 / 本地 vLLM 等）
# --------------------------------------------------------------------------- #
class OpenAICompatibleLLM(LLMBase):
    """任何 OpenAI 兼容端点。provider=openai_compatible。"""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        sub = config.get("openai_compatible", {})
        self.model = sub.get("model", "gpt-3.5-turbo")
        base_url = sub.get("base_url")
        api_key = (sub.get("api_key") or "").strip()
        if not api_key:
            raise ValueError(
                "Missing llm.openai_compatible.api_key. "
                "If you use ${GEMINI_API_KEY} in config/config.yaml, make sure that "
                "environment variable is available in the same shell that starts main.py."
            )
        if api_key.startswith("${") and api_key.endswith("}"):
            raise ValueError(
                "Invalid llm.openai_compatible.api_key placeholder. "
                "Use ${ENV_VAR_NAME} for environment variables, or provide a raw API key "
                "string without ${}."
            )
        # offline 早失败：端点必须本地（开发期 offline=false 时不限制）
        assert_local_endpoint(base_url, what="LLM(openai_compatible) base_url")
        from openai import OpenAI  # 延迟导入

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        oai_messages = _to_openai_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            if tool_choice:
                kwargs["tool_choice"] = (
                    tool_choice
                    if tool_choice in ("auto", "none")
                    else {"type": "function", "function": {"name": tool_choice}}
                )
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            raw_tc = tc.to_dict(mode="json", use_api_names=True) if hasattr(tc, "to_dict") else None
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    raw_arguments=raw_args,
                    raw=raw_tc,
                )
            )
        return LLMResponse(text=choice.content, tool_calls=calls, raw=resp)

    def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages, system),
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaLLM(OpenAICompatibleLLM):
    """本地 ollama（OpenAI 兼容）。provider=ollama。读 config['ollama'] 段。"""

    def __init__(self, config: dict) -> None:
        # 复用 OpenAI 兼容逻辑，但参数来自 ollama 段
        merged = dict(config)
        merged["openai_compatible"] = config.get("ollama", {})
        super().__init__(merged)


# --------------------------------------------------------------------------- #
# Anthropic Claude
# --------------------------------------------------------------------------- #
class AnthropicLLM(LLMBase):
    """Anthropic Claude 官方 SDK。provider=anthropic。

    注意：anthropic 默认指向公网端点；offline=true 时必须在 config 配本地 base_url，
    否则 net_guard 的 socket 兜底会拦截。开发期 offline=false 正常用 API。
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        sub = config.get("anthropic", {})
        self.model = sub.get("model", "claude-opus-4-8")
        api_key = sub.get("api_key")
        base_url = sub.get("base_url")  # offline 部署可指向本地兼容端点
        assert_local_endpoint(base_url, what="LLM(anthropic) base_url")
        import anthropic  # 延迟导入

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": _to_anthropic_messages(messages),
            # 医疗场景默认开启自适应思考；不传 temperature（4.8 已移除）
            "thinking": {"type": "adaptive"},
        }
        if system:
            kwargs["system"] = system
        if tools:
            # 统一 schema 已是 {name, description, input_schema}，即 anthropic 原生格式
            kwargs["tools"] = tools
            if tool_choice and tool_choice not in ("auto",):
                kwargs["tool_choice"] = (
                    {"type": tool_choice}
                    if tool_choice in ("any", "none")
                    else {"type": "tool", "name": tool_choice}
                )
        resp = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        text = "".join(text_parts) if text_parts else None
        return LLMResponse(text=text, tool_calls=calls, raw=resp)

    def stream_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": _to_anthropic_messages(messages),
            "thinking": {"type": "adaptive"},
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


# --------------------------------------------------------------------------- #
# 本地推理（移植服务器时实现）
# --------------------------------------------------------------------------- #
class LocalLLM(LLMBase):
    """本地模型推理占位。provider=local。

    移植到有显存的服务器后，在此用 transformers / llama-cpp-python 加载
    config['local']['model_path'] 的本地权重实现 chat/stream_chat。
    目前抛清晰错误，提示开发期请用 openai_compatible / anthropic / ollama。
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_path = config.get("local", {}).get("model_path")
        self.device = config.get("local", {}).get("device", "auto")

    def _not_implemented(self):
        raise NotImplementedError(
            "LocalLLM is not implemented yet. Set llm.provider to openai_compatible / "
            "anthropic / ollama during development, or implement "
            f"core/llm/providers.py::LocalLLM for deployment "
            f"(model_path={self.model_path}, device={self.device})."
        )

    def chat(self, messages, *, system=None, tools=None, tool_choice=None, max_tokens=None):  # noqa: D102,E501
        self._not_implemented()

    def stream_chat(self, messages, *, system=None, max_tokens=None):  # noqa: D102
        self._not_implemented()


# --------------------------------------------------------------------------- #
# 消息 / 工具格式转换
# --------------------------------------------------------------------------- #
def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                }
            )
        elif m.role == "assistant" and m.tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    tc.raw
                    if tc.raw is not None
                    else {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments
                            if tc.raw_arguments is not None
                            else json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ],
            }
            if m.content is not None:
                assistant_msg["content"] = m.content
            out.append(assistant_msg)
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def _to_openai_tool(tool: dict) -> dict:
    """统一 schema {name, description, input_schema} -> OpenAI function 格式。"""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """统一 Message 列表 -> anthropic messages（tool_use / tool_result 块）。"""
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content or "",
                        }
                    ],
                }
            )
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out
