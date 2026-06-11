"""core.llm 包入口与 provider 工厂。

create_llm(config) 按 config['llm']['provider'] 实例化对应 provider，
返回统一的 LLMBase。Agent、explainer 等只 import 这里和 base，
不感知具体 provider —— 移植服务器换本地部署只改 config.yaml。

LLMProvider 是 LLMBase 的显式别名（语义糖），方便对外引用；
禁止另立第二套方法名（见交叉验证结论）。
"""

from __future__ import annotations

from core.llm.base import LLMBase, LLMResponse, Message, ToolCall

# 语义别名：对外可用 LLMProvider 指代统一抽象，但它就是 LLMBase 本身。
LLMProvider = LLMBase

__all__ = [
    "LLMBase",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "create_llm",
]

# provider 名 -> 类。延迟在 create_llm 内取类，避免在未装某 SDK 时导入失败。
_PROVIDERS = {
    "openai_compatible": "OpenAICompatibleLLM",
    "anthropic": "AnthropicLLM",
    "ollama": "OllamaLLM",
    "local": "LocalLLM",
}


def create_llm(config: dict) -> LLMBase:
    """根据 config 装配 LLM provider。

    config 形如全局 config.yaml 的根（含 llm.provider 与各 provider 子段、
    顶层 temperature/max_tokens 也从 llm 段读）。
    """
    llm_cfg = config.get("llm", config)  # 容忍传入整份 config 或仅 llm 段
    provider = llm_cfg.get("provider", "openai_compatible")
    if provider not in _PROVIDERS:
        raise ValueError(
            f"未知 LLM provider: {provider!r}。可选: {sorted(_PROVIDERS)}"
        )
    from core.llm import providers as _impl

    cls = getattr(_impl, _PROVIDERS[provider])
    return cls(llm_cfg)
