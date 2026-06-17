"""TTS 子包入口与工厂。

仅注册本地引擎（Kokoro）；显式不注册联网引擎（edge-tts 等）。
"""

from __future__ import annotations

import importlib

from core.tts.base import TTSBase, TTSChunk

__all__ = [
    "TTSBase",
    "TTSChunk",
    "create_tts",
    "register_tts",
    "TTS_REGISTRY",
]

# 本地引擎注册表（不含任何联网引擎）
TTS_REGISTRY: dict[str, str] = {
    "KokoroTTS": "core.tts.kokoro:KokoroTTS",
}


def register_tts(name: str, import_path: str) -> None:
    TTS_REGISTRY[name] = import_path


def create_tts(name: str, config: dict) -> TTSBase:
    if name not in TTS_REGISTRY:
        raise ValueError(f"未知 TTS: {name!r}。可选本地: {sorted(TTS_REGISTRY)}")
    module_path, cls_name = TTS_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(config)
