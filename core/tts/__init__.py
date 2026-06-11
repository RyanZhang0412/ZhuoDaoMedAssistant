"""TTS 子包入口与工厂。

仅注册本地引擎（Kokoro）；显式不注册联网引擎（edge-tts 等）。
offline 守卫在工厂层二次拦截：offline=true 且选了联网/非本地 TTS 时，
抛 core.net_guard.OfflineViolationError（全系统唯一异常类），而非静默降级。
"""

from __future__ import annotations

import importlib

from core.net_guard import OfflineViolationError, is_offline_enforced
from core.tts.base import TTSBase, TTSChunk

__all__ = [
    "TTSBase",
    "TTSChunk",
    "create_tts",
    "register_tts",
    "TTS_REGISTRY",
    "OFFLINE_FORBIDDEN_TTS",
]

# 本地引擎注册表（不含任何联网引擎）
TTS_REGISTRY: dict[str, str] = {
    "KokoroTTS": "core.tts.kokoro:KokoroTTS",
}

# 已知联网 TTS 黑名单：offline=true 时即便被注册也拒绝
OFFLINE_FORBIDDEN_TTS = {"EdgeTTS", "AzureTTS", "OpenAITTS"}


def register_tts(name: str, import_path: str) -> None:
    TTS_REGISTRY[name] = import_path


def create_tts(name: str, config: dict) -> TTSBase:
    # offline 硬约束：选了联网 TTS 直接报错（不静默降级）
    if is_offline_enforced() and name in OFFLINE_FORBIDDEN_TTS:
        raise OfflineViolationError(
            f"offline=true 下不允许联网 TTS: {name}。请在 config.yaml 改用本地 TTS"
            f"（如 KokoroTTS）。可选本地: {sorted(TTS_REGISTRY)}"
        )
    if name not in TTS_REGISTRY:
        raise ValueError(f"未知 TTS: {name!r}。可选本地: {sorted(TTS_REGISTRY)}")
    module_path, cls_name = TTS_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    inst = cls(config)
    # 二次校验：实例必须声明本地
    if is_offline_enforced() and not getattr(inst, "is_local", False):
        raise OfflineViolationError(
            f"offline=true 下 TTS 引擎 {name} 未声明 is_local=True，已拒绝。"
        )
    return inst
