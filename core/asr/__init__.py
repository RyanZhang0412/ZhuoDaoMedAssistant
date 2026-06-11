"""ASR 子包入口与工厂。"""

from __future__ import annotations

import importlib

from core.asr.base import ASRBase, ASRResult

__all__ = ["ASRBase", "ASRResult", "create_asr", "register_asr", "ASR_REGISTRY"]

# 注册名 -> 类路径（延迟导入，未装 funasr 也能 import 本模块）
ASR_REGISTRY: dict[str, str] = {
    "SenseVoiceASR": "core.asr.sensevoice:SenseVoiceASR",
}


def register_asr(name: str, import_path: str) -> None:
    ASR_REGISTRY[name] = import_path


def create_asr(name: str, config: dict) -> ASRBase:
    if name not in ASR_REGISTRY:
        raise ValueError(f"未知 ASR: {name!r}。可选: {sorted(ASR_REGISTRY)}")
    module_path, cls_name = ASR_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(config)
