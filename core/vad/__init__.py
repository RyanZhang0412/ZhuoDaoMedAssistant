"""VAD 子包入口与工厂。"""

from __future__ import annotations

from core.vad.base import VADBase, VADEvent, VADEventType

__all__ = ["VADBase", "VADEvent", "VADEventType", "create_vad", "register_vad", "VAD_REGISTRY"]

VAD_REGISTRY: dict[str, str] = {
    "SileroVAD": "core.vad.silero:SileroVAD",
}


def register_vad(name: str, import_path: str) -> None:
    VAD_REGISTRY[name] = import_path


def create_vad(name: str, config: dict) -> VADBase:
    if name not in VAD_REGISTRY:
        raise ValueError(f"未知 VAD: {name!r}。可选: {sorted(VAD_REGISTRY)}")
    module_path, cls_name = VAD_REGISTRY[name].split(":")
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(config)
