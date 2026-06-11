"""语音模块共用的 device 解析（auto | cpu | cuda，默认 cuda）。"""

from __future__ import annotations

__all__ = ["resolve_device"]


def resolve_device(device: str | None, *, default: str = "cuda") -> str:
    import torch

    raw = device if device not in (None, "") else default
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "配置 device=cuda，但 torch.cuda.is_available()=False。"
            "请安装 CUDA 版 torch，或改回 device: cpu/auto。"
        )
    return raw
