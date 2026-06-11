"""TTS 抽象基类。输出 16-bit PCM；声明 is_local 供 offline 守卫校验。"""

from __future__ import annotations

import abc
from dataclasses import dataclass

__all__ = ["TTSBase", "TTSChunk"]


@dataclass
class TTSChunk:
    audio: bytes
    sample_rate: int = 16000
    is_last: bool = False


class TTSBase(abc.ABC):
    """语音合成抽象。

    is_local: 子类必须为本地引擎；联网引擎置 False，offline=true 时会被工厂拒绝。
    """

    is_local: bool = True
    SAMPLE_RATE: int = 16000

    def __init__(self, config: dict) -> None:
        self.config = config

    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """整段文本 -> PCM。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放资源。"""
