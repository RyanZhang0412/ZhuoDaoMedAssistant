"""TTS 抽象基类。输出 16-bit PCM；is_local 标记是否为本地引擎。"""

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

    def warmup(self) -> None:
        """预加载模型权重（语音模式启动时调用，避免首句合成卡顿）。"""

    def close(self) -> None:
        """释放资源。"""
