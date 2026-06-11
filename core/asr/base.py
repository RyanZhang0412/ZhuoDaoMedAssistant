"""ASR 抽象基类与结果结构。"""

from __future__ import annotations

import abc
from dataclasses import dataclass

__all__ = ["ASRBase", "ASRResult"]


@dataclass
class ASRResult:
    text: str
    is_final: bool = True
    confidence: float | None = None
    language: str | None = None


class ASRBase(abc.ABC):
    """语音识别引擎抽象。期望输入 16k 单声道 16-bit PCM。"""

    SAMPLE_RATE: int = 16000

    def __init__(self, config: dict) -> None:
        self.config = config

    @abc.abstractmethod
    def transcribe(self, audio: bytes) -> ASRResult:
        """整段 PCM -> 文本。"""
        raise NotImplementedError

    def reset(self) -> None:
        """清空流式内部状态。"""

    def close(self) -> None:
        """释放资源。"""
