"""VAD 抽象基类与语音段事件。"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass

__all__ = ["VADBase", "VADEvent", "VADEventType"]


class VADEventType(enum.Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SILENCE = "silence"


@dataclass
class VADEvent:
    type: VADEventType
    timestamp_ms: int = 0
    audio: bytes | None = None  # SPEECH_END 时附完整语音段


class VADBase(abc.ABC):
    """语音活动检测抽象。逐帧判定并在语音段边界吐事件，供切句与打断(barge-in)。"""

    SAMPLE_RATE: int = 16000

    def __init__(self, config: dict) -> None:
        self.config = config

    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """单帧是否为语音。"""
        raise NotImplementedError

    def reset(self) -> None:
        """清空有状态累积。"""
