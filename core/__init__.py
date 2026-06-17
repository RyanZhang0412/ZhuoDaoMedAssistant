"""core 包：语音对话引擎（ASR/VAD/TTS/LLM 可插拔 + Robot 协调层）。"""

from core.llm import LLMBase, create_llm
from core.robot import Robot, RobotResponse

__all__ = [
    "Robot",
    "RobotResponse",
    "create_llm",
    "LLMBase",
]
