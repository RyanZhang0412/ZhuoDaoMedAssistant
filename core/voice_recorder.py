"""语音会话录音：每轮保存 user/assistant 的 WAV + 文本 sidecar。"""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path

__all__ = ["VoiceSessionRecorder"]


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


class VoiceSessionRecorder:
    """一次 run_voice_session 对应一个子目录。"""

    def __init__(self, base_dir: str | Path, session_id: str, *, user_sample_rate: int = 16000) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_sid = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "default"
        self.session_dir = Path(base_dir) / f"{ts}_{safe_sid}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.user_sample_rate = user_sample_rate
        self._turn = 0

    @property
    def turn(self) -> int:
        return self._turn

    def save_user(self, pcm: bytes, text: str) -> Path:
        self._turn += 1
        prefix = f"{self._turn:03d}"
        wav_path = self.session_dir / f"{prefix}_user.wav"
        write_pcm_wav(wav_path, pcm, self.user_sample_rate)
        (self.session_dir / f"{prefix}_user.txt").write_text(text, encoding="utf-8")
        return wav_path

    def save_assistant(self, pcm: bytes, text: str, sample_rate: int, *, turn: int | None = None) -> Path:
        """保存助手音频。turn 应与本轮 save_user 返回的轮次一致（流式 TTS 异步回调时必须显式传入）。"""
        t = turn if turn is not None else self._turn
        prefix = f"{t:03d}"
        wav_path = self.session_dir / f"{prefix}_assistant.wav"
        if pcm:
            write_pcm_wav(wav_path, pcm, sample_rate)
        (self.session_dir / f"{prefix}_assistant.txt").write_text(text, encoding="utf-8")
        return wav_path
