"""默认本地 TTS：Kokoro（本地中文版 Kokoro-82M-v1.1-zh，离线）。is_local=True。

对齐官方 samples/make_zh.py 的关键设置：
  - repo_id=hexgrad/Kokoro-82M-v1.1-zh（v1.1 中文 G2P）
  - en_callable 处理句中英文字词
  - speed_callable 按音素长度调速，避免长句「赶字」
  - 保持 24kHz 输出（Kokoro 原生采样率，不再压到 16k）
  - 合成前清理 LLM Markdown，按句分段合成
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from core.device import resolve_device
from core.tts.base import TTSBase

__all__ = ["KokoroTTS", "prepare_tts_text"]

_KOKORO_SR = 24000
_WEIGHTS_NAME = "kokoro-v1_1-zh.pth"
_DEFAULT_REPO = "hexgrad/Kokoro-82M-v1.1-zh"
_SENTENCE_GAP_SAMPLES = 2400  # 24kHz 下约 0.1s 句间停顿


def prepare_tts_text(text: str) -> str:
    """去掉 LLM 常见 Markdown，避免 TTS 朗读符号。"""
    if not text:
        return ""
    t = text
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-*]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\d+\.\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _speed_callable(len_ps: int) -> float:
    """官方 make_zh.py：长句略减速，减轻咬字糊、赶字。"""
    speed = 0.8
    if len_ps <= 83:
        speed = 1.0
    elif len_ps < 183:
        speed = 1.0 - (len_ps - 83) / 500
    return speed * 1.1


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*", text)
    return [p.strip() for p in parts if p.strip()]


class KokoroTTS(TTSBase):
    is_local = True
    SAMPLE_RATE = _KOKORO_SR

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_dir = config.get("model_dir", "models/tts/kokoro")
        self.repo_id = config.get("repo_id", _DEFAULT_REPO)
        self.voice = config.get("voice", "zf_001")
        self.speed = config.get("speed", 1.0)
        self.device = resolve_device(config.get("device"))
        self._pipeline = None
        self._en_pipeline = None
        self._voice_path = None

    def _ensure(self):
        if self._pipeline is not None:
            return
        try:
            from kokoro import KModel, KPipeline
        except ImportError as e:
            raise ImportError(
                "KokoroTTS 需要 kokoro + misaki[zh]（可选语音依赖）。"
                ' 安装：pip install "kokoro>=0.9.4" "misaki[zh]>=0.9.4" soundfile，'
                f"模型已在 {self.model_dir}（本地）。或用 main.py --text 文本模式。"
            ) from e
        d = Path(self.model_dir)
        config_path = d / "config.json"
        weights_path = d / _WEIGHTS_NAME
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"未找到本地 Kokoro 模型：{config_path} / {weights_path}。"
                "请下载 hexgrad/Kokoro-82M-v1.1-zh 到 " + str(d)
            )
        model = KModel(
            repo_id=self.repo_id,
            config=str(config_path),
            model=str(weights_path),
        )
        model = model.to(self.device).eval()
        self._en_pipeline = KPipeline(lang_code="a", repo_id=self.repo_id, model=False)
        self._pipeline = KPipeline(
            lang_code="z",
            repo_id=self.repo_id,
            model=model,
            en_callable=self._en_callable,
        )
        vp = d / "voices" / f"{self.voice}.pt"
        if not vp.exists():
            raise FileNotFoundError(f"未找到音色：{vp}")
        self._voice_path = str(vp)

    def _en_callable(self, word: str) -> str:
        if self._en_pipeline is None:
            return word
        return next(self._en_pipeline(word)).phonemes

    def _resolve_speed(self) -> float | Callable[[int], float]:
        if self.speed in (None, "auto"):
            return _speed_callable
        return float(self.speed)

    def synthesize(self, text: str) -> bytes:
        self._ensure()
        import numpy as np

        clean = prepare_tts_text(text)
        if not clean:
            return b""
        sentences = _split_sentences(clean) or [clean]
        speed = self._resolve_speed()
        chunks: list[np.ndarray] = []
        for i, sentence in enumerate(sentences):
            for _gs, _ps, audio in self._pipeline(
                sentence, voice=self._voice_path, speed=speed
            ):
                chunks.append(_to_numpy(audio))
            if i < len(sentences) - 1 and chunks:
                chunks.append(np.zeros(_SENTENCE_GAP_SAMPLES, dtype=np.float32))
        if not chunks:
            return b""
        wave = np.concatenate(chunks).astype(np.float32)
        pcm16 = np.clip(wave, -1.0, 1.0)
        return (pcm16 * 32767.0).astype(np.int16).tobytes()


def _to_numpy(audio):
    import numpy as np

    if isinstance(audio, np.ndarray):
        return audio.reshape(-1).astype(np.float32)
    return audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
