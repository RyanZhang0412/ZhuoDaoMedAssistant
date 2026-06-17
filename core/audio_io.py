"""音频 IO —— 麦克风采集 + 扬声器播放（sounddevice，16k 单声道 16-bit PCM）。

全系统统一 16kHz 单声道 16-bit PCM，与 ASR/VAD/TTS 约定一致。
未装 sounddevice 时给出清晰提示；语音为可选模块。

- MicStream: 流式读麦克风，按固定窗口（默认 512 采样 = silero 帧长）产出 PCM 帧。
- play_pcm: 阻塞播放一段 PCM；支持中途打断（stop_event）。
- describe_input_device / describe_output_device: 打印当前默认输入/输出设备，便于排查。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

__all__ = [
    "MicStream",
    "play_pcm",
    "apply_input_gain",
    "describe_input_device",
    "describe_output_device",
    "SAMPLE_RATE",
    "FRAME_SAMPLES",
]

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512          # silero v5 在 16k 下的帧长
_BYTES_PER_SAMPLE = 2        # 16-bit


def _require_sd():
    try:
        import sounddevice as sd  # 延迟导入
    except ImportError as e:
        raise ImportError(
            "实时语音需要 sounddevice（可选语音依赖）。"
            "安装：pip install sounddevice。或用 main.py --text 文本模式。"
        ) from e
    return sd


def _describe_default_device(kind: str) -> str:
    sd = _require_sd()
    try:
        default_in, default_out = sd.default.device
        index = default_in if kind == "input" else default_out
    except Exception:
        return "default (unresolved)"
    if index is None:
        return "default (unresolved)"
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return f"default ({index})"
    if idx < 0:
        return f"default ({idx}, unresolved)"
    try:
        info = sd.query_devices(idx)
    except Exception:
        return f"#{idx}"

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    name = info.get("name") or f"device-{idx}"
    channels = info.get(channel_key, "?")
    return f"#{idx} {name} (channels={channels})"


def describe_input_device() -> str:
    """返回当前默认麦克风设备描述。"""
    return _describe_default_device("input")


def describe_output_device() -> str:
    """返回当前默认扬声器设备描述。"""
    return _describe_default_device("output")


class MicStream:
    """麦克风流式采集，迭代产出 FRAME_SAMPLES 大小的 16-bit PCM 帧。

    用法：
        with MicStream(input_gain=2.0) as mic:
            for frame in mic.frames():   # 每帧 512 采样的 bytes
                ...
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        *,
        input_gain: float = 1.0,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.input_gain = max(float(input_gain), 0.0)
        self.device = device
        self._stream = None

    def __enter__(self) -> "MicStream":
        sd = _require_sd()
        kwargs: dict = {
            "samplerate": self.sample_rate,
            "blocksize": self.frame_samples,
            "channels": 1,
            "dtype": "int16",
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self._stream = sd.RawInputStream(**kwargs)
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self) -> Iterator[bytes]:
        """持续产出 PCM 帧（阻塞读）。外层负责何时 break。"""
        assert self._stream is not None, "MicStream 需在 with 块内使用"
        gain = self.input_gain
        while True:
            data, _overflowed = self._stream.read(self.frame_samples)
            frame = bytes(data)
            if gain != 1.0:
                frame = apply_input_gain(frame, gain)
            yield frame

    def flush(self) -> int:
        """丢弃驱动缓冲里积压的帧（thinking 期间的漏音/底噪），返回丢弃帧数。"""
        assert self._stream is not None, "MicStream 需在 with 块内使用"
        dropped = 0
        try:
            while self._stream.read_available >= self.frame_samples:
                self._stream.read(self.frame_samples)
                dropped += 1
        except Exception:
            pass
        return dropped


def apply_input_gain(pcm: bytes, gain: float) -> bytes:
    """软件增益：放大麦克风 PCM，并限幅到 int16 范围。"""
    if not pcm or gain == 1.0:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768.0, 32767.0).astype(np.int16)
    return samples.tobytes()


def play_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE, stop_event: threading.Event | None = None) -> None:
    """播放一段 16-bit PCM。stop_event 置位时中途停止（barge-in）。"""
    if not pcm:
        return
    sd = _require_sd()
    import numpy as np

    wave = np.frombuffer(pcm, dtype=np.int16)
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()
    try:
        chunk = sample_rate // 10  # 100ms 一块，便于及时响应打断
        for i in range(0, len(wave), chunk):
            if stop_event is not None and stop_event.is_set():
                break
            stream.write(wave[i : i + chunk])
    finally:
        stream.stop()
        stream.close()
