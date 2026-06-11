"""默认 VAD：silero-vad（本地 onnx 权重，离线）。

silero-vad v5 onnx 签名（已实测）：
  inputs:  input [batch, samples], state [2, batch, 128], sr (int64 标量)
  outputs: output (语音概率), stateN (新状态)
16kHz 下每帧 512 采样（32ms）。is_speech 逐帧推理，概率 > threshold 即判为语音。

语音为可选模块；未装 onnxruntime 时给出清晰提示。
"""

from __future__ import annotations

from core.vad.base import VADBase

__all__ = ["SileroVAD"]

_WINDOW = 512  # 16kHz 下 silero v5 的帧长


class SileroVAD(VADBase):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_path = config.get("model_path", "models/vad/silero_vad.onnx")
        self.threshold = config.get("threshold", 0.5)
        self.sampling_rate = config.get("sampling_rate", 16000)
        self._session = None
        self._state = None  # [2, 1, 128]

    def _ensure(self):
        if self._session is not None:
            return
        try:
            import onnxruntime  # 延迟导入
        except ImportError as e:
            raise ImportError(
                "SileroVAD 需要 onnxruntime（可选语音依赖）。"
                "安装：pip install onnxruntime，并把 silero_vad.onnx 放到 "
                f"{self.model_path}（本地）。或用 main.py --text 文本模式。"
            ) from e
        self._session = onnxruntime.InferenceSession(self.model_path)
        self.reset()

    def reset(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, frame: bytes) -> bool:
        """单帧（512 采样的 16-bit PCM）是否为语音。"""
        self._ensure()
        import numpy as np

        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        # 不足/超出 512 采样则裁剪或补零到窗口
        if len(pcm) < _WINDOW:
            pcm = np.pad(pcm, (0, _WINDOW - len(pcm)))
        elif len(pcm) > _WINDOW:
            pcm = pcm[:_WINDOW]
        x = pcm.reshape(1, -1).astype(np.float32)
        sr = np.array(self.sampling_rate, dtype=np.int64)
        out, new_state = self._session.run(
            None, {"input": x, "state": self._state, "sr": sr}
        )
        self._state = new_state
        prob = float(out[0][0])
        return prob >= self.threshold
