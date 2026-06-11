"""默认本地 ASR：SenseVoice（FunASR 本地模型，离线）。

从 config['model_dir'] 加载本地权重（disable_update=True 禁止联网）。
transcribe 接收 16k 单声道 16-bit PCM bytes，转 float32 numpy 喂给 FunASR，
用 rich_transcription_postprocess 去掉情感/事件富文本标记，返回纯文本。

语音是可选模块；未装 funasr 时给出清晰提示，文本模式 (main --text) 不依赖它。
"""

from __future__ import annotations

from core.asr.base import ASRBase, ASRResult
from core.device import resolve_device

__all__ = ["SenseVoiceASR"]


class SenseVoiceASR(ASRBase):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_dir = config.get("model_dir", "models/asr/SenseVoiceSmall")
        self.device = resolve_device(config.get("device"))
        self.language = config.get("language", "zh")
        self._model = None  # 懒加载
        self._postprocess = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel  # 延迟导入
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as e:
            raise ImportError(
                "SenseVoiceASR 需要 funasr（可选语音依赖）。"
                "安装：pip install funasr，并把模型放到 "
                f"{self.model_dir}（本地，禁止联网下载）。"
                "或在 config.yaml 不启用语音，用 main.py --text 文本模式。"
            ) from e
        # 本地加载（禁用自动更新/下载：传本地路径 + disable_update）
        self._model = AutoModel(model=self.model_dir, device=self.device, disable_update=True)
        self._postprocess = rich_transcription_postprocess

    def transcribe(self, audio: bytes) -> ASRResult:
        self._ensure_model()
        import numpy as np

        # 16-bit PCM bytes -> float32 [-1,1)（FunASR 期望 16k float32）
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        res = self._model.generate(
            input=pcm,
            cache={},
            language=self.language,
            use_itn=True,            # 标点 + 逆文本规整
            batch_size_s=60,
        )
        raw = res[0]["text"] if res else ""
        text = self._postprocess(raw)  # 去掉 <|HAPPY|> 等富文本标记
        return ASRResult(text=text, is_final=True, language=self.language)
