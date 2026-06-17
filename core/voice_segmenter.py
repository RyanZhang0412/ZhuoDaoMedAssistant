"""VAD 切句状态机 —— 统一句首回溯、电平辅助、播报打断（barge-in）与句尾判定。

MicStream 每帧调用 VoiceSegmenter.push()；整句就绪时返回 PCM，由 Robot 送 ASR。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VoiceSegmenter",
    "VoiceSegmenterConfig",
    "SegmentPushResult",
    "frame_peak_stats",
]

# 16kHz / 512 samples ≈ 32ms per frame（与 core.audio_io.FRAME_SAMPLES 一致）
FRAME_MS = 32


def frame_peak_stats(frame: bytes) -> tuple[int, int]:
    """返回 (mean_abs, peak_abs) 16-bit PCM 电平。"""
    if not frame:
        return 0, 0
    try:
        samples = memoryview(frame).cast("h")
    except TypeError:
        return 0, 0
    total = 0
    peak = 0
    count = 0
    for sample in samples:
        amp = -sample if sample < 0 else sample
        total += amp
        count += 1
        if amp > peak:
            peak = amp
    mean = int(total / count) if count else 0
    return mean, peak


@dataclass(frozen=True)
class VoiceSegmenterConfig:
    """voice_loop 切句参数；未显式配置时从 VAD.min_*_ms 推导默认帧数。"""

    pre_speech_frames: int = 6
    barge_prefetch_frames: int = 12
    min_speech_frames: int = 5
    silence_frames_end: int = 20
    max_utterance_frames: int = 500
    interrupt_on_speech: bool = True
    interrupt_min_speech_frames: int = 3
    # 打断播报是否只认 VAD 确认的语音帧（不被纯能量峰值噪音误触发）。
    # True：barge-in 必须是 VAD speech；能量峰值仅用于非播报时的句首抢字。
    interrupt_require_vad: bool = True
    energy_onset_peak: int = 1500
    energy_tail_ratio: float = 0.45
    energy_tail_min_peak: int = 400
    onset_max_frames: int = 30
    onset_grace_silence_frames: int = 5
    post_barge_silence_frames: int = 12
    barge_streak_reset_silence: int = 2

    @property
    def energy_tail_peak(self) -> int:
        if self.energy_onset_peak <= 0:
            return self.energy_tail_min_peak
        return max(int(self.energy_onset_peak * self.energy_tail_ratio), self.energy_tail_min_peak)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VoiceSegmenterConfig:
        voice_cfg = config.get("voice_loop", {})
        vad_key = config.get("selected_module", {}).get("VAD", "SileroVAD")
        vad_cfg = config.get("VAD", {}).get(vad_key, {})

        default_silence = max(int(vad_cfg.get("min_silence_ms", 500) / FRAME_MS), 1)
        default_min_speech = max(int(vad_cfg.get("min_speech_ms", 250) / FRAME_MS), 1)

        pre_speech = max(int(voice_cfg.get("pre_speech_frames", 4)), 1)
        interrupt_min = max(int(voice_cfg.get("interrupt_min_speech_frames", 5)), 1)

        raw_barge_prefetch = voice_cfg.get("barge_prefetch_frames")
        if raw_barge_prefetch is None:
            barge_prefetch = max(pre_speech * 2 + interrupt_min, 12)
        else:
            barge_prefetch = max(int(raw_barge_prefetch), 1)

        energy_onset = max(int(voice_cfg.get("energy_onset_peak", 1500)), 0)
        onset_max = max(int(voice_cfg.get("onset_max_frames", 30)), pre_speech)

        return cls(
            pre_speech_frames=pre_speech,
            barge_prefetch_frames=barge_prefetch,
            min_speech_frames=int(voice_cfg.get("min_speech_frames", default_min_speech)),
            silence_frames_end=int(voice_cfg.get("silence_frames_end", default_silence)),
            max_utterance_frames=max(int(voice_cfg.get("max_utterance_frames", 500)), 1),
            interrupt_on_speech=bool(voice_cfg.get("interrupt_on_speech", True)),
            interrupt_min_speech_frames=interrupt_min,
            interrupt_require_vad=bool(voice_cfg.get("interrupt_require_vad", True)),
            energy_onset_peak=energy_onset,
            energy_tail_ratio=float(voice_cfg.get("energy_tail_ratio", 0.45)),
            energy_tail_min_peak=max(int(voice_cfg.get("energy_tail_min_peak", 400)), 0),
            onset_max_frames=onset_max,
            onset_grace_silence_frames=max(
                int(voice_cfg.get("onset_grace_silence_frames", 5)), 1
            ),
            post_barge_silence_frames=max(
                int(voice_cfg.get("post_barge_silence_frames", 12)), 0
            ),
            barge_streak_reset_silence=max(
                int(voice_cfg.get("barge_streak_reset_silence", 2)), 1
            ),
        )

    def debug_summary(self) -> str:
        return (
            f"pre_speech_frames={self.pre_speech_frames}, "
            f"barge_prefetch_frames={self.barge_prefetch_frames}, "
            f"min_speech_frames={self.min_speech_frames}, "
            f"silence_frames_end={self.silence_frames_end}, "
            f"interrupt_min_speech_frames={self.interrupt_min_speech_frames}, "
            f"interrupt_require_vad={self.interrupt_require_vad}, "
            f"energy_onset_peak={self.energy_onset_peak}, "
            f"onset_max_frames={self.onset_max_frames}, "
            f"post_barge_silence_frames={self.post_barge_silence_frames}, "
            f"max_utterance_frames={self.max_utterance_frames}"
        )


@dataclass
class SegmentPushResult:
    """单帧 push 的副作用与产出。"""

    utterance: bytes | None = None
    speech_frame_count: int = 0
    dropped_short: bool = False
    barge_in: bool = False
    events: list[str] = field(default_factory=list)


class VoiceSegmenter:
    """麦克风帧 → 整句 PCM；句首/打断/句尾逻辑集中在此。"""

    def __init__(self, cfg: VoiceSegmenterConfig) -> None:
        self.cfg = cfg
        self._ring: list[bytes] = []
        self._collecting = False
        self._frames: list[bytes] = []
        self._speech_count = 0
        self._silence_run = 0
        self._after_barge = False
        self._voice_streak = 0
        self._silence_streak = 0
        self._pending = False
        self._pending_frames: list[bytes] = []
        self._pending_speech_count = 0
        self._pending_low_energy = 0

    @property
    def collecting(self) -> bool:
        return self._collecting

    def on_playback_stopped(self) -> None:
        """播报结束：清打断计数与预取环，避免漏音影响下一句。"""
        self._voice_streak = 0
        self._silence_streak = 0
        if not self._collecting:
            self._ring.clear()

    def push(
        self,
        frame: bytes,
        *,
        is_speech: bool,
        peak: int,
        assistant_playing: bool,
    ) -> SegmentPushResult:
        result = SegmentPushResult()
        if self._collecting:
            return self._extend_collection(frame, is_speech, peak, result)

        self._append_ring(frame, assistant_playing)

        if assistant_playing:
            return self._handle_playback(frame, is_speech, peak, result)

        if self._voice_active(is_speech, peak):
            return self._handle_voice_onset(frame, is_speech, result)

        if self._pending:
            self._pending_low_energy += 1
            if self._pending_low_energy <= self.cfg.onset_grace_silence_frames:
                self._pending_frames.append(frame)
            else:
                self._clear_pending()
        return result

    def _voice_active(self, is_speech: bool, peak: int) -> bool:
        return is_speech or (
            self.cfg.energy_onset_peak > 0 and peak >= self.cfg.energy_onset_peak
        )

    def _ring_max(self, assistant_playing: bool) -> int:
        return (
            self.cfg.barge_prefetch_frames
            if assistant_playing
            else self.cfg.pre_speech_frames
        )

    def _append_ring(self, frame: bytes, assistant_playing: bool) -> None:
        self._ring.append(frame)
        limit = self._ring_max(assistant_playing)
        if len(self._ring) > limit:
            self._ring = self._ring[-limit:]

    def _handle_playback(
        self, frame: bytes, is_speech: bool, peak: int, result: SegmentPushResult
    ) -> SegmentPushResult:
        # 打断播报时默认只认 VAD 语音，避免高能量噪音误触发 barge-in。
        voice = (
            is_speech
            if self.cfg.interrupt_require_vad
            else self._voice_active(is_speech, peak)
        )
        if voice:
            self._silence_streak = 0
            self._voice_streak += 1
            if (
                self.cfg.interrupt_on_speech
                and self._voice_streak >= self.cfg.interrupt_min_speech_frames
            ):
                self._begin_collection(
                    list(self._ring),
                    speech_count=max(self._voice_streak, 1),
                    after_barge=True,
                )
                result.barge_in = True
                result.events.append(
                    f"barge-in ({self._voice_streak} voice frames, "
                    f"prefetch={len(self._frames)})"
                )
        else:
            self._silence_streak += 1
            if self._silence_streak >= self.cfg.barge_streak_reset_silence:
                self._voice_streak = 0
        return result

    def _handle_voice_onset(
        self, frame: bytes, is_speech: bool, result: SegmentPushResult
    ) -> SegmentPushResult:
        if not self._pending:
            self._pending = True
            self._pending_frames = list(self._ring)
            self._pending_speech_count = 1 if is_speech else 0
        else:
            self._pending_frames.append(frame)
            if is_speech:
                self._pending_speech_count += 1
            if len(self._pending_frames) > self.cfg.onset_max_frames:
                self._pending_frames = self._pending_frames[-self.cfg.onset_max_frames :]
        self._pending_low_energy = 0

        if is_speech:
            self._begin_collection(
                list(self._pending_frames),
                speech_count=max(self._pending_speech_count, 1),
                after_barge=False,
            )
            result.events.append(f"speech_start prefetch={len(self._frames)}")
        return result

    def _begin_collection(
        self, frames: list[bytes], *, speech_count: int, after_barge: bool
    ) -> None:
        self._collecting = True
        self._frames = frames
        self._speech_count = speech_count
        self._silence_run = 0
        self._after_barge = after_barge
        self._clear_pending()
        self._ring.clear()
        self._voice_streak = 0
        self._silence_streak = 0

    def _clear_pending(self) -> None:
        self._pending = False
        self._pending_frames = []
        self._pending_speech_count = 0
        self._pending_low_energy = 0

    def _extend_collection(
        self, frame: bytes, is_speech: bool, peak: int, result: SegmentPushResult
    ) -> SegmentPushResult:
        self._frames.append(frame)
        if is_speech:
            self._speech_count += 1
            self._silence_run = 0
        elif self.cfg.energy_onset_peak > 0 and peak >= self.cfg.energy_tail_peak:
            pass
        else:
            self._silence_run += 1

        silence_end = self.cfg.silence_frames_end + (
            self.cfg.post_barge_silence_frames if self._after_barge else 0
        )
        reached_end = self._silence_run >= silence_end
        reached_limit = len(self._frames) >= self.cfg.max_utterance_frames
        if not reached_end and not reached_limit:
            return result

        if reached_end and self._silence_run > 0:
            utterance_frames = self._frames[: -self._silence_run] or self._frames
        else:
            utterance_frames = self._frames
        utterance = b"".join(utterance_frames)
        speech_count = self._speech_count
        silence_run = self._silence_run

        self._reset_collection()

        if speech_count < self.cfg.min_speech_frames or not utterance:
            result.dropped_short = True
            result.events.append(
                f"drop utterance speech_frames={speech_count} "
                f"min={self.cfg.min_speech_frames} bytes={len(utterance)}"
            )
            return result

        result.utterance = utterance
        result.speech_frame_count = speech_count
        result.events.append(
            f"speech_end frames={speech_count} bytes={len(utterance)} "
            f"silence_run={silence_run}/{silence_end}"
        )
        return result

    def _reset_collection(self) -> None:
        self._collecting = False
        self._frames = []
        self._speech_count = 0
        self._silence_run = 0
        self._after_barge = False
        self._ring.clear()
