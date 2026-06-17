"""VoiceSegmenter 切句状态机单测。"""

import struct

from core.voice_segmenter import VoiceSegmenter, VoiceSegmenterConfig


def _frame(peak: int = 0) -> bytes:
    if peak == 0:
        return b"\x00\x00" * 512
    return struct.pack("<h", peak) * 512


def _cfg(**overrides) -> VoiceSegmenterConfig:
    base = dict(
        pre_speech_frames=4,
        barge_prefetch_frames=10,
        min_speech_frames=2,
        silence_frames_end=3,
        max_utterance_frames=100,
        interrupt_on_speech=True,
        interrupt_min_speech_frames=2,
        energy_onset_peak=1500,
        energy_tail_ratio=0.45,
        energy_tail_min_peak=400,
        onset_max_frames=20,
        onset_grace_silence_frames=2,
        post_barge_silence_frames=0,
        barge_streak_reset_silence=2,
    )
    base.update(overrides)
    return VoiceSegmenterConfig(**base)


def _finish_utterance(seg: VoiceSegmenter, *, speech_frames: int = 3, silence_frames: int = 3) -> bytes:
    pcm = b""
    for _ in range(speech_frames):
        r = seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=False)
        assert r.utterance is None
    for _ in range(silence_frames):
        r = seg.push(_frame(), is_speech=False, peak=0, assistant_playing=False)
        if r.utterance is not None:
            return r.utterance
    raise AssertionError("utterance not finalized")


def test_idle_vad_speech_start_and_end():
    seg = VoiceSegmenter(_cfg())
    for _ in range(3):
        seg.push(_frame(), is_speech=False, peak=0, assistant_playing=False)
    seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=False)
    utterance = _finish_utterance(seg, speech_frames=2, silence_frames=3)
    assert len(utterance) > 0


def test_energy_onset_before_vad_confirms():
    seg = VoiceSegmenter(_cfg())
    for _ in range(3):
        seg.push(_frame(), is_speech=False, peak=0, assistant_playing=False)
    seg.push(_frame(peak=2000), is_speech=False, peak=2000, assistant_playing=False)
    seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=False)
    utterance = _finish_utterance(seg, speech_frames=2, silence_frames=3)
    assert len(utterance) >= 512 * 2 * 2


def test_barge_in_triggers_and_collects_prefetch():
    seg = VoiceSegmenter(_cfg(interrupt_min_speech_frames=2))
    seg.push(_frame(), is_speech=False, peak=0, assistant_playing=True)
    r1 = seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=True)
    assert not r1.barge_in
    r2 = seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=True)
    assert r2.barge_in
    assert seg.collecting
    utterance = _finish_utterance(seg, speech_frames=1, silence_frames=3)
    assert len(utterance) > 0


def test_energy_noise_does_not_barge_in_when_require_vad():
    """高能量噪音（VAD=silence）默认不应打断播报。"""
    seg = VoiceSegmenter(_cfg(interrupt_min_speech_frames=2, interrupt_require_vad=True))
    seg.push(_frame(), is_speech=False, peak=0, assistant_playing=True)
    for _ in range(5):
        r = seg.push(_frame(peak=3000), is_speech=False, peak=3000, assistant_playing=True)
        assert not r.barge_in
    assert not seg.collecting


def test_energy_noise_barges_in_when_require_vad_disabled():
    """关掉 require_vad 时退回旧行为：高能量峰值即可打断。"""
    seg = VoiceSegmenter(_cfg(interrupt_min_speech_frames=2, interrupt_require_vad=False))
    seg.push(_frame(), is_speech=False, peak=0, assistant_playing=True)
    seg.push(_frame(peak=3000), is_speech=False, peak=3000, assistant_playing=True)
    r = seg.push(_frame(peak=3000), is_speech=False, peak=3000, assistant_playing=True)
    assert r.barge_in


def test_post_barge_extra_silence_before_end():
    seg = VoiceSegmenter(_cfg(silence_frames_end=2, post_barge_silence_frames=4, min_speech_frames=1))
    seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=True)
    seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=True)
    utterance = None
    for _ in range(8):
        r = seg.push(_frame(), is_speech=False, peak=0, assistant_playing=False)
        if r.utterance is not None:
            utterance = r.utterance
            break
    assert utterance is not None


def test_drop_short_utterance():
    seg = VoiceSegmenter(_cfg(min_speech_frames=5))
    seg.push(_frame(peak=2000), is_speech=True, peak=2000, assistant_playing=False)
    dropped = False
    for _ in range(6):
        r = seg.push(_frame(), is_speech=False, peak=0, assistant_playing=False)
        if r.dropped_short:
            dropped = True
            break
    assert dropped


def test_config_from_yaml_defaults():
    cfg = VoiceSegmenterConfig.from_config(
        {
            "selected_module": {"VAD": "SileroVAD"},
            "VAD": {"SileroVAD": {"min_silence_ms": 650, "min_speech_ms": 150}},
            "voice_loop": {"pre_speech_frames": 6, "interrupt_min_speech_frames": 3},
        }
    )
    assert cfg.pre_speech_frames == 6
    assert cfg.silence_frames_end == 20  # 650/32
    assert cfg.min_speech_frames == 4  # 150/32
    assert cfg.barge_prefetch_frames == 15  # 6*2+3
