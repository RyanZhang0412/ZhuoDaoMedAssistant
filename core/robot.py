"""对话协调层 Robot —— 串联语音/文本输入 → Agent/决策 → 回复/TTS。

按交叉验证结论收口：
  - Agent 对外稳定名为 Agent（= LocalAgent 别名，见 agent/__init__.py）。
  - 推荐结果类型统一为 RecommendationResult（不另立 RehabRecommendation）。
  - 会话历史单一所有者 = Robot 持有 DialogueMemory；调用 Agent.chat(query, history=...)，
    Agent 自身不持久化历史。
  - 推荐 + 解释 + guard 只经 Recommender.recommend(record, explain=True) 一次；
    Robot 不单独再调 explainer。

文本回路 handle_text 可独立运行；语音模式在此基础上叠加：
  麦克风采集 -> VAD 切句 -> ASR -> Agent -> TTS -> 扬声器播放。
"""

from __future__ import annotations

import enum
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型提示，避免运行时强依赖
    from agent.agent import LocalAgent
    from core.asr.base import ASRBase
    from core.tts.base import TTSBase
    from core.vad.base import VADBase
    from medical.service import RecommendationResult
    from memory.dialogue_memory import DialogueMemory

__all__ = ["Robot", "RobotResponse", "DialogueState"]


class DialogueState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class RobotResponse:
    """单轮处理结果。"""

    text: str
    audio: bytes | None = None
    recommendation: "RecommendationResult | None" = None
    source: str = "agent"  # 'agent' | 'rule' | 'chat'


class Robot:
    """对话协调层。文本/语音两种入口的统一编排者。"""

    def __init__(
        self,
        config: dict,
        *,
        agent: "LocalAgent",
        memory: "DialogueMemory | None" = None,
        asr: "ASRBase | None" = None,
        vad: "VADBase | None" = None,
        tts: "TTSBase | None" = None,
    ) -> None:
        self.config = config
        self.agent = agent
        self.memory = memory
        self.asr = asr
        self.vad = vad
        self.tts = tts
        self.state = DialogueState.IDLE
        self._play_stop_event = threading.Event()
        self._play_thread: threading.Thread | None = None

    # ---- 文本一问一答（server/CLI 复用，无麦克风也能跑） ----
    def handle_text(self, text: str, *, patient_id: str | None = None, session_id: str = "default") -> RobotResponse:
        self.state = DialogueState.THINKING
        history = (
            self.memory.context_for_llm(session_id) if self.memory is not None else None
        )
        reply = self.agent.chat(text, history=history, patient_id=patient_id)
        if self.memory is not None:
            self.memory.append(session_id, "user", text)
            self.memory.append(session_id, "assistant", reply)
        self.state = DialogueState.IDLE
        return RobotResponse(text=reply, source="agent")

    # ---- 语音单轮：识别 -> 文本回路 -> 合成 ----
    def handle_audio(self, audio: bytes, *, patient_id: str | None = None, session_id: str = "default") -> RobotResponse:
        if self.asr is None:
            raise RuntimeError("未装配 ASR，无法处理语音；请用 handle_text 或配置语音模块")
        self.state = DialogueState.LISTENING
        asr_result = self.asr.transcribe(audio)
        resp = self.handle_text(asr_result.text, patient_id=patient_id, session_id=session_id)
        if self.tts is not None:
            self.state = DialogueState.SPEAKING
            resp.audio = self.tts.synthesize(resp.text)
        self.state = DialogueState.IDLE
        return resp

    # ---- 实时语音循环：麦克风 -> VAD 切句 -> ASR -> Agent -> TTS -> 播放 ----
    def run_voice_session(self, *, patient_id: str | None = None, session_id: str = "default") -> None:
        if self.asr is None or self.vad is None:
            raise RuntimeError("语音模式需要同时装配 ASR + VAD")

        from core.audio_io import MicStream, describe_input_device, describe_output_device, play_pcm

        voice_cfg = self.config.get("voice_loop", {})
        pre_speech_frames = int(voice_cfg.get("pre_speech_frames", 4))
        min_speech_frames = int(voice_cfg.get("min_speech_frames", 6))
        silence_frames_end = int(voice_cfg.get("silence_frames_end", 20))
        max_utterance_frames = int(voice_cfg.get("max_utterance_frames", 500))
        interrupt_on_speech = bool(voice_cfg.get("interrupt_on_speech", True))
        debug_voice = bool(voice_cfg.get("debug", True))
        debug_frame_interval = max(int(voice_cfg.get("debug_frame_interval", 20)), 1)
        debug_level_threshold = max(int(voice_cfg.get("debug_level_threshold", 150)), 0)
        input_gain = max(float(voice_cfg.get("input_gain", 1.0)), 0.0)
        input_device = voice_cfg.get("input_device")
        record_enabled = bool(voice_cfg.get("record", False))
        record_assistant = bool(voice_cfg.get("record_assistant", True))
        record_dir = voice_cfg.get("record_dir", "data/recordings")

        pre_roll: deque[bytes] = deque(maxlen=max(pre_speech_frames, 1))
        speech_frames: list[bytes] = []
        in_speech = False
        speech_count = 0
        silence_run = 0
        frame_index = 0
        last_is_speech = False
        last_audio_hint_frame = -debug_frame_interval

        def _debug(message: str) -> None:
            if debug_voice:
                print(f"[voice-debug] {message}")

        def _frame_stats(frame: bytes) -> tuple[int, int]:
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

        def _start_playback(pcm: bytes) -> None:
            if not pcm:
                return
            self.interrupt()
            self._play_stop_event = threading.Event()
            self.state = DialogueState.SPEAKING
            _debug(f"start playback: output={describe_output_device()} bytes={len(pcm)}")
            sr = getattr(self.tts, "SAMPLE_RATE", 16000)
            self._play_thread = threading.Thread(
                target=play_pcm,
                kwargs={"pcm": pcm, "sample_rate": sr, "stop_event": self._play_stop_event},
                daemon=True,
                name="robot-tts-playback",
            )
            self._play_thread.start()

        print("=== 卓道康复助手（语音模式）===")
        print("开始监听。直接说话即可，Ctrl+C 退出。\n")
        if patient_id:
            print(f"[当前聚焦患者: {patient_id}]\n")
        _debug(f"input device: {describe_input_device()}")
        if self.tts is not None:
            _debug(f"output device: {describe_output_device()}")
        _debug(
            "voice cfg: "
            f"vad=SileroVAD threshold={getattr(self.vad, 'threshold', '?')}, "
            f"input_gain={input_gain}, interrupt_on_speech={interrupt_on_speech}, "
            f"pre_speech_frames={pre_speech_frames}, "
            f"min_speech_frames={min_speech_frames}, "
            f"silence_frames_end={silence_frames_end}, "
            f"max_utterance_frames={max_utterance_frames}"
        )

        recorder = None
        if record_enabled:
            from core.audio_io import SAMPLE_RATE
            from core.voice_recorder import VoiceSessionRecorder

            recorder = VoiceSessionRecorder(record_dir, session_id, user_sample_rate=SAMPLE_RATE)
            _debug(f"recording enabled: {recorder.session_dir}")

        self.state = DialogueState.LISTENING
        with MicStream(input_gain=input_gain, device=input_device) as mic:
            _debug("microphone stream opened")
            for frame in mic.frames():
                frame_index += 1
                self._reap_playback()

                is_speech = self.vad.is_speech(frame)
                level, peak = _frame_stats(frame)
                if is_speech != last_is_speech:
                    _debug(
                        f"vad -> {'speech' if is_speech else 'silence'} "
                        f"frame={frame_index} state={self.state.value} level={level} peak={peak}"
                    )
                elif level >= debug_level_threshold and not is_speech and (
                    frame_index - last_audio_hint_frame >= debug_frame_interval
                ):
                    _debug(
                        f"mic has audio but VAD kept silence: "
                        f"frame={frame_index} state={self.state.value} level={level} peak={peak}"
                    )
                    last_audio_hint_frame = frame_index
                elif frame_index % (debug_frame_interval * 3) == 0 and not in_speech:
                    _debug(f"heartbeat frame={frame_index} state={self.state.value} level={level} peak={peak}")
                last_is_speech = is_speech

                if self.state == DialogueState.SPEAKING and is_speech and interrupt_on_speech:
                    _debug("barge-in detected; interrupt playback")
                    self.interrupt()

                if not in_speech:
                    pre_roll.append(frame)
                    if not is_speech:
                        continue
                    if self.state == DialogueState.SPEAKING and not interrupt_on_speech:
                        continue
                    in_speech = True
                    speech_count = 1
                    silence_run = 0
                    speech_frames = list(pre_roll)
                    self.state = DialogueState.LISTENING
                    _debug(
                        f"speech_start frame={frame_index} pre_roll_frames={len(pre_roll)} "
                        f"level={level} peak={peak}"
                    )
                    continue

                speech_frames.append(frame)
                if is_speech:
                    speech_count += 1
                    silence_run = 0
                else:
                    silence_run += 1

                reached_end = silence_run >= silence_frames_end
                reached_limit = len(speech_frames) >= max_utterance_frames
                if not reached_end and not reached_limit:
                    continue

                if reached_end and silence_run > 0:
                    utterance_frames = speech_frames[:-silence_run] or speech_frames
                else:
                    utterance_frames = speech_frames
                utterance = b"".join(utterance_frames)
                final_speech_count = speech_count
                _debug(
                    f"speech_end frame={frame_index} speech_frames={final_speech_count} "
                    f"buffer_frames={len(utterance_frames)} bytes={len(utterance)}"
                )

                in_speech = False
                speech_frames = []
                speech_count = 0
                silence_run = 0
                pre_roll.clear()

                if final_speech_count < min_speech_frames or not utterance:
                    _debug(
                        f"drop utterance: speech_frames={final_speech_count}, "
                        f"min_required={min_speech_frames}, bytes={len(utterance)}"
                    )
                    self.state = DialogueState.LISTENING
                    continue

                self.state = DialogueState.THINKING
                _debug(f"transcribe start: bytes={len(utterance)}")
                asr_result = self.asr.transcribe(utterance)
                text = asr_result.text.strip()
                if not text:
                    _debug("ASR returned empty text")
                    self.state = DialogueState.LISTENING
                    continue

                print(f"你 > {text}")
                if recorder is not None:
                    user_wav = recorder.save_user(utterance, text)
                    _debug(f"saved user audio: {user_wav}")

                resp = self.handle_text(text, patient_id=patient_id, session_id=session_id)
                print(f"助手 > {resp.text}\n")

                if self.tts is not None:
                    resp.audio = self.tts.synthesize(resp.text)
                    if recorder is not None and record_assistant and resp.audio:
                        asst_wav = recorder.save_assistant(
                            resp.audio, resp.text, getattr(self.tts, "SAMPLE_RATE", 16000)
                        )
                        _debug(f"saved assistant audio: {asst_wav}")
                    _start_playback(resp.audio)
                else:
                    self.state = DialogueState.LISTENING

    def _reap_playback(self) -> None:
        if self._play_thread is not None and not self._play_thread.is_alive():
            self._play_thread.join(timeout=0.01)
            self._play_thread = None
            if self.state == DialogueState.SPEAKING:
                self.state = DialogueState.LISTENING

    def interrupt(self) -> None:
        """barge-in：停止当前播报，回到聆听。"""
        self._play_stop_event.set()
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None
        self.state = DialogueState.INTERRUPTED

    def shutdown(self) -> None:
        self.interrupt()
        for comp in (self.asr, self.tts):
            if comp is not None:
                comp.close()
