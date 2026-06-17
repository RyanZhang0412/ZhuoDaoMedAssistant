"""对话协调层 Robot —— 串联语音/文本输入 → Agent/决策 → 回复/TTS。

按交叉验证结论收口：
  - Agent 对外稳定名为 Agent（= LocalAgent 别名，见 agent/__init__.py）。
  - 推荐结果类型统一为 RecommendationResult（不另立 RehabRecommendation）。
  - 会话历史单一所有者 = Robot 持有 DialogueMemory；调用 Agent.chat(query, history=...)，
    Agent 自身不持久化历史。
  - 推荐 + 解释 + guard 只经 Recommender.recommend(record, explain=True) 一次；
    Robot 不单独再调 explainer。

文本回路 handle_text 可独立运行；语音模式在此基础上叠加：
  麦克风采集 -> VoiceSegmenter 切句 -> ASR -> Agent 流式输出 -> 按句 TTS 管线 -> 扬声器播放。

低延迟语音管线（对齐百聆 chat_tool + tts_queue 思路）：
  Agent.chat_stream 产出文本增量 -> 按句切分 -> TTSPipeline 两级队列
  （合成第 N+1 句与播放第 N 句重叠），第一句合成完即开播。
barge-in 用连续语音帧门控（interrupt_min_speech_frames），避免耳机漏音误触发。
空闲时检查 Scheduler 到期提醒并主动播报。
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # 仅类型提示，避免运行时强依赖
    from agent.agent import LocalAgent
    from core.asr.base import ASRBase
    from core.tts.base import TTSBase
    from core.vad.base import VADBase
    from medical.service import RecommendationResult
    from memory.dialogue_memory import DialogueMemory
    from memory.long_term import LongTermMemory

__all__ = ["Robot", "RobotResponse", "DialogueState", "split_ready_sentences"]

# 句末标点：凑满一句就送 TTS（顿号/逗号不切，保持语气连贯）
_SENT_END_CHARS = "。！？!?；;\n"


def split_ready_sentences(buf: str, *, min_len: int = 6) -> tuple[list[str], str]:
    """从流式文本缓冲里切出"已完整"的句子，返回 (句子列表, 剩余缓冲)。

    min_len: 句子最短字符数，避免把"好。"这种碎片单独送 TTS（合成开销大、停顿怪）。
    """
    ready: list[str] = []
    start = 0
    for i, ch in enumerate(buf):
        if ch in _SENT_END_CHARS and (i + 1 - start) >= min_len:
            seg = buf[start : i + 1].strip()
            if seg:
                ready.append(seg)
            start = i + 1
    return ready, buf[start:]


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
        long_term: "LongTermMemory | None" = None,
        asr: "ASRBase | None" = None,
        vad: "VADBase | None" = None,
        tts: "TTSBase | None" = None,
    ) -> None:
        self.config = config
        self.agent = agent
        self.memory = memory
        self.long_term = long_term
        self.asr = asr
        self.vad = vad
        self.tts = tts
        self.state = DialogueState.IDLE
        self._play_stop_event = threading.Event()
        self._play_thread: threading.Thread | None = None
        self._pipeline = None  # TTSPipeline，语音会话首次使用时创建
        lt_cfg = config.get("long_term_memory", {})
        self._lt_update_every = max(int(lt_cfg.get("update_every_turns", 8)), 1)
        self._turns_since_lt_update: dict[str, int] = {}

    # ---- 启动预加载（避免首句说话时才加载 SenseVoice/Kokoro 卡顿）----
    def warmup(self) -> None:
        """预加载 ASR / VAD / TTS 权重，并初始化 TTS 播放管线。"""
        if self.vad is not None:
            print("  [预加载] VAD …", flush=True)
            self.vad.warmup()
        if self.asr is not None:
            print("  [预加载] ASR …", flush=True)
            self.asr.warmup()
        if self.tts is not None:
            print("  [预加载] TTS …", flush=True)
            self.tts.warmup()
            if self._pipeline is None:
                from core.tts_pipeline import TTSPipeline

                self._pipeline = TTSPipeline(self.tts)

    # ---- 文本一问一答（server/CLI 复用，无麦克风也能跑） ----
    def handle_text(self, text: str, *, patient_id: str | None = None, session_id: str = "default") -> RobotResponse:
        self.state = DialogueState.THINKING
        history = (
            self.memory.context_for_llm(session_id) if self.memory is not None else None
        )
        reply = self.agent.chat(
            text,
            history=history,
            patient_id=patient_id,
            extra_system=self._long_term_context(session_id),
            session_id=session_id,
        )
        self._after_turn(session_id, text, reply)
        self.state = DialogueState.IDLE
        return RobotResponse(text=reply, source="agent")

    # ---- 文本流式（语音管线 / 流式 REPL 用）----
    def handle_text_stream(
        self, text: str, *, patient_id: str | None = None, session_id: str = "default"
    ) -> Iterator[str]:
        """与 handle_text 等价，但以增量产出回复文本（供按句 TTS 尽早开播）。"""
        self.state = DialogueState.THINKING
        history = (
            self.memory.context_for_llm(session_id) if self.memory is not None else None
        )
        parts: list[str] = []
        for delta in self.agent.chat_stream(
            text,
            history=history,
            patient_id=patient_id,
            extra_system=self._long_term_context(session_id),
            session_id=session_id,
        ):
            parts.append(delta)
            yield delta
        self._after_turn(session_id, text, "".join(parts))

    # ---- 每轮收尾：写对话记忆 + 周期性后台更新长期记忆 ----
    def _after_turn(self, session_id: str, user_text: str, reply: str) -> None:
        if self.memory is None:
            return
        self.memory.append(session_id, "user", user_text)
        self.memory.append(session_id, "assistant", reply)
        if self.long_term is None:
            return
        n = self._turns_since_lt_update.get(session_id, 0) + 1
        if n >= self._lt_update_every:
            self._turns_since_lt_update[session_id] = 0
            history = self.memory.context_for_llm(session_id)
            threading.Thread(
                target=self.long_term.update,
                args=(session_id, history, self.agent.llm),
                daemon=True,
                name="long-term-memory-update",
            ).start()
        else:
            self._turns_since_lt_update[session_id] = n

    def _long_term_context(self, session_id: str) -> str | None:
        if self.long_term is None:
            return None
        return self.long_term.context_block(session_id)

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

        from core.audio_io import MicStream, describe_input_device, describe_output_device
        from core.voice_segmenter import VoiceSegmenter, VoiceSegmenterConfig, frame_peak_stats

        voice_cfg = self.config.get("voice_loop", {})
        seg_cfg = VoiceSegmenterConfig.from_config(self.config)
        tts_min_sentence_chars = max(int(voice_cfg.get("tts_min_sentence_chars", 6)), 1)
        reminder_interval_s = float(voice_cfg.get("reminder_check_interval_s", 30))
        reminder_check_frames = int(reminder_interval_s / 0.032) if reminder_interval_s > 0 else 0
        debug_voice = bool(voice_cfg.get("debug", False))
        debug_frame_interval = max(int(voice_cfg.get("debug_frame_interval", 20)), 1)
        debug_level_threshold = max(int(voice_cfg.get("debug_level_threshold", 150)), 0)
        input_gain = max(float(voice_cfg.get("input_gain", 1.0)), 0.0)
        input_device = voice_cfg.get("input_device")
        record_enabled = bool(voice_cfg.get("record", False))
        record_assistant = bool(voice_cfg.get("record_assistant", True))
        record_dir = voice_cfg.get("record_dir", "data/recordings")

        segmenter = VoiceSegmenter(seg_cfg)
        frame_index = 0
        last_is_speech = False
        last_audio_hint_frame = -debug_frame_interval
        last_reminder_frame = 0
        was_speaking = False

        def _debug(message: str) -> None:
            if debug_voice:
                print(f"[voice-debug] {message}")

        if self.tts is not None and self._pipeline is None:
            from core.tts_pipeline import TTSPipeline

            self._pipeline = TTSPipeline(self.tts)

        def _speaking() -> bool:
            return self._pipeline is not None and self._pipeline.is_active()

        def _check_reminders() -> None:
            """空闲时检查到期康复提醒并主动播报（百聆 TaskManager 的'时间到主动说'）。"""
            try:
                from agent.tools.context import get_context

                scheduler = get_context().scheduler
            except Exception:
                return
            if not hasattr(scheduler, "due_all"):
                return
            for item in scheduler.due_all():
                note = f"康复提醒：患者{item.get('patient_id', '')}，{item.get('content', '')}"
                print(f"助手 > {note}\n")
                if self._pipeline is not None:
                    self._pipeline.submit(note)
                    self._pipeline.end_reply()

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
            f"input_gain={input_gain}, interrupt_on_speech={seg_cfg.interrupt_on_speech}, "
            f"{seg_cfg.debug_summary()}"
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
                if self.state == DialogueState.SPEAKING and not _speaking():
                    self.state = DialogueState.LISTENING

                is_speech = self.vad.is_speech(frame)
                level, peak = frame_peak_stats(frame)
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
                elif (
                    frame_index % (debug_frame_interval * 3) == 0
                    and not segmenter.collecting
                ):
                    _debug(
                        f"heartbeat frame={frame_index} state={self.state.value} "
                        f"level={level} peak={peak}"
                    )
                last_is_speech = is_speech

                speaking_now = _speaking()
                if was_speaking and not speaking_now:
                    segmenter.on_playback_stopped()
                was_speaking = speaking_now

                if (
                    reminder_check_frames
                    and not segmenter.collecting
                    and not speaking_now
                    and frame_index - last_reminder_frame >= reminder_check_frames
                ):
                    last_reminder_frame = frame_index
                    _check_reminders()

                seg_result = segmenter.push(
                    frame,
                    is_speech=is_speech,
                    peak=peak,
                    assistant_playing=speaking_now,
                )
                for event in seg_result.events:
                    _debug(f"{event} frame={frame_index}")

                if seg_result.barge_in:
                    self.interrupt()
                    self.vad.reset()
                    self.state = DialogueState.LISTENING

                if seg_result.dropped_short:
                    self.state = DialogueState.LISTENING
                    continue

                utterance = seg_result.utterance
                if utterance is None:
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
                current_turn = 0
                if recorder is not None:
                    recorder.save_user(utterance, text)
                    current_turn = recorder.turn
                    _debug(f"saved user audio: {recorder.session_dir / f'{current_turn:03d}_user.wav'}")

                buf = ""
                reply_parts: list[str] = []
                first_seg_sent = False
                for delta in self.handle_text_stream(
                    text, patient_id=patient_id, session_id=session_id
                ):
                    reply_parts.append(delta)
                    if self._pipeline is None:
                        continue
                    buf += delta
                    ready, buf = split_ready_sentences(buf, min_len=tts_min_sentence_chars)
                    for seg in ready:
                        if not first_seg_sent:
                            first_seg_sent = True
                            self.state = DialogueState.SPEAKING
                            _debug("first sentence -> tts pipeline (playback starts early)")
                        self._pipeline.submit(seg)
                reply = "".join(reply_parts)
                print(f"助手 > {reply}\n")

                if self._pipeline is not None and reply.strip():
                    if buf.strip():
                        self._pipeline.submit(buf)
                    if recorder is not None and record_assistant and current_turn:
                        turn_for_save = current_turn
                        tts_sr = getattr(self.tts, "SAMPLE_RATE", 16000)
                        record_done = threading.Event()
                        asst_txt = recorder.session_dir / f"{turn_for_save:03d}_assistant.txt"
                        asst_txt.write_text(reply, encoding="utf-8")

                        def _save_assistant(pcm: bytes, _turn=turn_for_save, _text=reply) -> None:
                            try:
                                wav = recorder.save_assistant(
                                    pcm, _text, tts_sr, turn=_turn
                                )
                                _debug(f"saved assistant audio: {wav}")
                            finally:
                                record_done.set()

                        self._pipeline.end_reply(_save_assistant)
                        if not record_done.wait(timeout=120):
                            _debug("assistant recording callback timeout")
                    else:
                        self._pipeline.end_reply()
                    self.state = DialogueState.SPEAKING
                elif recorder is not None and record_assistant and current_turn and reply.strip():
                    asst_txt = recorder.session_dir / f"{current_turn:03d}_assistant.txt"
                    asst_txt.write_text(reply, encoding="utf-8")
                    self.state = DialogueState.LISTENING
                else:
                    self.state = DialogueState.LISTENING

                if not _speaking():
                    dropped = mic.flush()
                    if dropped:
                        _debug(f"flushed {dropped} backlogged mic frames")

    def interrupt(self) -> None:
        """barge-in：停止当前播报（清空 TTS 管线），回到聆听。"""
        if self._pipeline is not None:
            self._pipeline.stop()
        self._play_stop_event.set()
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None
        self.state = DialogueState.INTERRUPTED

    def shutdown(self) -> None:
        self.interrupt()
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
        for comp in (self.asr, self.tts):
            if comp is not None:
                comp.close()
