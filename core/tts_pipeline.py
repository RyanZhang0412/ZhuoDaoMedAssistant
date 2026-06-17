"""流式 TTS 播放管线 —— 文本段队列 + PCM 播放队列，两级流水（对齐百聆 tts_queue 思路）。

    submit("第一句。") ─┐
    submit("第二句。") ─┤→ [synth 线程] → PCM → [playback 线程] → 扬声器
    end_reply(cb)     ─┘

要点：
  - 第 N+1 句合成与第 N 句播放重叠；第一句合成完即开播，显著降低首响延迟。
  - barge-in：stop() 提升 generation，两级队列里的旧代项被直接丢弃，
    当前播放经 stop_event 立刻中断。
  - end_reply(callback)：一段回复的句子全部合成完后，把拼接 PCM 回调出去
    （Robot 用它保存助手录音），不阻塞主循环。
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from core.audio_io import play_pcm

__all__ = ["TTSPipeline"]


class TTSPipeline:
    """两级流水：text -> PCM -> 播放。线程安全；stop() 用于打断。"""

    def __init__(self, tts) -> None:
        self.tts = tts
        self.sample_rate = getattr(tts, "SAMPLE_RATE", 16000)
        self._text_q: queue.Queue = queue.Queue()
        self._pcm_q: queue.Queue = queue.Queue()
        self._gen = 0                       # barge-in 代数；旧代项一律丢弃
        self._lock = threading.Lock()
        self._pending = 0                   # 未完成项计数（text+pcm）
        self._idle = threading.Event()
        self._idle.set()
        self._play_stop = threading.Event()
        self._closed = False
        self._synth_thread = threading.Thread(
            target=self._synth_loop, daemon=True, name="tts-synth"
        )
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="tts-play"
        )
        self._synth_thread.start()
        self._play_thread.start()

    # ---- 生产端（Robot 主循环调用）----
    def submit(self, text: str) -> None:
        """提交一句待合成文本。立即返回。"""
        text = (text or "").strip()
        if not text or self._closed:
            return
        self._inc()
        self._text_q.put((self._gen, "text", text, None))

    def end_reply(self, on_audio: Callable[[bytes], None] | None = None) -> None:
        """标记一段回复结束。本段全部 PCM 合成完后回调 on_audio(拼接的 PCM)。"""
        if self._closed:
            return
        self._inc()
        self._text_q.put((self._gen, "end", None, on_audio))

    def stop(self) -> None:
        """barge-in：丢弃所有排队项并中断当前播放。"""
        with self._lock:
            self._gen += 1
        self._play_stop.set()

    def is_active(self) -> bool:
        """是否还有未合成/未播完的内容（= 助手"正在说话"）。"""
        return not self._idle.is_set()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def close(self) -> None:
        self._closed = True
        self.stop()
        self._text_q.put((self._gen, "quit", None, None))
        self._pcm_q.put((self._gen, "quit", None))

    # ---- 计数 ----
    def _inc(self) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()

    def _dec(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._pending <= 0:
                self._pending = 0
                self._idle.set()

    # ---- synth 线程：text -> PCM ----
    def _synth_loop(self) -> None:
        reply_parts: list[bytes] = []
        reply_gen = -1
        while True:
            gen, kind, text, cb = self._text_q.get()
            if kind == "quit":
                return
            if gen != self._gen:  # 已被 stop() 作废
                if gen != reply_gen:
                    reply_parts = []
                self._dec()
                continue
            if gen != reply_gen:  # 新一段回复（或打断后的新代）
                reply_parts = []
                reply_gen = gen
            if kind == "end":
                if cb is not None:
                    try:
                        cb(b"".join(reply_parts))
                    except Exception as e:  # 录音回调失败不影响播报
                        print(f"[tts-pipeline] end_reply 回调出错: {e}")
                reply_parts = []
                self._dec()
                continue
            try:
                pcm = self.tts.synthesize(text)
            except Exception as e:
                print(f"[tts-pipeline] 合成失败（跳过该句）: {e}")
                pcm = b""
            if pcm and gen == self._gen:
                reply_parts.append(pcm)
                self._inc()
                self._pcm_q.put((gen, "pcm", pcm))
            self._dec()

    # ---- playback 线程：PCM -> 扬声器 ----
    def _play_loop(self) -> None:
        while True:
            gen, kind, pcm = self._pcm_q.get()
            if kind == "quit":
                return
            if gen != self._gen:  # 已被 stop() 作废
                self._dec()
                continue
            # 新代第一项开播前复位打断标志（stop() 之后的新内容要能正常播）
            self._play_stop.clear()
            if gen != self._gen:  # 复位与 stop() 竞争的兜底重检
                self._dec()
                continue
            try:
                play_pcm(pcm, sample_rate=self.sample_rate, stop_event=self._play_stop)
            except Exception as e:
                print(f"[tts-pipeline] 播放失败: {e}")
            finally:
                self._dec()
