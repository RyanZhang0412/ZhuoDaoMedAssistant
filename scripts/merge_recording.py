"""合并一次语音会话的 user/assistant 录音为完整音频 + SRT 字幕。"""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import numpy as np

OUTPUT_SR = 24000
GAP_MS = 400


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return sr, pcm


def resample(pcm: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or len(pcm) == 0:
        return pcm
    duration = len(pcm) / from_sr
    new_len = max(int(round(duration * to_sr)), 1)
    x_old = np.linspace(0.0, duration, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, pcm.astype(np.float64)).astype(np.int16)


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def turn_ids(session: Path) -> list[int]:
    ids: set[int] = set()
    for p in session.glob("*_user.wav"):
        m = re.match(r"(\d+)_user\.wav$", p.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def merge_session(session: Path) -> tuple[Path, Path, Path, float]:
    out_wav = session / "merged_conversation.wav"
    out_srt = session / "merged_conversation.srt"
    out_txt = session / "merged_conversation.txt"

    gap = np.zeros(int(OUTPUT_SR * GAP_MS / 1000), dtype=np.int16)
    chunks: list[np.ndarray] = []
    srt_lines: list[str] = []
    txt_lines: list[str] = []
    timeline = 0.0
    idx = 1

    for turn in turn_ids(session):
        user_wav = session / f"{turn:03d}_user.wav"
        asst_wav = session / f"{turn:03d}_assistant.wav"
        user_txt = session / f"{turn:03d}_user.txt"
        asst_txt = session / f"{turn:03d}_assistant.txt"
        if not user_wav.exists() or not asst_wav.exists():
            continue

        user_text = user_txt.read_text(encoding="utf-8").strip() if user_txt.exists() else ""
        asst_text = asst_txt.read_text(encoding="utf-8").strip() if asst_txt.exists() else ""

        for role, wav_path, text in (
            ("用户", user_wav, user_text),
            ("助手", asst_wav, asst_text),
        ):
            sr, pcm = read_wav(wav_path)
            pcm = resample(pcm, sr, OUTPUT_SR)
            dur = len(pcm) / OUTPUT_SR
            start, end = timeline, timeline + dur

            srt_lines += [str(idx), f"{fmt_srt_time(start)} --> {fmt_srt_time(end)}", f"{role}：{text}", ""]
            txt_lines.append(
                f"[{fmt_srt_time(start).replace(',', '.')} - {fmt_srt_time(end).replace(',', '.')}] "
                f"{role}：{text}"
            )
            chunks += [pcm, gap]
            timeline = end + GAP_MS / 1000
            idx += 1

    if chunks:
        chunks.pop()

    merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    write_wav(out_wav, merged, OUTPUT_SR)
    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")
    return out_wav, out_srt, out_txt, len(merged) / OUTPUT_SR


def main() -> None:
    parser = argparse.ArgumentParser(description="合并会话录音并生成字幕")
    parser.add_argument("session_dir", type=Path, help="录音会话目录")
    args = parser.parse_args()
    session = args.session_dir.resolve()
    if not session.is_dir():
        raise SystemExit(f"目录不存在: {session}")

    wav, srt, txt, dur = merge_session(session)
    print(f"轮次: {len(turn_ids(session))}")
    print(f"时长: {dur:.1f}s")
    print(f"音频: {wav}")
    print(f"字幕: {srt}")
    print(f"文本: {txt}")


if __name__ == "__main__":
    main()
