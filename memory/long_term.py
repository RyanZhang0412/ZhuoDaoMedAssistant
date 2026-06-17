"""长期偏好记忆 —— 跨会话的沟通偏好摘要（对齐百聆 Memory 思路，加医疗约束）。

与百聆的关键差异：医疗场景下，长期记忆**只存沟通偏好类信息**
（称呼、语速详略偏好、作息、常聊话题），**绝不存医疗结论**
（诊断/用药/训练参数必须走正式病历 data/patients/，由工具读写）。

存储：{dir}/{session_id}.json -> {"summary", "updated", "turns_seen"}
更新：每 N 轮对话由 Robot 在后台线程调 update()，用 LLM 把旧摘要+近期对话
合并为 ≤200 字新摘要；失败静默保留旧摘要（记忆是增强项，不能拖垮主流程）。
注入：Robot 取 context_block() 作为 extra_system 追加到 system prompt。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

__all__ = ["LongTermMemory"]

_UPDATE_PROMPT = """你在维护一个语音助手的长期记忆。把【旧记忆】与【最近对话】合并成一份新的记忆。

要求：
- 不超过200字，直接输出记忆正文，不要解释、不要标题。
- 只保留：用户称呼/身份（医生还是患者）、沟通偏好（详略、语气）、作息习惯、常提的话题。
- 禁止写入：诊断、用药、训练方案、肌力等任何医疗内容——这些有专门的病历系统。
- 没有值得记的内容就输出原旧记忆；旧记忆也为空就输出空字符串。

【旧记忆】
{old}

【最近对话】
{dialogue}"""


class LongTermMemory:
    """跨会话沟通偏好摘要。线程安全（单文件级锁）。"""

    def __init__(self, dir_path: str, *, max_chars: int = 300) -> None:
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._updating: set[str] = set()  # 防同一会话并发更新

    # ---- 读 ----
    def get(self, session_id: str) -> str:
        data = self._load(session_id)
        return data.get("summary", "")

    def context_block(self, session_id: str) -> str | None:
        """注入 system prompt 的格式化块；无记忆时返回 None。"""
        summary = self.get(session_id)
        if not summary:
            return None
        return (
            "【长期记忆｜仅沟通偏好，非病历，不得作为医疗依据】\n" + summary
        )

    # ---- 写（调用方放后台线程；内部已防并发）----
    def update(self, session_id: str, history: list[dict], llm) -> None:
        """用 LLM 把旧摘要与 history（[{"role","content"}]）合并为新摘要。"""
        with self._lock:
            if session_id in self._updating:
                return
            self._updating.add(session_id)
        try:
            old = self.get(session_id)
            dialogue = "\n".join(
                f"{h.get('role', '?')}: {h.get('content', '')}" for h in history[-16:]
            )
            if not dialogue.strip():
                return
            prompt = _UPDATE_PROMPT.format(old=old or "（空）", dialogue=dialogue)
            from core.llm.base import Message

            resp = llm.chat([Message(role="user", content=prompt)], max_tokens=300)
            summary = (resp.text or "").strip()
            if summary in ("", "（空）"):
                return
            self._save(session_id, summary[: self.max_chars], len(history))
        except Exception as e:  # 记忆更新失败绝不影响主流程
            print(f"[long-term-memory] 更新失败（保留旧记忆）: {e}")
        finally:
            with self._lock:
                self._updating.discard(session_id)

    # ---- 存储 ----
    def _path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_")
        return self.dir / f"{safe}.json"

    def _load(self, session_id: str) -> dict:
        path = self._path(session_id)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, session_id: str, summary: str, turns_seen: int) -> None:
        path = self._path(session_id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    "turns_seen": turns_seen,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        tmp.replace(path)
