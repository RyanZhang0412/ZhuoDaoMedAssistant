"""对话记忆 —— 会话上下文，与正式病历严格隔离。

关键设计：这是【临时对话上下文】，绝不等于、绝不写入正式病历。
百聆把"记忆"当闲聊偏好用；医疗场景里，LLM 闲聊产生的内容绝不能污染患者病历。
因此本模块只管多轮对话上下文，存到 data/dialogues/，与 data/patients/ 物理隔离。

Robot 是会话历史的唯一所有者（见 core/robot.py 与交叉验证结论）：
Robot 调 context_for_llm(session_id) 取历史喂给 Agent.chat，Agent 自身不持久化历史。

context_for_llm 返回 [{"role","content"}]（边界转换在 agent.agent._to_messages）。
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

__all__ = ["DialogueMemory"]


class DialogueMemory:
    """多会话的滚动对话上下文。"""

    def __init__(self, dialogue_dir: str | None = None, *, max_turns: int = 20) -> None:
        # 内存中的滚动窗口：session_id -> deque[{"role","content"}]
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))
        self.max_turns = max_turns
        self.dir = Path(dialogue_dir) if dialogue_dir else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条对话。role: 'user' | 'assistant'。"""
        self._buffers[session_id].append({"role": role, "content": content})
        if self.dir is not None:
            self._persist(session_id)

    def context_for_llm(self, session_id: str) -> list[dict]:
        """返回该会话的上下文历史，供喂给 LLM（不含正式病历）。"""
        return list(self._buffers.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)

    # ---- 落盘（与病历目录隔离） ----
    def _persist(self, session_id: str) -> None:
        if self.dir is None:
            return
        safe = session_id.replace("/", "_").replace("\\", "_")
        path = self.dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(list(self._buffers[session_id]), f, ensure_ascii=False, indent=2)
        tmp.replace(path)
