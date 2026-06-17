"""康复训练提醒/排期 —— 本地 Scheduler + 工具，全程不联网。

排期存为本地 JSON：{schedule_dir}/{patient_id}.json。仅本地持久化，
不接任何推送/日历云服务。对齐百聆 schedule_task 但去掉联网。

工具是薄适配层：通过 ToolContext 注入的 ctx.scheduler 调用，不自建依赖。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.agent import ToolAction, ToolResult, register_tool
from agent.tools.context import get_context

__all__ = [
    "Scheduler",
    "schedule_rehab_reminder",
    "list_reminders",
    "cancel_reminder",
]


class Scheduler:
    """本地 JSON 排期存储。"""

    def __init__(self, schedule_dir: str) -> None:
        self.dir = Path(schedule_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, patient_id: str) -> Path:
        if not patient_id or any(c in patient_id for c in ("/", "\\", "..", "\0")):
            raise ValueError(f"非法 patient_id: {patient_id!r}")
        return self.dir / f"{patient_id}.json"

    def _load(self, patient_id: str) -> list[dict]:
        path = self._path(patient_id)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, patient_id: str, items: list[dict]) -> None:
        path = self._path(patient_id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def add(self, patient_id: str, time_str: str, content: str, repeat: str | None = None) -> str:
        items = self._load(patient_id)
        schedule_id = f"{patient_id}-{len(items) + 1}"
        items.append(
            {
                "schedule_id": schedule_id,
                "patient_id": patient_id,
                "time": time_str,
                "content": content,
                "repeat": repeat,
                "active": True,
            }
        )
        self._save(patient_id, items)
        return schedule_id

    def list(self, patient_id: str) -> list[dict]:
        return [it for it in self._load(patient_id) if it.get("active", True)]

    def cancel(self, schedule_id: str) -> bool:
        # schedule_id 形如 {patient_id}-{n}
        patient_id = schedule_id.rsplit("-", 1)[0]
        items = self._load(patient_id)
        found = False
        for it in items:
            if it["schedule_id"] == schedule_id:
                it["active"] = False
                found = True
        if found:
            self._save(patient_id, items)
        return found

    # ---- 到期检查（语音空闲时主动播报用，对齐百聆 TaskManager 的"时间到了主动说"）----
    def due_all(self, now: datetime | None = None) -> list[dict]:
        """扫描全部患者排期，返回此刻到期且未播报过的提醒，并标记已播报。

        时间格式支持：
          - "YYYY-MM-DD HH:MM"   一次性提醒，触发后置 inactive
          - "每天HH:MM" / "HH:MM" 每日重复（或 repeat 字段非空），每天最多触发一次
        到期窗口 2 分钟：错过窗口不补播，避免重启后陈年提醒轰炸。
        """
        now = now or datetime.now()
        fired: list[dict] = []
        for path in self.dir.glob("*.json"):
            patient_id = path.stem
            items = self._load(patient_id)
            changed = False
            for it in items:
                if not it.get("active", True):
                    continue
                if not self._is_due(it, now):
                    continue
                fired.append(dict(it))
                it["last_fired"] = now.strftime("%Y-%m-%d %H:%M")
                if not self._is_recurring(it):
                    it["active"] = False
                changed = True
            if changed:
                self._save(patient_id, items)
        return fired

    @staticmethod
    def _is_recurring(item: dict) -> bool:
        t = str(item.get("time", "")).strip()
        return bool(item.get("repeat")) or "每天" in t or not re.search(r"\d{4}-\d{2}-\d{2}", t)

    @staticmethod
    def _is_due(item: dict, now: datetime) -> bool:
        t = str(item.get("time", "")).strip()
        m = re.search(r"(\d{1,2}):(\d{2})", t)
        if not m:
            return False
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            return False
        date_m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
        if date_m:  # 一次性：日期+时间
            target = datetime(
                int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)), hh, mm
            )
        else:  # 每日重复：今天的该时刻
            if str(item.get("last_fired", "")).startswith(now.strftime("%Y-%m-%d")):
                return False  # 今天已播报
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return 0 <= (now - target).total_seconds() < 120


# --------------------------------------------------------------------------- #
# 工具（薄适配层，调 ctx.scheduler）
# --------------------------------------------------------------------------- #
@register_tool(
    {
        "name": "schedule_rehab_reminder",
        "description": "为患者设置康复训练提醒/排期。当用户要求安排训练时间或提醒时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者ID"},
                "time_str": {"type": "string", "description": "提醒时间，如 '每天09:00' 或 '2026-06-12 15:00'"},
                "content": {"type": "string", "description": "提醒内容"},
                "repeat": {"type": "string", "description": "重复规则，可选，如 'daily'"},
            },
            "required": ["patient_id", "time_str", "content"],
        },
    }
)
def schedule_rehab_reminder(patient_id: str, time_str: str, content: str, repeat: str | None = None) -> ToolResult:
    sid = get_context().scheduler.add(patient_id, time_str, content, repeat)
    return ToolResult(
        action=ToolAction.RESPONSE,
        result={"schedule_id": sid},
        response=f"好的，已为患者 {patient_id} 设置提醒：{time_str} {content}",
    )


@register_tool(
    {
        "name": "list_reminders",
        "description": "查询某患者的全部康复提醒/排期。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def list_reminders(patient_id: str) -> ToolResult:
    items = get_context().scheduler.list(patient_id)
    return ToolResult(action=ToolAction.REQLLM, result={"reminders": items})


@register_tool(
    {
        "name": "cancel_reminder",
        "description": "取消一条康复提醒。",
        "input_schema": {
            "type": "object",
            "properties": {"schedule_id": {"type": "string", "description": "提醒ID"}},
            "required": ["schedule_id"],
        },
    }
)
def cancel_reminder(schedule_id: str) -> ToolResult:
    ok = get_context().scheduler.cancel(schedule_id)
    return ToolResult(
        action=ToolAction.RESPONSE,
        result={"cancelled": ok},
        response="已取消该提醒" if ok else "未找到该提醒",
    )
