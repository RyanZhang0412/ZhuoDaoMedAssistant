"""康复训练提醒/排期 —— 本地 Scheduler + 工具，全程不联网。

排期存为本地 JSON：{schedule_dir}/{patient_id}.json。仅本地持久化，
不接任何推送/日历云服务。对齐百聆 schedule_task 但去掉联网。

工具是薄适配层：通过 ToolContext 注入的 ctx.scheduler 调用，不自建依赖。
"""

from __future__ import annotations

import json
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
