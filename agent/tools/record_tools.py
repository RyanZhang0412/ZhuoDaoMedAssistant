"""病历录入/查询工具（薄适配层）。

复用 medical.repository 的最小数据接口，不在工具层重复实现存储逻辑。
当前目标是保持工具入口稳定，同时让底层更像未来可替换的数据库访问层。
"""

from __future__ import annotations

from agent.agent import ToolAction, ToolResult, register_tool
from agent.tools.context import get_context
from medical.repository import PatientNotFoundError, PatientRecord, new_record

__all__ = [
    "get_patient_record",
    "create_patient_record",
    "update_patient_record",
    "delete_patient_record",
    "list_patients",
    "append_training",
]


@register_tool(
    {
        "name": "get_patient_record",
        "description": "按患者ID读取完整病历。需要查看某患者既往病历时调用。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def get_patient_record(patient_id: str) -> ToolResult:
    repo = get_context().repository
    try:
        rec = repo.get(patient_id)
    except PatientNotFoundError:
        return ToolResult(action=ToolAction.REQLLM, result={"error": f"未找到患者 {patient_id}"})
    return ToolResult(action=ToolAction.REQLLM, result=rec.to_dict())


@register_tool(
    {
        "name": "create_patient_record",
        "description": "为新患者建档。录入新患者基本信息与当前临床状态时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者ID（唯一）"},
                "name": {"type": "string", "description": "姓名"},
                "age": {"type": "integer", "description": "年龄"},
                "diagnosis": {"type": "string", "description": "诊断，如 脑卒中"},
                "affected_limb": {"type": "string", "description": "患肢，如 上肢/下肢"},
                "muscle_strength": {"type": "integer", "description": "肌力 0-5 级"},
            },
            "required": ["patient_id", "name"],
        },
    }
)
def create_patient_record(patient_id: str, name: str, **fields) -> ToolResult:
    repo = get_context().repository
    if repo.exists(patient_id):
        return ToolResult(
            action=ToolAction.RESPONSE,
            response=f"患者 {patient_id} 已存在，未重复建档",
        )
    clean = {k: v for k, v in fields.items() if v is not None}
    rec = new_record(patient_id, name, **clean)
    repo.create(rec)
    return ToolResult(
        action=ToolAction.RESPONSE,
        result={"patient_id": patient_id},
        response=f"已为患者 {name}（ID {patient_id}）建档",
    )


@register_tool(
    {
        "name": "update_patient_record",
        "description": "更新某患者的临床状态字段（如肌力、平衡、术后天数）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者ID"},
                "updates": {
                    "type": "object",
                    "description": "要更新的字段，如 {\"muscle_strength\": 3}",
                },
            },
            "required": ["patient_id", "updates"],
        },
    }
)
def update_patient_record(patient_id: str, updates: dict) -> ToolResult:
    repo = get_context().repository
    try:
        rec = repo.get(patient_id)
    except PatientNotFoundError:
        return ToolResult(action=ToolAction.RESPONSE, response=f"未找到患者 {patient_id}")

    allowed = set(PatientRecord.__dataclass_fields__)
    blocked = {"patient_id", "training_sessions", "created_at", "updated_at"}
    applied = {}
    for k, v in (updates or {}).items():
        if k in allowed and k not in blocked:
            setattr(rec, k, v)
            applied[k] = v
    if not applied:
        return ToolResult(action=ToolAction.RESPONSE, response="没有可更新的有效字段")

    repo.update(rec)
    return ToolResult(
        action=ToolAction.RESPONSE,
        result={"updated": applied},
        response=f"已更新患者 {patient_id} 的字段：{applied}",
    )


@register_tool(
    {
        "name": "delete_patient_record",
        "description": "按患者ID删除本地病历。确认不再需要该患者的本地 JSON 档案时调用。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def delete_patient_record(patient_id: str) -> ToolResult:
    repo = get_context().repository
    deleted = repo.delete(patient_id)
    if not deleted:
        return ToolResult(action=ToolAction.RESPONSE, response=f"未找到患者 {patient_id}，无需删除")
    return ToolResult(action=ToolAction.RESPONSE, response=f"已删除患者 {patient_id} 的本地病历")


@register_tool(
    {
        "name": "list_patients",
        "description": "列出患者，可按姓名或ID关键字检索。",
        "input_schema": {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "检索关键字，可选"}},
            "required": [],
        },
    }
)
def list_patients(keyword: str | None = None) -> ToolResult:
    repo = get_context().repository
    return ToolResult(action=ToolAction.REQLLM, result={"patients": repo.search(keyword)})


@register_tool(
    {
        "name": "append_training",
        "description": "为患者追加一条康复训练记录到既往训练史。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者ID"},
                "date": {"type": "string", "description": "训练日期 YYYY-MM-DD"},
                "mode": {"type": "string", "description": "训练模式"},
                "response": {"type": "string", "description": "患者反应/效果，可选"},
            },
            "required": ["patient_id", "date", "mode"],
        },
    }
)
def append_training(patient_id: str, date: str, mode: str, response: str | None = None) -> ToolResult:
    repo = get_context().repository
    try:
        rec = repo.get(patient_id)
    except PatientNotFoundError:
        return ToolResult(action=ToolAction.RESPONSE, response=f"未找到患者 {patient_id}")

    rec.training_sessions.append({"date": date, "mode": mode, "response": response})
    repo.update(rec)
    return ToolResult(
        action=ToolAction.RESPONSE,
        response=f"已为患者 {patient_id} 追加训练记录：{date} {mode}",
    )
