"""病历录入/查询工具（薄适配层）。

复用 medical.repository 的最小数据接口，不在工具层重复实现存储逻辑。
当前目标是保持工具入口稳定，同时让底层更像未来可替换的数据库访问层。
"""

from __future__ import annotations

from agent.agent import ToolAction, ToolResult, register_tool
from agent.tools.context import get_context
from medical.repository import (
    CLINICAL_CREATE_FIELDS,
    PatientNotFoundError,
    PatientRecord,
    allocate_patient_id,
    new_record,
)

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
        "description": "按患者ID或姓名读取完整病历。用户说姓名时也可直接传入姓名。",
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
        rec = repo.resolve(patient_id)
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
                "patient_id": {
                    "type": "string",
                    "description": "患者ID（可选；缺省按姓名拼音首字母自动生成，如卓小道->zxd001）",
                },
                "name": {"type": "string", "description": "姓名"},
                "age": {"type": "integer", "description": "年龄"},
                "gender": {"type": "string", "description": "性别，男/女"},
                "diagnosis": {"type": "string", "description": "诊断，如 脑卒中、脑瘫"},
                "affected_limb": {"type": "string", "description": "患肢，如 上肢/下肢"},
                "muscle_strength": {"type": "integer", "description": "肌力 0-5 级"},
                "balance_level": {"type": "string", "description": "平衡等级：差/中/良/正常"},
            },
            "required": ["name"],
        },
    }
)
def create_patient_record(name: str, patient_id: str | None = None, **fields) -> ToolResult:
    repo = get_context().repository
    clean = {k: v for k, v in fields.items() if v is not None}
    if not any(k in clean for k in CLINICAL_CREATE_FIELDS):
        return ToolResult(
            action=ToolAction.RESPONSE,
            response=f"请提供{name}的年龄、性别、诊断、患肢、肌力及平衡等级，再为其建档。",
        )
    pid = allocate_patient_id(repo, name, preferred=patient_id)
    if repo.exists(pid):
        return ToolResult(
            action=ToolAction.RESPONSE,
            response=f"患者 {pid} 已存在，未重复建档",
        )
    rec = new_record(pid, name, **clean)
    repo.create(rec)
    return ToolResult(
        action=ToolAction.RESPONSE,
        result={"patient_id": pid},
        response=f"已为患者 {name}（ID {pid}）建档",
    )


@register_tool(
    {
        "name": "update_patient_record",
        "description": "更新患者病历字段（姓名、性别、年龄、诊断、肌力、平衡等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "患者ID或姓名"},
                "updates": {
                    "type": "object",
                    "description": "要更新的字段，如 {\"name\":\"汪昊祾\",\"gender\":\"男\",\"muscle_strength\": 3}",
                },
            },
            "required": ["patient_id", "updates"],
        },
    }
)
def update_patient_record(patient_id: str, updates: dict) -> ToolResult:
    repo = get_context().repository
    try:
        rec = repo.resolve(patient_id)
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

    from datetime import datetime

    rec.updated_at = datetime.now().isoformat(timespec="seconds")
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
    # 先 resolve 到真实 patient_id（兼容 LLM 传 "wzc" 而文件名是 "wzc001" 的情况），
    # 否则 delete 只做精确文件名匹配会误报"未找到"。
    try:
        rec = repo.resolve(patient_id)
        real_id = rec.patient_id
    except PatientNotFoundError:
        return ToolResult(action=ToolAction.RESPONSE, response=f"未找到患者 {patient_id}，无需删除")
    deleted = repo.delete(real_id)
    if not deleted:
        return ToolResult(action=ToolAction.RESPONSE, response=f"未找到患者 {real_id}，无需删除")
    return ToolResult(action=ToolAction.RESPONSE, response=f"已删除患者 {real_id}（{rec.name}）的本地病历")


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
