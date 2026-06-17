"""Agent 入口守卫：拦截空建档请求、无工具写入幻觉，并为写库操作加确认层。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.direct_tools import extract_create_name, parse_clinical_fields
from core.llm.base import ToolCall

_WRITE_CLAIM_RE = re.compile(
    r"已(为患者|将|帮您|为您|成功)?.*(建档|更新|记录|保存|删除|录入|重新建档)"
)
_CREATE_BARE_RE = re.compile(
    r"^(增加|新建|添加|创建)(?:患者)?(?:病历)?[。.!！?？\s]*$"
)

# 会落库的写工具：执行前必须经用户二次确认。
# 读类工具（get_patient_record / list_patients / list_reminders /
# recommend_rehab_plan / explain_recommendation / list_applicable_rules）不在此列。
WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "create_patient_record",
        "update_patient_record",
        "delete_patient_record",
        "append_training",
        "schedule_rehab_reminder",
        "cancel_reminder",
    }
)

# 用户明确同意执行上一条待确认写操作。
_CONFIRM_RE = re.compile(
    r"^\s*(确认|确定|确认执行|确认修改|确认更新|确认删除|确认建档|确认设置|确认追加|"
    r"对[的了]?|是的?|没错|好的?|可以|行[的吧]?|嗯+|执行[吧]?|没问题|就这样|同意)\s*"
    r"[。.!！?？嗯行了的吧]*\s*$"
)
# 用户明确拒绝执行。
_DENY_RE = re.compile(
    r"^\s*(取消|不要|不执行|算了|改主意|不需要|不用了?|不对|不是|否认|先别|先不要|"
    r"别(删|改|更|建|设|加)|不确认|不同意)\s*[。.!！?？]*\s*$"
)


@dataclass
class PendingWrite:
    """一条待用户确认的写操作（按 session 隔离暂存于 Agent 实例）。"""

    call: ToolCall
    description: str


def precheck_user_query(query: str) -> str | None:
    """信息不足时直接追问，不交给 LLM（避免编造张三等）。"""
    q = query.strip()
    if not q:
        return None
    if _CREATE_BARE_RE.match(q):
        return "请告诉我患者姓名（可选患者 ID），以及年龄、性别、诊断等信息。"
    name = extract_create_name(q)
    if name and not parse_clinical_fields(q):
        return f"请提供{name}的年龄、性别、诊断、患肢、肌力及平衡等级，再为其建档。"
    return None


def claims_write_without_tool(text: str | None) -> bool:
    return bool(text and _WRITE_CLAIM_RE.search(text))


def reject_hallucinated_write() -> str:
    return "我还没有写入数据库。请提供患者姓名和必要信息，或说明要查/改哪一位患者。"


def is_confirmation(text: str | None) -> bool:
    """用户是否明确同意执行上一条待确认写操作（先排否认，避免"不确认"误判）。"""
    if not text:
        return False
    t = text.strip()
    if _DENY_RE.match(t):
        return False
    return bool(_CONFIRM_RE.match(t))


def is_denial(text: str | None) -> bool:
    if not text:
        return False
    return bool(_DENY_RE.match(text.strip()))


def describe_write(call: ToolCall) -> str:
    """把写工具调用翻译成给用户看的「即将做什么」自然语言摘要。"""
    args = call.arguments or {}
    name = call.name
    if name == "create_patient_record":
        pid = args.get("patient_id") or "（自动生成ID）"
        fields = ", ".join(f"{k}={v}" for k, v in args.items() if k not in ("patient_id",))
        return f"为 {args.get('name', '新患者')}（ID {pid}）建档，字段：{fields}"
    if name == "update_patient_record":
        updates = args.get("updates") or {}
        kv = ", ".join(f"{k}={v}" for k, v in updates.items())
        return f"更新患者 {args.get('patient_id', '?')} 的字段：{kv}"
    if name == "delete_patient_record":
        return f"删除患者 {args.get('patient_id', '?')} 的本地病历"
    if name == "append_training":
        return (
            f"为患者 {args.get('patient_id', '?')} 追加训练记录："
            f"{args.get('date', '?')} {args.get('mode', '?')}"
        )
    if name == "schedule_rehab_reminder":
        return (
            f"为患者 {args.get('patient_id', '?')} 设置提醒："
            f"{args.get('time_str', '?')} {args.get('content', '')}"
        )
    if name == "cancel_reminder":
        return f"取消提醒 {args.get('schedule_id', '?')}"
    return f"执行写操作 {name}"


def confirmation_prompt(call: ToolCall) -> str:
    """生成「即将 X，确认执行吗？」的追问话术。"""
    return f"即将{describe_write(call)}。确认执行吗？（回复“确认”执行，“取消”放弃）"
