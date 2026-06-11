"""康复方案推荐工具（薄适配层）。

复用 medical.service.Recommender：工具只负责取病历、调推荐、回灌结果。
规则仍是确定性结论来源，LLM 仅负责把结果组织成更自然的回答。
"""

from __future__ import annotations

from dataclasses import asdict

from agent.agent import ToolAction, ToolResult, register_tool
from agent.tools.context import get_context
from medical.repository import PatientNotFoundError

__all__ = ["recommend_rehab_plan", "explain_recommendation", "list_applicable_rules"]


def _get_record(patient_id: str):
    repo = get_context().repository
    try:
        return repo.get(patient_id), None
    except PatientNotFoundError:
        return None, ToolResult(
            action=ToolAction.RESPONSE,
            response=f"未找到患者 {patient_id}，无法推荐",
        )


@register_tool(
    {
        "name": "recommend_rehab_plan",
        "description": "根据患者既往病历，推荐康复训练模式。这是从病历判断该做什么康复训练的核心工具。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def recommend_rehab_plan(patient_id: str) -> ToolResult:
    record, err = _get_record(patient_id)
    if err:
        return err
    rec = get_context().recommender.recommend(record)
    return ToolResult(action=ToolAction.REQLLM, result=asdict(rec))


@register_tool(
    {
        "name": "explain_recommendation",
        "description": "对某患者的康复推荐给出详细解释（为什么这样安排）。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def explain_recommendation(patient_id: str) -> ToolResult:
    record, err = _get_record(patient_id)
    if err:
        return err
    rec = get_context().recommender.recommend(record)
    return ToolResult(
        action=ToolAction.REQLLM,
        result={"rationale": rec.explanation, "reasons": rec.reasons},
    )


@register_tool(
    {
        "name": "list_applicable_rules",
        "description": "展示某患者命中的所有康复规则（可解释性，给医生看）。",
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "患者ID"}},
            "required": ["patient_id"],
        },
    }
)
def list_applicable_rules(patient_id: str) -> ToolResult:
    record, err = _get_record(patient_id)
    if err:
        return err
    rules = get_context().recommender.matched_rules(record)
    return ToolResult(action=ToolAction.REQLLM, result={"applicable_rules": rules})
