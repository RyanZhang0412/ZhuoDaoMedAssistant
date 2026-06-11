"""最小推荐层：取规则结论，必要时用 LLM 解释，再返回统一结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.llm.base import LLMBase, Message
from medical.repository import PatientRecord
from medical.rules import RuleEngine, RuleEngineResult

__all__ = ["RecommendationResult", "Recommender"]

_SYSTEM_PROMPT = (
    "你是康复科助手。下面给你一个已经由规则确定的康复训练方案，"
    "请你用通俗、简洁、有同理心的语言解释这个方案。"
    "不要更改训练模式、频率、强度；你只能解释，不能改方案。"
)


@dataclass
class RecommendationResult:
    patient_id: str
    matched: bool
    rule_name: str
    plan: dict[str, Any]
    explanation: str
    reasons: list[str] = field(default_factory=list)
    applicable_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safe: bool = True

    @property
    def mode(self) -> str | None:
        return self.plan.get("mode")

    @property
    def summary(self) -> str:
        return self.explanation


class Recommender:
    def __init__(self, rule_engine: RuleEngine, llm: LLMBase | None = None) -> None:
        self.rule_engine = rule_engine
        self.llm = llm

    def recommend(self, record: PatientRecord, *, explain: bool = True) -> RecommendationResult:
        result = self.rule_engine.evaluate(record.condition_view())
        explanation = self._fallback_text(result)
        warnings: list[str] = []
        if explain and self.llm is not None:
            llm_text = self._explain(result, patient_name=record.name)
            if llm_text:
                explanation, warnings = self._normalize_explanation(result, llm_text)

        return RecommendationResult(
            patient_id=record.patient_id,
            matched=result.matched,
            rule_name=result.rule_name,
            plan=result.plan,
            explanation=explanation,
            reasons=result.reasons,
            applicable_rules=[m.rule_name for m in result.all_matches],
            warnings=warnings,
            safe=True,
        )

    def recommend_dry_run(self, record: PatientRecord) -> RuleEngineResult:
        return self.rule_engine.evaluate(record.condition_view())

    def matched_rules(self, record: PatientRecord) -> list[str]:
        return [m.rule_name for m in self.recommend_dry_run(record).all_matches]

    def _explain(self, result: RuleEngineResult, patient_name: str | None = None) -> str | None:
        who = f"患者{patient_name}" if patient_name else "该患者"
        plan = result.plan
        user = (
            f"请为{who}解释以下康复方案：\n"
            f"规则：{result.rule_name}\n"
            f"模式：{plan.get('mode', '（无）')}\n"
            f"频率：{plan.get('frequency', '（无）')}\n"
            f"强度：{plan.get('intensity', '（无）')}\n"
            f"目标：{plan.get('goal', '（无）')}\n"
            f"注意事项：{'；'.join(plan.get('precautions', []) or []) or '（无）'}\n"
            f"原因：{'；'.join(result.reasons) or '（兜底）'}"
        )
        try:
            resp = self.llm.chat([Message(role="user", content=user)], system=_SYSTEM_PROMPT)
            return (resp.text or "").strip() or None
        except Exception:
            return None

    def _normalize_explanation(self, result: RuleEngineResult, text: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        explanation = text.strip()
        rule_mode = result.mode or ""
        if rule_mode and rule_mode not in explanation:
            explanation = f"{explanation}\n\n（康复方案：{rule_mode}）"
            warnings.append("explanation did not include rule mode; appended automatically")
        return explanation, warnings

    def _fallback_text(self, result: RuleEngineResult) -> str:
        plan = result.plan
        parts = []
        if plan.get("mode"):
            parts.append(f"建议康复训练模式：{plan['mode']}")
        if plan.get("frequency"):
            parts.append(f"频率：{plan['frequency']}")
        if plan.get("intensity"):
            parts.append(f"强度：{plan['intensity']}")
        if plan.get("goal"):
            parts.append(f"目标：{plan['goal']}")
        precautions = plan.get("precautions") or []
        if precautions:
            parts.append("注意事项：" + "；".join(precautions))
        if not result.matched:
            parts.insert(0, "（未匹配到标准方案，以下为兜底建议）")
        return "。".join(parts)
