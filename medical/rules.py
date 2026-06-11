"""最小规则层：加载 rules.yaml 并做确定性匹配。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medical.repository import RULE_CONDITION_FIELDS

__all__ = ["Rule", "RuleMatch", "RuleEngineResult", "RuleEngine"]


@dataclass
class Rule:
    name: str
    priority: int
    conditions: dict[str, Any]
    recommend: dict[str, Any]


@dataclass
class RuleMatch:
    rule_name: str
    priority: int
    reasons: list[str]


@dataclass
class RuleEngineResult:
    matched: bool
    rule_name: str
    plan: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    all_matches: list[RuleMatch] = field(default_factory=list)

    @property
    def mode(self) -> str | None:
        return self.plan.get("mode")


class RuleEngine:
    def __init__(self, rules: list[Rule], fallback: dict[str, Any]) -> None:
        self.rules = rules
        self.fallback = fallback

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleEngine":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_rules = data.get("rules", [])
        fallback = data.get("fallback", {"mode": "综合评估", "precautions": ["请人工评估"]})

        rules: list[Rule] = []
        for r in raw_rules:
            conditions = r.get("conditions", {})
            unknown = set(conditions) - RULE_CONDITION_FIELDS
            if unknown:
                raise ValueError(
                    f"规则 {r.get('name')!r} 含未知条件字段 {sorted(unknown)}；允许字段: {sorted(RULE_CONDITION_FIELDS)}"
                )
            rules.append(
                Rule(
                    name=r.get("name", "未命名规则"),
                    priority=int(r.get("priority", 0)),
                    conditions=conditions,
                    recommend=r.get("recommend", {}),
                )
            )
        rules.sort(key=lambda x: -x.priority)
        return cls(rules, fallback)

    def evaluate(self, condition_view: dict[str, Any]) -> RuleEngineResult:
        matches: list[tuple[Rule, RuleMatch]] = []
        for rule in self.rules:
            ok, reasons = self._match_rule(rule, condition_view)
            if ok:
                matches.append((rule, RuleMatch(rule.name, rule.priority, reasons)))

        if not matches:
            return RuleEngineResult(
                matched=False,
                rule_name="fallback",
                plan=dict(self.fallback),
                reasons=["无任何规则命中，返回兜底方案"],
                all_matches=[],
            )

        best_rule, best_match = matches[0]
        return RuleEngineResult(
            matched=True,
            rule_name=best_rule.name,
            plan=dict(best_rule.recommend),
            reasons=best_match.reasons,
            all_matches=[m for _, m in matches],
        )

    def _match_rule(self, rule: Rule, view: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for field_name, expected in rule.conditions.items():
            actual = view.get(field_name)
            if actual is None:
                return False, []
            ok, reason = _match_condition(field_name, actual, expected)
            if not ok:
                return False, []
            reasons.append(reason)
        return True, reasons


_CMP_RE = re.compile(r"^\s*(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def _match_condition(field_name: str, actual: Any, expected: Any) -> tuple[bool, str]:
    if isinstance(expected, list):
        ok = actual in expected
        return ok, f"{field_name}={actual} 属于 {expected}" if ok else ""

    if isinstance(expected, str):
        m = _CMP_RE.match(expected)
        if m:
            op, num = m.group(1), float(m.group(2))
            av = _as_number(actual)
            if av is None:
                return False, ""
            ok = _apply_cmp(av, op, num)
            return ok, f"{field_name}={actual} 满足 {expected}" if ok else ""
        rng = _RANGE_RE.match(expected)
        if rng:
            lo, hi = float(rng.group(1)), float(rng.group(2))
            av = _as_number(actual)
            if av is None:
                return False, ""
            ok = lo <= av <= hi
            return ok, f"{field_name}={actual} 在 {lo}-{hi} 范围内" if ok else ""

    ok = actual == expected
    return ok, f"{field_name}={actual} 等于 {expected}" if ok else ""


def _as_number(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _apply_cmp(a: float, op: str, b: float) -> bool:
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b}[op]
