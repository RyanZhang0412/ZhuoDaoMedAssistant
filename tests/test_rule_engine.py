"""规则引擎测试 + recommend 工具薄适配测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from medical.repository import PatientRecord
from medical.rules import RuleEngine
from medical.service import Recommender

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.yaml"


@pytest.fixture
def engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULES_PATH)


def test_rule_match_basic(engine):
    rec = PatientRecord(
        patient_id="P1", name="张三", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=2
    )
    result = engine.evaluate(rec.condition_view())
    assert result.matched
    assert "被动关节活动" in (result.mode or "")
    assert result.reasons


def test_amputation_rule_match(engine):
    rec = PatientRecord(
        patient_id="zry001",
        name="张瑞阳",
        diagnosis="截肢",
        affected_limb="上肢",
        muscle_strength=3,
        balance_level="良",
    )
    result = engine.evaluate(rec.condition_view())
    assert result.matched
    assert "残肢" in (result.mode or "")


def test_rule_determinism(engine):
    rec = PatientRecord(
        patient_id="P1", name="张三", diagnosis="脑卒中", affected_limb="下肢",
        muscle_strength=3, balance_level="中",
    )
    r1 = engine.evaluate(rec.condition_view())
    r2 = engine.evaluate(rec.condition_view())
    assert r1.rule_name == r2.rule_name
    assert r1.plan == r2.plan


def test_priority_selection(engine):
    rec = PatientRecord(
        patient_id="P1", name="张三", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=2
    )
    result = engine.evaluate(rec.condition_view())
    assert result.rule_name == "脑卒中-偏瘫早期-上肢被动训练"


def test_no_rule_match_returns_fallback(engine):
    rec = PatientRecord(patient_id="P1", name="张三", diagnosis="其他")
    result = engine.evaluate(rec.condition_view())
    assert not result.matched
    assert result.rule_name == "fallback"
    assert result.plan.get("mode")


def test_numeric_comparison(engine):
    early = PatientRecord(
        patient_id="P1", name="A", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=1
    )
    later = PatientRecord(
        patient_id="P2", name="B", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=4
    )
    assert "被动" in (engine.evaluate(early.condition_view()).mode or "")
    assert "主动辅助" in (engine.evaluate(later.condition_view()).mode or "")


class _MockLLM:
    def __init__(self):
        self.called = False

    def chat(self, messages, *, system=None, tools=None, tool_choice=None, max_tokens=None):
        from core.llm.base import LLMResponse

        self.called = True
        return LLMResponse(text="这是LLM生成的通俗解释。")

    def stream_chat(self, messages, *, system=None, max_tokens=None):
        yield "解释"


def test_recommend_conclusion_from_rules_not_llm(engine):
    mock = _MockLLM()
    rec = Recommender(engine, llm=mock)
    record = PatientRecord(
        patient_id="P1", name="张三", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=2
    )
    out = rec.recommend(record, explain=True)

    assert "被动关节活动" in (out.mode or "")
    assert mock.called
    assert out.explanation


def test_recommend_dry_run_no_llm(engine):
    rec = Recommender(engine, llm=None)
    record = PatientRecord(
        patient_id="P1", name="张三", diagnosis="脑卒中", affected_limb="下肢",
        muscle_strength=3, balance_level="差",
    )
    result = rec.recommend_dry_run(record)
    assert result.matched
