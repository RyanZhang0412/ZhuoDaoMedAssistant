"""写操作确认层：所有写库工具调用前必须经用户二次确认才真正落库。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.agent import LocalAgent
from agent.tools import context as ctx_mod
from agent.tools.context import ToolContext, set_context
from core.llm.base import LLMBase, LLMResponse, ToolCall
from medical.repository import PatientRepository


class _StubLLM(LLMBase):
    """按预设队列返回 LLMResponse，便于驱动 agent.chat 的工具循环。"""

    def __init__(self, replies):
        super().__init__({})
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages, *, system=None, tools=None, tool_choice=None, max_tokens=None):
        self.calls += 1
        return self._replies.pop(0)

    def stream_chat(self, messages, *, system=None, max_tokens=None):
        yield self.chat(messages).text or ""


def _make_agent(tmp_path: Path, replies) -> tuple[LocalAgent, PatientRepository]:
    repo = PatientRepository(str(tmp_path / "patients"))
    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))
    return LocalAgent(_StubLLM(replies), system_prompt="x"), repo


def test_update_blocked_until_confirmation(tmp_path):
    """LLM 想更新患者 -> 首轮被拦、未落库；确认后才真正写入。"""
    agent, repo = _make_agent(
        tmp_path,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="update_patient_record",
                        arguments={"patient_id": "zry001", "updates": {"muscle_strength": 3}},
                    )
                ]
            ),
        ],
    )
    # 先放一个种子患者，否则 resolve 会找不到
    seed = tmp_path / "patients" / "zry001.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        json.dumps(
            {
                "patient_id": "zry001",
                "name": "张瑞阳",
                "age": 22,
                "gender": "男",
                "diagnosis": "截肢",
                "affected_limb": "下肢",
                "muscle_strength": 2,
                "balance_level": "中",
                "training_sessions": [],
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pending_reply = agent.chat("把张瑞阳肌力改成3级")
    assert "确认" in pending_reply and "更新患者 zry001" in pending_reply
    # 未确认前不能落库
    assert json.loads(seed.read_text(encoding="utf-8"))["muscle_strength"] == 2

    ok_reply = agent.chat("确认")
    assert "已更新" in ok_reply and "muscle_strength" in ok_reply
    assert json.loads(seed.read_text(encoding="utf-8"))["muscle_strength"] == 3


def test_cancel_does_not_write(tmp_path):
    """取消 -> 不落库，pending 被清空。"""
    agent, repo = _make_agent(
        tmp_path,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="delete_patient_record",
                        arguments={"patient_id": "zry001"},
                    )
                ]
            ),
        ],
    )
    seed = tmp_path / "patients" / "zry001.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(json.dumps({"patient_id": "zry001", "name": "张瑞阳"}, ensure_ascii=False), encoding="utf-8")

    pending_reply = agent.chat("删除张瑞阳")
    assert "删除患者 zry001" in pending_reply and "确认" in pending_reply
    assert seed.exists()  # 未落库删除

    cancel_reply = agent.chat("取消")
    assert "已取消" in cancel_reply
    assert seed.exists()
    assert agent._pending_writes.get("default") is None


def test_topic_change_clears_pending(tmp_path):
    """改话题（非确认非取消）-> 清掉 pending，本轮按新意图正常走 LLM。"""
    agent, repo = _make_agent(
        tmp_path,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="update_patient_record",
                        arguments={"patient_id": "zry001", "updates": {"muscle_strength": 4}},
                    )
                ]
            ),
            # 第二轮（改话题后）LLM 纯文本回复
            LLMResponse(text="好的，我不改了。需要别帮忙吗？"),
        ],
    )
    pending_reply = agent.chat("把张瑞阳肌力改成4级")
    assert "确认" in pending_reply
    assert agent._pending_writes.get("default") is not None

    new_reply = agent.chat("今天天气怎么样")
    assert new_reply == "好的，我不改了。需要别帮忙吗？"
    assert agent._pending_writes.get("default") is None


def test_read_tool_not_blocked(tmp_path):
    """读类工具不被确认层拦截，直接执行。"""
    agent, repo = _make_agent(
        tmp_path,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="get_patient_record", arguments={"patient_id": "zry001"})
                ]
            ),
            # 第二轮 LLM 组织语言
            LLMResponse(text="张瑞阳，男，22岁。"),
        ],
    )
    seed = tmp_path / "patients" / "zry001.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        json.dumps(
            {"patient_id": "zry001", "name": "张瑞阳", "age": 22, "gender": "男"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reply = agent.chat("查看张瑞阳病历")
    assert "张瑞阳" in reply
    # 没有 pending 产生
    assert agent._pending_writes.get("default") is None


def test_confirmation_isolated_per_session(tmp_path):
    """不同 session 的 pending 互不干扰：B 的普通查询不影响 A 的待确认写操作。"""
    agent, repo = _make_agent(
        tmp_path,
        [
            # session A：发起写工具 -> 产生 pending
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="update_patient_record",
                        arguments={"patient_id": "zry001", "updates": {"muscle_strength": 5}},
                    )
                ]
            ),
            # session B：普通闲聊 -> 纯文本
            LLMResponse(text="你好，有什么可以帮你？"),
        ],
    )
    # session A 产生 pending
    agent.chat("改张瑞阳肌力", session_id="A")
    assert agent._pending_writes.get("A") is not None
    # session B 没有 pending，走自己的 LLM 流程
    reply_b = agent.chat("你好", session_id="B")
    assert reply_b == "你好，有什么可以帮你？"
    assert agent._pending_writes.get("B") is None
    # A 的 pending 依然在，未被 B 误触发或清除
    assert agent._pending_writes.get("A") is not None


def test_delete_resolves_short_id(tmp_path):
    """LLM 传短名 "wzc" 时，delete 应 resolve 到 wzc001 再删，而非误报未找到。

    复现终端 bug：文件名是 wzc001.json，delete("wzc") 因精确匹配失败。
    """
    agent, repo = _make_agent(
        tmp_path,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="delete_patient_record",
                        arguments={"patient_id": "wzc"},
                    )
                ]
            ),
        ],
    )
    seed = tmp_path / "patients" / "wzc001.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        json.dumps({"patient_id": "wzc001", "name": "王智超"}, ensure_ascii=False),
        encoding="utf-8",
    )

    pending_reply = agent.chat("删除患者wzc")
    # 确认话术里应显示 resolve 后的真实 ID，而非原始短名
    assert "wzc001" in pending_reply and "确认" in pending_reply
    assert seed.exists()  # 未确认前不删

    ok_reply = agent.chat("确认")
    assert "已删除" in ok_reply and "wzc001" in ok_reply
    assert not seed.exists()
