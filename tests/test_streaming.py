"""流式管线相关单测：句子切分 / agent 流式 / 排期到期 / 流式工具聚合退化。"""

from datetime import datetime

from agent.agent import LocalAgent
from agent.tools.schedule_tools import Scheduler
from core.llm.base import LLMBase, LLMResponse
from core.robot import split_ready_sentences


class _StubLLM(LLMBase):
    """非流式 stub：验证基类 stream_chat_tools 退化路径与 agent.chat_stream。"""

    def __init__(self, replies):
        super().__init__({})
        self._replies = list(replies)

    def chat(self, messages, *, system=None, tools=None, tool_choice=None, max_tokens=None):
        return self._replies.pop(0)

    def stream_chat(self, messages, *, system=None, max_tokens=None):
        yield self.chat(messages).text or ""


def test_split_ready_sentences_basic():
    ready, rest = split_ready_sentences("今天天气不错。我们开始训练吧！还没说完", min_len=4)
    assert ready == ["今天天气不错。", "我们开始训练吧！"]
    assert rest == "还没说完"


def test_split_ready_sentences_min_len():
    # 句长不足 min_len 时不切，与后文合并
    ready, rest = split_ready_sentences("好。明白了，马上开始训练。", min_len=6)
    assert ready == ["好。明白了，马上开始训练。"]
    assert rest == ""


def test_agent_chat_stream_plain_reply():
    llm = _StubLLM([LLMResponse(text="你好，有什么可以帮你？")])
    agent = LocalAgent(llm, system_prompt="x")
    out = "".join(agent.chat_stream("你好"))
    assert out == "你好，有什么可以帮你？"


def test_agent_chat_extra_system_compose():
    agent = LocalAgent(_StubLLM([LLMResponse(text="ok")]), system_prompt="base")
    assert agent._compose_system("extra") == "base\n\nextra"
    assert agent._compose_system(None) == "base"


def test_voice_recorder_turn_pairing(tmp_path):
    from core.voice_recorder import VoiceSessionRecorder

    rec = VoiceSessionRecorder(tmp_path, "test")
    rec.save_user(b"\x00\x01" * 100, "用户话")
    assert rec.turn == 1
    rec.save_assistant(b"\x00\x02" * 200, "助手话", 24000, turn=1)
    rec.save_user(b"\x00\x03" * 100, "第二句")
    assert (rec.session_dir / "001_assistant.wav").exists()
    assert (rec.session_dir / "001_assistant.txt").read_text(encoding="utf-8") == "助手话"
    assert (rec.session_dir / "002_user.wav").exists()


def test_scheduler_due_all_daily_and_once(tmp_path):
    s = Scheduler(str(tmp_path))
    s.add("P001", "每天09:00", "上肢被动训练", repeat="daily")
    s.add("P002", "2026-06-11 15:00", "复诊评估")

    now = datetime(2026, 6, 11, 9, 0, 30)
    fired = s.due_all(now)
    assert [f["patient_id"] for f in fired] == ["P001"]
    # 同一天内不重复触发
    assert s.due_all(datetime(2026, 6, 11, 9, 1)) == []

    fired2 = s.due_all(datetime(2026, 6, 11, 15, 1))
    assert [f["patient_id"] for f in fired2] == ["P002"]
    # 一次性提醒触发后失效
    assert s.list("P002") == []
    # 错过窗口（>2分钟）不补播
    s.add("P003", "2026-06-11 10:00", "晚到提醒")
    assert s.due_all(datetime(2026, 6, 11, 10, 5)) == []
