"""病历仓库测试 + record 工具薄适配测试。"""

from __future__ import annotations

import pytest

from medical.repository import PatientNotFoundError, PatientRepository, new_record


@pytest.fixture
def repo(tmp_path) -> PatientRepository:
    return PatientRepository(str(tmp_path / "patients"))


def test_create_then_get_roundtrip(repo):
    rec = new_record("P001", "张三", diagnosis="脑卒中", affected_limb="上肢", muscle_strength=2)
    repo.create(rec)
    got = repo.get("P001")
    assert got.name == "张三"
    assert got.diagnosis == "脑卒中"
    assert got.muscle_strength == 2


def test_create_duplicate_raises(repo):
    repo.create(new_record("P001", "张三"))
    with pytest.raises(FileExistsError):
        repo.create(new_record("P001", "李四"))


def test_missing_patient_raises(repo):
    with pytest.raises(PatientNotFoundError):
        repo.get("nope")


def test_update_roundtrip(repo):
    rec = new_record("P001", "张三", muscle_strength=2)
    repo.create(rec)
    got = repo.get("P001")
    got.muscle_strength = 3
    repo.update(got)
    assert repo.get("P001").muscle_strength == 3


def test_append_training_persists(repo):
    rec = new_record("P001", "张三")
    repo.create(rec)
    got = repo.get("P001")
    got.training_sessions.append({"date": "2026-06-10", "mode": "被动关节活动训练"})
    repo.update(got)
    reloaded = repo.get("P001")
    assert len(reloaded.training_sessions) == 1
    assert reloaded.training_sessions[0]["mode"] == "被动关节活动训练"


def test_search(repo):
    repo.create(new_record("P001", "张三", diagnosis="脑卒中"))
    repo.create(new_record("P002", "李四", diagnosis="脊髓损伤"))
    all_p = repo.search()
    assert len(all_p) == 2
    zhang = repo.search("张三")
    assert len(zhang) == 1 and zhang[0]["patient_id"] == "P001"


def test_resolve_by_name_or_id(repo):
    repo.create(new_record("zry001", "张瑞阳", diagnosis="截肢"))
    got = repo.resolve("张瑞阳")
    assert got.patient_id == "zry001"
    assert repo.resolve("zry001").name == "张瑞阳"
    assert repo.resolve("zry").patient_id == "zry001"
    with pytest.raises(PatientNotFoundError):
        repo.resolve("张璐阳")


def test_record_tool_get_by_name(repo):
    from agent.agent import ToolAction
    from agent.tools import record_tools
    from agent.tools.context import ToolContext, set_context

    repo.create(new_record("zry001", "张瑞阳", diagnosis="截肢"))
    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))
    g = record_tools.get_patient_record("张瑞阳")
    assert g.action == ToolAction.REQLLM
    assert g.result["patient_id"] == "zry001"


def test_path_traversal_blocked(repo):
    with pytest.raises(ValueError):
        repo.get("../etc/passwd")


def test_allocate_patient_id_from_name(repo):
    from medical.repository import allocate_patient_id

    assert allocate_patient_id(repo, "卓小道") == "zxd001"
    assert allocate_patient_id(repo, "卓小道", preferred="pat005") == "zxd001"
    assert allocate_patient_id(repo, "王五", preferred="P001") == "p001"


def test_record_tool_create_blocks_name_only(repo):
    from agent.agent import ToolAction
    from agent.tools import record_tools
    from agent.tools.context import ToolContext, set_context

    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))
    r = record_tools.create_patient_record("卓小道", patient_id="pat005")
    assert r.action == ToolAction.RESPONSE
    assert "请提供" in (r.response or "")
    assert not repo.exists("pat005")
    assert not repo.exists("zxd001")


def test_record_tool_create_and_get(repo):
    from agent.agent import ToolAction
    from agent.tools import record_tools
    from agent.tools.context import ToolContext, set_context

    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))

    r = record_tools.create_patient_record("王五", patient_id="P001", age=70, diagnosis="脑卒中")
    assert r.action == ToolAction.RESPONSE
    assert repo.exists("p001")

    g = record_tools.get_patient_record("p001")
    assert g.action == ToolAction.REQLLM
    assert g.result["name"] == "王五"


def test_record_tool_delete(repo):
    from agent.agent import ToolAction
    from agent.tools import record_tools
    from agent.tools.context import ToolContext, set_context

    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))
    repo.create(new_record("P009", "待删除患者"))

    out = record_tools.delete_patient_record("P009")
    assert out.action == ToolAction.RESPONSE
    assert not repo.exists("P009")


def test_pinyin_key_normalizes():
    from medical.repository import pinyin_key

    # 同音不同字 -> 同一拼音键（语音 ASR 误识别的关键）
    assert pinyin_key("汪昊林") == pinyin_key("汪浩林")
    assert pinyin_key("张瑞阳") == pinyin_key("张瑞扬")
    assert pinyin_key("卓小道") == "zhuoxiaodao"


def test_resolve_by_pinyin_homophone(repo):
    """ASR 把"汪昊林"转成同音的"汪浩林"时，拼音匹配应命中同一患者。"""
    repo.create(new_record("whl001", "汪昊林", diagnosis="脑卒中"))
    got = repo.resolve("汪浩林")
    assert got.patient_id == "whl001"
    assert got.name == "汪昊林"


def test_resolve_by_pinyin_partial(repo):
    """拼音子串匹配：说"汪昊"（拼音 wanghao）应命中"汪昊林"。"""
    repo.create(new_record("whl001", "汪昊林"))
    assert repo.resolve("汪浩").patient_id == "whl001"


def test_resolve_pinyin_multiple_raises_with_candidates(repo):
    """拼音命中多个患者时不猜，异常消息列出候选（含 ID/姓名）。

    查询用第三种同音写法"汪豪林"，不等于任一患者名，迫使走拼音层；
    两个同音患者"汪昊林""汪浩林"都命中 -> 抛异常并列出候选。
    """
    repo.create(new_record("whl001", "汪昊林"))
    repo.create(new_record("whl002", "汪浩林"))  # 同音不同字
    with pytest.raises(PatientNotFoundError) as exc_info:
        repo.resolve("汪豪林")
    msg = str(exc_info.value)
    # 候选信息带出来，供上层（LLM/用户）补充 ID
    assert "whl001" in msg and "whl002" in msg


def test_search_by_pinyin(repo):
    """list_patients 走 search，拼音匹配让语音误识别也能检索到候选。"""
    repo.create(new_record("whl001", "汪昊林", diagnosis="脑卒中", age=45))
    repo.create(new_record("zry001", "张瑞阳", diagnosis="截肢"))
    found = repo.search("汪浩林")
    assert len(found) == 1
    assert found[0]["patient_id"] == "whl001"
    assert found[0]["name"] == "汪昊林"


def test_resolve_patient_id_by_pinyin(repo):
    """直连路径 _resolve_patient_id：语音误识别姓名时靠拼音命中。

    模拟：用户说"查一下汪浩林的肌力"，ASR 把"汪昊林"转成同音"汪浩林"，
    字符匹配失败，拼音子串匹配应命中 whl001。
    """
    from agent.direct_tools import _resolve_patient_id
    from agent.tools.context import ToolContext, set_context

    repo.create(new_record("whl001", "汪昊林", diagnosis="脑卒中"))
    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))

    pid = _resolve_patient_id(None, None, query="查一下汪浩林的肌力")
    assert pid == "whl001"


def test_resolve_by_initials_near_homophone(repo):
    """近音字：祾(ling) vs 林(lin) 全拼不同，但首字母均为 whl。"""
    repo.create(new_record("whl001", "汪昊祾", diagnosis="偏瘫"))
    got = repo.resolve("汪浩林")
    assert got.patient_id == "whl001"
    assert got.name == "汪昊祾"


def test_search_by_initials_near_homophone(repo):
    repo.create(new_record("whl001", "汪昊祾", age=22, diagnosis="偏瘫"))
    found = repo.search("汪浩林")
    assert len(found) == 1
    assert found[0]["patient_id"] == "whl001"


def test_resolve_patient_id_by_initials_near_homophone(repo):
    """直连路径：ASR 近音误识别 汪昊祾 -> 汪浩林。"""
    from agent.direct_tools import _resolve_patient_id
    from agent.tools.context import ToolContext, set_context

    repo.create(new_record("whl001", "汪昊祾"))
    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))

    pid = _resolve_patient_id(None, None, query="查一下汪浩林的肌力")
    assert pid == "whl001"
