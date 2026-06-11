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


def test_path_traversal_blocked(repo):
    with pytest.raises(ValueError):
        repo.get("../etc/passwd")


def test_record_tool_create_and_get(repo):
    from agent.agent import ToolAction
    from agent.tools import record_tools
    from agent.tools.context import ToolContext, set_context

    set_context(ToolContext(repository=repo, recommender=None, scheduler=None))

    r = record_tools.create_patient_record("P001", "王五", age=70, diagnosis="脑卒中")
    assert r.action == ToolAction.RESPONSE
    assert repo.exists("P001")

    g = record_tools.get_patient_record("P001")
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
