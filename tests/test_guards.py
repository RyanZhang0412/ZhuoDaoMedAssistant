"""建档入口守卫。"""

from agent.guards import claims_write_without_tool, precheck_user_query


def test_bare_create_asks_for_name():
    assert "姓名" in (precheck_user_query("增加患者") or "")


def test_name_only_create_asks_for_fields():
    reply = precheck_user_query("我要给卓小道建档")
    assert reply and "卓小道" in reply and "年龄" in reply


def test_create_patient_record_name_only_asks_for_fields():
    reply = precheck_user_query("创建患者病历卓小道")
    assert reply and "卓小道" in reply and "年龄" in reply


def test_write_claim_detected():
    assert claims_write_without_tool("已为患者张三（ID P001）建档。")
