"""直连工具：语音短答肌力/平衡更新 + 建档流程防误更新。"""

from agent.direct_tools import parse_clinical_fields, parse_muscle_strength, try_direct_tool_call
from main import build_agent, load_config


def test_parse_muscle_strength():
    assert parse_muscle_strength("一") == 1
    assert parse_muscle_strength("3级") == 3
    assert parse_muscle_strength("肌力一级") == 1


def test_parse_clinical_fields_create_sentence():
    fields = parse_clinical_fields("11，男，脖子下截瘫，下身，1，1")
    assert fields["age"] == 11
    assert fields["gender"] == "男"
    assert fields["affected_limb"] == "下肢"
    assert fields["muscle_strength"] == 1


def test_parse_clinical_fields_space_separated():
    """空格分隔也要能抽全字段（终端实测 bug）。"""
    fields = parse_clinical_fields("11 男 腿部骨折 下肢 1 1")
    assert fields["age"] == 11
    assert fields["gender"] == "男"
    assert fields["affected_limb"] == "下肢"
    assert fields["muscle_strength"] == 1
    assert fields["balance_level"] == "差"
    assert fields["diagnosis"] == "腿部骨折"


def test_create_flow_generates_pinyin_id():
    """新建患者 ID 应为姓名拼音首字母 + 序号（卓小道 -> zxd001），而非 patNNN。"""
    build_agent(load_config())
    history = [
        {"role": "user", "content": "增加患者卓小道"},
        {"role": "assistant", "content": "请补充卓小道的年龄、性别、诊断、患肢、肌力和平衡等级。"},
    ]
    call = try_direct_tool_call("11 男 腿部骨折 下肢 1 1", history)
    assert call is not None
    assert call.name == "create_patient_record"
    assert call.arguments["patient_id"].startswith("zxd")
    assert call.arguments["age"] == 11
    assert call.arguments["diagnosis"] == "腿部骨折"


def test_direct_update_after_muscle_prompt():
    build_agent(load_config())
    history = [
        {"role": "user", "content": "查找患者ZRY001"},
        {"role": "assistant", "content": "患者张瑞阳 (ID: zry001)。请提供肌力等级。"},
    ]
    call = try_direct_tool_call("一", history)
    assert call is not None
    assert call.name == "update_patient_record"
    assert call.arguments["updates"]["muscle_strength"] == 1


def test_parse_balance_level_numeric():
    from agent.direct_tools import parse_balance_level

    assert parse_balance_level("张瑞阳平衡等级五级") == "正常"
    assert parse_balance_level("平衡良好") == "良"
    assert parse_balance_level("平衡等级3级") == "良"


def test_parse_clinical_fields_balance_not_muscle():
    fields = parse_clinical_fields("张瑞阳平衡等级五级")
    assert fields.get("balance_level") == "正常"
    assert "muscle_strength" not in fields


def test_parse_clinical_fields_amputation_update():
    fields = parse_clinical_fields("截肢部位：腿部，由于被车撞，患肢在右侧")
    assert fields["diagnosis"] == "截肢"
    assert fields["affected_limb"] == "下肢"
    assert fields.get("diagnosis") != "具体诊断"


def test_direct_balance_update_with_patient_name():
    build_agent(load_config())
    history = [
        {"role": "user", "content": "查看张瑞阳的病历"},
        {"role": "assistant", "content": "张瑞阳，男，22岁。诊断：截肢。"},
    ]
    call = try_direct_tool_call("张瑞阳平衡等级五级", history)
    assert call is not None
    assert call.arguments["patient_id"] == "zry001"
    assert call.arguments["updates"]["balance_level"] == "正常"
    assert "muscle_strength" not in call.arguments["updates"]


def test_direct_correction_not_muscle():
    build_agent(load_config())
    history = [
        {"role": "user", "content": "张瑞阳平衡等级五级"},
        {"role": "assistant", "content": "已更新肌力"},
        {"role": "user", "content": "不是肌力"},
    ]
    call = try_direct_tool_call("不是肌力", history)
    assert call is not None
    assert call.arguments["updates"]["balance_level"] == "正常"


def test_create_flow_does_not_update_existing_patient():
    build_agent(load_config())
    history = [
        {"role": "assistant", "content": "当前系统中有3名患者：汪昊祾、王一熙、张瑞阳。"},
        {"role": "user", "content": "增加患者 卓小道"},
        {"role": "assistant", "content": "请提供卓小道的年龄、性别、诊断、患肢、肌力及平衡等级。"},
    ]
    call = try_direct_tool_call("11，男，脖子下截瘫，下身，1，1", history)
    assert call is not None
    assert call.name == "create_patient_record"
    assert call.arguments["name"] == "卓小道"
    assert call.arguments["age"] == 11
    assert call.arguments["gender"] == "男"
    assert call.arguments["patient_id"] != "whl001"


def test_delete_intent_not_hijacked_by_direct_tools():
    """「删除患者 whl002」不应被直连层误当成肌力更新。"""
    build_agent(load_config())
    history = [
        {"role": "user", "content": "whl"},
        {
            "role": "assistant",
            "content": "汪昊祾 (whl001) 病历：22岁，男，偏瘫，患肢上肢，肌力5级。",
        },
    ]
    assert try_direct_tool_call("删除患者whl002", history) is None


def test_parse_muscle_strength_ignores_patient_id_digits():
    assert parse_muscle_strength("删除患者whl002") is None
    assert parse_muscle_strength("3级") == 3


def test_resolve_patient_id_prefers_query_over_history():
    build_agent(load_config())
    history = [
        {
            "role": "assistant",
            "content": "汪昊祾 (whl001) 病历：22岁，男，偏瘫，患肢上肢，肌力5级。",
        },
    ]
    from agent.direct_tools import _resolve_patient_id

    assert _resolve_patient_id(history, None, query="删除患者whl002") == "whl002"
