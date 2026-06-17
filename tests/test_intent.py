"""患者数据意图检测：多轮场景下强制 tool_choice=required。"""

from agent.intent import infer_tool_choice


def test_list_patients_requires_tool():
    assert infer_tool_choice("查看患者列表") == "required"


def test_patient_name_after_prompt_requires_tool():
    history = [
        {"role": "user", "content": "查看患者病历"},
        {"role": "assistant", "content": "请提供患者姓名或ID。"},
    ]
    assert infer_tool_choice("张瑞阳", history) == "required"


def test_patient_id_requires_tool():
    assert infer_tool_choice("whl001") == "required"


def test_chitchat_no_force():
    assert infer_tool_choice("你好") is None
