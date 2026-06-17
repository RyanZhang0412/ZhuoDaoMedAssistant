"""用户意图启发式：多轮对话时 LLM 常凭上下文编造病历，需强制走工具。"""

from __future__ import annotations

import re

__all__ = ["infer_tool_choice"]

_DATA_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"患者列表|列出.*患者|列举患者|所有患者|有哪些患者|几个患者|查看患者"),
    re.compile(r"查.*病历|查看.*病历|患者病历|看病历|读病历|查找患者|查询患者|患者信息"),
    re.compile(r"康复方案|推荐.*训练|训练建议|适用规则|解释推荐"),
    re.compile(r"提醒|排期|日程|取消提醒"),
    re.compile(r"建档|新建患者|录入|更新病历|删除患者|补充|记录|修改|保存|性别|更正|漏了"),
    re.compile(r"数据库|没.*查|是不是.*编|凭.*记忆|没.*记录|肌入"),
)

_PATIENT_ID_RE = re.compile(r"^[A-Za-z]{2,8}\d{2,6}$", re.I)
_CHITCHAT_RE = re.compile(r"^(你好|您好|谢谢|再见|退出|没事|不用了?|好的|嗯+|哦+|啊+)[。.!！?？]*$")
_CJK_NAME_RE = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
_NON_NAME_WORDS = re.compile(r"^(训练|康复|方案|提醒|列表|病历|患者|查看|推荐|上肢|下肢|平衡|确认)$")
_FIELD_ASK_RE = re.compile(r"肌力|平衡|术后|痉挛|认知|补充|提供.*(等级|信息|数值)")
_STRENGTH_ANSWER_RE = re.compile(r"^[0-5一二三四五][级]?[。.!！?？]*$")


def _assistant_asked_for_patient(history: list[dict] | None) -> bool:
    if not history:
        return False
    for turn in reversed(history):
        if turn.get("role") != "assistant":
            continue
        text = turn.get("content") or ""
        return bool(
            re.search(
                r"(提供|告诉|输入).*(患者|姓名|ID|编号)|"
                r"(哪位|哪个).*(患者|人)|"
                r"患者.*(姓名|ID|编号)",
                text,
            )
        )
    return False


def _assistant_awaiting_field(history: list[dict] | None) -> bool:
    last = ""
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            last = turn.get("content") or ""
            break
    return bool(_FIELD_ASK_RE.search(last))


def _assistant_awaiting_create_info(history: list[dict] | None) -> bool:
    last = ""
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            last = turn.get("content") or ""
            break
    return bool(
        re.search(r"提供.+的(?:年龄|性别|诊断|患肢|肌力)|重新提供.+建档", last)
    )


def infer_tool_choice(query: str, history: list[dict] | None = None) -> str | None:
    """返回 OpenAI tool_choice；患者/排期等数据类问题用 required 防编造。"""
    q = query.strip()
    if not q or _CHITCHAT_RE.match(q):
        return None
    for pat in _DATA_QUERY_PATTERNS:
        if pat.search(q):
            return "required"
    if _PATIENT_ID_RE.match(q):
        return "required"
    if _CJK_NAME_RE.match(q) and not _NON_NAME_WORDS.match(q):
        return "required"
    if _assistant_asked_for_patient(history) and len(q) <= 16:
        return "required"
    if _assistant_awaiting_field(history) and (
        _STRENGTH_ANSWER_RE.match(q) or len(q) <= 8
    ):
        return "required"
    if _assistant_awaiting_create_info(history) and re.search(r"\d|男|女|截瘫|上身|下身|上肢|下肢", q):
        return "required"
    return None
