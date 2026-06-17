"""语音/文本碎片直连工具，避免 LLM 口头「已更新」但未落库。"""

from __future__ import annotations

import re

from core.llm.base import ToolCall
from medical.repository import (
    BALANCE_LEVELS,
    DIAGNOSIS_VALUES,
    PatientNotFoundError,
    allocate_patient_id,
    pinyin_key,
)

__all__ = [
    "try_direct_tool_call",
    "parse_clinical_fields",
    "parse_muscle_strength",
    "extract_create_name",
]

_MUSCLE_ASK = re.compile(r"肌力|几分")
_BALANCE_ASK = re.compile(r"平衡")
_LIMB_ASK = re.compile(r"患肢|上肢|下肢|下身")
_SUPPLEMENT = re.compile(r"补充")
_GENDER_CORRECTION = re.compile(r"性别|说过.*男|说过.*女|漏了|没记|没写|更正")
_PID_RE = re.compile(r"([A-Za-z]{2,8}\d{2,6})", re.I)
_LIMB_VALUES = ("双上肢", "双下肢", "四肢", "上肢", "下肢")
_BALANCE_BY_SCORE = {0: "差", 1: "差", 2: "中", 3: "良", 4: "良", 5: "正常"}
_DIAGNOSIS_SKIP_PARTS = frozenset(
    {"具体诊断", "诊断", "患肢", "部位", "由于", "被车撞", "右边", "左侧", "右侧", "左边"}
)
_INVALID_CREATE_NAMES = frozenset({"患者", "新患者", "一位", "一个", "谁", "他", "她", "病历"})
_CREATE_NAME_RE = re.compile(
    r"^(?:增加|新建|添加|创建)(?:患者)?(?:病历)?\s*(?P<name>[^\s，,。]+)\s*$|"
    r"^(?:增加|新建)患者\s+(?P<name2>[^\s，,。]+)\s*$|"
    r"(?:给|为|帮)(?P<name3>[^\s，,。]+)建档"
)
_CN_STRENGTH = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


def extract_create_name(query: str) -> str | None:
    """从「增加/创建患者…」类语句提取待建档姓名。"""
    match = _CREATE_NAME_RE.search(query.strip())
    if not match:
        return None
    name = (match.group("name") or match.group("name2") or match.group("name3") or "").strip()
    if not name or name in _INVALID_CREATE_NAMES:
        return None
    return name


def _last_assistant_text(history: list[dict] | None) -> str:
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            return turn.get("content") or ""
    return ""


def _recent_blob(history: list[dict] | None, n: int = 6) -> str:
    return "\n".join((turn.get("content") or "") for turn in (history or [])[-n:])


def _is_create_flow(history: list[dict] | None) -> bool:
    """正在为新患者建档/补资料，不应误更新历史里提到的其他患者。"""
    for turn in reversed((history or [])[-10:]):
        text = turn.get("content") or ""
        if turn.get("role") == "user" and re.search(
            r"增加患者|新建患者|创建患者|创建病历|重新建档|应该是[a-z]", text, re.I
        ):
            return True
        if turn.get("role") == "assistant" and re.search(
            r"提供.+的(?:年龄|性别|诊断|患肢|肌力)|重新提供.+建档|重新建档|"
            r"无法直接修改|已为您删除|已删除原记录",
            text,
        ):
            return True
    return False


def _extract_pending_create_name(history: list[dict] | None) -> str | None:
    last = _last_assistant_text(history)
    for pat in (
        r"请提供\s*([^\s的]+)\s*的",
        r"提供\s*([^\s的]+)\s*的(?:年龄|性别|诊断)",
        r"重新提供([^\s的]+)的",
        r"为患者\s*([^\s（(]+)",
    ):
        match = re.search(pat, last)
        if match:
            name = match.group(1).strip()
            if name not in {"您", "患者", "新", "该", "这位"}:
                return name
    for turn in reversed((history or [])[-10:]):
        if turn.get("role") != "user":
            continue
        text = turn.get("content") or ""
        name = extract_create_name(text)
        if name:
            return name
        match = re.search(r"增加患者\s*([^\s，,。]+)|新建患者\s*([^\s，,。]+)", text)
        if match:
            return (match.group(1) or match.group(2)).strip()
    return None


def _extract_pending_patient_id(history: list[dict] | None) -> str | None:
    blob = _recent_blob(history, 12)
    for pat in (
        r"应该是\s*([a-z]{2,8}\d*)",
        r"ID[是为：:\s]+([a-z]{2,8}\d*)",
        r"\(ID[:\s]*([a-z0-9]+)\)",
        r"已为患者\S+（ID\s*([a-z0-9]+)）",
    ):
        match = re.search(pat, blob, re.I)
        if match:
            return _normalize_patient_id(match.group(1))
    match = re.search(r"应该是\s*([a-z]{2,6})(?:\d{3})?", blob, re.I)
    if match:
        return _normalize_patient_id(match.group(1))
    return None


def _normalize_patient_id(pid: str) -> str:
    pid = pid.lower()
    if re.fullmatch(r"[a-z]{2,8}", pid):
        return f"{pid}001"
    return pid


def _iter_cjk_sliding_chunks(text: str, *, min_len: int = 2, max_len: int = 4):
    """从 CJK 连续段提取 2-4 字滑动窗口（避免贪婪 regex 把姓名切碎）。"""
    for run in re.finditer(r"[\u4e00-\u9fff]+", text):
        s = run.group()
        for n in range(min(max_len, len(s)), min_len - 1, -1):
            for i in range(len(s) - n + 1):
                yield s[i : i + n]


def _resolve_patient_id(
    history: list[dict] | None,
    patient_id: str | None,
    *,
    query: str = "",
) -> str | None:
    if _is_create_flow(history):
        return None

    from agent.tools.context import get_context

    repo = get_context().repository
    if patient_id:
        try:
            return repo.resolve(patient_id).patient_id
        except PatientNotFoundError:
            pass

    focus = f"{query}\n{_recent_blob(history, 6)}"
    focus_py = pinyin_key(focus)
    for pid in repo.list_ids():
        try:
            rec = repo.get(pid)
        except (PatientNotFoundError, Exception):
            continue
        if rec.name and rec.name in focus:
            return rec.patient_id
        if rec.patient_id.lower() in focus.lower():
            return rec.patient_id
        # 拼音层：语音 ASR 把姓名转成同音/近音字时，字符匹配失败但拼音可命中。
        name_py = pinyin_key(rec.name or "")
        if name_py and len(name_py) >= 4 and focus_py and name_py in focus_py:
            return rec.patient_id

    # 从当前句抽姓名片段走 resolve（含拼音/首字母层），覆盖 ASR 近音误识别。
    seen_chunk: set[str] = set()
    for chunk in _iter_cjk_sliding_chunks(query):
        if chunk in seen_chunk:
            continue
        seen_chunk.add(chunk)
        try:
            return repo.resolve(chunk).patient_id
        except PatientNotFoundError:
            continue

    match = _PID_RE.search(query) or _PID_RE.search(_recent_blob(history, 4))
    if match:
        try:
            return repo.resolve(match.group(1)).patient_id
        except PatientNotFoundError:
            pass
    return None


def parse_muscle_strength(query: str) -> int | None:
    text = query.strip().rstrip("。.!！?？")
    if text in _CN_STRENGTH:
        return _CN_STRENGTH[text]
    compact = text.replace("级", "").strip()
    if compact in _CN_STRENGTH:
        return _CN_STRENGTH[compact]
    if compact.isdigit():
        val = int(compact)
        return val if 0 <= val <= 5 else None
    match = re.search(r"([0-5一二三四五])[级]?", text)
    if match:
        ch = match.group(1)
        if ch.isdigit():
            return int(ch)
        return _CN_STRENGTH.get(ch)
    return None


def parse_balance_level(query: str) -> str | None:
    text = query.strip().rstrip("。.!！?？")
    if re.search(r"良\s*好|良好", text):
        return "良"
    for level in BALANCE_LEVELS:
        if level in text:
            return level
    m = re.search(r"平衡(?:等级)?[^0-5一二三四五]*([0-5一二三四五])\s*级?", text)
    if m:
        ch = m.group(1)
        n = int(ch) if ch.isdigit() else _CN_STRENGTH.get(ch, 0)
        return _BALANCE_BY_SCORE.get(n)
    if re.search(r"平衡", text) and re.search(r"五级", text):
        return _BALANCE_BY_SCORE[5]
    return None


def _score_from_text(text: str, *, keyword: str | None = None) -> int | None:
    """从「肌力三级」「平衡等级五级」等片段取 0-5 分值。"""
    if keyword:
        m = re.search(rf"{keyword}[^0-5一二三四五]*([0-5一二三四五])\s*级?", text)
        if m:
            ch = m.group(1)
            return int(ch) if ch.isdigit() else _CN_STRENGTH.get(ch)
        if keyword == "平衡" and re.search(r"五级", text):
            return 5
        return None
    m = re.search(r"([0-5一二三四五])\s*级", text)
    if m:
        ch = m.group(1)
        return int(ch) if ch.isdigit() else _CN_STRENGTH.get(ch)
    return None


def _free_text_diagnosis(parts: list[str]) -> str | None:
    """从分词里挑一个像「诊断」的自由文本（非数字/性别/肢体/年龄）。"""
    for p in parts:
        if p.isdigit() or p in _CN_STRENGTH or re.fullmatch(r"[0-5]级?", p):
            continue
        if p in {"男", "女", "男性", "女性"} or "岁" in p:
            continue
        if any(limb in p for limb in (*_LIMB_VALUES, "下身", "上身")):
            continue
        if p in _DIAGNOSIS_SKIP_PARTS or p.endswith("：") or p.endswith(":"):
            continue
        if 1 < len(p) <= 12:
            return p
    return None


def parse_clinical_fields(query: str) -> dict:
    """从自然语言里抽取可落库字段。

    分词按「逗号/顿号/分号/空白」统一切，兼容 "11，男，..." 与 "11 男 ..." 两种说法。
    """
    text = query.strip()
    fields: dict = {}

    # 性别
    if re.search(r"性别\s*男|(?<![a-zA-Z])男(?:性)?(?![a-zA-Z])", text):
        fields["gender"] = "男"
    elif re.search(r"性别\s*女|(?<![a-zA-Z])女(?:性)?(?![a-zA-Z])", text):
        fields["gender"] = "女"

    # 患肢（腿部/右侧腿 → 下肢，优先于泛匹配）
    if re.search(r"腿部|大腿|小腿|膝|踝|脚|下身|右腿|左腿|下肢", text):
        fields["affected_limb"] = "下肢"
    elif "下身" in text:
        fields["affected_limb"] = "下肢"
    else:
        for limb in _LIMB_VALUES:
            if limb in text:
                fields["affected_limb"] = limb
                break

    parts = [p.strip() for p in re.split(r"[，,、;；\s]+", text) if p.strip()]
    digit_nums = [int(p) for p in parts if p.isdigit()]
    small_nums: list[int] = []
    for p in parts:
        m = re.fullmatch(r"([0-5])级?", p)
        if m:
            small_nums.append(int(m.group(1)))
        elif p in _CN_STRENGTH:
            small_nums.append(_CN_STRENGTH[p])

    # 年龄：显式"岁"优先；否则取第一个 >5 的整数（肌力/平衡都落在 0-5）
    age_match = re.search(r"(\d{1,3})\s*岁", text)
    if age_match:
        fields["age"] = int(age_match.group(1))
    else:
        age_candidates = [n for n in digit_nums if 5 < n <= 120]
        if age_candidates:
            fields["age"] = age_candidates[0]

    is_balance_ctx = bool(re.search(r"平衡", text))
    is_muscle_ctx = bool(re.search(r"肌力", text))
    bal_score = _score_from_text(text, keyword="平衡")
    mus_score = _score_from_text(text, keyword="肌力")

    if bal_score is not None:
        fields["balance_level"] = _BALANCE_BY_SCORE.get(bal_score)
    if mus_score is not None:
        fields["muscle_strength"] = mus_score

    if bal_score is None and mus_score is None:
        if small_nums:
            if is_balance_ctx and not is_muscle_ctx:
                fields["balance_level"] = _BALANCE_BY_SCORE.get(small_nums[0])
            elif is_muscle_ctx and not is_balance_ctx:
                fields["muscle_strength"] = small_nums[0]
            else:
                fields["muscle_strength"] = small_nums[0]
                if len(small_nums) >= 2:
                    fields["balance_level"] = _BALANCE_BY_SCORE.get(small_nums[1])
        else:
            m = re.search(r"肌力\s*([0-5一二三四五])", text)
            if m:
                ch = m.group(1)
                fields["muscle_strength"] = int(ch) if ch.isdigit() else _CN_STRENGTH.get(ch)

    explicit_balance = parse_balance_level(text)
    if explicit_balance is not None:
        fields["balance_level"] = explicit_balance

    # 诊断：关键词 > 标准枚举 > 截瘫别名 > 字段列表兜底
    if "截肢" in text:
        fields["diagnosis"] = "截肢"
    for diag in DIAGNOSIS_VALUES:
        if diag in text and "diagnosis" not in fields:
            fields["diagnosis"] = diag
            break
    if "diagnosis" not in fields:
        if "截瘫" in text:
            fields["diagnosis"] = "脊髓损伤"
        elif len(parts) >= 3:
            # 仅在「字段列表」式输入里兜底自由文本诊断，避免把整句话当诊断
            candidate = _free_text_diagnosis(parts)
            if candidate:
                fields["diagnosis"] = candidate

    name_match = re.search(r"修改.*名为\s*([^\s，,。]+)|改名为\s*([^\s，,。]+)", text)
    if name_match:
        fields["name"] = (name_match.group(1) or name_match.group(2) or "").strip()
    return fields


def try_direct_tool_call(
    query: str,
    history: list[dict] | None = None,
    *,
    patient_id: str | None = None,
) -> ToolCall | None:
    """识别碎片回答并直连 update/get/create。"""
    q = query.strip()
    if not q:
        return None

    fields = parse_clinical_fields(q)
    last = _last_assistant_text(history)

    # ---- 新建患者流程：优先 create，避免误改 whl001 等旧患者 ----
    if _is_create_flow(history) and fields:
        name = _extract_pending_create_name(history)
        if name:
            from agent.tools.context import get_context

            repo = get_context().repository
            pid = _extract_pending_patient_id(history)
            if pid:
                pid = _normalize_patient_id(pid)
            else:
                pid = allocate_patient_id(repo, name)
            create_fields = {k: v for k, v in fields.items() if k != "name"}
            if repo.exists(pid):
                return ToolCall(
                    id="direct",
                    name="update_patient_record",
                    arguments={"patient_id": pid, "updates": create_fields},
                )
            return ToolCall(
                id="direct",
                name="create_patient_record",
                arguments={"patient_id": pid, "name": name, **create_fields},
            )

    pid = _resolve_patient_id(history, patient_id, query=q)

    # 用户纠正「不是肌力」：回看上一条含「平衡」的表述
    if re.search(r"不是肌力|不是肌肉|搞错.*肌力", q):
        for turn in reversed(history or []):
            if turn.get("role") != "user":
                continue
            prev = (turn.get("content") or "").strip()
            if prev == q or "平衡" not in prev:
                continue
            pid2 = _resolve_patient_id(history, patient_id, query=prev)
            level = parse_balance_level(prev)
            if pid2 and level:
                return ToolCall(
                    id="direct",
                    name="update_patient_record",
                    arguments={"patient_id": pid2, "updates": {"balance_level": level}},
                )
            break

    # 句中写明平衡/肌力时直连（不依赖上轮助手是否在追问）
    if pid and re.search(r"平衡", q):
        updates: dict = {}
        level = parse_balance_level(q)
        if level:
            updates["balance_level"] = level
        cf = parse_clinical_fields(q)
        cf.pop("muscle_strength", None)
        for k, v in cf.items():
            if k not in updates and k != "name":
                updates[k] = v
        if updates:
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": updates},
            )

    if pid and re.search(r"肌力", q):
        updates = {}
        strength = parse_muscle_strength(q)
        if strength is not None:
            updates["muscle_strength"] = strength
        cf = parse_clinical_fields(q)
        cf.pop("balance_level", None)
        for k, v in cf.items():
            if k not in updates and k != "name":
                updates[k] = v
        if updates:
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": updates},
            )

    if _SUPPLEMENT.search(q) and pid:
        return ToolCall(id="direct", name="get_patient_record", arguments={"patient_id": pid})

    if pid and fields:
        if _GENDER_CORRECTION.search(q) and "gender" in fields:
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": {"gender": fields["gender"]}},
            )
        if len(fields) >= 1 and (
            _MUSCLE_ASK.search(last)
            or _LIMB_ASK.search(last)
            or _BALANCE_ASK.search(last)
            or re.search(r"提供|补充|信息", last)
        ):
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": fields},
            )
        if _GENDER_CORRECTION.search(q) and "gender" not in fields:
            if re.search(r"男", q):
                fields["gender"] = "男"
            elif re.search(r"女", q):
                fields["gender"] = "女"
            if "gender" in fields:
                return ToolCall(
                    id="direct",
                    name="update_patient_record",
                    arguments={"patient_id": pid, "updates": fields},
                )

    if pid and _MUSCLE_ASK.search(last):
        strength = parse_muscle_strength(q)
        if strength is not None:
            updates = {"muscle_strength": strength}
            for limb in _LIMB_VALUES:
                if limb in q:
                    updates["affected_limb"] = limb
                    break
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": updates},
            )

    if pid and _BALANCE_ASK.search(last):
        level = parse_balance_level(q)
        if level is not None:
            return ToolCall(
                id="direct",
                name="update_patient_record",
                arguments={"patient_id": pid, "updates": {"balance_level": level}},
            )

    return None
