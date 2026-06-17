"""最小病历数据层：开发期本地 JSON，未来可直接替换成数据库实现。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PatientRecord",
    "PatientRepository",
    "PatientNotFoundError",
    "new_record",
    "RULE_CONDITION_FIELDS",
    "DIAGNOSIS_VALUES",
    "AFFECTED_LIMB_VALUES",
    "BALANCE_LEVELS",
    "CLINICAL_CREATE_FIELDS",
    "name_initials",
    "pinyin_key",
    "allocate_patient_id",
]

DIAGNOSIS_VALUES = [
    "脑卒中",
    "脊髓损伤",
    "颅脑损伤",
    "截肢",
    "膝关节置换术后",
    "髋关节置换术后",
    "骨折术后",
    "肌少症",
    "帕金森病",
    "其他",
]
AFFECTED_LIMB_VALUES = ["上肢", "下肢", "双上肢", "双下肢", "四肢", "无"]
BALANCE_LEVELS = ["差", "中", "良", "正常"]
CLINICAL_CREATE_FIELDS = frozenset(
    {"age", "gender", "diagnosis", "affected_limb", "muscle_strength", "balance_level"}
)

RULE_CONDITION_FIELDS = {
    "diagnosis",
    "affected_limb",
    "muscle_strength",
    "balance_level",
    "age",
    "injury_completeness",
    "post_op_day",
    "spasticity",
    "cognition",
}


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: int | None = None
    gender: str | None = None
    diagnosis: str | None = None
    affected_limb: str | None = None
    muscle_strength: int | None = None
    balance_level: str | None = None
    injury_completeness: str | None = None
    post_op_day: int | None = None
    spasticity: str | None = None
    cognition: str | None = None
    notes: str | None = None
    training_sessions: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None

    def condition_view(self) -> dict[str, Any]:
        return {
            f: getattr(self, f, None)
            for f in RULE_CONDITION_FIELDS
            if getattr(self, f, None) is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatientRecord":
        known = {k: data.get(k) for k in cls.__dataclass_fields__}
        known["training_sessions"] = list(data.get("training_sessions", []) or [])
        return cls(**known)


def new_record(patient_id: str, name: str, *, now: str | None = None, **fields) -> PatientRecord:
    rec = PatientRecord(patient_id=patient_id, name=name, **fields)
    if now is None:
        from datetime import datetime

        now = datetime.now().isoformat(timespec="seconds")
    rec.created_at = now
    rec.updated_at = now
    return rec


class PatientNotFoundError(KeyError):
    """请求的 patient_id 不存在。"""


class PatientRepository:
    """最小本地仓库。接口尽量贴近未来数据库访问层。"""

    def __init__(self, patients_dir: str) -> None:
        self.dir = Path(patients_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, patient_id: str) -> Path:
        return self.dir / f"{_safe_id(patient_id)}.json"

    def exists(self, patient_id: str) -> bool:
        return self._path(patient_id).exists()

    def get(self, patient_id: str) -> PatientRecord:
        path = self._path(patient_id)
        if not path.exists():
            raise PatientNotFoundError(patient_id)
        with path.open("r", encoding="utf-8") as f:
            return PatientRecord.from_dict(json.load(f))

    def resolve(self, query: str) -> PatientRecord:
        """按 patient_id 或姓名解析患者。

        匹配顺序：精确 ID → 姓名精确/子串 → 姓名拼音全等/子串 → 拼音首字母全等/子串。
        拼音/首字母层专为语音场景设计。命中多个时不猜，抛 PatientNotFoundError。
        """
        q = (query or "").strip()
        if not q:
            raise PatientNotFoundError(q)
        if self.exists(q):
            return self.get(q)

        exact: list[PatientRecord] = []
        partial: list[PatientRecord] = []
        for pid in self.list_ids():
            try:
                rec = self.get(pid)
            except (PatientNotFoundError, json.JSONDecodeError):
                continue
            name = rec.name or ""
            if name == q:
                exact.append(rec)
            elif q in pid or q in name:
                partial.append(rec)

        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise PatientNotFoundError(q)
        if len(partial) == 1:
            return partial[0]

        # 拼音层：查询与姓名都归一为全拼后再比（全等优先于子串）。
        q_py = pinyin_key(q)
        if q_py:
            py_exact: list[PatientRecord] = []
            py_partial: list[PatientRecord] = []
            for pid in self.list_ids():
                try:
                    rec = self.get(pid)
                except (PatientNotFoundError, json.JSONDecodeError):
                    continue
                name_py = pinyin_key(rec.name or "")
                if not name_py:
                    continue
                if name_py == q_py:
                    py_exact.append(rec)
                elif q_py in name_py:
                    py_partial.append(rec)
            if len(py_exact) == 1:
                return py_exact[0]
            if len(py_exact) > 1:
                names = "、".join(f"{r.name}({r.patient_id})" for r in py_exact)
                raise PatientNotFoundError(f"{q} 命中多个患者：{names}")
            if len(py_partial) == 1:
                return py_partial[0]
            if len(py_partial) > 1:
                names = "、".join(f"{r.name}({r.patient_id})" for r in py_partial)
                raise PatientNotFoundError(f"{q} 命中多个患者：{names}")

        # 首字母层：全拼对不上的近音字（祾 ling / 林 lin）首字母仍一致。
        init_exact: list[PatientRecord] = []
        init_partial: list[PatientRecord] = []
        for pid in self.list_ids():
            try:
                rec = self.get(pid)
            except (PatientNotFoundError, json.JSONDecodeError):
                continue
            kind = _initials_match_kind(q, rec.name or "")
            if kind == "exact":
                init_exact.append(rec)
            elif kind == "partial":
                init_partial.append(rec)
        if len(init_exact) == 1:
            return init_exact[0]
        if len(init_exact) > 1:
            names = "、".join(f"{r.name}({r.patient_id})" for r in init_exact)
            raise PatientNotFoundError(f"{q} 命中多个患者：{names}")
        if len(init_partial) == 1:
            return init_partial[0]
        if len(init_partial) > 1:
            names = "、".join(f"{r.name}({r.patient_id})" for r in init_partial)
            raise PatientNotFoundError(f"{q} 命中多个患者：{names}")

        raise PatientNotFoundError(q)

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def search(self, keyword: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        kw_py = pinyin_key(keyword) if keyword else ""
        kw_init = name_initials(keyword) if keyword else ""
        for pid in self.list_ids():
            try:
                rec = self.get(pid)
            except (PatientNotFoundError, json.JSONDecodeError):
                continue
            if keyword:
                name = rec.name or ""
                name_py = pinyin_key(name)
                name_init = name_initials(name)
                initials_hit = (
                    len(kw_init) >= 3
                    and len(name_init) >= 3
                    and (kw_init == name_init or kw_init in name_init)
                )
                # 字符子串、拼音子串或首字母命中均可（语音 ASR 同音/近音字场景）
                if (
                    keyword not in pid
                    and keyword not in name
                    and not (kw_py and name_py and kw_py in name_py)
                    and not initials_hit
                ):
                    continue
            out.append(
                {
                    "patient_id": rec.patient_id,
                    "name": rec.name,
                    "age": rec.age,
                    "diagnosis": rec.diagnosis,
                }
            )
        return out

    def create(self, record: PatientRecord) -> PatientRecord:
        if self.exists(record.patient_id):
            raise FileExistsError(f"患者已存在: {record.patient_id}")
        self._write(record)
        return record

    def upsert(self, record: PatientRecord) -> PatientRecord:
        self._write(record)
        return record

    def update(self, record: PatientRecord, *, expected_revision: int | None = None) -> PatientRecord:
        if not self.exists(record.patient_id):
            raise PatientNotFoundError(record.patient_id)
        self._write(record)
        return record

    def delete(self, patient_id: str) -> bool:
        path = self._path(patient_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _write(self, record: PatientRecord) -> None:
        path = self._path(record.patient_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)


def _safe_id(patient_id: str) -> str:
    if not patient_id or any(c in patient_id for c in ("/", "\\", "..", "\0")):
        raise ValueError(f"非法 patient_id: {patient_id!r}")
    return patient_id


def name_initials(name: str) -> str:
    """姓名 -> 拼音首字母（卓小道 -> zxd）。"""
    try:
        from pypinyin import Style, lazy_pinyin

        letters = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER))
    except Exception:
        letters = name
    letters = re.sub(r"[^a-z]", "", letters.lower())
    return letters or "pat"


def pinyin_key(name: str) -> str:
    """姓名 -> 全拼归一化（去声调、小写、无分隔）。

    用于语音场景的读音匹配：ASR 把"汪昊祾"转成"汪浩伦"时，字符对不上，
    但拼音都是 wanghao lun，归一后即可命中。非汉字原样小写保留。
    """
    try:
        from pypinyin import Style, lazy_pinyin

        parts = lazy_pinyin(name, style=Style.NORMAL)
    except Exception:
        parts = [name]
    return re.sub(r"[^a-z]", "", "".join(parts).lower())


def _initials_match_kind(query: str, name: str) -> str | None:
    """拼音首字母匹配：全等 > 子串。至少 3 字母才参与，避免短名误命中。

    覆盖全拼对不上的近音 ASR 误识别（如 祾 ling / 林 lin -> 首字母均为 l）。
    """
    qi = name_initials(query)
    ni = name_initials(name)
    if len(qi) < 3 or len(ni) < 3:
        return None
    if qi == ni:
        return "exact"
    if qi in ni:
        return "partial"
    return None


def allocate_patient_id(
    repo: "PatientRepository",
    name: str,
    preferred: str | None = None,
) -> str:
    """分配患者 ID：优先合法显式 ID，否则按姓名拼音首字母 + 序号（zxd001）。

    忽略 LLM 编造的 patNNN 占位 ID。
    """
    pid = (preferred or "").strip().lower()
    if pid and not re.fullmatch(r"pat\d{3,}", pid) and not repo.exists(pid):
        return pid
    base = name_initials(name)
    n = 1
    while True:
        candidate = f"{base}{n:03d}"
        if not repo.exists(candidate):
            return candidate
        n += 1
