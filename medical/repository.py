"""最小病历数据层：开发期本地 JSON，未来可直接替换成数据库实现。"""

from __future__ import annotations

import json
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
]

DIAGNOSIS_VALUES = [
    "脑卒中",
    "脊髓损伤",
    "颅脑损伤",
    "膝关节置换术后",
    "髋关节置换术后",
    "骨折术后",
    "肌少症",
    "帕金森病",
    "其他",
]
AFFECTED_LIMB_VALUES = ["上肢", "下肢", "双上肢", "双下肢", "四肢", "无"]
BALANCE_LEVELS = ["差", "中", "良", "正常"]

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
    rec.created_at = now
    rec.updated_at = now
    return rec


class PatientNotFoundError(KeyError):
    """请求的 patient_id 不存在。"""


class PatientRepository:
    """最小本地仓库。接口尽量贴近未来数据库访问层。"""

    def __init__(self, patients_dir: str, *, keep_backup: bool = True) -> None:
        self.dir = Path(patients_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_backup = keep_backup  # 兼容旧装配参数；当前不使用

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

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def search(self, keyword: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pid in self.list_ids():
            try:
                rec = self.get(pid)
            except (PatientNotFoundError, json.JSONDecodeError):
                continue
            if keyword and keyword not in pid and keyword not in (rec.name or ""):
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
