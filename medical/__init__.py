"""medical 包：最小医疗核心（数据层 + 规则层 + 薄服务层）。"""

from medical.repository import PatientNotFoundError, PatientRecord, PatientRepository, new_record
from medical.rules import RuleEngine, RuleEngineResult
from medical.service import RecommendationResult, Recommender

__all__ = [
    "PatientRecord",
    "PatientRepository",
    "PatientNotFoundError",
    "new_record",
    "RuleEngine",
    "RuleEngineResult",
    "Recommender",
    "RecommendationResult",
]
