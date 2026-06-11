"""工具运行时上下文容器。

保存已装配好的依赖单例（repository / recommender / scheduler），
供各 tools 模块的薄适配函数取用，避免工具自行构造依赖。
由 LocalAgent.bind_context() 在装配末期填充。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tools.schedule_tools import Scheduler
    from medical.repository import PatientRepository
    from medical.service import Recommender

__all__ = ["ToolContext", "set_context", "get_context"]


@dataclass
class ToolContext:
    repository: "PatientRepository"
    recommender: "Recommender"
    scheduler: "Scheduler"


_ctx: ToolContext | None = None


def set_context(c: ToolContext) -> None:
    global _ctx
    _ctx = c


def get_context() -> ToolContext:
    if _ctx is None:
        raise RuntimeError("ToolContext 未绑定。请先调用 LocalAgent.bind_context()")
    return _ctx
