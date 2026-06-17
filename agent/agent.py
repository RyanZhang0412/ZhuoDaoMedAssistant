"""本地 Agent 主循环 —— 基于 LLM function calling，注册本地工具，无任何联网工具。

工具注册与 schema 绑定：@register_tool(schema={...}) 在工具同文件一次声明，
写入全局 TOOL_REGISTRY；Agent.get_tool_schemas() 直接收集。
（不学百聆把 schema 放独立 json，避免与代码漂移。）

ToolResult.action 语义：
  REQLLM   —— 结果需回灌 LLM 二次组织语言（读病历/推方案/列表类）
  RESPONSE —— 话术已就绪，可直接用（建档/排期确认类）
Agent.chat 内部处理 REQLLM 循环（max_tool_rounds 兜底防死循环）。

依赖统一的 core.llm.base.LLMBase（与 explainer 共用同一抽象）。
对外稳定名 Agent = LocalAgent（见文件末尾别名），供 core/robot 引用。
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from agent.direct_tools import try_direct_tool_call
from agent.guards import (
    WRITE_TOOL_NAMES,
    PendingWrite,
    claims_write_without_tool,
    confirmation_prompt,
    is_confirmation,
    is_denial,
    precheck_user_query,
    reject_hallucinated_write,
)
from agent.intent import infer_tool_choice
from core.llm.base import LLMBase, Message, ToolCall

__all__ = [
    "LocalAgent",
    "Agent",
    "ToolResult",
    "ToolAction",
    "RegisteredTool",
    "register_tool",
    "TOOL_REGISTRY",
]


class ToolAction(enum.Enum):
    REQLLM = "reqllm"        # 结果回灌 LLM 二次组织语言
    RESPONSE = "response"    # 话术已就绪
    NOTFOUND = "notfound"    # 工具未注册
    ERROR = "error"          # 执行出错
    PENDING = "pending"      # 写操作待用户确认（未落库）


@dataclass
class ToolResult:
    """工具执行后的结果（区别于 core.llm.ToolCall：那是模型"想调用"的意图）。

    action=PENDING 时 pending 字段携带待确认的写调用与摘要话术，
    由 Agent 暂存到会话状态、下一轮按用户确认/取消决定是否真正执行。
    """

    action: ToolAction
    result: Any = None          # 结构化数据（回灌 LLM 用）
    response: str | None = None  # 直接话术（RESPONSE 时）
    pending: "PendingWrite | None" = None


@dataclass
class RegisteredTool:
    name: str
    func: Callable[..., ToolResult]
    schema: dict


# 全局注册表：name -> RegisteredTool
TOOL_REGISTRY: dict[str, RegisteredTool] = {}


def register_tool(schema: dict) -> Callable[[Callable], Callable]:
    """装饰器：把工具函数与其 schema 一次性绑定并注册。

    schema 形如 {"name", "description", "input_schema": {...JSON Schema...}}
    （anthropic 原生格式；LLM 适配层会按 provider 转换）。
    """

    def deco(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        name = schema["name"]
        TOOL_REGISTRY[name] = RegisteredTool(name=name, func=func, schema=schema)
        return func

    return deco


class LocalAgent:
    """本地 agent。LLM 决定调用哪个工具，Agent 执行并循环到无工具调用。"""

    def __init__(
        self,
        llm: LLMBase,
        *,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        # 写操作确认层：按 session_id 暂存「上一轮被拦下的写调用」。
        # Agent 自身不持久化，重启即清空；会话历史仍由 Robot/DialogueMemory 持有。
        self._pending_writes: dict[str, PendingWrite] = {}
        self._bypass_confirm: set[str] = set()

    # ---- 工具 schema 收集 ----
    def get_tool_schemas(self) -> list[dict]:
        return [t.schema for t in TOOL_REGISTRY.values()]

    # ---- 依赖绑定（装配末期调用，把 repository/recommender/scheduler 注入工具上下文）----
    @staticmethod
    def bind_context(repository, recommender, scheduler) -> None:  # noqa: ANN001
        from agent.tools.context import ToolContext, set_context

        set_context(ToolContext(repository=repository, recommender=recommender, scheduler=scheduler))

    # ---- 主入口：吃文本、吐文本 ----
    def chat(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        patient_id: str | None = None,
        extra_system: str | None = None,
        session_id: str = "default",
    ) -> str:
        """处理一轮用户输入，返回最终文本回复。

        history: 形如 [{"role","content"}] 的历史（由 Robot/DialogueMemory 提供，
                 Agent 自身不持久化历史 —— 单一所有者是 Robot）。
        patient_id: 当前会话聚焦的患者（注入系统提示，便于 LLM 省略追问）。
        extra_system: 附加系统上下文（如长期偏好记忆），追加在 system prompt 之后。
        session_id: 会话标识，用于隔离各会话的待确认写操作状态。
        """
        # 写操作确认层：先看用户是否在回应上一条待确认的写操作。
        resolved = self._resolve_pending(query, session_id)
        if resolved is not None:
            return resolved

        messages = self._build_messages(query, history, patient_id)
        system = self._compose_system(extra_system)
        tools = self.get_tool_schemas()
        blocked = precheck_user_query(query)
        if blocked:
            return blocked
        direct = try_direct_tool_call(query, history, patient_id=patient_id)
        if direct is not None:
            reply = self._run_direct_tool(
                messages, direct, system=system, tools=tools, session_id=session_id
            )
            if reply is not None:
                return reply
        direct_response: str | None = None
        for round_i in range(self.max_tool_rounds):
            tool_choice = infer_tool_choice(query, history) if round_i == 0 else None
            resp = self.llm.chat(
                messages, system=system, tools=tools, tool_choice=tool_choice
            )

            # 无工具调用 -> 直接返回文本
            if not resp.tool_calls:
                text = resp.text or ""
                if claims_write_without_tool(text):
                    if round_i == 0:
                        retry = self.llm.chat(
                            messages,
                            system=system,
                            tools=tools,
                            tool_choice="required",
                        )
                        if retry.tool_calls:
                            dr = self._apply_tool_round(
                                messages, retry, session_id=session_id
                            )
                            direct_response = dr or direct_response
                            continue
                    return reject_hallucinated_write()
                return text

            dr = self._apply_tool_round(messages, resp, session_id=session_id)
            direct_response = dr or direct_response
            # 写操作被拦成待确认：不再继续 LLM 循环，直接返回"即将X，确认吗"。
            if self._pending_writes.get(session_id) is not None:
                return direct_response or ""
            # 继续下一轮，让 LLM 基于工具结果组织语言

        # 达到最大轮数仍未收敛：返回最近一次直接话术或提示
        return direct_response or "（已达到最大工具调用轮数，请补充信息后重试）"

    # ---- 流式入口：吐文本增量（语音管线用，对齐百聆 chat_tool 思路）----
    def chat_stream(
        self,
        query: str,
        *,
        history: list[dict] | None = None,
        patient_id: str | None = None,
        extra_system: str | None = None,
        session_id: str = "default",
    ) -> Iterator[str]:
        """与 chat 等价，但以增量方式产出文本（供按句 TTS 尽早开播）。

        工具轮内的文本（"我查一下…"之类前导语）也会被产出，符合语音直觉。
        """
        # 写操作确认层：确认/取消话术一次性产出，不走流式 LLM。
        resolved = self._resolve_pending(query, session_id)
        if resolved is not None:
            yield resolved
            return

        messages = self._build_messages(query, history, patient_id)
        system = self._compose_system(extra_system)
        tools = self.get_tool_schemas()
        blocked = precheck_user_query(query)
        if blocked:
            yield blocked
            return
        direct = try_direct_tool_call(query, history, patient_id=patient_id)
        if direct is not None:
            reply = self._run_direct_tool(
                messages, direct, system=system, tools=tools, session_id=session_id
            )
            if reply is not None:
                yield reply
                return
        direct_response: str | None = None
        for round_i in range(self.max_tool_rounds):
            tool_choice = infer_tool_choice(query, history) if round_i == 0 else None
            resp = yield from self.llm.stream_chat_tools(
                messages, system=system, tools=tools, tool_choice=tool_choice
            )

            if not resp.tool_calls:
                # stream_chat_tools 已逐字 yield 过 resp.text，这里不再重复产出。
                # 仅在本轮模型没说话但前序工具已有现成话术时兜底播报。
                if not (resp.text or "").strip() and direct_response:
                    yield direct_response
                return

            dr = self._apply_tool_round(messages, resp, session_id=session_id)
            direct_response = dr or direct_response
            # 写操作被拦成待确认：不再继续 LLM 循环，产出"即将X，确认吗"。
            if self._pending_writes.get(session_id) is not None:
                if direct_response:
                    yield direct_response
                return

        yield direct_response or "（已达到最大工具调用轮数，请补充信息后重试）"

    def _run_direct_tool(
        self,
        messages: list[Message],
        call: ToolCall,
        *,
        system: str | None,
        tools: list[dict],
        session_id: str = "default",
    ) -> str | None:
        """执行直连工具；update 类直接返回话术，get 类再走一轮 LLM 组织语言。"""
        tr = self._dispatch_tool(call, session_id=session_id)
        if tr.action == ToolAction.PENDING and tr.pending is not None:
            return tr.pending.description
        if tr.action == ToolAction.RESPONSE and tr.response:
            return tr.response
        if tr.action != ToolAction.REQLLM:
            return tr.response or str(tr.result)
        messages.append(Message(role="assistant", content=None, tool_calls=[call]))
        messages.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                content=json.dumps(tr.result if tr.result is not None else {}, ensure_ascii=False),
            )
        )
        resp = self.llm.chat(messages, system=system, tools=tools)
        return resp.text or tr.response or ""

    # ---- 单轮工具调用的回灌（chat / chat_stream 共用）----
    def _apply_tool_round(
        self, messages: list[Message], resp, *, session_id: str = "default"
    ) -> str | None:
        """执行 resp 中的工具调用并把结果回灌 messages，返回可兜底的直接话术。"""
        # 记录 assistant 的工具调用意图
        messages.append(
            Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
        )
        direct_response: str | None = None
        for call in resp.tool_calls:
            tr = self._dispatch_tool(call, session_id=session_id)
            if tr.action == ToolAction.PENDING and tr.pending is not None:
                # 写操作被拦下待确认：把"即将X，确认吗"作为本轮直接话术，
                # 同时回灌一个 tool 消息提示 LLM 等待用户确认，不要继续编造结果。
                direct_response = tr.pending.description
                payload = {"status": "pending_confirmation", "message": tr.pending.description}
            elif tr.action == ToolAction.RESPONSE and tr.response:
                # 话术已就绪，但仍回灌让 LLM 串成连贯回复；同时记下兜底
                direct_response = tr.response
                payload = {"status": "ok", "message": tr.response}
            elif tr.action == ToolAction.REQLLM:
                payload = tr.result if tr.result is not None else {}
            elif tr.action == ToolAction.NOTFOUND:
                payload = {"error": f"工具未注册: {call.name}"}
            else:  # ERROR
                payload = {"error": str(tr.result)}
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(payload, ensure_ascii=False),
                )
            )
        return direct_response

    def _build_messages(
        self, query: str, history: list[dict] | None, patient_id: str | None
    ) -> list[Message]:
        messages: list[Message] = _to_messages(history)
        # 当前患者上下文注入（轻量）
        user_text = query
        if patient_id:
            user_text = f"[当前患者ID: {patient_id}]\n{query}"
        messages.append(Message(role="user", content=user_text))
        return messages

    def _compose_system(self, extra_system: str | None) -> str | None:
        parts = [p for p in (self.system_prompt, extra_system) if p]
        return "\n\n".join(parts) if parts else None

    # ---- 工具分发 ----
    def _dispatch_tool(
        self, call: ToolCall, *, session_id: str = "default"
    ) -> ToolResult:
        tool = TOOL_REGISTRY.get(call.name)
        if tool is None:
            return ToolResult(action=ToolAction.NOTFOUND, result=call.name)
        # 写操作确认层：除非本轮是用户确认后的 bypass，否则不立即落库，
        # 而是暂存待确认、返回「即将X，确认吗」。
        if (
            call.name in WRITE_TOOL_NAMES
            and session_id not in self._bypass_confirm
        ):
            # 拦截前先规范化 patient_id（如把 LLM 传的 "wzc" resolve 成 "wzc001"），
            # 让确认话术与后续执行基于同一真实 ID，避免「显示wzc、实际删wzc001」的不一致。
            normalized = self._normalize_write_call(call)
            self._pending_writes[session_id] = PendingWrite(
                call=normalized, description=confirmation_prompt(normalized)
            )
            return ToolResult(
                action=ToolAction.PENDING,
                pending=self._pending_writes[session_id],
            )
        # 进入实际执行：若是 bypass，消费一次性标志，避免后续写操作绕过确认。
        self._bypass_confirm.discard(session_id)
        try:
            return tool.func(**call.arguments)
        except Exception as e:  # 工具内异常不应崩溃整个 agent
            return ToolResult(action=ToolAction.ERROR, result=f"{type(e).__name__}: {e}")

    @staticmethod
    def _normalize_write_call(call: ToolCall) -> ToolCall:
        """把写工具调用里的 patient_id resolve 成仓库中的真实 ID。

        复现场景：LLM 传 "wzc" 而文件名是 "wzc001"。不规范化会导致确认话术
        显示短名、工具内部却删另一个 ID，确认失去意义。resolve 失败时原样返回，
        由工具执行时给出「未找到」提示。
        """
        from agent.tools.context import get_context
        from medical.repository import PatientNotFoundError

        args = dict(call.arguments or {})
        raw = args.get("patient_id")
        if not raw:
            return call
        try:
            rec = get_context().repository.resolve(raw)
        except (PatientNotFoundError, Exception):
            return call
        if rec.patient_id and rec.patient_id != raw:
            args["patient_id"] = rec.patient_id
            return ToolCall(
                id=call.id,
                name=call.name,
                arguments=args,
                raw_arguments=call.raw_arguments,
                raw=call.raw,
            )
        return call

    # ---- 写操作确认层：处理用户对上一条待确认写操作的回应 ----
    def _resolve_pending(self, query: str, session_id: str) -> str | None:
        """若用户正在回应待确认写操作，返回应答话术；否则返回 None 走正常流程。

        - 确认 -> 取出 pending，置 bypass 标志后真正执行该写工具。
        - 取消 -> 丢弃 pending。
        - 既非确认也非取消（改话题）-> 丢弃 pending，返回 None 让本轮正常处理。
        """
        pending = self._pending_writes.get(session_id)
        if pending is None:
            return None

        if is_confirmation(query):
            self._pending_writes.pop(session_id, None)
            self._bypass_confirm.add(session_id)
            tr = self._dispatch_tool(pending.call, session_id=session_id)
            if tr.action == ToolAction.PENDING and tr.pending is not None:
                # 理论上不会发生（已置 bypass）；防御性返回提示。
                return tr.pending.description
            return tr.response or str(tr.result or "已执行")

        if is_denial(query):
            self._pending_writes.pop(session_id, None)
            return "已取消，未做任何修改。"

        # 用户改话题：清掉 pending，让本轮按新意图正常处理。
        self._pending_writes.pop(session_id, None)
        return None


def _to_messages(history: list[dict] | None) -> list[Message]:
    """把 [{"role","content"}] 历史转为统一 Message 列表（边界转换点）。"""
    if not history:
        return []
    return [Message(role=h["role"], content=h.get("content", "")) for h in history]


# 对外稳定名（core/robot 等引用 Agent）
Agent = LocalAgent
