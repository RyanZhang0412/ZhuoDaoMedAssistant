"""offline 强约束 —— 全系统唯一的禁网真相源。

按交叉验证结论收口：整个项目只有这一个 ``OfflineViolationError`` 定义，
core/llm、core/tts、medical 等所有需要表达"违反离线约束"的地方都从这里 import，
保证上层 ``except OfflineViolationError`` 不会因异常类不同而漏网。

两道防线：
  1. socket 级硬兜底：``enforce_offline()`` monkey-patch ``socket.socket.connect``，
     拦截一切指向非本地主机的连接。即便某 LLM provider 误配了公网 base_url，
     也会在真正建立连接时抛异常，而不是把患者隐私静默外发。这是最外层、装配最早期就装上。
  2. 端点早失败层：core/llm、core/tts 在构造时校验 base_url / 引擎是否本地，
     命中即抛同一个 ``OfflineViolationError``，给出清晰报错（友好层，非安全边界）。

装配顺序固定：load_config -> enforce_offline(if offline) -> 其余组件构造。
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable

__all__ = [
    "OfflineViolationError",
    "enforce_offline",
    "is_offline_enforced",
    "is_local_host",
    "assert_local_endpoint",
]


class OfflineViolationError(RuntimeError):
    """违反离线约束时抛出（全系统唯一定义）。

    任何模块表达"试图联网/配置了公网地址/选了联网引擎"都用这个异常，
    上层只需 ``except OfflineViolationError`` 即可一网打尽。
    """


# 进程级状态：是否已经装上 socket 拦截
_ENFORCED = False
_ALLOW_HOSTS: set[str] = set()
_ORIGINAL_CONNECT = None  # 保存原始 socket.connect 以便（测试中）恢复


def is_local_host(host: str | None) -> bool:
    """判断主机是否为本地（loopback）。

    放行 localhost、127.0.0.0/8、::1，以及显式配置在 allow_hosts 里的主机
    （例如本地 ollama 所在的内网地址，若用户主动加入）。
    """
    if not host:
        return False
    host = host.strip().strip("[]")  # 去掉 IPv6 字面量的方括号
    if host in _ALLOW_HOSTS or host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        # 不是 IP 字面量的主机名（且不在白名单）一律视为非本地
        return False


def _guarded_connect(self, address):  # noqa: ANN001 - 兼容 socket.connect 原签名
    """替换 socket.socket.connect：非本地目标直接拒绝。"""
    host = None
    try:
        if isinstance(address, tuple) and address:
            host = address[0]
    except Exception:  # pragma: no cover - 极端地址形态
        host = None

    if not is_local_host(host):
        raise OfflineViolationError(
            f"offline=true：已拦截对非本地地址的网络连接 -> {address!r}。"
            f"仅放行 {sorted(_ALLOW_HOSTS) or ['127.0.0.1', 'localhost']}。"
            f"如需联网请在 config.yaml 关闭 offline，或检查 LLM/TTS 是否误配了公网地址。"
        )
    return _ORIGINAL_CONNECT(self, address)


def enforce_offline(allow_hosts: Iterable[str] = ("127.0.0.1", "localhost")) -> None:
    """装上 socket 级禁网兜底（幂等）。应在任何组件构造之前调用。

    monkey-patch ``socket.socket.connect``，只放行本地与 allow_hosts。
    这是最外层硬防线；具体 provider 的端点校验是更友好的早失败层。
    """
    global _ENFORCED, _ORIGINAL_CONNECT
    _ALLOW_HOSTS.clear()
    _ALLOW_HOSTS.update(h.strip() for h in allow_hosts if h and h.strip())

    if _ENFORCED:
        return
    _ORIGINAL_CONNECT = socket.socket.connect
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    _ENFORCED = True


def is_offline_enforced() -> bool:
    """当前进程是否已启用 socket 级禁网兜底。"""
    return _ENFORCED


def assert_local_endpoint(url: str | None, *, what: str = "endpoint") -> None:
    """端点早失败校验：若已启用 offline，则要求 url 指向本地，否则抛异常。

    供 core/llm、core/tts 在构造时调用，给出比 socket 拦截更清晰的报错。
    未启用 offline 时不做限制（开发期用 API）。
    """
    if not _ENFORCED:
        return
    host = _extract_host(url)
    if not is_local_host(host):
        raise OfflineViolationError(
            f"offline=true：{what} 指向非本地地址（{url!r}），已拒绝。"
            f"离线部署请改用本地服务（如本地 ollama / 本地模型）。"
        )


def _extract_host(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.hostname
    except Exception:  # pragma: no cover
        return None
