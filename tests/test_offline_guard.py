"""offline 禁网守卫测试。

验证 enforce_offline 后：本地地址放行、非本地地址被拦截抛 OfflineViolationError，
以及 create_tts 在 offline 下拒绝联网 TTS。
"""

from __future__ import annotations

import socket

import pytest

import core.net_guard as ng


@pytest.fixture(autouse=True)
def _reset_guard():
    """每个用例后恢复 socket.connect，避免污染其他测试。"""
    original = socket.socket.connect
    ng._ENFORCED = False
    ng._ORIGINAL_CONNECT = None
    yield
    socket.socket.connect = original
    ng._ENFORCED = False


def test_local_host_allowed():
    ng.enforce_offline()
    assert ng.is_local_host("127.0.0.1")
    assert ng.is_local_host("localhost")
    assert not ng.is_local_host("api.deepseek.com")


def test_assert_local_endpoint_blocks_public():
    ng.enforce_offline()
    with pytest.raises(ng.OfflineViolationError):
        ng.assert_local_endpoint("https://api.deepseek.com/v1", what="LLM")
    # 本地端点放行（不抛）
    ng.assert_local_endpoint("http://127.0.0.1:11434/v1", what="LLM")


def test_offline_tts_rejects_networked_engine():
    ng.enforce_offline()
    from core.net_guard import OfflineViolationError
    from core.tts import create_tts

    with pytest.raises(OfflineViolationError):
        create_tts("EdgeTTS", {})


def test_not_enforced_allows_any_endpoint():
    """offline 未启用（开发期）时不限制端点。"""
    assert not ng.is_offline_enforced()
    ng.assert_local_endpoint("https://api.deepseek.com/v1", what="LLM")  # 不抛
