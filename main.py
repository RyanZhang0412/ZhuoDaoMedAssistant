"""ZhuoDaoMedAssistant 程序主入口。

装配顺序严格自底向上（见交叉验证结论）：
  load_config
    -> enforce_offline (if offline)      # 最外层 socket 级禁网兜底
    -> PatientRepository
    -> RuleEngine / Recommender
    -> LLM (create_llm 工厂，端点早失败校验)
    -> Scheduler
    -> LocalAgent + bind_context         # 把 repository/recommender/scheduler 注入工具
    -> Robot (注入 agent + memory + 可选 asr/vad/tts)

提供 --text 纯文本模式：不起语音，循环 input()->agent.chat()->print，
无麦克风也能跑通业务与测试。API key 从环境变量读（config 写 ${ENV} 占位）。

运行：
  python main.py --text                 # 文本对话（推荐开发期）
  python main.py --text --patient P001  # 聚焦某患者
  python main.py                        # 语音模式（需装语音依赖与本地模型）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from agent.agent import LocalAgent
from agent.tools.schedule_tools import Scheduler
from core.llm import create_llm
from core.net_guard import enforce_offline
from core.robot import Robot
from medical.repository import PatientRepository
from medical.rules import RuleEngine
from medical.service import Recommender
from memory.dialogue_memory import DialogueMemory

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 配置加载（含环境变量插值）
# --------------------------------------------------------------------------- #
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate_env(obj):
    """递归把配置里的 ${ENV_VAR} 替换为环境变量值（API key 不写明文）。"""
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    return obj


def load_config(path: str = "config/config.yaml") -> dict:
    import yaml

    cfg_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return _interpolate_env(config)


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #
def build_agent(config: dict) -> LocalAgent:
    """自底向上装配并返回已 bind_context 的 LocalAgent。"""
    storage = config.get("storage", {})

    # 1) 病历仓库
    repository = PatientRepository(
        str(ROOT / storage.get("patients_dir", "data/patients")),
        keep_backup=storage.get("keep_backup", True),
    )

    # 2) 规则 + 推荐服务
    rule_engine = RuleEngine.from_yaml(
        str(ROOT / config.get("rules", {}).get("path", "config/rules.yaml"))
    )

    # 3) LLM（provider 工厂；端点早失败校验在 provider 构造内）
    llm = create_llm(config)

    recommender = Recommender(rule_engine, llm=llm)

    # 4) 排期
    scheduler = Scheduler(str(ROOT / storage.get("schedule_dir", "data/schedules")))

    # 5) Agent + 绑定工具上下文
    agent_cfg = config.get("agent", {})
    system_prompt = _read_system_prompt(agent_cfg.get("system_prompt_file"))
    agent = LocalAgent(
        llm,
        system_prompt=system_prompt,
        max_tool_rounds=agent_cfg.get("max_tool_rounds", 5),
    )
    agent.bind_context(repository, recommender, scheduler)
    return agent


def build_robot(config: dict, agent: LocalAgent) -> Robot:
    """装配 Robot（注入 agent + memory + 可选语音模块）。"""
    storage = config.get("storage", {})
    memory = DialogueMemory(str(ROOT / storage.get("dialogue_dir", "data/dialogues")))

    asr = vad = tts = None
    # 语音模块按需装配（文本模式不需要）
    selected = config.get("selected_module", {})
    if selected.get("_enable_voice"):  # 显式开关，默认关闭
        from core.asr import create_asr
        from core.tts import create_tts
        from core.vad import create_vad

        asr = create_asr(selected["ASR"], config.get("ASR", {}).get(selected["ASR"], {}))
        vad = create_vad(selected["VAD"], config.get("VAD", {}).get(selected["VAD"], {}))
        tts = create_tts(selected["TTS"], config.get("TTS", {}).get(selected["TTS"], {}))

    return Robot(config, agent=agent, memory=memory, asr=asr, vad=vad, tts=tts)


def _read_system_prompt(rel: str | None) -> str | None:
    if not rel:
        return None
    path = ROOT / rel
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# --------------------------------------------------------------------------- #
# 文本 REPL（无语音调试入口）
# --------------------------------------------------------------------------- #
def run_text_repl(robot: Robot, patient_id: str | None) -> None:
    print("=== 卓道康复助手（文本模式）===")
    print("输入问题，输入 exit/quit 退出。\n")
    if patient_id:
        print(f"[当前聚焦患者: {patient_id}]\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if text.lower() in ("exit", "quit", "退出"):
            print("再见。")
            break
        if not text:
            continue
        resp = robot.handle_text(text, patient_id=patient_id)
        print(f"助手 > {resp.text}\n")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="卓道康复助手")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--text", action="store_true", help="纯文本模式（无语音）")
    parser.add_argument("--patient", default=None, help="聚焦的患者ID")
    args = parser.parse_args()

    config = load_config(args.config)

    # offline 硬约束：任何组件构造之前先装上 socket 兜底
    if config.get("offline"):
        enforce_offline(config.get("offline_allow_hosts", ("127.0.0.1", "localhost")))
        print("[offline] 已启用禁网兜底，仅放行本地地址")

    agent = build_agent(config)
    robot = build_robot(config, agent)

    if args.text or not config.get("selected_module", {}).get("_enable_voice"):
        run_text_repl(robot, args.patient)
    else:
        try:
            robot.run_voice_session(patient_id=args.patient)
        except KeyboardInterrupt:
            print("\n再见。")
        finally:
            robot.shutdown()


if __name__ == "__main__":
    main()
