"""可选本地 Web 界面（FastAPI）。

复用 main.build_agent / build_robot 同一装配，避免两套装配逻辑。
仅绑 127.0.0.1，不对外。offline=true 时同样受 net_guard 约束。

运行：
  pip install fastapi uvicorn
  python server/server.py

接口：
  POST /chat      {query, patient_id?}      -> {reply}
  GET  /patients                            -> 患者列表
  GET  /patients/{id}                       -> 病历详情
  POST /recommend {patient_id}              -> 康复推荐
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

# 让 server.py 能 import 到项目根的包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import build_agent, build_robot, load_config  # noqa: E402


def create_app(config: dict):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="卓道康复助手", description="本地、不联网的医疗康复助手")

    agent = build_agent(config)
    robot = build_robot(config, agent)
    # 复用 agent 已绑定的工具上下文取 repository/recommender
    from agent.tools.context import get_context

    ctx = get_context()

    class ChatIn(BaseModel):
        query: str
        patient_id: str | None = None

    class RecommendIn(BaseModel):
        patient_id: str

    @app.post("/chat")
    def chat(body: ChatIn):
        resp = robot.handle_text(body.query, patient_id=body.patient_id)
        return {"reply": resp.text}

    @app.get("/patients")
    def patients():
        return {"patients": ctx.repository.search()}

    @app.get("/patients/{patient_id}")
    def patient(patient_id: str):
        from medical.repository import PatientNotFoundError

        try:
            return ctx.repository.get(patient_id).to_dict()
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail="患者不存在")

    @app.post("/recommend")
    def recommend(body: RecommendIn):
        from medical.repository import PatientNotFoundError

        try:
            record = ctx.repository.get(body.patient_id)
        except PatientNotFoundError:
            raise HTTPException(status_code=404, detail="患者不存在")
        return asdict(ctx.recommender.recommend(record))

    return app


def run_server(config: dict, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    run_server(
        cfg,
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8000),
    )
