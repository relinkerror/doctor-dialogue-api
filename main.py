# doctor_dialogue_api.py
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from pydantic import BaseModel, Field

# ===== 你的项目依赖（保持不变）=====
from utils.LLMClient import LLMClient
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent

# ---------------- Config ----------------
CONFIG_DIR = os.environ.get("CONFIG_DIR", "config")
QUESTIONNAIRE_DIR = os.environ.get("QUESTIONNAIRE_DIR", "questionnaire")
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
APP_NAME = "Doctor Dialogue API"
APP_VERSION = "2.0.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 必要：根路径 + 健康检查 ----------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """给 Render/浏览器返回 200，并提供快速入口。"""
    return f"""
    <html>
    <head><title>{APP_NAME}</title></head>
    <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
      <h1>{APP_NAME}</h1>
      <p>status: <b>ok</b></p>
      <ul>
        <li>version: <code>{APP_VERSION}</code></li>
        <li>time (UTC): <code>{datetime.utcnow().isoformat()}Z</code></li>
      </ul>
      <p>
        <a href="/docs">/docs (Swagger)</a> |
        <a href="/redoc">/redoc</a> |
        <a href="/openapi.json">/openapi.json</a> |
        <a href="/healthz">/healthz</a> |
        <a href="/readyz">/readyz</a>
      </p>
    </body>
    </html>
    """

@app.get("/healthz", include_in_schema=False)
async def healthz():
    """存活检查（轻量、总是 200）"""
    return {"status": "ok", "service": APP_NAME, "version": APP_VERSION}

@app.get("/readyz", include_in_schema=False)
async def readyz():
    """
    就绪检查：如果你需要检查外部依赖（模型、配置、网络）可以在这里做。
    目前简单返回 200；需要的话在此加入实际探测逻辑。
    """
    checks = {
        "config_dir_exists": os.path.isdir(CONFIG_DIR),
        "questionnaire_dir_exists": os.path.isdir(QUESTIONNAIRE_DIR),
    }
    ok = all(checks.values())
    status = 200 if ok else 503
    return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=status)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """避免浏览器请求 /favicon.ico 产生 404 噪音。"""
    return Response(status_code=204)

# FastAPI 会自动为 GET 注册 HEAD，因此对 HEAD / 也会返回 200。

# ---------------- 下面保留/示例你的业务模型与路由 ----------------

class DialogueTurn(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str

class DialogueRequest(BaseModel):
    session_id: str
    turns: List[DialogueTurn]
    meta: Optional[Dict[str, Any]] = None

class DialogueResponse(BaseModel):
    session_id: str
    reply: str
    summary: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# 初始化你的组件（如需懒加载可移动到路由内）
llm_client = LLMClient()
summary_gen = LLMSummaryGenerator(llm_client=llm_client)
doctor_agent = DialogueDoctorAgent(llm_client=llm_client)
survey_agent = SurveyAgent(questionnaire_dir=QUESTIONNAIRE_DIR)

@app.post("/api/v1/dialogue", response_model=DialogueResponse)
async def dialogue(req: DialogueRequest):
    try:
        reply = doctor_agent.reply([t.model_dump() for t in req.turns], meta=req.meta or {})
        summary = summary_gen.summarize([t.model_dump() for t in req.turns] + [{"role":"assistant","content":reply}])
        return DialogueResponse(session_id=req.session_id, reply=reply, summary=summary, meta={"version": APP_VERSION})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dialogue error: {e}")

class SurveyRequest(BaseModel):
    session_id: str
    action: str = Field(..., description="'next' | 'prev' | 'answer'")
    answer: Optional[bool] = None

@app.post("/api/v1/survey")
async def survey(req: SurveyRequest):
    try:
        if req.action == "next":
            q = survey_agent.next_question(req.session_id)
            return {"session_id": req.session_id, "question": q}
        elif req.action == "prev":
            q = survey_agent.prev_question(req.session_id)
            return {"session_id": req.session_id, "question": q}
        elif req.action == "answer":
            survey_agent.answer(req.session_id, req.answer)
            q = survey_agent.next_question(req.session_id)
            return {"session_id": req.session_id, "question": q}
        else:
            raise HTTPException(status_code=400, detail="unknown action")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"survey error: {e}")
