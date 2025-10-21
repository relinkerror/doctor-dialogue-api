# main.py
import os
import json
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# ======== Import your own project dependencies ========
from utils.LLMClient import LLMClient
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent


# ======================================================
# Config
# ======================================================
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")
QUESTIONNAIRE_DIR = os.getenv("QUESTIONNAIRE_DIR", "questionnaire")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
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

# ======================================================
# Helper for env and lazy initialization
# ======================================================
_READY = {"llm": False, "survey": False}


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.getenv(name, default)
    if required and not v:
        return None
    return v


@lru_cache
def get_llm_client() -> LLMClient:
    api_key = _get_env("LLM_API_KEY", required=True)
    base_url = _get_env("LLM_BASE_URL", "https://api.openai.com/v1")
    model = _get_env("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise HTTPException(status_code=503, detail="LLM not configured (missing LLM_API_KEY)")
    client = LLMClient(api_key=api_key, base_url=base_url, model=model)
    _READY["llm"] = True
    return client


@lru_cache
def get_summary_gen() -> LLMSummaryGenerator:
    return LLMSummaryGenerator(llm_client=get_llm_client())


@lru_cache
def get_doctor_agent() -> DialogueDoctorAgent:
    return DialogueDoctorAgent(llm_client=get_llm_client())


@lru_cache
def get_survey_agent() -> SurveyAgent:
    agent = SurveyAgent(questionnaire_dir=QUESTIONNAIRE_DIR)
    _READY["survey"] = True
    return agent


# ======================================================
# Models
# ======================================================
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


class SurveyRequest(BaseModel):
    session_id: str
    action: str = Field(..., description="'next' | 'prev' | 'answer'")
    answer: Optional[bool] = None


# ======================================================
# Routes
# ======================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Root path for Render/Browser"""
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
        <a href="/docs">/docs</a> |
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
    """Lightweight liveness probe"""
    return {"status": "ok", "service": APP_NAME, "version": APP_VERSION}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    """Readiness check"""
    checks = {
        "config_dir_exists": os.path.isdir(CONFIG_DIR),
        "questionnaire_dir_exists": os.path.isdir(QUESTIONNAIRE_DIR),
        "llm_configured": bool(os.getenv("LLM_API_KEY")),
    }
    ok = all(checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid noisy 404s"""
    return Response(status_code=204)


# ======================================================
# Core API Endpoints
# ======================================================

@app.post("/api/v1/dialogue", response_model=DialogueResponse)
async def dialogue(
    req: DialogueRequest,
    doctor_agent: DialogueDoctorAgent = Depends(get_doctor_agent),
    summary_gen: LLMSummaryGenerator = Depends(get_summary_gen),
):
    try:
        reply = doctor_agent.reply([t.model_dump() for t in req.turns], meta=req.meta or {})
        summary = summary_gen.summarize(
            [t.model_dump() for t in req.turns] + [{"role": "assistant", "content": reply}]
        )
        return DialogueResponse(
            session_id=req.session_id, reply=reply, summary=summary, meta={"version": APP_VERSION}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dialogue error: {e}")


@app.post("/api/v1/survey")
async def survey(
    req: SurveyRequest,
    survey_agent: SurveyAgent = Depends(get_survey_agent),
):
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"survey error: {e}")


# ======================================================
# Run (for local dev)
# ======================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
