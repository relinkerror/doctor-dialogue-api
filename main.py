# main.py
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== Your project dependencies (unchanged) =====
from utils.LLMClient import LLMClient
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent

# ---------------- Config ----------------
CONFIG_DIR = os.environ.get("CONFIG_DIR", "config")
QUESTIONNAIRE_DIR = os.environ.get("QUESTIONNAIRE_DIR", "questionnaire")
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
APP_VERSION = os.environ.get("APP_VERSION", "2.0.0")

app = FastAPI(title="Doctor Dialogue API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Runtime state ----------------
class AppState:
    def __init__(self):
        self.llm: Optional[LLMClient] = None
        self.doctor_meta: Dict[str, Any] = {}
        self.eval_prompt: str = ""
        self.survey_meta_json_path: str = ""
        self.sessions: Dict[str, "SessionState"] = {}

state = AppState()

class SessionState:
    def __init__(self, survey_agent: SurveyAgent, doctor_agent: DialogueDoctorAgent):
        self.survey_agent = survey_agent
        self.doctor_agent = doctor_agent
        self.display_history: List[Dict[str, str]] = []  # [{"role": "user/assistant", "content": "..."}]
        self.turn: int = 0  # authoritative turn counter on server
        self.created_at: datetime = datetime.utcnow()
        self.last_active: datetime = datetime.utcnow()

# ---------------- Pydantic models ----------------
class InitRequest(BaseModel):
    user_id: str = "demo-user"

class InitResponse(BaseModel):
    session_id: str
    message: str

class DialogueRequest(BaseModel):
    session_id: str
    patient_reply: str = Field(min_length=1)
    turn: int = Field(1, description="Client-sent turn; server maintains authoritative counter")

class DialogueResponse(BaseModel):
    turn: int
    doctor_ask: str
    current_question: str  # question TEXT (not ID)
    finished: bool
    progress_text: str

class FinishRequest(BaseModel):
    session_id: str

class FinishResponse(BaseModel):
    session_id: str
    finished: bool
    answers: Dict[str, List[Dict[str, Any]]]
    progress_text: Optional[str] = None

# ---------------- Helpers ----------------
def get_session(session_id: str) -> "SessionState":
    s = state.sessions.get(session_id)
    if not s:
        raise HTTPException(
            status_code=403,
            detail="Session not found or expired. Please call /api/patient/init first.",
        )
    s.last_active = datetime.utcnow()
    return s

def to_progress_text(p: Any) -> str:
    try:
        if isinstance(p, (dict, list)):
            return json.dumps(p, ensure_ascii=False)
        return str(p)
    except Exception:
        return "N/A"

# ---------------- Lifecycle ----------------
@app.on_event("startup")
def startup_event():
    # LLM
    state.llm = LLMClient.from_config(os.path.join(CONFIG_DIR, "llm_config.json"))

    # Survey metadata
    state.survey_meta_json_path = os.path.join(CONFIG_DIR, "dsm5_depression_surveys.json")

    # Doctor config & evaluation prompt
    with open(os.path.join(CONFIG_DIR, "doctor_agent.json"), encoding="utf-8") as f:
        state.doctor_meta = json.load(f)
    with open(os.path.join(CONFIG_DIR, "survey_uncertainty_eval_prompt.txt"), encoding="utf-8") as f:
        state.eval_prompt = f.read()

# ---------------- System & Health Routes ----------------
@app.get("/", tags=["system"])
def root():
    """Root must return 200 for Render/ELB health checks."""
    return {
        "status": "ok",
        "app": "Doctor Dialogue API",
        "version": APP_VERSION,
        "time": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "ok"}

@app.get("/livez", tags=["system"])
def livez():
    return {"status": "alive"}

@app.get("/readyz", tags=["system"])
def readyz():
    ready = state.llm is not None
    return {"ready": ready}

@app.get("/version", tags=["system"])
def version():
    return {"version": APP_VERSION}

# ---------------- API: Patient Session ----------------
@app.post("/api/patient/init", response_model=InitResponse)
def patient_init(req: InitRequest):
    if not state.llm:
        raise HTTPException(status_code=500, detail="LLM is not initialized.")

    # 每轮评估：eval_every_n=1（保持你现有逻辑）
    survey_agent = SurveyAgent(
        state.survey_meta_json_path,
        QUESTIONNAIRE_DIR,
        state.llm,
        eval_prompt=state.eval_prompt,
        threshold=0.2,
        eval_every_n=1,
    )

    doctor_agent = DialogueDoctorAgent(
        client=state.llm,
        name="demo-doctor",
        summary_generator=LLMSummaryGenerator(state.llm),
        config=state.doctor_meta,
    )

    session_id = str(uuid.uuid4())
    state.sessions[session_id] = SessionState(survey_agent, doctor_agent)
    return InitResponse(session_id=session_id, message="Session initialized.")

@app.post("/api/patient/dialogue", response_model=DialogueResponse)
def patient_dialogue(req: DialogueRequest):
    s = get_session(req.session_id)

    # Authoritative turn++ on server
    s.turn += 1

    # 1) Record user input
    s.display_history.append({"role": "user", "content": req.patient_reply})

    # 1.5) 打点计数（eval_every_n=1 → 每轮评估）
    s.survey_agent.tick_round()

    # 2) Advance survey, return QUESTION TEXT
    current_question_text = s.survey_agent.auto_advance(s.display_history)

    # 2.1) Survey finished
    if not current_question_text:
        progress_text = to_progress_text(s.survey_agent.get_progress())
        return DialogueResponse(
            turn=s.turn,
            doctor_ask="",
            current_question="",
            finished=True,
            progress_text=progress_text,
        )

    # 3) Doctor generates response
    doctor_ask = s.doctor_agent.generate_dialog(
        user_input=req.patient_reply,
        current_question=current_question_text,  # pass question TEXT
        question_context=s.survey_agent.get_question_context(current_question_text),
    )

    # 4) Record doctor reply
    s.display_history.append({"role": "assistant", "content": doctor_ask})

    return DialogueResponse(
        turn=s.turn,
        doctor_ask=doctor_ask,
        current_question=current_question_text,
        finished=False,
        progress_text=to_progress_text(s.survey_agent.get_progress()),
    )

@app.post("/api/patient/finish", response_model=FinishResponse)
def patient_finish(req: FinishRequest):
    s = state.sessions.pop(req.session_id, None)
    if not s:
        # Session already gone: return empty structure by convention
        return FinishResponse(
            session_id=req.session_id,
            finished=True,
            answers={},
            progress_text=None,
        )

    # 1) Structured answers JSON from your SurveyAgent
    try:
        answers_json = s.survey_agent.get_all_answers()
    except Exception:
        answers_json = {}

    # 2) Final progress text
    progress_text = to_progress_text(s.survey_agent.get_progress())

    return FinishResponse(
        session_id=req.session_id,
        finished=True,
        answers=answers_json,
        progress_text=progress_text,
    )

@app.get("/api/patient/session/{session_id}")
def session_snapshot(session_id: str):
    s = get_session(session_id)
    return {
        "session_id": session_id,
        "created_at": s.created_at.isoformat(),
        "last_active": s.last_active.isoformat(),
        "history_len": len(s.display_history),
        "turn": s.turn,
        "progress_text": to_progress_text(s.survey_agent.get_progress()),
        "finished": False,
    }

# ---------------- main ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
