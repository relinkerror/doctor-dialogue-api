import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load local .env only for dev
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

LLM_API_KEY = os.getenv("LLM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# fallback to example file for non-sensitive defaults (not for keys)
try:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "llm_config.example.json"
    default_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
except Exception:
    default_cfg = {}
