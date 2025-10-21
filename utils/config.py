import os
from dotenv import load_dotenv
from pathlib import Path
import json

# ---------------------------
# 1) 本地开发：自动加载根目录 .env
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

# ---------------------------
# 2) 通用环境变量
# ---------------------------
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

# 允许两种密钥名：LLM_API_KEY / OPENAI_API_KEY（二选一）
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 保持原接口不变

# ---------------------------
# 3) LLM 配置文件路径（支持覆盖）
# ---------------------------
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")
CONFIG_DIR_PATH = ROOT / CONFIG_DIR
LLM_CONFIG_FILE = CONFIG_DIR_PATH / "llm_config.json"

# ---------------------------
# 4) 读取 example 默认（非敏感）
# ---------------------------
try:
    example_path = ROOT / "config" / "llm_config.example.json"
    default_cfg = json.loads(example_path.read_text(encoding="utf-8")) if example_path.exists() else {}
except Exception:
    default_cfg = {}

def load_llm_file_config() -> dict | None:
    """
    如果存在 config/llm_config.json 则读取；否则返回 None。
    """
    try:
        if LLM_CONFIG_FILE.exists():
            return json.loads(LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def build_llm_runtime_config() -> dict:
    """
    当 llm_config.json 不存在时，使用环境变量 + example 默认值 组装一个运行时配置。
    不包含任何硬编码密钥，密钥从环境变量读取。
    """
    cfg = dict(default_cfg)  # 先拷贝 example 默认
    # 覆盖/补充环境变量
    cfg["provider"] = os.getenv("LLM_PROVIDER", cfg.get("provider", "openai"))
    cfg["model"] = os.getenv("LLM_MODEL", cfg.get("model", "gpt-4o-mini"))
    cfg["base_url"] = os.getenv("LLM_BASE_URL", cfg.get("base_url"))
    # 超时时间
    try:
        cfg["timeout"] = int(os.getenv("LLM_TIMEOUT", cfg.get("timeout", 60)))
    except Exception:
        cfg["timeout"] = 60
    # 密钥不写入文件：只在运行时注入
    cfg["api_key_env"] = os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY")
    return cfg
