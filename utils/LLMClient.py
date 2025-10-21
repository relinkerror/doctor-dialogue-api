import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import openai


class LLMClient:
    """
    通用 LLM 客户端，支持从配置文件或环境变量初始化。
    优先级：
    1. config/llm_config.json 存在时读取
    2. 若文件不存在或字段为空，则从环境变量兜底
    支持环境变量：
      OPENAI_API_KEY / LLM_API_KEY / LLM_API_KEY_ENV / LLM_BASE_URL / LLM_MODEL / LLM_TEMPERATURE
    """

    def __init__(self, api_key: str, base_url: Optional[str], model: str, temperature: float = 0.3):
        if not api_key or not api_key.strip():
            raise RuntimeError("API key is missing. Please set OPENAI_API_KEY or LLM_API_KEY in Render Environment.")

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(api_key=api_key.strip(), base_url=base_url)
        self.model = model
        self.temperature = temperature

    # ---------------------------
    # 工具函数
    # ---------------------------
    @staticmethod
    def _norm(val: Optional[str]) -> Optional[str]:
        """去除空串/空白"""
        if not val or not str(val).strip():
            return None
        return val.strip()

    @classmethod
    def _resolve_api_key(cls, cfg: dict) -> Optional[str]:
        """统一解析 api_key"""
        # 1) 文件里直接写了
        key = cls._norm(cfg.get("api_key"))
        if key:
            return key

        # 2) 文件里写了 api_key_env
        api_env = cls._norm(cfg.get("api_key_env"))
        if api_env:
            key = cls._norm(os.getenv(api_env))
            if key:
                return key

        # 3) 环境变量 LLM_API_KEY / OPENAI_API_KEY
        key = cls._norm(os.getenv("LLM_API_KEY")) or cls._norm(os.getenv("OPENAI_API_KEY"))
        if key:
            return key

        # 4) 允许通过 LLM_API_KEY_ENV 指定自定义环境变量名
        env_name = cls._norm(os.getenv("LLM_API_KEY_ENV"))
        if env_name:
            key = cls._norm(os.getenv(env_name))
            if key:
                return key

        return None

    # ---------------------------
    # 初始化
    # ---------------------------
    @classmethod
    def from_config(cls, config_path: str):
        """
        从 json 配置文件初始化；
        如果文件不存在或缺关键字段，则从环境变量兜底。
        示例：
        {
          "api_key_env": "OPENAI_API_KEY",
          "base_url": null,
          "model": "gpt-4o-mini",
          "temperature": 0.3
        }
        """
        p = Path(config_path)
        cfg = {}
        if p.exists():
            try:
                with p.open(encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")

        api_key = cls._resolve_api_key(cfg)
        base_url = cls._norm(cfg.get("base_url")) or cls._norm(os.getenv("LLM_BASE_URL"))
        model = cls._norm(cfg.get("model")) or os.getenv("LLM_MODEL", "gpt-4o-mini")

        try:
            temperature = float(cfg.get("temperature", os.getenv("LLM_TEMPERATURE", 0.3)))
        except Exception:
            temperature = 0.3

        if not api_key:
            raise RuntimeError(
                "Missing API key. Please set OPENAI_API_KEY / LLM_API_KEY "
                "or specify api_key_env in config."
            )

        return cls(api_key=api_key, base_url=base_url, model=model, temperature=temperature)

    @classmethod
    def from_env(cls):
        """纯环境变量初始化"""
        api_key = cls._resolve_api_key({})
        base_url = cls._norm(os.getenv("LLM_BASE_URL"))
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        return cls(api_key=api_key, base_url=base_url, model=model, temperature=temperature)

    # ---------------------------
    # 聊天接口
    # ---------------------------
    def call(self, messages: List[Dict[str, str]]) -> str:
        """普通聊天返回字符串"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def call_json(self, messages: List[Dict[str, str]]) -> str:
        """调用 chat completion，要求返回 JSON 字符串"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        return response.choices[0].message.content


# ---------------------------
# 本地测试
# ---------------------------
if __name__ == "__main__":
    # 尝试从文件或环境变量初始化
    cfg_path = "config/llm_config.json"
    if Path(cfg_path).exists():
        llm = LLMClient.from_config(cfg_path)
    else:
        llm = LLMClient.from_env()

    messages = [
        {"role": "system", "content": "你是一位简明的助手"},
        {"role": "user", "content": "你好，请用一句话介绍大模型"},
    ]
    print("普通回复:", llm.call(messages))

    messages_json = [
        {"role": "system", "content": "请用JSON格式回答: {'answer': ... }"},
        {"role": "user", "content": "介绍一下GPT-4"},
    ]
    print("JSON回复:", llm.call_json(messages_json))
