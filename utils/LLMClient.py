import openai
import json

class LLMClient:
    """
    通用 LLM 客户端，支持 config 初始化与 JSON 返回格式。
    """
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.3):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_config(cls, config_path: str):
        """
        从json config文件初始化 LLMClient。
        config格式: {
            "api_key": "...",
            "base_url": "...",
            "model": "...",
            "temperature": 0.5
        }
        """
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.5)
        )

    def call(self, messages: list) -> str:
        """
        普通聊天返回字符串。
        Args:
            messages: List[Dict] 格式: [{"role":"user", "content":"..."}, ...]
        Returns:
            str: assistant回复内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def call_json(self, messages: list) -> str:
        """
        调用chat completion，要求模型返回JSON字符串。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # 假设 config/llm_config.json 已存在
    config_path = "config/llm_config.json"
    llm = LLMClient.from_config(config_path)
    
    # 示例消息
    messages = [
        {"role": "system", "content": "你是一位简明的助手"},
        {"role": "user", "content": "你好，请用一句话介绍大模型"}
    ]
    reply = llm.call(messages)
    print("普通回复:", reply)

    # JSON 格式回复
    messages_json = [
        {"role": "system", "content": "请用JSON格式回答: {'answer': ... }"},
        {"role": "user", "content": "介绍一下GPT-4"}
    ]
    reply_json = llm.call_json(messages_json)
    print("JSON回复:", reply_json)