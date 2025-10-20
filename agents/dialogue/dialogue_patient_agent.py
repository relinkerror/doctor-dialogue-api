import json
import random
from textwrap import dedent
from collections import deque
from typing import List, Dict, Optional
import os

from utils.LLMClient import LLMClient
from agents.dialogue.interfaces import IPatientAgent
from agents.dialogue.summary_generator import ISummaryGenerator

class DialoguePatientAgent(IPatientAgent):
    """
    模拟患者角色，支持指定id或随机选择persona，模板全部外部化。
    """

    def __init__(
        self,
        client: LLMClient,
        name: str,
        summary_generator: ISummaryGenerator,
        config: Optional[dict] = None,
        persona_id: Optional[str] = None
    ):
        cfg = config or {}
        self.client = client
        self.name = name
        self.max_history = cfg.get("max_history", 3)
        self.summary_interval = cfg.get("summary_interval", 3)
        self.persona_templates = cfg.get("persona_templates") or []
        self.few_shot = cfg.get("few_shot")
        # persona_id 选择优先级: init参数 > config文件
        pid = persona_id if persona_id is not None else cfg.get("persona_id")
        # 必须有模板
        if not self.persona_templates:
            raise ValueError("必须提供 persona_templates 列表")
        # 选择 persona
        if pid is None or pid == "random":
            self.persona = random.choice(self.persona_templates)
        else:
            self.persona = next((t for t in self.persona_templates if t["id"] == pid), None)
            if self.persona is None:
                print(f"[WARNING] persona_id '{pid}' 未找到，将随机选择一个 persona！")
                self.persona = random.choice(self.persona_templates)
        # 其它对话属性
        self.summary_generator = summary_generator
        self.raw_history: deque = deque(maxlen=self.max_history * 2)
        self.current_summary: str = ""
        self.dialog_count: int = 0
        print(f"[INFO] Selected persona: {self.persona['id']}")

    def generate_dialog(
        self,
        user_input: str,
        history_size: Optional[int] = None
    ) -> str:
        self.raw_history.append({"role": "user", "content": user_input})

        if history_size is None:
            history_size = 3
        slice_len = history_size * 2
        history_to_send = list(self.raw_history)[-slice_len:] if history_size > 0 else list(self.raw_history)

        messages = [{"role": "system", "content": self._build_system_prompt()}] + history_to_send

        try:
            # print(f"[DEBUG][{self.name}][API 调用] 发送消息: {messages}")
            resp = self.client.call(messages)
        except Exception as e:
            print(f"[ERROR][{self.name}] API call failed: {e}")
            resp = "暂时无法回应，请稍后再试。"

        self.raw_history.append({"role": "assistant", "content": resp})
        self.dialog_count += 1

        if self.dialog_count % self.summary_interval == 0:
            self.current_summary = self.summary_generator.summarize(
                list(self.raw_history), self.current_summary, self.summary_interval
            )

        return resp

    def _build_system_prompt(self) -> str:
        summary = self.current_summary or "无历史摘要。"
        base = dedent(f"""
        你是一个{self.persona['disorder']}患者，正在进行线上对话式的心理咨询。
        【背景信息】：
        {self.persona['background']}
        【历史对话摘要】：
        {summary}
        **只允许以一对一线上咨询对话的方式表达**
        回复要求：
        - 严格禁止在回复中添加任何括号（包括(), [], 【】, （）等）内的动作、表情、情绪等描述。
        - 不允许出现表情符号或括号标注。
        - 只用纯文本自然对话风格回复。
        """).strip()
        if self.few_shot:
            base += "\n" + self.few_shot
        return base

    def get_history(self) -> List[Dict]:
        return list(self.raw_history)

    def get_last_assistant_response(self) -> Optional[str]:
        if self.raw_history and self.raw_history[-1]["role"] == "assistant":
            return self.raw_history[-1]["content"]
        return None

    @staticmethod
    def get_all_persona_ids(config: dict) -> List[str]:
        return [t["id"] for t in config.get("persona_templates", [])]

# =================== 测试用 main =====================
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../../config/patient_agent.json")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    class DummyLLMClient:
        def call(self, messages):
            print("\n======= LLM 调用 =======")
            for m in messages:
                print(f"{m['role']}:\n{m['content']}\n")
            return "（模拟LLM回复）我最近一直很难入睡，情绪低落。"

    class DummySummaryGenerator:
        def summarize(self, raw_history, existing_summary, interval):
            return "【模拟摘要】用户表达了失眠和低落。"

    # 手动指定id（正常选中）
    agent1 = DialoguePatientAgent(
        client=DummyLLMClient(),
        name="patient1",
        summary_generator=DummySummaryGenerator(),
        config=cfg,
        persona_id="Depressive_Disorder_Due_to_Another_Medical_Condition"
    )

    # 不存在的id（警告并随机选）
    agent2 = DialoguePatientAgent(
        client=DummyLLMClient(),
        name="patient2",
        summary_generator=DummySummaryGenerator(),
        config=cfg,
        persona_id="不存在的id"
    )

    # 随机选
    agent3 = DialoguePatientAgent(
        client=DummyLLMClient(),
        name="patient3",
        summary_generator=DummySummaryGenerator(),
        config=cfg,
        persona_id="random"
    )

    # 测试对话
    user_input = "你好，我最近晚上很难睡着。"
    print("\n[Agent1] 指定id生成回复：")
    print(agent1.generate_dialog(user_input=user_input, history_size=3))

    print("\n[Agent2] 不存在id自动fallback：")
    print(agent2.generate_dialog(user_input=user_input, history_size=3))

    print("\n[Agent3] 随机id：")
    print(agent3.generate_dialog(user_input=user_input, history_size=3))

    # 展示全部id
    print("\n所有可用persona_id：", DialoguePatientAgent.get_all_persona_ids(cfg))
