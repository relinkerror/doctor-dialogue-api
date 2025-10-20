# -*- coding: utf-8 -*-
import os, json, datetime
from textwrap import dedent
from collections import deque
from typing import List, Dict, Optional

from utils.LLMClient import LLMClient
from agents.dialogue.interfaces import IConversationalAgent
from agents.dialogue.summary_generator import ISummaryGenerator

class DialogueDoctorAgent(IConversationalAgent):
    def __init__(
        self,
        client: LLMClient,
        name: str,
        summary_generator: ISummaryGenerator,
        config: Optional[Dict] = None,
    ):
        self.client = client
        self.name = name
        self.summary = summary_generator

        cfg = config or {}
        self.system_template: str = cfg.get("system_template", dedent(
            """
            你是一名温和、有同理心的医生助理，帮助用户完成精神健康问卷。
            - 仅围绕问卷题意进行自然提问或澄清，不扩展无关话题；
            - 语言口语化、简洁；
            - 不使用量表分数字样，改用自然语言选项或追问具体例子；
            """
        ).strip())
        self.guide_template: str = cfg.get("guide_template", "【引导提示】请顺畅地把对话引向：\n{question_context}")
        self.max_history: int = int(cfg.get("max_history", 6))  # user/assistant 交替消息计数
        self.few_shot: str = cfg.get("few_shot", "")

        # —— 固定路径 JSONL 调试日志 ——
        self.debug_log_path: str = cfg.get("debug_log_path", "logs/doctor_debug.jsonl")
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)

        self.raw_history: deque = deque(maxlen=self.max_history)

    # ---- 调试日志 ----
    def _append_log(self, rec: Dict):
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- 系统提示拼装 ----
    def _build_system_prompt(self, current_question: Optional[str], question_context: Optional[str]) -> str:
        base = self.system_template
        # 追加一个轻量的“历史摘要”
        summary_text = getattr(self.summary, "get_summary", lambda: "")() or ""
        if summary_text:
            base += "\n\n# 历史摘要\n" + summary_text
        if current_question and question_context:
            base += "\n\n# 当前问题上下文\n" + question_context
        if self.few_shot:
            base += "\n\n" + self.few_shot
        return base

    # ---- 主调用 ----
    def generate_dialog(
        self,
        user_input: str,
        current_question: Optional[str] = None,
        question_context: Optional[str] = None,
        history_size: Optional[int] = None
    ) -> str:
        # 把用户输入推进到历史（允许为空串，因 VALIDATE 走引导）
        if user_input is not None:
            self.raw_history.append({"role": "user", "content": user_input})

        guide_prompt = self.guide_template.format(question_context=question_context or "") if question_context else ""
        system_prompt = self._build_system_prompt(current_question, question_context)

        # 取历史窗口
        if history_size is None:
            history_size = min(3, self.max_history // 2)
        msgs = list(self.raw_history)[-history_size * 2:] if history_size > 0 else list(self.raw_history)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if guide_prompt:
            messages.append({"role": "system", "content": guide_prompt})
        messages.extend(msgs)
        # 避免空 user 结尾导致部分模型拒答：如最后一条是 user，则补一条空 assistant
        if messages and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": "好的，我来确认一下。"})

        # 调试：入参消息
        self._append_log({"t": datetime.datetime.utcnow().isoformat(),
                          "agent": self.name, "event": "llm_call",
                          "messages": messages})

        # 真正调用 LLM
        reply = self.client.call(messages)

        # 调试：出参
        self._append_log({"t": datetime.datetime.utcnow().isoformat(),
                          "agent": self.name, "event": "llm_resp",
                          "text": reply})

        # 写回历史
        self.raw_history.append({"role": "assistant", "content": reply})
        return reply

    # ---- 便捷方法 ----
    def get_history(self) -> List[Dict]:
        return list(self.raw_history)

    def get_last_assistant_response(self) -> Optional[str]:
        if self.raw_history and self.raw_history[-1]["role"] == "assistant":
            return self.raw_history[-1]["content"]
        return None


# ==== 独立测试 ====
if __name__ == "__main__":
    # 你可以按需改成你的 config 路径
    CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
    llm = LLMClient.from_config(os.path.join(CONFIG_DIR, "llm_config.json"))

    class DummySummary(ISummaryGenerator):
        def get_summary(self) -> str:
            return "(demo) 上次讨论：睡眠差、兴趣减退。"

    agent = DialogueDoctorAgent(
        client=llm,
        name="demo-doctor",
        summary_generator=DummySummary(),
        config={"debug_log_path": "logs/doctor_debug.jsonl"}
    )

    print(agent.generate_dialog(
        user_input="最近两周总是睡不着",
        current_question="MDD_Q1",
        question_context="请用自然语言复述/澄清用户刚才关于‘几乎每天都情绪低落’的描述。"
    ))