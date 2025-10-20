from utils import LLMClient
from textwrap import dedent
from typing import Deque, Dict
from .summary_generator import ISummaryGenerator

class LLMSummaryGenerator(ISummaryGenerator):
    """
    基于 LLM 的增量式摘要生成器实现。
    """
    def __init__(self, llm_client: LLMClient):
        self.client = llm_client  # 用于调用 OpenAI 接口

    def summarize(self, raw_history: Deque[Dict], existing_summary: str, interval: int) -> str:
        """
        构建摘要提示，并调用 LLM 生成新的摘要。
        """
        # 从 raw_history 中提取最近 interval 轮的对话
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in list(raw_history)[-2*interval:]
        )
        # 构建系统提示
        system_msg = dedent("""
        # 角色
        你是专业对话记录分析师，负责提取关键信息并生成结构化摘要

        # 任务要求
        1. 识别以下内容：
            - 用户的个人信息
            - 用户明确提供的事实数据（时间/地点/偏好等）
            - 需要澄清的模糊表述
        2. 使用Markdown格式：
           * 已确认信息用列表项表示
           [待处理] 需要跟进的事项
        3. 保留时间上下文（如"用户刚刚提到..."）
        4. 长度严格控制在150字内
        """).strip()
        user_msg = dedent(f"""
        ## 原始对话记录（最近{interval}轮）
        {history_text}

        ## 已有摘要（供参考）
        {existing_summary or "无历史摘要"}
        """)
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            response = self.client.call(messages)
            return response
        except Exception as e:
            print(f"摘要生成失败: {e}")
            return existing_summary  # 失败时返回已有摘要