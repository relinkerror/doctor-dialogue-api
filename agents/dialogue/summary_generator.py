from abc import ABC, abstractmethod
from typing import Deque, Dict

class ISummaryGenerator(ABC):
    """
    摘要生成器接口，负责对对话历史进行增量式摘要。
    """
    @abstractmethod
    def summarize(self, raw_history: Deque[Dict], existing_summary: str, interval: int) -> str:
        """
        根据给定对话历史与已有摘要，生成新的摘要文本。
        
        :param raw_history: 对话历史队列，包含用户和助理的消息
        :param existing_summary: 上一次的摘要字符串
        :param interval: 触发摘要生成的轮次数
        :return: 新的摘要字符串
        """
        pass  # 子类需实现此方法