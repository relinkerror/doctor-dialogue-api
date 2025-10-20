from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class ISurveyAgent(ABC):
    @abstractmethod
    def get_current_question(self) -> Optional[str]:
        """返回当前应继续推进/等待用户答复的题目文本。"""
        pass

    @abstractmethod
    def auto_advance(self, dialog_history) -> Optional[str]:
        """
        用于“推进到下一步”：自动检测当前题是否已完成，若完成则切到下一个最佳问题，否则返回当前题。
        Returns: 当前要问的问题文本，全部完成时返回 None
        """
        pass

    @abstractmethod
    def get_question_context(self, question: str) -> str:
        pass

    @abstractmethod
    def is_completed(self) -> bool:
        pass

    @abstractmethod
    def get_all_answers(self) -> Dict:
        pass

