from abc import ABC, abstractmethod
from typing import Optional, List, Dict

class IChatAgent(ABC):
    @abstractmethod
    def get_history(self) -> List[Dict]:
        pass

    @abstractmethod
    def get_last_assistant_response(self) -> Optional[str]:
        pass

class IConversationalAgent(IChatAgent):
    @abstractmethod
    def generate_dialog(
        self,
        user_input: str,
        current_question: Optional[str] = None,
        question_context: Optional[str] = None,
        history_size: Optional[int] = None
    ) -> str:
        pass

class IPatientAgent(IChatAgent):
    @abstractmethod
    def generate_dialog(
        self,
        user_input: str,
        history_size: Optional[int] = None
    ) -> str:
        pass
