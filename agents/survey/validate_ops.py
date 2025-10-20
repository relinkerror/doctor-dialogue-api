# agents/survey/validate_ops.py
# 三类 ValidateOp + 模糊词检测 + Prompt 生成 + 降级时的策略选择
from enum import Enum
from typing import List

class ValidateOp(str, Enum):
    CLARIFY_MEANING = "CLARIFY_MEANING"          # 澄清模糊含义
    PROBE_EXAMPLE = "PROBE_EXAMPLE"              # 追问具体例子/最近一次
    CONFIRM_RESTATEMENT = "CONFIRM_RESTATEMENT"  # 选项式复述确认（2~3项，自然语言，不用Likert分制）

# 你定的模糊词表
VAGUE_TERMS: List[str] = ["经常", "偶尔", "有时候", "很多", "很少", "一直", "从不", "差不多"]

def contains_vague_term(text: str) -> bool:
    if not text:
        return False
    return any(term in text for term in VAGUE_TERMS)

def make_validate_prompt(op: str, question: str, context: str) -> str:
    """
    生成用于对话引导的提示，包含题目原文 + 上下文（依赖题/remarks）
    """
    if op == ValidateOp.CLARIFY_MEANING:
        return (
            f"针对问题：{question}\n"
            "当前任务：请用一句自然中文澄清用户回答的具体含义："
            "例如‘你刚才说“经常”，你指每天吗，还是每周/每月/偶尔？’。\n"
            f"相关上下文：\n{context}"
        )
    if op == ValidateOp.PROBE_EXAMPLE:
        return (
            f"针对问题：{question}\n"
            "当前任务：请要求用户给出一个具体的例子或最近一次发生的情况，"
            "用一句话，不求很长，避免多问。\n"
            f"相关上下文：\n{context}"
        )
    if op == ValidateOp.CONFIRM_RESTATEMENT:
        return (
            f"针对问题：{question}\n"
            "当前任务：请把用户的开放式回答转为2~3个自然语言选项，让用户选择其一，"
            "避免用具体的分数。\n"
            f"相关上下文：\n{context}"
        )
    return f"请就问题：{question} 继续讨论。\n相关上下文：\n{context}"

def choose_validate_op_for_failure(p: float, margin: float, last_user_text: str) -> ValidateOp:
    """
    当 COMPLETE(τ) 未放行时（例如 p<τ 或 margin 过小），按失败原因选择一个验证操作。
    - 若检测到模糊词：先澄清含义
    - 若分辨率低（margin小）：用选项式复述来“收敛答案”
    - 其他情况：要求举例以增加信息密度
    """
    if contains_vague_term(last_user_text):
        return ValidateOp.CLARIFY_MEANING
    if margin < 0.10:
        return ValidateOp.CONFIRM_RESTATEMENT
    return ValidateOp.PROBE_EXAMPLE
