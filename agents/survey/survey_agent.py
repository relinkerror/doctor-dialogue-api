# agents/survey/survey_agent.py
from typing import List, Dict, Any, Optional

from .file_survey_loader import FileSurveyLoader
from .memory_survey_state_manager import MemorySurveyStateManager
from .survey_selector import SurveySelector
from .survey_uncertainty_evaluator import SurveyUncertaintyEvaluator
from utils.embedding_similarity import EmbeddingSimilarityTool
from .validate_ops import ValidateOp, make_validate_prompt

class SurveyAgent:
    """
    SurveyAgent：自动推进主控。
    - 惰性评估：VALIDATE/其他动作不立刻评估，累计到 N 步或遇到 COMPLETE/JUMP 时再评估
    - 计数策略：每“对话轮”+1、每“动作”+1；触发阈值看两者的 max；评估后清零
    - SSOT：所有状态以 state_manager 为准
    """

    def __init__(self,
                 meta_json_path: str,
                 questionnaire_dir: str,
                 llm_client,
                 eval_prompt: str,
                 survey_depends_on_prompt: Optional[str] = None,
                 embedder_model: str = "models/paraphrase-multilingual-MiniLM-L12-v2",
                 threshold: float = 0.1,
                 # 惰性评估参数
                 lazy_eval: bool = True,
                 eval_every_n: int = 3):
        # === 初始化各组件 ===
        self.loader = FileSurveyLoader(meta_json_path, llm_client, survey_depends_on_prompt)
        self.surveys = self.loader.load_surveys(questionnaire_dir)
        self.state_manager = MemorySurveyStateManager()
        self.state_manager.initialize_states(self.surveys)
        self.sim_tool = EmbeddingSimilarityTool(embedder_model)
        self.selector = SurveySelector(self.state_manager, self.sim_tool)
        self.evaluator = SurveyUncertaintyEvaluator(
            llm_client, eval_prompt, threshold=threshold, embedder_model=embedder_model
        )

        # 运行态
        self.current_question: Optional[str] = None
        self.current_survey_id: Optional[str] = None

        # 惰性评估配置
        self.lazy_eval: bool = lazy_eval
        self.eval_every_n: int = max(1, int(eval_every_n))

        # —— 新：双计数（环境轮 & 动作步）——
        self._rounds_since_eval: int = 0
        self._actions_since_eval: int = 0

    # ===== 供外部环境/脚本打点 =====
    def tick_round(self):
        """每轮对话结束后由外部调用一次（环境步+1）"""
        self._rounds_since_eval += 1
        print(f"[TICK][round] rounds={self._rounds_since_eval} actions={self._actions_since_eval}")

    def _tick_action(self):
        """每执行一次动作（RL步+1）"""
        self._actions_since_eval += 1
        print(f"[TICK][action] rounds={self._rounds_since_eval} actions={self._actions_since_eval}")

    # ===== 兼容原有 API =====
    def get_current_question(self) -> Optional[str]:
        return self.current_question

    def evaluate_and_update(self, dialog_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        把“当前问卷”的全部题目（含状态）送评，并写回。
        """
        if not self.current_survey_id:
            return []
        items = self.state_manager.survey_states[self.current_survey_id]['items']
        questions = [
            {
                "id": it["id"],
                "question": it["question"],
                "depends_on": it.get("depends_on", []),
                "status": it.get("status", None),
            }
            for it in items
        ]
        print(f"[EVAL][begin] sid={self.current_survey_id} q_count={len(questions)}")
        updates = self.evaluator.evaluate(questions, dialog_history)
        print(f"[EVAL][end] updates={updates}")
        if updates:
            self.state_manager.update_states(self.current_survey_id, updates)
            print(f"[EVAL][writeback] progress={self.get_progress()}")
        return updates

    def _find_best_next_question(self, dialog_history: List[Dict[str, str]]):
        """
        返回（最佳下一题, 所属问卷id），多问卷下全局优选（与历史语义最相似）。
        —— 注意：从 state_manager 读取最新 items，避免状态不同步。
        """
        max_sim = -1.0
        best_q, best_sid = None, None
        ctx_now = " ".join([d["content"] for d in dialog_history]) if dialog_history else ""
        for survey_id, state in self.state_manager.survey_states.items():
            if state['completed']:
                continue
            items = state['items']
            pending_qs = [it["question"] for it in items if it.get("status") is None]
            if not pending_qs:
                continue
            for q in pending_qs:
                sim = self.sim_tool.similarity(ctx_now, q)
                if sim > max_sim:
                    max_sim, best_q, best_sid = sim, q, survey_id
        return best_q, best_sid

    def _evaluate_and_update_if_due(self, dialog_history: List[Dict[str, str]], force: bool = False) -> List[Dict[str, Any]]:
        """
        惰性评估：达到阈值或 force=True 时才评估，否则跳过。
        触发阈值看 pending = max(rounds_since_eval, actions_since_eval)
        """
        if not self.current_survey_id:
            print("[EVAL?] skipped (no current survey)")
            return []
        pending = max(self._rounds_since_eval, self._actions_since_eval)
        print(f"[EVAL?] lazy={self.lazy_eval} force={force} pending={pending}/{self.eval_every_n} "
              f"sid={self.current_survey_id} q='{self.current_question}'")
        if (not self.lazy_eval) or force or (pending >= self.eval_every_n):
            updates = self.evaluate_and_update(dialog_history)
            # —— 评估后清零 —— 
            self._rounds_since_eval = 0
            self._actions_since_eval = 0
            print(f"[EVAL!] done. reset counters -> rounds=0, actions=0")
            return updates
        print("[EVAL?] skipped (not due)")
        return []

    def auto_advance(self, dialog_history: List[Dict[str, str]]) -> Optional[str]:
        """
        推进流程：
          1) 按需评估
          2) 若当前题未完成，返回当前题
          3) 否则从全局pending中选最优下一题
        """
        print(f"[AUTO] enter: sid={self.current_survey_id} q='{self.current_question}'")
        self._evaluate_and_update_if_due(dialog_history, force=False)

        if self.current_question and self.current_survey_id:
            done = self.state_manager.is_question_completed(self.current_question)
            print(f"[AUTO] current_done={done}")
            if not done:
                return self.current_question

        next_q, survey_id = self._find_best_next_question(dialog_history)
        print(f"[AUTO] pick_next: sid={survey_id} q='{next_q}'")
        if next_q:
            self.current_question = next_q
            self.current_survey_id = survey_id
            return next_q

        print("[AUTO] no more questions. reset current.")
        self.current_question = None
        self.current_survey_id = None
        return None

    def get_question_context(self, question: str) -> str:
        return self.state_manager.get_question_context(question)

    def is_completed(self) -> bool:
        return self.state_manager.is_diagnosis_completed()

    def get_all_answers(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.state_manager.get_all_answers()

    def get_progress(self) -> str:
        return self.state_manager.get_progress()

    # ===== 供RL/策略调用的动作接口 =====
    def get_pending_candidates(self, dialog_history: List[Dict[str, str]]) -> List[tuple]:
        leaves = self.selector.get_all_ready_leaves(dialog_history)  # [(question, sim, chain)]
        pairs = []
        for (q, _, _) in leaves:
            for sid, s in self.state_manager.survey_states.items():
                for it in s['items']:
                    if it['question'] == q:
                        pairs.append((it['id'], q))
                        break
        return pairs

    def apply_action(self,
                     action_type: str,
                     param: Dict[str, Any],
                     dialog_history: List[Dict[str, str]],
                     convo_agent=None) -> List[Dict[str, Any]]:
        """
        执行动作：
          - "validate": 只追加一条验证话术，不评估（惰性累计）
          - "jump":    若有挂起则先软强评估，再切题
          - "complete":总是强评
        返回 updates；非评估步返回 []
        """
        updates: List[Dict[str, Any]] = []
        atype = (action_type or "").lower()

        # === validate ===
        if atype == "validate":
            # 1) 决定 validate_op（默认澄清）
            op = param.get("op", ValidateOp.CLARIFY_MEANING)
            if isinstance(op, str):
                try:
                    op = ValidateOp(op)
                except ValueError:
                    op = ValidateOp.CLARIFY_MEANING

            # 2) 若未选中当前题，先推进一次（不强评）
            question = self.current_question or self.auto_advance(dialog_history)
            if not question:
                print("[VALIDATE] no current question, return.")
                return []

            # 3) 拼接引导文本
            context = self.get_question_context(question)

            # 小工具：打印长文本的摘要
            def _short(txt: str, limit: int = 800) -> str:
                if not txt:
                    return ""
                return txt if len(txt) <= limit else (txt[:limit] + f" ...[len={len(txt)}]")

            # 尝试找出当前题的 qid 仅用于日志可读性
            qid = None
            for sid, s in self.state_manager.survey_states.items():
                for it in s['items']:
                    if it['question'] == question:
                        qid = it['id']; break
                if qid: break

            guide_text = make_validate_prompt(op, question, context)

            print(f"[VALIDATE] sid={self.current_survey_id} qid={qid} op={op}")
            print(f"[VALIDATE][question]\n{_short(question, 400)}")
            print(f"[VALIDATE][context]\n{_short(context, 1200)}")
            print(f"[VALIDATE][guide_text]\n{_short(guide_text, 1200)}")

            # 4) 生成一句自然话术（把 guide_text 当 question_context 传给医生）
            if convo_agent is not None:
                utterance = convo_agent.generate_dialog(
                    user_input="",
                    current_question=None,
                    question_context=guide_text,  # <== 关键
                    history_size=3
                )
            else:
                utterance = guide_text

            print(f"[VALIDATE][doctor_utterance]\n{_short(utterance, 800)}")

            # 5) 写入对话历史；动作计数+1
            dialog_history.append({"role": "assistant", "content": utterance})
            self._tick_action()
            return []

        elif atype == "jump":
            qid = param.get("qid")
            if not qid:
                return []
            # 有挂起则先来一次软强评估
            pending = max(self._rounds_since_eval, self._actions_since_eval)
            if pending > 0:
                self._evaluate_and_update_if_due(dialog_history, force=True)

            # 切题（从原始 surveys 找到所属sid，但题文本从 state_manager 保证一致）
            for sid, state in self.state_manager.survey_states.items():
                for it in state['items']:
                    if it["id"] == qid:
                        self.current_survey_id = sid
                        self.current_question = it["question"]
                        print(f"[JUMP] to sid={sid} qid={qid}")
                        self._tick_action()
                        return []
            return []

        elif atype == "complete":
            # 总是强制评估
            updates = self._evaluate_and_update_if_due(dialog_history, force=True)
            self._tick_action()  # 记一次动作（虽已清零，但计入本次 action 以便外层统计算法步数）
            return updates

        # 未知动作：no-op
        return []
