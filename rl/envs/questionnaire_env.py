"""
QuestionnaireEnv (final)
-----------------------
- 接入医生代理：VALIDATE 时把 guide_text 交给 doctor 生成自然话术
- 强化三类奖励：
  (1) LLM 调用成本（COMPLETE=2，其余=1）
  (2) 表格式“相关问卷正确性”奖励（相关=应 True；无关=0）
  (3) 对话质量：validate_op_head + coherence_head 两颗头
- 追加“结论错误强惩”：相关问卷 True 比例不是全局最高 → 默认 -10
- 固定路径 JSONL 调试日志：logs/env_debug.jsonl（含 reward_breakdown/动作/阈值/概率）
- 与现有 SurveyAgent / Evaluator / Selector / SSOT 完全兼容
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import os, json, datetime
import numpy as np

# === 你项目已有模块 ===
from agents.survey.survey_agent import SurveyAgent
from agents.survey.validate_ops import ValidateOp, choose_validate_op_for_failure

# ========== 动作编码 ==========
# (type_id, validate_op_id, jump_choice_idx, tau_T_scaled, tau_F_scaled)
# type_id: 0=VALIDATE, 1=JUMP, 2=COMPLETE
# validate_op_id: 0=CLARIFY_MEANING, 1=PROBE_EXAMPLE, 2=CONFIRM_RESTATEMENT（仅 type=0 使用）
# jump_choice_idx: 候选池索引（仅 type=1 使用）
# tau_*_scaled ∈ [0,1]，线性映射到 [tau_min, tau_max]（仅 type=2 使用）


# ========== 环境配置 ==========
@dataclass
class EnvConfig:
    # 阈值范围（双阈值）
    tau_min: float = 0.90
    tau_max: float = 0.99
    # 分辨力要求
    min_margin: float = 0.10
    # 投影维度（0 = 不投影）
    proj_dim: int = 64
    # 每步代价 & 多次验证惩罚
    step_cost: float = -0.05
    extra_validate_cost: float = -0.10
    # 质量奖励系数（max_p 提升）
    quality_coef: float = 0.2
    # Episode 上限
    max_steps: int = 200

    # === LLM 调用成本（规范化记账） ===
    llm_cost_per_call: float = -0.06
    complete_llm_calls: int = 2
    validate_llm_calls: int = 1
    jump_llm_calls: int = 1

    # === 相关问卷的表格式奖励（加大权重） ===
    relevant_surveys: List[str] = field(default_factory=list)
    reward_true_relevant: float = 1.0
    penalty_false_relevant: float = -1.0

    # === 两颗头（validate_op / coherence） ===
    validate_head_path: Optional[str] = "models/validate_op_head.joblib"
    coherence_head_path: Optional[str] = "models/coherence_head.joblib"
    emb_model_path: Optional[str] = "models/paraphrase-multilingual-MiniLM-L12-v2"
    op_match_coef: float = 0.20
    coherence_coef: float = 0.20

    # === 结论正确性（比例最高）检查 ===
    conclusion_check_mode: str = "both"  # on_complete / on_done / both
    wrong_conclusion_penalty: float = -10.0

    # === 日志 ===
    debug_log_path: str = "logs/env_debug.jsonl"


# ========== 运行时缓存 ==========
@dataclass
class RunningStats:
    turns_since_eval: int = 0
    actions_since_eval: int = 0
    validate_count_on_current: int = 0
    last_tau_T: float = 0.0
    last_tau_F: float = 0.0
    prev_max_p: float = 0.0
    eval_cursor: int = 0


# ========== 两颗头的轻量适配器 ==========
class _RewardHeadsAdapter:
    def __init__(self, validate_path: Optional[str], coherence_path: Optional[str], emb_model: Optional[str]):
        self.validate = None
        self.coherence = None
        try:
            import joblib
            if validate_path and os.path.isfile(validate_path):
                self.validate = joblib.load(validate_path)
            if coherence_path and os.path.isfile(coherence_path):
                self.coherence = joblib.load(coherence_path)
        except Exception:
            self.validate = None
            self.coherence = None
        self.emb_model = emb_model

    def score_validate_op(self, context_text: str, theme: str, chosen_op_label: str) -> float:
        if self.validate is None:
            return 0.0
        try:
            x = [f"[THEME]{theme}\n[CTX]{context_text}"]
            proba = self.validate.predict_proba(x)[0]
            label2idx = {"CLARIFY_MEANING": 0, "PROBE_EXAMPLE": 1, "CONFIRM_RESTATEMENT": 2}
            idx = label2idx.get(chosen_op_label, 0)
            return float(proba[idx])
        except Exception:
            return 0.0

    def score_coherence(self, theme: str, utterance: str) -> float:
        if self.coherence is None:
            return 0.0
        try:
            x = [f"[THEME]{theme}\n[UTT]{utterance}"]
            proba = self.coherence.predict_proba(x)[0][1]
            return float(proba)
        except Exception:
            return 0.0


class QuestionnaireEnv:
    """最小 Gym 风格接口：reset / step / observation。"""

    def __init__(self, agent: SurveyAgent, dialog_history: Optional[List[Dict[str, str]]] = None,
                 config: Optional[EnvConfig] = None, random_seed: int = 42, doctor=None):
        self.agent = agent
        self.cfg = config or EnvConfig()
        self.rng = np.random.RandomState(random_seed)
        self.sim_tool = getattr(self.agent, "sim_tool", None)

        self.dialog_history: List[Dict[str, str]] = dialog_history or []
        self.stats = RunningStats()
        self._done = False
        self._steps = 0
        self._proj_W = None

        # 医生代理（用于 VALIDATE 生成自然话术）
        self.doctor = doctor

        # 头模型适配器
        self.heads = _RewardHeadsAdapter(
            validate_path=self.cfg.validate_head_path,
            coherence_path=self.cfg.coherence_head_path,
            emb_model=self.cfg.emb_model_path
        )

        # 日志路径
        os.makedirs(os.path.dirname(self.cfg.debug_log_path), exist_ok=True)

    # ---------- 工具：动作打包/解包 ----------
    @staticmethod
    def pack_action(type_id: int, validate_op_id: int = 0, jump_choice_idx: int = 0,
                    tau_T_scaled: float = 0.8, tau_F_scaled: float = 0.8) -> np.ndarray:
        return np.array([type_id, validate_op_id, jump_choice_idx, tau_T_scaled, tau_F_scaled], dtype=float)

    @staticmethod
    def unpack_action(a: np.ndarray) -> Tuple[int, int, int, float, float]:
        type_id = int(round(float(a[0])))
        validate_op_id = int(round(float(a[1])))
        jump_choice_idx = int(round(float(a[2])))
        tau_T_scaled = float(a[3]); tau_F_scaled = float(a[4])
        return type_id, validate_op_id, jump_choice_idx, tau_T_scaled, tau_F_scaled

    # ---------- 公开 API ----------
    def reset(self) -> Dict[str, Any]:
        self.dialog_history = []
        self.stats = RunningStats()
        self._done = False
        self._steps = 0
        _ = self.agent.auto_advance(self.dialog_history)
        return self._build_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            return self._build_observation(), 0.0, True, {}

        self._steps += 1
        self.stats.turns_since_eval += 1

        atype, op_id, jump_idx, tauT_s, tauF_s = self.unpack_action(action)
        reward = self.cfg.step_cost
        info: Dict[str, Any] = {}
        breakdown = {"llm_cost": 0.0, "op_match": 0.0, "coherence": 0.0, "table": 0.0, "conclusion_penalty": 0.0}

        # 阈值缩放
        tau_T = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) * np.clip(tauT_s, 0.0, 1.0)
        tau_F = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) * np.clip(tauF_s, 0.0, 1.0)
        self.stats.last_tau_T, self.stats.last_tau_F = float(tau_T), float(tau_F)

        # 当前题（无则推进）
        cur_q = self.agent.get_current_question() or self.agent.auto_advance(self.dialog_history)
        if not cur_q:
            self._done = True
            self._maybe_apply_conclusion_penalty(trigger="on_done", reward_acc=lambda d: None)
            return self._build_observation(), 0.0, True, {"reason": "no_more_questions"}

        # === LLM 成本 ===
        calls = 0
        if atype == 0:   calls = self.cfg.validate_llm_calls
        elif atype == 1: calls = self.cfg.jump_llm_calls
        elif atype == 2: calls = self.cfg.complete_llm_calls
        cost = self.cfg.llm_cost_per_call * calls
        reward += cost; breakdown["llm_cost"] += float(cost)

        # --- 执行动作 ---
        if atype == 0:  # VALIDATE
            op_id = int(np.clip(op_id, 0, 2))
            op = [ValidateOp.CLARIFY_MEANING, ValidateOp.PROBE_EXAMPLE, ValidateOp.CONFIRM_RESTATEMENT][op_id]

            # 上下文组装供头模型评分
            last_assistant = self._get_last_assistant_text() or ""
            last_user = self._get_last_user_text() or ""
            cur_theme = self.agent.current_survey_id or "GENERAL"
            ctx_text = (last_assistant + "\n" + last_user).strip()

            # ★ 使用医生代理生成自然话术
            self.agent.apply_action("validate", {"op": op}, self.dialog_history, convo_agent=self.doctor)
            self.stats.actions_since_eval += 1
            self.stats.validate_count_on_current += 1

            # 取刚写入的医生话术作为候选
            generated = self._get_last_assistant_text() or ""
            p_op = self.heads.score_validate_op(ctx_text, cur_theme, op.value)
            p_coh = self.heads.score_coherence(cur_theme, generated)
            reward += self.cfg.op_match_coef * p_op; breakdown["op_match"] += self.cfg.op_match_coef * p_op
            reward += self.cfg.coherence_coef * p_coh; breakdown["coherence"] += self.cfg.coherence_coef * p_coh

        elif atype == 1:  # JUMP
            candidates = self.agent.get_pending_candidates(self.dialog_history)
            if candidates:
                jump_idx = int(np.clip(jump_idx, 0, len(candidates) - 1))
                qid, _ = candidates[jump_idx]
                self.agent.apply_action("jump", {"qid": qid}, self.dialog_history)
                self.stats.validate_count_on_current = 0
                self.stats.actions_since_eval += 1

        elif atype == 2:  # COMPLETE(τ_T, τ_F)
            passed, reason = self._gate_complete()
            info["complete_gate"] = {"passed": passed, "reason": reason, "tau_T": float(tau_T), "tau_F": float(tau_F)}
            if passed:
                updates = self.agent.apply_action("complete", {}, self.dialog_history)
                self.stats.actions_since_eval = 0
                self.stats.turns_since_eval = 0
                self.stats.eval_cursor = len(self.dialog_history)

                # 相关问卷的表格式奖励
                delta_table = self._table_rewards_for_updates(updates)
                reward += delta_table; breakdown["table"] += float(delta_table)

                # 若当前题从 None→落定，给一次完成奖励
                if updates and self.agent.state_manager.is_question_completed(cur_q):
                    reward += 1.0

                # 质量奖励：max_p 提升
                new_obs = self._build_observation()
                max_p = max(new_obs["p_T"], new_obs["p_F"])
                reward += self.cfg.quality_coef * float(max(0.0, max_p - self.stats.prev_max_p))
                self.stats.prev_max_p = max_p

                # 结论正确性检查（on_complete）
                box = {"pen": 0.0}
                def acc(d):
                    box["pen"] += float(d)
                self._maybe_apply_conclusion_penalty(trigger="on_complete", reward_acc=acc)
                reward += box["pen"]; breakdown["conclusion_penalty"] += box["pen"]
            else:
                # 失败降级为一次 VALIDATE（同样使用医生代理）
                last_user = self._get_last_user_text()
                p_T, p_F, margin, _decided = self._read_eval_state()
                op = choose_validate_op_for_failure(max(p_T, p_F), margin, last_user or "")
                self.agent.apply_action("validate", {"op": op}, self.dialog_history, convo_agent=self.doctor)
                self.stats.actions_since_eval += 1
                self.stats.validate_count_on_current += 1

        # 多次验证的额外惩罚（同题第3次及以上）
        if self.stats.validate_count_on_current > 2:
            reward += self.cfg.extra_validate_cost

        # 终止条件
        done = False
        if self.agent.is_completed() or self._steps >= self.cfg.max_steps:
            done = True
            self._done = True
            # 终局再做一次结论检查
            box = {"pen": 0.0}
            def acc2(d): box["pen"] += float(d)
            self._maybe_apply_conclusion_penalty(trigger="on_done", reward_acc=acc2)
            reward += box["pen"]; breakdown["conclusion_penalty"] += box["pen"]

        obs = self._build_observation()

        # —— 固定路径 JSONL 调试日志 ——
        try:
            with open(self.cfg.debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": datetime.datetime.utcnow().isoformat(),
                    "step": int(self._steps),
                    "sid": self.agent.current_survey_id,
                    "action": {"type": int(atype), "validate_op_id": int(op_id), "jump_idx": int(jump_idx),
                                "tauT": float(self.stats.last_tau_T), "tauF": float(self.stats.last_tau_F)},
                    "obs": {"p_T": float(obs["p_T"]), "p_F": float(obs["p_F"]), "margin": float(obs["margin"])},
                    "reward": float(reward),
                    "breakdown": {k: float(v) for k, v in breakdown.items()}
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return obs, float(reward), bool(done), info

    # ---------- 掩码 ----------
    def get_action_masks(self) -> Dict[str, Any]:
        decided = self._is_current_decided()
        validate_enabled = 0 if decided else 1

        candidates = self.agent.get_pending_candidates(self.dialog_history)
        jump_mask = [1] * len(candidates) if candidates else []

        complete_enabled = 0 if decided else 1
        return {
            "validate_enabled": validate_enabled,
            "jump_mask": jump_mask,
            "complete_enabled": complete_enabled,
            "candidate_count": len(candidates)
        }

    # ---------- 观测 ----------
    def _build_observation(self) -> Dict[str, Any]:
        p_T, p_F, margin, decided = self._read_eval_state()
        u_last_emb = self._embed_last_user()
        u_span_emb, a_span_emb = self._embed_since_eval_means()
        q_emb = self._embed_question()
        sim_last = self._cos(u_last_emb, q_emb)
        sim_span = self._cos(u_span_emb, q_emb)

        if self.cfg.proj_dim and self.cfg.proj_dim > 0:
            u_last_proj = self._project(u_last_emb)
            u_span_proj = self._project(u_span_emb)
            a_span_proj = self._project(a_span_emb)
            q_proj = self._project(q_emb)
        else:
            u_last_proj, u_span_proj, a_span_proj, q_proj = u_last_emb, u_span_emb, a_span_emb, q_emb

        masks = self.get_action_masks()
        has_pending = 1 if (self.stats.turns_since_eval > 0 or self.stats.actions_since_eval > 0) else 0

        return {
            "p_T": float(p_T), "p_F": float(p_F), "margin": float(margin),
            "decided": int(decided),
            "tau_prev_T": float(self.stats.last_tau_T), "tau_prev_F": float(self.stats.last_tau_F),
            "sim_last": float(sim_last), "sim_span": float(sim_span),
            "has_pending": int(has_pending),
            "turns_since_eval": int(self.stats.turns_since_eval),
            "actions_since_eval": int(self.stats.actions_since_eval),
            "candidate_count": int(masks["candidate_count"]),
            "u_last": u_last_proj, "u_span": u_span_proj, "a_span": a_span_proj, "q_emb": q_proj,
            "masks": masks,
        }

    # ---------- 内部：COMPLETE 放行 ----------
    def _gate_complete(self) -> Tuple[bool, str]:
        p_T, p_F, margin, decided = self._read_eval_state()
        if decided:
            return False, "already_decided"
        tau_T, tau_F = self.stats.last_tau_T, self.stats.last_tau_F
        pass_basic = (p_T >= tau_T) or (p_F >= tau_F)
        if not pass_basic:
            return False, "below_threshold"
        if margin < self.cfg.min_margin:
            return False, "low_margin"
        return True, "ok"

    # ---------- 内部：评估态读取 ----------
    def _read_eval_state(self) -> Tuple[float, float, float, bool]:
        cur_q = self.agent.get_current_question()
        decided = self.agent.state_manager.is_question_completed(cur_q) if cur_q else False
        fu_t = fu_f = 1.0
        if cur_q:
            # 先找题目的 id 作为 prob_cache key
            key_id = None
            for sid, s in self.agent.state_manager.survey_states.items():
                for it in s['items']:
                    if it['question'] == cur_q:
                        key_id = it['id']; break
                if key_id: break
            if key_id and key_id in self.agent.evaluator.prob_cache:
                rec = self.agent.evaluator.prob_cache[key_id]
                fu_t = float(rec.get("uncertainty_true", fu_t))
                fu_f = float(rec.get("uncertainty_false", fu_f))
            elif cur_q in self.agent.evaluator.prob_cache:
                rec = self.agent.evaluator.prob_cache[cur_q]
                fu_t = float(rec.get("uncertainty_true", fu_t))
                fu_f = float(rec.get("uncertainty_false", fu_f))
        p_T = 1.0 - fu_t
        p_F = 1.0 - fu_f
        margin = abs(p_T - p_F)
        return p_T, p_F, margin, decided

    # ---------- 内部：向量/投影 ----------
    def _ensure_projection(self, dim: int):
        if self._proj_W is not None or dim <= 0:
            return
        # 假取一个向量维度（依赖 sim_tool 的实际 embed 维）
        sample = self._embed_text("样本", persist_q=False)
        d_in = sample.shape[0] if sample is not None else 768
        self._proj_W = self.rng.normal(0, 1.0 / np.sqrt(self.cfg.proj_dim), size=(self.cfg.proj_dim, d_in))

    def _project(self, v: Optional[np.ndarray]) -> np.ndarray:
        if v is None:
            return np.zeros(self.cfg.proj_dim, dtype=np.float32)
        if not (self.cfg.proj_dim and self.cfg.proj_dim > 0):
            return v.astype(np.float32)
        self._ensure_projection(self.cfg.proj_dim)
        return (self._proj_W @ v.astype(np.float32))

    def _embed_text(self, text: str, persist_q: bool = False) -> Optional[np.ndarray]:
        if not text or self.sim_tool is None:
            return None
        try:
            return self.sim_tool._get_embedding(text, persist=persist_q)
        except Exception:
            return None

    def _embed_last_user(self) -> Optional[np.ndarray]:
        for d in reversed(self.dialog_history):
            if d.get("role") == "user":
                return self._embed_text(d.get("content", ""), persist_q=False)
        return None

    def _embed_since_eval_means(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = max(0, self.stats.eval_cursor)
        user_vecs, asst_vecs = [], []
        for d in self.dialog_history[start:]:
            txt = d.get("content", "")
            if not txt:
                continue
            v = self._embed_text(txt, persist_q=False)
            if d.get("role") == "user":
                if v is not None: user_vecs.append(v)
            else:
                if v is not None: asst_vecs.append(v)
        u_mean = np.mean([v for v in user_vecs if v is not None], axis=0) if user_vecs else None
        a_mean = np.mean([v for v in asst_vecs if v is not None], axis=0) if asst_vecs else None
        return u_mean, a_mean

    def _embed_question(self) -> Optional[np.ndarray]:
        q = self.agent.get_current_question()
        return self._embed_text(q, persist_q=True) if q else None

    @staticmethod
    def _cos(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
        if v1 is None or v2 is None:
            return 0.0
        try:
            return float(np.dot(v1, v2))
        except Exception:
            return 0.0

    def _is_current_decided(self) -> bool:
        q = self.agent.get_current_question()
        return self.agent.state_manager.is_question_completed(q) if q else False

    def _get_last_user_text(self) -> Optional[str]:
        for d in reversed(self.dialog_history):
            if d.get("role") == "user":
                return d.get("content")
        return None

    def _get_last_assistant_text(self) -> Optional[str]:
        for d in reversed(self.dialog_history):
            if d.get("role") == "assistant":
                return d.get("content")
        return None

    # ---------- 表格式奖励（相关问卷 True/False） ----------
    def _table_rewards_for_updates(self, updates: Optional[List[Dict[str, Any]]]) -> float:
        if not updates:
            return 0.0
        delta = 0.0
        cur_sid = self.agent.current_survey_id
        if cur_sid and (cur_sid in set(self.cfg.relevant_surveys)):
            for u in updates:
                st = u.get("status", None)
                if st is True:
                    delta += self.cfg.reward_true_relevant
                elif st is False:
                    delta += self.cfg.penalty_false_relevant
        return float(delta)

    # ---------- 结论正确性检查 ----------
    def _maybe_apply_conclusion_penalty(self, trigger: str, reward_acc):
        mode = (self.cfg.conclusion_check_mode or "").lower()
        if mode not in ("on_complete", "on_done", "both"):
            return
        if (mode == "on_complete" and trigger != "on_complete") or \
           (mode == "on_done" and trigger != "on_done"):
            return
        if not self.cfg.relevant_surveys:
            return
        ratios = self._compute_true_ratios()
        if not ratios:
            return
        best_sid = max(ratios.items(), key=lambda kv: kv[1])[0]
        rel_best_sid = max(self.cfg.relevant_surveys, key=lambda sid: ratios.get(sid, -1.0))
        if ratios.get(rel_best_sid, -1.0) < ratios.get(best_sid, -1.0) - 1e-8 and best_sid not in self.cfg.relevant_surveys:
            reward_acc(self.cfg.wrong_conclusion_penalty)

    def _compute_true_ratios(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for sid, state in self.agent.state_manager.survey_states.items():
            items = state.get('items', [])
            total = len(items)
            if total == 0:
                ratios[sid] = -1.0
                continue
            trues = sum(1 for it in items if it.get('status') is True)
            ratios[sid] = trues / total
        return ratios


if __name__ == "__main__":
    print("QuestionnaireEnv (final) loaded.")