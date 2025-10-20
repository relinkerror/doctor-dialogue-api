# -*- coding: utf-8 -*-
"""
Compact Bandit trainer with explicit Doctor↔Patient loop + Resume & Checkpoints
- Linear contextual bandit over small action templates (REINFORCE + baseline + L2)
- Uses QuestionnaireEnv for reward/obs; injects doctor↔patient turns like experiments
- JSONL logs + end-of-run weights + periodic checkpoints

Run (example):
python -m rl.train_bandit_questionnaire --personas MDD,PDD,SMDID,DDAMC,DMDD,PMDD --steps 2000 --no_doctor 0 --save_every 200
"""
from __future__ import annotations

import os, json, time, random, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# project modules
from utils.LLMClient import LLMClient
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent
from agents.dialogue.dialogue_patient_agent import DialoguePatientAgent
from rl.envs.questionnaire_env import QuestionnaireEnv, EnvConfig  # env 内含动作/奖励/日志

FEATURE_KEYS = [
    "p_T","p_F","margin","decided","sim_last","sim_span",
    "candidate_count","has_pending","turns_since_eval","actions_since_eval",
]

def _scale_tau(x: float) -> float:
    return (x - 0.90) / 0.09  # map [0.90,0.99] -> [0,1]

TEMPLATES: List[np.ndarray] = [
    np.array([0,0,0,0.50,0.50], np.float32),                          # VALIDATE-CLARIFY
    np.array([0,1,0,0.50,0.50], np.float32),                          # VALIDATE-PROBE
    np.array([0,2,0,0.50,0.50], np.float32),                          # VALIDATE-CONFIRM
    np.array([2,0,0,_scale_tau(0.95),_scale_tau(0.95)], np.float32),  # COMPLETE balanced
    np.array([2,0,0,_scale_tau(0.97),_scale_tau(0.90)], np.float32),  # COMPLETE bias T
    np.array([2,0,0,_scale_tau(0.90),_scale_tau(0.97)], np.float32),  # COMPLETE bias F
    np.array([1,0,0,0.50,0.50], np.float32),                          # JUMP idx=0
]

PERSONA_TO_REL = {
    "MDD":["MDD"],"PDD":["PDD"],"SMDID":["SMDID"],"DDAMC":["DDAMC"],"DMDD":["DMDD"],"PMDD":["PMDD"]
}

class Bandit:
    def __init__(self, K:int, D:int, alpha=0.05, l2=1e-4, eps=0.2, temperature=1.0, seed=42):
        self.rng = np.random.RandomState(seed)
        self.W = np.zeros((K, D), np.float32)
        self.alpha, self.l2, self.eps, self.temp = alpha, l2, eps, max(1e-6, temperature)
        self.baseline, self.beta = 0.0, 0.99

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        m = float(scores.max()) if scores.size else 0.0
        e = np.exp((scores - m) / self.temp)
        s = float(e.sum())
        return e / s if s > 0 else np.ones_like(scores) / max(1, scores.size)

    def choose(self, x: np.ndarray, avail: List[int]) -> Tuple[int, np.ndarray]:
        scores = np.array([self.W[i].dot(x) for i in avail], np.float32)
        pi = self._softmax(scores)
        if self.eps > 0 and self.rng.rand() < self.eps:
            idx = self.rng.randint(len(avail))
        else:
            idx = int(self.rng.choice(len(avail), p=pi))
        return avail[idx], pi

    def update(self, a_id: int, x: np.ndarray, avail: List[int], pi: np.ndarray, r: float):
        self.baseline = self.beta * self.baseline + (1.0 - self.beta) * r
        adv = r - self.baseline
        for j, k in enumerate(avail):
            grad = (-pi[j]) * x
            if k == a_id:
                grad = (1.0 - pi[j]) * x
            self.W[k] += self.alpha * (adv * grad - self.l2 * self.W[k])

class Trainer:
    def __init__(self, personas: List[str], persona_mode: str, weights: Optional[List[float]],
                 steps: int, alpha: float, l2: float, eps: float, temperature: float,
                 proj_dim: int, use_doctor: bool, save_every: int = 0,
                 resume_npz: Optional[str] = None):
        self.personas, self.persona_mode, self.weights = personas, persona_mode, weights
        self.steps = steps
        self.save_every = int(save_every)
        self.resume_npz = resume_npz

        # paths
        self.root = Path(__file__).resolve().parents[1]
        self.logs_dir = self.root / "logs"; self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.runs_dir = self.root / "runs"; self.runs_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir = self.runs_dir / "checkpoints"; self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.bandit_log = open(self.logs_dir / "bandit_debug.jsonl", "a", encoding="utf-8", buffering=1)

        # llm client
        self.CONFIG_DIR = self.root / "config"
        self.QUESTIONNAIRE_DIR = self.root / "questionnaire"
        self.llm = LLMClient.from_config(str(self.CONFIG_DIR / "llm_config.json"))

        # doctor (可选)
        self.use_doctor = bool(use_doctor)
        self.doctor: Optional[DialogueDoctorAgent] = None
        if self.use_doctor:
            self.doctor = DialogueDoctorAgent(
                self.llm, name="bandit-doctor",
                summary_generator=LLMSummaryGenerator(self.llm),
                config={"debug_log_path": str(self.logs_dir / "doctor_debug.jsonl")},
            )

        # 先建一个 SurveyAgent（reset_episode 时会重建并替换）
        with open(self.CONFIG_DIR / "survey_uncertainty_eval_prompt.txt", encoding="utf-8") as f:
            eval_prompt = f.read()
        meta_json = str(self.CONFIG_DIR / "dsm5_depression_surveys.json")
        self.survey = SurveyAgent(meta_json, str(self.QUESTIONNAIRE_DIR), self.llm,
                                  eval_prompt=eval_prompt, threshold=0.2,
                                  lazy_eval=True, eval_every_n=5)

        # patient 配置（每个 episode 重建 patient）
        with open(self.CONFIG_DIR / "patient_agent.json", encoding="utf-8") as f:
            self.patient_cfg = json.load(f)
        self.patient: Optional[DialoguePatientAgent] = None

        # env（核心奖励/动作/日志都在 Env 内实现）
        cfg = EnvConfig(
            relevant_surveys=["MDD"], proj_dim=proj_dim,
            llm_cost_per_call=-0.06, reward_true_relevant=1.0, penalty_false_relevant=-1.0,
            op_match_coef=0.20, coherence_coef=0.20, conclusion_check_mode="both",
            wrong_conclusion_penalty=-10.0, debug_log_path=str(self.logs_dir / "env_debug.jsonl"),
        )
        self.env = QuestionnaireEnv(self.survey, dialog_history=[], config=cfg, random_seed=42, doctor=self.doctor)

        # bandit
        self.D = 1 + len(FEATURE_KEYS)
        self.K = len(TEMPLATES)
        self.bandit = Bandit(self.K, self.D, alpha=alpha, l2=l2, eps=eps, temperature=temperature)

    # -------- resume & checkpoint --------
    def maybe_resume(self):
        p = Path(self.resume_npz) if self.resume_npz else None
        if not p:
            return
        if not p.exists():
            print(f"[resume] file not found: {p}")
            return
        try:
            data = np.load(str(p), allow_pickle=True)
            W = data["W"]
            if W.shape != self.bandit.W.shape:
                print(f"[resume] shape mismatch: ckpt {W.shape} vs current {self.bandit.W.shape}")
                return
            self.bandit.W[...] = W
            if "feature_keys" in data.files:
                fk = list(data["feature_keys"])
                if [*fk] != FEATURE_KEYS:
                    print("[resume] warning: feature_keys mismatch; weights loaded anyway.")
            print(f"[resume] loaded weights from {p}")
        except Exception as e:
            print(f"[resume] failed to load {p}: {e}")

    def maybe_checkpoint(self, t: int):
        if self.save_every <= 0: return
        if t % self.save_every != 0: return
        ckpt = self.ckpt_dir / f"bandit_step_{t}.npz"
        np.savez(str(ckpt), W=self.bandit.W, feature_keys=np.array(FEATURE_KEYS, dtype=object))
        print(f"[ckpt] saved {ckpt}")

    # -------- sampling & features --------
    def _sample_persona(self) -> str:
        if len(self.personas) == 1 or self.persona_mode == "single":
            return self.personas[0]
        if self.persona_mode == "weighted" and self.weights:
            return random.choices(self.personas, weights=self.weights, k=1)[0]
        return random.choice(self.personas)

    def _phi(self, obs: Dict[str, Any]) -> np.ndarray:
        x = [1.0]
        for k in FEATURE_KEYS:
            v = float(obs.get(k, 0.0))
            if k in ("turns_since_eval","actions_since_eval"): v = min(v,10.0)/10.0
            if k == "candidate_count": v = min(v,5.0)/5.0
            x.append(v)
        return np.asarray(x, np.float32)

    def _avail(self, obs: Dict[str, Any]) -> List[int]:
        avail = list(range(self.K))
        if int(obs.get("candidate_count",0)) <= 0 and 6 in avail:
            avail.remove(6)
        return avail

    # ------ experiment-like round: auto_advance -> doctor.ask -> patient.reply ------
    def _pre_round_exchange(self):
        cur_q = self.survey.auto_advance(self.env.dialog_history)
        if not cur_q:
            return False
        # doctor ask, then patient reply —— 写回 history（assistant → user）
        if self.use_doctor and self.doctor is not None:
            doc_ask = self.doctor.generate_dialog(
                user_input=self._last_user_text() or "你好，我想继续上次的咨询。",
                current_question=cur_q,
                question_context=self.survey.get_question_context(cur_q),
            )
            self.env.dialog_history.append({"role":"assistant","content":doc_ask})
        else:
            doc_ask = f"Could you tell me more about: {cur_q}"
            self.env.dialog_history.append({"role":"assistant","content":doc_ask})
        if self.patient is not None:
            pat_reply = self.patient.generate_dialog(user_input=doc_ask)
            self.env.dialog_history.append({"role":"user","content":pat_reply})
        return True

    def _last_user_text(self) -> Optional[str]:
        for d in reversed(self.env.dialog_history):
            if d.get("role") == "user":
                return d.get("content")
        return None

    # -------- episode reset (rebuild SurveyAgent like experiments) --------
    def reset_episode(self):
        # 1) 采样 persona & relevant
        persona = self._sample_persona()
        rel = PERSONA_TO_REL.get(persona, [persona])

        # 2) —— 重建 SurveyAgent（对齐实验脚本的做法）——
        with open(self.CONFIG_DIR / "survey_uncertainty_eval_prompt.txt", encoding="utf-8") as f:
            eval_prompt = f.read()
        meta_json = str(self.CONFIG_DIR / "dsm5_depression_surveys.json")

        new_agent = SurveyAgent(
            meta_json, str(self.QUESTIONNAIRE_DIR), self.llm,
            eval_prompt=eval_prompt, threshold=0.2,
            lazy_eval=True, eval_every_n=5
        )
        try:
            new_agent.selector.set_relevant_surveys(rel)
        except Exception:
            pass

        # 3) 替换到 env，并同步 env 的 relevant_surveys
        self.survey = new_agent
        self.env.agent = new_agent
        self.env.cfg.relevant_surveys = rel

        # 4) 清空对话历史；doctor 也重置（或重建）以清历史
        self.env.dialog_history.clear()
        if self.use_doctor:
            try:
                self.doctor.reset()  # 如果实现了 reset
            except Exception:
                # 重建 doctor（与实验里每轮重建 doctor 的习惯一致）
                self.doctor = DialogueDoctorAgent(
                    self.llm, name="bandit-doctor",
                    summary_generator=LLMSummaryGenerator(self.llm),
                    config={"debug_log_path": str(self.logs_dir / "doctor_debug.jsonl")},
                )
            self.env.doctor = self.doctor  # 确保 env 内 doctor 指针同步

        # 5) 重建 patient（与实验一致）
        self.patient = DialoguePatientAgent(
            client=self.llm,
            name=f"bandit-patient-{persona}",
            summary_generator=LLMSummaryGenerator(self.llm),
            config=self.patient_cfg,
            persona_id=persona,
        )

        # 6) env.reset() 取得一个干净的首观测
        obs = self.env.reset()
        return persona, obs

    # -------- training loop --------
    def train(self):
        # resume weights if provided
        self.maybe_resume()

        persona, obs = self.reset_episode()
        for t in range(1, self.steps+1):
            # pre-round convo
            if not self._pre_round_exchange():
                persona, obs = self.reset_episode()
                continue

            # choose action
            x = self._phi(obs)
            avail = self._avail(obs)
            a_id, pi = self.bandit.choose(x, avail)
            action = TEMPLATES[a_id]

            # env step
            next_obs, r, done, info = self.env.step(action)

            # after VALIDATE: patient replies to validation utterance (close loop)
            if int(action[0]) == 0 and self.env.dialog_history and self.env.dialog_history[-1].get("role") == "assistant":
                val_utt = self.env.dialog_history[-1]["content"]
                pat = self.patient.generate_dialog(user_input=val_utt)
                self.env.dialog_history.append({"role":"user","content":pat})

            # lazy eval tick like experiments（与两份实验脚本节奏一致）
            self.survey.tick_round()

            # update
            self.bandit.update(a_id, x, avail, pi, float(r))

            # logging
            rec = {"t": time.time(), "step": t, "persona": persona, "a_id": int(a_id), "reward": float(r)}
            self.bandit_log.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.bandit_log.flush()
            try: os.fsync(self.bandit_log.fileno())
            except Exception: pass

            # periodic checkpoint
            self.maybe_checkpoint(t)

            obs = next_obs
            if done:
                persona, obs = self.reset_episode()

        # final save
        tag = "mix" if (len(self.personas) > 1 or self.persona_mode != "single") else self.personas[0]
        out = self.runs_dir / f"bandit_policy_{tag}.npz"
        np.savez(str(out), W=self.bandit.W, feature_keys=np.array(FEATURE_KEYS, dtype=object))
        self.bandit_log.close()
        print(f"✅ Done. Saved policy to {out}\nLogs at: {self.logs_dir / 'bandit_debug.jsonl'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="MDD", help="comma list, e.g. MDD,PDD,PMDD")
    ap.add_argument("--persona_mode", choices=["single","random","weighted"], default="random")
    ap.add_argument("--weights", default="", help="comma weights if weighted")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--proj_dim", type=int, default=32)
    ap.add_argument("--no_doctor", type=int, default=1, help="1 to disable doctor LLM for speed")
    ap.add_argument("--save_every", type=int, default=0, help="save checkpoint every N steps (0=disable)")
    ap.add_argument("--resume_npz", default="", help="path to saved npz to resume (e.g., runs/bandit_policy_mix.npz)")
    args = ap.parse_args()

    personas = [p.strip() for p in args.personas.split(',') if p.strip()]
    weights = [float(x) for x in args.weights.split(',')] if args.weights else None

    trainer = Trainer(
        personas=personas,
        persona_mode=args.persona_mode,
        weights=weights,
        steps=args.steps,
        alpha=args.alpha,
        l2=args.l2,
        eps=args.eps,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        use_doctor=(args.no_doctor==0),
        save_every=args.save_every,
        resume_npz=(args.resume_npz or None),
    )
    trainer.train()

if __name__ == "__main__":
    main()
