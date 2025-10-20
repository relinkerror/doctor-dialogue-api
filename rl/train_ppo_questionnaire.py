# -*- coding: utf-8 -*-
import os, json, argparse, logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ==== 你的工程内模块 ====
from utils.LLMClient import LLMClient
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent
from rl.envs.questionnaire_env import QuestionnaireEnv, EnvConfig

# ---------- 把 obs(dict) 展平为 1D 向量 ----------
NUM_SCALARS = [
    "p_T","p_F","margin","decided","tau_prev_T","tau_prev_F",
    "sim_last","sim_span","has_pending","turns_since_eval",
    "actions_since_eval","candidate_count"
]
VEC_KEYS = ["u_last","u_span","a_span","q_emb"]

def flatten_obs(obs):
    scalars = np.array([float(obs[k]) for k in NUM_SCALARS], dtype=np.float32)
    vecs = []
    for k in VEC_KEYS:
        v = obs.get(k, None)
        if v is None: v = np.zeros_like(obs["q_emb"])
        vecs.append(np.array(v, dtype=np.float32))
    return np.concatenate([scalars] + vecs, axis=0)

# ---------- SB3 包装器 ----------
class SB3QuestionnaireEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, persona_id: str, relevant_surveys, cfg_overrides=None, doctor_cfg=None):
        super().__init__()
        # === 目录 ===
        CONFIG_DIR = "config"; QUESTIONNAIRE_DIR = "questionnaire"

        # === LLM / Doctor ===
        llm = LLMClient.from_config(os.path.join(CONFIG_DIR, "llm_config.json"))
        doctor = DialogueDoctorAgent(
            llm,
            name=f"rl-doctor-{persona_id}",
            summary_generator=LLMSummaryGenerator(llm),
            config=(doctor_cfg or {"debug_log_path": "logs/doctor_debug.jsonl"})
        )

        # === SurveyAgent ===
        with open(os.path.join(CONFIG_DIR, "survey_uncertainty_eval_prompt.txt"), encoding="utf-8") as f:
            eval_prompt = f.read()
        meta_json_path = os.path.join(CONFIG_DIR, "dsm5_depression_surveys.json")

        self.survey_agent = SurveyAgent(
            meta_json_path, QUESTIONNAIRE_DIR, llm,
            eval_prompt=eval_prompt,
            threshold=0.2,
            lazy_eval=True,
            eval_every_n=5
        )

        # === EnvConfig ===
        cfg = EnvConfig(
            relevant_surveys=list(relevant_surveys),
            llm_cost_per_call=-0.06,
            reward_true_relevant=1.0,
            penalty_false_relevant=-1.0,
            op_match_coef=0.20,
            coherence_coef=0.20,
            conclusion_check_mode="both",
            wrong_conclusion_penalty=-10.0,
        )
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        # === Env Core ===
        self.core = QuestionnaireEnv(self.survey_agent, dialog_history=[], config=cfg, random_seed=42, doctor=doctor)

        # SB3 spaces —— 动作：5维 Box（[type, validate_op, jump_idx, tauT, tauF]）
        self.action_space = spaces.Box(low=np.array([0,0,0,0,0],np.float32),
                                       high=np.array([2,2,100,1,1],np.float32), dtype=np.float32)

        # 先 reset 拿到 obs 形状
        o = self.core.reset()
        flat = flatten_obs(o)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        o = self.core.reset()
        return flatten_obs(o), {}

    def step(self, action):
        a = np.array(action, dtype=np.float32)
        obs, rew, done, info = self.core.step(a)
        return flatten_obs(obs), float(rew), bool(done), False, info  # truncated=False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="MDD")
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--n_steps", type=int, default=1024)
    ap.add_argument("--checkpoint_every", type=int, default=5000)
    ap.add_argument("--resume", type=int, default=0)
    args = ap.parse_args()

    # persona -> 相关问卷（示例映射）
    persona_to_rel = {"MDD": ["MDD"], "GAD": ["GAD"], "PDD": ["PDD"]}
    relevant = persona_to_rel.get(args.persona, ["MDD"])  # 默认 MDD

    # 固定训练日志
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename="logs/train.log", level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    env = SB3QuestionnaireEnv(persona_id=args.persona, relevant_surveys=relevant,
                              cfg_overrides={"proj_dim": args.proj_dim})

    # PPO 实例化 + TensorBoard
    model_path = f"runs/ppo_questionnaire_{args.persona}.zip"
    os.makedirs("runs/checkpoints", exist_ok=True)

    if args.resume and os.path.exists(model_path):
        logging.info(f"Resuming from {model_path}")
        model = PPO.load(model_path, env=env)
        model.set_parameters(model.get_parameters())  # 显式覆盖（兼容性写法）
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=256,
            gamma=args.gamma,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="runs/tb",
            verbose=1
        )

    ckpt = CheckpointCallback(save_freq=args.checkpoint_every, save_path="runs/checkpoints",
                              name_prefix=f"ppo_{args.persona}")
    try:
        model.learn(total_timesteps=args.total_steps, callback=ckpt, progress_bar=True,
                    reset_num_timesteps=not (args.resume and os.path.exists(model_path)))
    except KeyboardInterrupt:
        logging.warning("Interrupted. Saving model...")
    finally:
        model.save(model_path)
        logging.info(f"Saved to {model_path}")
        print("✅ 训练结束，模型已保存。")


if __name__ == "__main__":
    main()