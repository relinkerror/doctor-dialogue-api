# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- 运行时 sys.path 兜底：确保项目根在路径里 ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ------------------------------------------------

# ===== 你项目里的模块（顶层）=====
from agents.survey.survey_agent import SurveyAgent
from agents.dialogue.dialogue_patient_agent import DialoguePatientAgent
from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent
from utils.LLMClient import LLMClient

# ===== 通用依赖 =====
import numpy as np
import joblib

# ========== 嵌入与模板（替代 rl_heads.common / infer_demo） ==========

DEFAULT_EMB = os.getenv("EMB_MODEL_PATH", "models/paraphrase-multilingual-MiniLM-L12-v2")

# 三类 VALIDATE 模板（如需，改成你项目里的 validate_ops 模板）
TEMPLATES = {
    "CLARIFY_MEANING": "To clarify about {symptom}, could you tell me when it started or what affects it?",
    "PROBE_EXAMPLE": "Could you give an example or quantify the {symptom}, like how often or how much?",
    "CONFIRM_RESTATEMENT": "So just to confirm regarding {symptom}, is my understanding correct?"
}

_embedder = None
def embed_texts(texts: List[str], model_path: str = DEFAULT_EMB) -> np.ndarray:
    """L2-normalized sentence embeddings"""
    global _embedder
    if _embedder is None:
        # 懒加载，避免编辑器报错
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_path)
    embs = _embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return embs

def _template_sims(context_texts: List[str], themes: List[str], emb_model: str) -> np.ndarray:
    """
    为 validate_op_head 构造 3 维模板相似度特征：cos(context, template(op themed))
    训练时我们使用了这 3 维，因此推理也要一致。
    """
    ctx = embed_texts(context_texts, emb_model)  # (n,d)
    sims = []
    for i, th in enumerate(themes):
        templs = [
            TEMPLATES["CLARIFY_MEANING"].format(symptom=th or "symptom"),
            TEMPLATES["PROBE_EXAMPLE"].format(symptom=th or "symptom"),
            TEMPLATES["CONFIRM_RESTATEMENT"].format(symptom=th or "symptom"),
        ]
        temb = embed_texts(templs, emb_model)    # (3,d), 已 L2
        s = (ctx[i:i+1] @ temb.T).ravel()        # (3,)
        sims.append(s)
    return np.vstack(sims)  # (n,3)

def _featurize_validate(context_text: str, theme: str, emb_model: str) -> np.ndarray:
    """
    与训练时的 featurize 对齐： [ctx_emb, theme_emb, clar_present, quant_present, sims3]
    在线时没有结构统计，clar_present/quant_present 置 0。
    """
    ctx = embed_texts([context_text], emb_model)
    thm = embed_texts([theme], emb_model)
    counts = np.zeros((1,2), dtype=np.float32)     # clar_present, quant_present
    sims = _template_sims([context_text], [theme], emb_model)  # (1,3)
    X = np.hstack([ctx, thm, counts, sims])        # (1, 2d + 2 + 3)
    return X

def _featurize_coherence(theme: str, candidate_text: str, emb_model: str) -> np.ndarray:
    """
    与训练时的 featurize 对齐： [theme_emb, cand_emb, cos]
    """
    thm = embed_texts([theme], emb_model)
    cnd = embed_texts([candidate_text], emb_model)
    cos = np.sum(thm * cnd, axis=1, keepdims=True)  # (1,1)
    X = np.hstack([thm, cnd, cos])                  # (1, 2d + 1)
    return X

def score_validate_op(context_text: str, theme: str, model_path: str, emb_model: str) -> Dict[str, float]:
    """
    返回三类 VALIDATE 的概率分布：{"CLARIFY_MEANING": p1, "PROBE_EXAMPLE": p2, "CONFIRM_RESTATEMENT": p3}
    """
    clf = joblib.load(model_path)   # Pipeline(scaler+logreg)；logreg 在 step 名为 "logreg"
    X = _featurize_validate(context_text, theme, emb_model)
    proba = clf.predict_proba(X)[0]
    classes = clf.named_steps["logreg"].classes_.tolist()
    return dict(zip(classes, proba))

def score_coherence(theme: str, candidate_text: str, model_path: str, emb_model: str) -> float:
    """
    返回连贯度概率（正类=1）
    """
    clf = joblib.load(model_path)
    X = _featurize_coherence(theme, candidate_text, emb_model)
    prob = float(clf.predict_proba(X)[0,1])
    return prob

def choose_validate_op(context_text: str, theme: str,
                       validate_model: str, coherence_model: str, emb_model: str,
                       alpha: float = 0.6, eps: float = 0.1) -> Dict[str, Any]:
    """
    合成分数 = alpha * p(op) + (1-alpha) * coherence(theme, template(op))
    返回 {"best": OP, "op_scores": {...}, "coherence": {...}, "combined": {...}}
    """
    op_scores = score_validate_op(context_text, theme, validate_model, emb_model)
    coh_scores = {}
    for op, tmpl in TEMPLATES.items():
        cand = tmpl.format(symptom=theme or "symptom")
        coh_scores[op] = score_coherence(theme, cand, coherence_model, emb_model)
    combined = {op: alpha*op_scores.get(op, 0.0) + (1-alpha)*coh_scores[op] for op in op_scores.keys()}

    import random
    best = max(combined, key=combined.get)
    # epsilon-greedy
    if random.random() < eps:
        best = random.choice(list(combined.keys()))
    return {"best": best, "op_scores": op_scores, "coherence": coh_scores, "combined": combined}

# ========== 统计 True 比例、实用函数 ==========

def compute_true_ratio(all_answers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """统计每份问卷与全局的 True 比例"""
    per_survey = {}
    total_true = total_ans = 0
    for sid, items in all_answers.items():
        tr = sum(1 for it in items if it.get("status") is True)
        an = sum(1 for it in items if it.get("status") is not None)
        per_survey[sid] = {"true": tr, "answered": an, "true_ratio": (tr / an) if an else 0.0}
        total_true += tr
        total_ans  += an
    overall = {"true": total_true, "answered": total_ans, "true_ratio": (total_true / total_ans) if total_ans else 0.0}
    return {"per_survey": per_survey, "overall": overall}

def last_turns_text(dialog_history: List[Dict[str, str]], n_pair: int = 1) -> str:
    """拼接最近 n 对（user+assistant）的文本作为 context_text"""
    if not dialog_history:
        return ""
    k = min(len(dialog_history), max(2 * n_pair, 2))
    return " [SEP] ".join([d["content"] for d in dialog_history[-k:]])

def safe_tick(agent: SurveyAgent):
    """兼容不同实现的轮次计数方法"""
    if hasattr(agent, "tick_round"):
        agent.tick_round()
    elif hasattr(agent, "_tick_round"):
        agent._tick_round()

# ========== Heads 组合策略 ==========

class HeadsValidatePolicy:
    """
    用 validate_op_head + coherence_head 合成分数选择 VALIDATE 子类
    combined = alpha * p(op) + (1 - alpha) * coherence(theme, template(op))
    """
    def __init__(self, validate_model: str, coherence_model: str, emb_model: str,
                 alpha: float = 0.6, eps: float = 0.1):
        self.validate_model = validate_model
        self.coherence_model = coherence_model
        self.emb_model = emb_model
        self.alpha = float(alpha)
        self.eps   = float(eps)

    def select_op(self, context_text: str, theme: str) -> Dict[str, Any]:
        return choose_validate_op(
            context_text=context_text,
            theme=theme,
            validate_model=self.validate_model,
            coherence_model=self.coherence_model,
            emb_model=self.emb_model,
            alpha=self.alpha,
            eps=self.eps
        )

# ========== 实验主程序 ==========

def run(args):
    # 1) LLMClient
    llm = LLMClient.from_config(args.llm_config)

    # 2) SurveyAgent（关闭内部自动评估；我们用“下一轮 COMPLETE”）
    with open(args.eval_prompt, "r", encoding="utf-8") as f:
        eval_prompt = f.read()
    agent = SurveyAgent(
        meta_json_path=args.meta_json,
        questionnaire_dir=args.questionnaire_dir,
        llm_client=llm,
        eval_prompt=eval_prompt,
        survey_depends_on_prompt=None,
        embedder_model=args.emb_model,
        threshold=args.threshold,
        lazy_eval=True,
        eval_every_n=10**9,   # 很大值，禁用内部的 N 步评估
    )

    # 3) （可选）医生自然话术
    doctor = None
    if int(args.use_doctor_llm) == 1:
        from agents.dialogue.summary_generator import ISummaryGenerator
        class DummySummary(ISummaryGenerator):
            def get_summary(self): return ""
        doctor = DialogueDoctorAgent(client=llm, name="doctor", summary_generator=DummySummary(),
                                     config={"debug_log_path": "logs/doctor_debug.jsonl"})

    # 4) 两个头组合的 VALIDATE 策略
    policy = HeadsValidatePolicy(
        validate_model=args.validate_model,
        coherence_model=args.coherence_model,
        emb_model=args.emb_model,
        alpha=args.alpha,
        eps=args.eps
    )

    dialog_history: List[Dict[str, str]] = []
    step = 0
    eval_counter = 0
    log_rows = []
    pending_complete = False

    print("\n==== True 比例实验开始（VALIDATE → 下一轮 COMPLETE 模式）====\n")

    while step < args.max_steps:
        if agent.is_completed():
            print("[DONE] 所有问题已完成。")
            break

        # A) 若上一轮做过 VALIDATE，则本轮开头做 COMPLETE 强评
        if pending_complete:
            updates = agent.apply_action("complete", {}, dialog_history, convo_agent=None)
            eval_counter += 1
            answers = agent.get_all_answers()
            stats = compute_true_ratio(answers)
            print(f"[EVAL#{eval_counter}] updates={len(updates)} overall_true_ratio={stats['overall']['true_ratio']:.3f}")
            for sid, row in stats["per_survey"].items():
                print(f"  - {sid}: {row['true']}/{row['answered']} (ratio={row['true_ratio']:.3f})")
            log_rows.append({"eval_id": eval_counter, "step": step, "updates": updates, "stats": stats})
            pending_complete = False

            # 计步（把 complete 也计入一步，形成严格交替）
            safe_tick(agent)
            step += 1
            if step >= args.max_steps:
                break

        # B) 选择/确认当前题（不评估）
        q = agent.auto_advance(dialog_history)  # 内部会按选择器挑 pending 叶子题
        if not q:
            # 没题可问了，做一次最终评估后退出
            updates = agent.apply_action("complete", {}, dialog_history, convo_agent=None)
            answers = agent.get_all_answers()
            stats = compute_true_ratio(answers)
            print(f"[FINAL-EVAL] updates={len(updates)} overall_true_ratio={stats['overall']['true_ratio']:.3f}")
            break

        # C) 用两个头选择 VALIDATE 子类
        theme = q
        ctx_text = last_turns_text(dialog_history, n_pair=1)
        sel = policy.select_op(context_text=ctx_text, theme=theme)
        chosen_op = sel["best"]
        print(f"[STEP {step}] Q='{str(q)[:50]}...'  VALIDATE<{chosen_op}>  scores={sel['combined']}")

        # D) 执行 VALIDATE（仅追加引导，不评估）
        #    param={"op": op_str}；你的 SurveyAgent 支持 str/Enum 均可
        agent.apply_action(
            action_type="validate",
            param={"op": chosen_op},
            dialog_history=dialog_history,
            convo_agent=doctor
        )

        # 下一轮开头要 COMPLETE
        pending_complete = True

        # 计步
        safe_tick(agent)
        step += 1

    # 收尾：再做一次 COMPLETE 并总结 True 比例
    updates = agent.apply_action("complete", {}, dialog_history, convo_agent=None)
    answers = agent.get_all_answers()
    stats = compute_true_ratio(answers)
    print("\n==== 实验结束 ====")
    print(f"[FINAL] overall: {stats['overall']['true']}/{stats['overall']['answered']} "
          f"(ratio={stats['overall']['true_ratio']:.3f})")
    for sid, row in stats["per_survey"].items():
        print(f"        {sid}: {row['true']}/{row['answered']} (ratio={row['true_ratio']:.3f})")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"final": stats, "logs": log_rows}, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] 已写入 {args.out_json}")


def build_argparser():
    ap = argparse.ArgumentParser()
    # 必填：项目配置 & 问卷/评估资源
    ap.add_argument("--llm_config", required=True, help="LLMClient 配置 json")
    ap.add_argument("--meta_json", required=True, help="问卷 meta json（含 surveys 列表）")
    ap.add_argument("--questionnaire_dir", required=True, help="问卷原文目录")
    ap.add_argument("--eval_prompt", required=True, help="评估 prompt 文件路径")
    # 两个头 + 向量模型
    ap.add_argument("--validate_model", default="models/validate_op_head.joblib")
    ap.add_argument("--coherence_model", default="models/coherence_head.joblib")
    ap.add_argument("--emb_model", default=os.getenv("EMB_MODEL_PATH", "models/paraphrase-multilingual-MiniLM-L12-v2"))
    # 策略参数
    ap.add_argument("--alpha", type=float, default=0.6, help="p(op) 与 coherence 的加权")
    ap.add_argument("--eps", type=float, default=0.1, help="epsilon-greedy 探索率")
    # 评估与轮次
    ap.add_argument("--max_steps", type=int, default=60, help="最多动作步（VALIDATE/COMPLETE 各计 1 步）")
    ap.add_argument("--threshold", type=float, default=0.1, help="SurveyUncertaintyEvaluator 置信阈值")
    # 其他
    ap.add_argument("--use_doctor_llm", type=int, default=0, help="1=用医生代理生成自然话术；0=直接用模板注入")
    ap.add_argument("--out_json", default="./runs/true_ratio_heads_exp.json", help="日志保存路径")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
