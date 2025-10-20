# experiment/run_rl_action_sanity_exp.py
import cProfile
import sys
import os
import time
import signal
import random
import csv

def main_experiment():
    import json
    from utils.LLMClient import LLMClient
    from agents.survey.survey_agent import SurveyAgent
    from agents.survey.validate_ops import ValidateOp  # 枚举: CLARIFY_MEANING / PROBE_EXAMPLE / CONFIRM_RESTATEMENT
    from agents.dialogue.dialogue_patient_agent import DialoguePatientAgent
    from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
    from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent

    # ========= 目录/配置 =========
    CONFIG_DIR = "config"
    QUESTIONNAIRE_DIR = "questionnaire"
    TABLES_DIR = "tables"
    LOGS_DIR = "experiment/rl_logs"
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # —— 元数据与模型配置（与原实验一致）——
    with open(os.path.join(CONFIG_DIR, "patient_agent.json"), encoding="utf-8") as f:
        patient_meta = json.load(f)
    persona_ids = [p["id"] for p in patient_meta["persona_templates"]]

    with open(os.path.join(CONFIG_DIR, "doctor_agent.json"), encoding="utf-8") as f:
        doctor_meta = json.load(f)

    llm = LLMClient.from_config(os.path.join(CONFIG_DIR, "llm_config.json"))
    with open(os.path.join(CONFIG_DIR, "survey_uncertainty_eval_prompt.txt"), encoding="utf-8") as f:
        eval_prompt = f.read()
    meta_json_path = os.path.join(CONFIG_DIR, "dsm5_depression_surveys.json")

    # ========= 一个很简单的“策略” =========
    class MockRLPolicy:
        """
        目的：打通 validate / jump / complete 的动作通路
        - 奇数步：validate（随机选择一种 ValidateOp，并以一定概率先 jump 到一题）
        - 偶数步：complete（触发一次强评估，清零计数）
        """
        def __init__(self, p_jump: float = 0.25):
            self.step = 0
            self.p_jump = float(p_jump)
            self.ops = [ValidateOp.CLARIFY_MEANING,
                        ValidateOp.PROBE_EXAMPLE,
                        ValidateOp.CONFIRM_RESTATEMENT]  # 三类校验操作：澄清/举例/选项式复述
            random.seed()

        def choose(self, agent, dialog_history):
            self.step += 1
            # 偶尔先 jump 一下（只在 validate 轮尝试）
            do_jump = (self.step % 2 == 1) and (random.random() < self.p_jump)
            if do_jump:
                cands = agent.get_pending_candidates(dialog_history)  # [(qid, question), ...]
                if cands:
                    qid, _ = random.choice(cands)
                    return [("jump", {"qid": qid}),
                            ("validate", {"op": random.choice(self.ops)})]
            # 常规交替：validate / complete
            if self.step % 2 == 1:
                return [("validate", {"op": random.choice(self.ops)})]
            else:
                return [("complete", {})]

    # ========= 运行设置 =========
    N_RUNS_PER_PERSONA = 5   # 每个 persona 跑 5 轮（可自行增减）
    LAZY_EVAL = True         # 惰性评估打开，用 complete 来做强评（交替验证计数清零逻辑）
    EVAL_EVERY_N = 5

    # ========= 全局 profiler（与原实验一致） =========
    global profiler, profile_path
    profiler = globals().get("profiler", None)
    profile_path = globals().get("profile_path", "experiment/experiment.prof")

    for persona_id in persona_ids:
        print(f"\n==== RL自检实验: {persona_id} ====")

        # CSV：记录每一步的动作与评估结果，方便后续用作训练样本/回放
        csv_path = os.path.join(TABLES_DIR, f"rl_actions_{persona_id}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "run", "step",
                "action_seq",         # e.g., "jump->validate" / "complete"
                "validate_op",        # 若本轮包含 validate
                "jump_qid",           # 若本轮包含 jump
                "updates_len",        # 本轮评估（complete）返回的 updates 数
                "progress"            # 写回后的总体进度
            ])

            for run_idx in range(1, N_RUNS_PER_PERSONA + 1):
                print(f"\n--- persona_id={persona_id}, 第{run_idx}轮 ---")

                survey_agent = SurveyAgent(
                    meta_json_path, QUESTIONNAIRE_DIR, llm,
                    eval_prompt=eval_prompt,
                    threshold=0.2,
                    lazy_eval=LAZY_EVAL,
                    eval_every_n=EVAL_EVERY_N
                )
                patient = DialoguePatientAgent(
                    client=llm,
                    name=f"{persona_id}_rl_run{run_idx}",
                    summary_generator=LLMSummaryGenerator(llm),
                    config=patient_meta,
                    persona_id=persona_id
                )
                doctor = DialogueDoctorAgent(
                    client=llm,
                    name="demo-doctor",
                    summary_generator=LLMSummaryGenerator(llm),
                    config=doctor_meta,
                )

                policy = MockRLPolicy(p_jump=0.30)  # 30% 的 validate 轮前插 jump
                dialog_history = []
                round_idx = 0
                patient_response = "你好，我想继续上次的咨询。"

                while True:
                    round_idx += 1
                    print(f"\n==== 第{round_idx}轮 ====")

                    # 1) 按当前对话推进选题（可能也会触发到期评估）
                    dialog_history = doctor.get_history()
                    current_question = survey_agent.auto_advance(dialog_history)
                    if not current_question:
                        print(f"\n[问卷完成] 运行 {run_idx} 已完成。")
                        break
                    print(f"[问卷] 当前问题: {current_question}")

                    # 2) 医生提问、病人回答（环境一步）
                    doctor_ask = doctor.generate_dialog(
                        user_input=patient_response,
                        current_question=current_question,
                        question_context=survey_agent.get_question_context(current_question),
                    )
                    print(f"[医生] {doctor_ask}")
                    patient_response = patient.generate_dialog(user_input=doctor_ask)
                    print(f"[病人] {patient_response}")

                    # 3) RL 策略选择并执行动作（可能包含 jump+validate 或单步 complete）
                    actions = policy.choose(survey_agent, dialog_history)
                    action_names = "->".join([a for (a, _) in actions])
                    vop_name, jump_qid = "", ""
                    updates = []

                    for (act, param) in actions:
                        if act == "validate":
                            vop = param.get("op")
                            vop_name = str(vop)
                            print(f"[RL] 执行动作: validate ({vop_name})")
                            survey_agent.apply_action("validate", {"op": vop}, dialog_history, convo_agent=doctor)
                            # 让病人对“验证话术”回一嘴，形成闭环
                            if dialog_history and dialog_history[-1]["role"] == "assistant":
                                val_utter = dialog_history[-1]["content"]
                                patient_response = patient.generate_dialog(user_input=val_utter)
                                print(f"[病人·针对验证] {patient_response}")

                        elif act == "jump":
                            jump_qid = param.get("qid", "")
                            print(f"[RL] 执行动作: jump (qid={jump_qid})")
                            survey_agent.apply_action("jump", {"qid": jump_qid}, dialog_history, convo_agent=doctor)

                        elif act == "complete":
                            print(f"[RL] 执行动作: complete（强评估）")
                            updates = survey_agent.apply_action("complete", {}, dialog_history, convo_agent=doctor)
                            print(f"[RL] complete 返回 updates: {updates}")

                    # 4) 回合日志 & tick_round（保证“每轮+1；评估后清零”）
                    log_path = os.path.join(LOGS_DIR, f"{persona_id}_rl_run{run_idx}_step{round_idx}.json")
                    with open(log_path, "w", encoding="utf-8") as f_log:
                        json.dump({
                            "persona_id": persona_id,
                            "run": run_idx,
                            "step": round_idx,
                            "dialog_history": dialog_history,
                            "current_question": current_question,
                            "doctor_ask": doctor_ask,
                            "patient_reply": patient_response,
                            "actions": actions,
                            "updates": updates,
                            "progress": survey_agent.get_progress(),
                        }, f_log, ensure_ascii=False, indent=2)

                    # —— 环境计数 +1（惰性评估保底；complete 已清零计数）——
                    survey_agent.tick_round()

                    # 5) 写入 CSV（方便训练/可视化）
                    writer.writerow([
                        run_idx, round_idx, action_names, vop_name, jump_qid,
                        len(updates) if updates else 0, survey_agent.get_progress()
                    ])

                    # profiling（可选）
                    if profiler is not None:
                        profiler.dump_stats(profile_path)
                        print(f"[StepProfile] 已保存 {profile_path}。")

                # 轮末打印下本轮回答
                answers = survey_agent.get_all_answers()
                print(f"[统计] 回答汇总：{answers}")

        print(f"[保存] {csv_path} 完成。")

    print("\n[RL 自检实验全部完成]")

def sigint_handler(signum, frame):
    raise KeyboardInterrupt

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        os.makedirs("experiment", exist_ok=True)
        global profiler, profile_path
        profile_path = "experiment/experiment.prof"
        profiler = cProfile.Profile()
        print("[Profile] 采样中（每轮自动保存），Ctrl+C 结束。")
        signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            profiler.enable()
            main_experiment()
        except KeyboardInterrupt:
            print("\n[Profile] 收到 Ctrl+C，保存采样数据...")
        finally:
            profiler.disable()
            profiler.dump_stats(profile_path)
            print(f"[Profile] 已保存 {profile_path}（用 snakeviz 打开）。")
    else:
        main_experiment()
