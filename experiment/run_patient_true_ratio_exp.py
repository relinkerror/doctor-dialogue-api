# agents/survey/run_patient_true_ratio_exp.py
import cProfile
import sys
import os
import threading
import time
import signal

def main_experiment():
    import os
    import json
    import csv
    import time
    from utils.LLMClient import LLMClient
    from agents.survey.survey_agent import SurveyAgent
    from agents.dialogue.dialogue_patient_agent import DialoguePatientAgent
    from agents.dialogue.llm_summary_generator import LLMSummaryGenerator
    from agents.dialogue.dialogue_doctor_agent import DialogueDoctorAgent

    CONFIG_DIR = "config"
    QUESTIONNAIRE_DIR = "questionnaire"
    TABLES_DIR = "tables"
    LOGS_DIR = "experiment/logs"
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(os.path.join(CONFIG_DIR, "patient_agent.json"), encoding="utf-8") as f:
        patient_meta = json.load(f)
    persona_templates = patient_meta["persona_templates"]
    persona_ids = [p["id"] for p in persona_templates]

    with open(os.path.join(CONFIG_DIR, "doctor_agent.json"), encoding="utf-8") as f:
        doctor_meta = json.load(f)

    llm = LLMClient.from_config(os.path.join(CONFIG_DIR, "llm_config.json"))
    with open(os.path.join(CONFIG_DIR, "survey_uncertainty_eval_prompt.txt"), encoding="utf-8") as f:
        eval_prompt = f.read()

    meta_json_path = os.path.join(CONFIG_DIR, "dsm5_depression_surveys.json")

    class DummySummaryGenerator:
        def summarize(self, raw_history, existing_summary, interval):
            return ""

    N = 10  # 每个病人跑10次

    global profiler, profile_path
    profiler = globals().get("profiler", None)
    profile_path = globals().get("profile_path", "experiment/experiment.prof")

    for persona_id in persona_ids:
        print(f"\n==== 开始实验: {persona_id} ====")
        stats = {}

        for run_idx in range(1, N + 1):
            print(f"\n--- persona_id={persona_id}, 第{run_idx}轮 ---")
            survey_agent = SurveyAgent(
                meta_json_path, QUESTIONNAIRE_DIR, llm,
                eval_prompt=eval_prompt,
                threshold=0.2,
                lazy_eval=True,       # 惰性评估开关（如需每轮评估可改为 False 或 eval_every_n=1）
                eval_every_n=5
            )
            patient = DialoguePatientAgent(
                client=llm,
                name=f"{persona_id}_run{run_idx}",
                summary_generator=LLMSummaryGenerator(llm),
                config=patient_meta,
                persona_id=persona_id
            )
            dialog_history = []
            round_idx = 0

            doctor = DialogueDoctorAgent(
                client=llm,
                name="demo-doctor",
                summary_generator=LLMSummaryGenerator(llm),
                config=doctor_meta,
            )

            patient_response = "你好，我最近睡眠不好，想咨询一下。"
            last_doctor_ask = None

            while True:
                round_idx += 1
                print(f"\n==== 第{round_idx}轮对话 ====")
                dialog_history = doctor.get_history()
                current_question = survey_agent.auto_advance(dialog_history)
                if not current_question:
                    print(f"\n[问卷完成] 运行 {run_idx} 已完成。")
                    break
                print(f"\n[问卷] 当前问题: {current_question}")

                doctor_response = doctor.generate_dialog(
                    user_input=patient_response,
                    current_question=current_question,
                    question_context=survey_agent.get_question_context(current_question),
                )
                last_doctor_ask = doctor_response
                print(f"[医生] 回复: {doctor_response}")

                patient_response = patient.generate_dialog(
                    user_input=doctor_response,
                )
                print(f"[病人] 回复: {patient_response}")

                log_path = os.path.join(LOGS_DIR, f"{persona_id}_run{run_idx}_step{round_idx}.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "persona_id": persona_id,
                        "run": run_idx,
                        "step": round_idx,
                        "dialog_history": dialog_history,
                        "current_question": current_question,
                        "doctor_ask": doctor_response,
                        "patient_reply": patient_response,
                        "progress": survey_agent.get_progress()
                    }, f, ensure_ascii=False, indent=2)
                time.sleep(0.5)

                # —— 每轮对话结束：环境计数+1（惰性评估保底）——
                survey_agent.tick_round()

                if profiler is not None:
                    profiler.dump_stats(profile_path)
                    print(f"[StepProfile] 已保存 {profile_path}。")

            print(f"\n[统计] 运行 {run_idx} 完成，开始统计问卷答案...")
            answers = survey_agent.get_all_answers()
            print(f"[统计] 回答大致为：{answers}")
            for survey_id, items in answers.items():
                stats.setdefault(survey_id, {})
                for item in items:
                    qid = item["id"]
                    if qid not in stats[survey_id]:
                        stats[survey_id][qid] = {"true": 0, "count": 0, "question": item["question"]}
                    if item.get("status") is True:
                        stats[survey_id][qid]["true"] += 1
                    stats[survey_id][qid]["count"] += 1

        csv_path = os.path.join(TABLES_DIR, f"ratio_{persona_id}.csv")
        with open(csv_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["persona_id", "survey_id", "qid", "question", "true_count", "total", "true_ratio"])
            for survey_id, qs in stats.items():
                for qid, s in qs.items():
                    ratio = s["true"] / s["count"] if s["count"] > 0 else 0
                    writer.writerow([persona_id, survey_id, qid, s["question"], s["true"], s["count"], f"{ratio:.4f}"])
        print(f"[保存] {csv_path} 完成。")

    print("\n[所有实验完成]")

def sigint_handler(signum, frame):
    raise KeyboardInterrupt

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        os.makedirs("experiment", exist_ok=True)
        global profiler, profile_path
        profile_path = "experiment/experiment.prof"
        profiler = cProfile.Profile()
        print("[Profile] 正在采样（每轮自动保存一次），可随时 Ctrl+C 或 kill 停止。")
        signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            profiler.enable()
            main_experiment()
        except KeyboardInterrupt:
            print("\n[Profile] 收到 Ctrl+C，正在保存采样数据...")
        finally:
            profiler.disable()
            profiler.dump_stats(profile_path)
            print(f"[Profile] 已保存为 {profile_path}，可用 snakeviz {profile_path} 查看火焰图。")
    else:
        main_experiment()
