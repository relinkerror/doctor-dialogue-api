import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"

def api_init():
    resp = requests.post(f"{API_BASE}/api/patient/init", json={"user_id": "gradio_user"})
    data = resp.json()
    return data["session_id"], data.get("display_history", [])

def api_dialogue(session_id, user_msg):
    payload = {"session_id": session_id, "patient_reply": user_msg}
    resp = requests.post(f"{API_BASE}/api/patient/dialogue", json=payload)
    return resp.json()

def gradio_reset():
    session_id, display_history = api_init()
    return display_history, [session_id, display_history]

def gradio_chat(user_input, state):
    session_id, chat_history = state if state else (None, [])
    if not session_id:
        session_id, chat_history = api_init()
    data = api_dialogue(session_id, user_input)
    display_history = data.get("display_history", [])
    return display_history, [session_id, display_history]

with gr.Blocks() as demo:
    gr.Markdown("# AI Doctor – Mental Health Screening Demo\n与AI医生对话，系统将自动引导你完成心理健康筛查问卷。")
    chatbot = gr.Chatbot(label="对话记录", type="messages")
    user_input = gr.Textbox(label="请输入您的症状或回复", placeholder="如：最近总是失眠……", lines=2)
    state = gr.State([None, []])

    demo.load(gradio_reset, [], [chatbot, state])
    send_btn = gr.Button("发送")
    send_btn.click(gradio_chat, [user_input, state], [chatbot, state])
    user_input.submit(gradio_chat, [user_input, state], [chatbot, state])
    gr.Button("重新开始").click(gradio_reset, [], [chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860)



# http://127.0.0.1:7860
