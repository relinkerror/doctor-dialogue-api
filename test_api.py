# test_api.py
import requests

base_url = "http://127.0.0.1:8000"

# 1. 初始化会话
resp = requests.post(base_url + "/api/patient/init", json={"user_id": "test_user"})
session_id = resp.json()["session_id"]
print("Session:", session_id)

# 2. 第一轮对话
payload = {
    "session_id": session_id,
    "patient_reply": "我最近心情很低落"
}
resp = requests.post(base_url + "/api/patient/dialogue", json=payload)
print("[Round 1] Response:")
print(resp.json())

# 3. 第二轮对话（模拟患者继续回复）
payload = {
    "session_id": session_id,
    "patient_reply": "已经持续了好几周了，几乎每天都很难受"
}
resp = requests.post(base_url + "/api/patient/dialogue", json=payload)
print("\n[Round 2] Response:")
print(resp.json())
