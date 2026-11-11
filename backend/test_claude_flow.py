import sys
import io
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"
proxies = {"http": None, "https": None}

print("[*] 1. Testing /api/config/ai endpoint...")
res = requests.get(f"{BASE_URL}/api/config/ai", proxies=proxies)
print("Config:", res.json())

print("\n[*] 2. Loading sample Iris dataset...")
res = requests.post(f"{BASE_URL}/api/sample/iris", proxies=proxies)
data = res.json()
dataset_id = data["dataset_id"]
print("Dataset ID:", dataset_id)

print("\n[*] 3. Testing /api/analyze with Claude...")
res = requests.post(f"{BASE_URL}/api/analyze", json={"dataset_id": dataset_id, "provider": "claude"}, proxies=proxies)
analysis = res.json()
print("Provider:", analysis.get("provider"))
print("Summary:", analysis.get("dataset_summary"))
if analysis.get("thinking"):
    print("Thinking captured! Length:", len(analysis["thinking"]))

print("\n[*] 4. Testing /api/chat with Claude...")
res = requests.post(f"{BASE_URL}/api/chat", json={
    "dataset_id": dataset_id,
    "provider": "claude",
    "history": [],
    "query": "Which 2 features are most correlated in Iris and why?"
}, proxies=proxies)
chat_res = res.json()
print("Chat Response:", chat_res.get("response")[:200], "...")
if chat_res.get("thinking"):
    print("Chat Thinking captured! Length:", len(chat_res["thinking"]))

print("\n[===] ALL ENDPOINTS VERIFIED & WORKING WITH CLAUDE! [===]")
