"""Verify Feature 9: Session Storage API"""
import requests
import json

BASE = "http://127.0.0.1:8001/api/v1/sessions"

# 1. Create a session
print("=== CREATE SESSION ===")
r = requests.post(f"{BASE}/create", json={"session_type": "live", "title": "Test Lecture", "language": "en"})
print(r.json())
session_id = r.json()["session_id"]

# 2. Add captions
print("\n=== ADD CAPTIONS ===")
captions = [
    {"text": "Hello students, welcome to the class.", "language": "en"},
    {"text": "Today we will learn about machine learning.", "language": "en", "keywords": [{"keyword": "machine learning", "emoji": "🔑"}]},
    {"text": "This is very important for your exam.", "language": "en", "tone": {"emotion": "neutral", "intent": "statement", "emoji": "😐"}},
]
for cap in captions:
    r = requests.post(f"{BASE}/{session_id}/caption", json=cap)
    print(f"  Added: {r.json()}")

# 3. Get session (while active)
print("\n=== GET SESSION (ACTIVE) ===")
r = requests.get(f"{BASE}/{session_id}")
data = r.json()
print(f"  Session: {data['session_id']}, Captions: {data['caption_count']}, Active: {data['is_active']}")

# 4. End session
print("\n=== END SESSION ===")
r = requests.post(f"{BASE}/{session_id}/end")
print(r.json())

# 5. List all sessions
print("\n=== LIST SESSIONS ===")
r = requests.get(f"{BASE}/")
sessions = r.json()
print(f"  Total: {sessions['total']}")
for s in sessions["sessions"]:
    print(f"    - {s['session_id']}: {s['title']} ({s['caption_count']} captions)")

# 6. Get session (from disk)
print("\n=== GET SESSION (FROM DISK) ===")
r = requests.get(f"{BASE}/{session_id}")
data = r.json()
print(f"  Session: {data['session_id']}, Active: {data['is_active']}, Captions: {data['caption_count']}")

print("\n✅ Feature 9 Verification Complete!")
