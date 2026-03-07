"""Verify Feature 10: Caption Export/Download API"""
import requests

BASE = "http://127.0.0.1:8001/api/v1"

# 1. First create and populate a session
print("=== CREATING TEST SESSION ===")
r = requests.post(f"{BASE}/sessions/create", json={
    "session_type": "live", "title": "AI Lecture Demo", "language": "en"
})
session_id = r.json()["session_id"]
print(f"Session: {session_id}")

# Add captions
captions = [
    {"text": "Welcome to today's lecture on artificial intelligence.", "keywords": [{"keyword": "artificial intelligence", "emoji": "🔑"}], "tone": {"emotion": "neutral", "intent": "statement", "emoji": "😐"}},
    {"text": "Machine learning is very important for the exam.", "keywords": [{"keyword": "exam", "emoji": "📘"}, {"keyword": "machine learning", "emoji": "🔑"}], "tone": {"emotion": "neutral", "intent": "urgent", "emoji": "⚠️"}},
    {"text": "Don't forget to submit your assignment by tomorrow.", "keywords": [{"keyword": "submit", "emoji": "📤"}, {"keyword": "tomorrow", "emoji": "📆"}]},
]
for cap in captions:
    requests.post(f"{BASE}/sessions/{session_id}/caption", json=cap)

# End session
requests.post(f"{BASE}/sessions/{session_id}/end")
print("Session ended and saved.\n")

# 2. Test each export format
print("=== SRT EXPORT ===")
r = requests.get(f"{BASE}/export/{session_id}/srt")
print(r.text[:300])

print("\n=== VTT EXPORT ===")
r = requests.get(f"{BASE}/export/{session_id}/vtt")
print(r.text[:300])

print("\n=== TXT EXPORT ===")
r = requests.get(f"{BASE}/export/{session_id}/txt")
print(r.text[:500])

print("\n=== SUMMARY EXPORT ===")
r = requests.get(f"{BASE}/export/{session_id}/summary")
print(r.text[:500])

print("\n=== ALL FORMATS (PREVIEW) ===")
r = requests.get(f"{BASE}/export/{session_id}/all")
data = r.json()
print(f"Download links:")
for fmt, link in data["download_links"].items():
    print(f"  {fmt}: {link}")

print("\n✅ Feature 10 Verification Complete!")
