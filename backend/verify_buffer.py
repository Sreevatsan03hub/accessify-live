"""Verify Feature 11: Low-Internet Resilience API"""
import requests

BASE = "http://127.0.0.1:8001/api/v1/buffer"

# 1. Register a client
print("=== REGISTER CLIENT ===")
r = requests.post(f"{BASE}/register", json={"session_id": "test-session"})
data = r.json()
print(data)
client_id = data["client_id"]

# 2. Check status
print("\n=== CLIENT STATUS ===")
r = requests.get(f"{BASE}/status/{client_id}")
print(r.json())

# 3. Simulate reconnection
print("\n=== RECONNECT ===")
r = requests.post(f"{BASE}/reconnect", json={"client_id": client_id})
print(r.json())

# 4. List all clients
print("\n=== LIST CLIENTS ===")
r = requests.get(f"{BASE}/clients")
data = r.json()
print(f"Total clients: {data['total']}")
for c in data["clients"]:
    print(f"  - {c['client_id']}: connected={c['is_connected']}, reconnects={c['reconnect_count']}")

# 5. Cleanup
print("\n=== CLEANUP ===")
r = requests.delete(f"{BASE}/{client_id}")
print(r.json())

print("\n✅ Feature 11 Verification Complete!")
print("Note: Batch processing (/process-batch) requires actual audio data,")
print("which will be tested via the frontend's offline buffering system.")
