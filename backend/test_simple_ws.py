import asyncio
import websockets
import json
import requests

async def test_simple():
    # 1. Create room
    res = requests.post("http://127.0.0.1:8001/api/v1/rooms/create", json={"title": "Test", "teacher_name": "T"})
    room_code = res.json()["room_code"]
    print(f"Room: {room_code}")

    # 2. Connect
    url = f"ws://127.0.0.1:8001/ws/room/{room_code}/teacher"
    async with websockets.connect(url) as ws:
        print("Connected")
        # Receive welcome
        msg = await ws.recv()
        print(f"Welcome: {msg}")

        # Send test message
        print("Sending ping...")
        await ws.send(json.dumps({"type": "ping"}))
        
        # Receive pong
        msg = await ws.recv()
        print(f"Pong: {msg}")

        # Send audio chunk (empty)
        print("Sending audio_chunk...")
        await ws.send(json.dumps({"type": "audio_chunk", "data": "", "sample_rate": 16000}))
        
        await asyncio.sleep(1)
        print("Done")

if __name__ == "__main__":
    asyncio.run(test_simple())
