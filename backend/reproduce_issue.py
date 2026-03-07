import asyncio
import websockets
import json
import base64
import numpy as np
import requests
import time
import sys

API_URL = "http://127.0.0.1:8001"
WS_URL = "ws://127.0.0.1:8001"

async def test_live_captions():
    print("1. Creating a room...")
    # Generate random room code logic
    room_code = f"TEST-{int(time.time())}"
    
    # Try creating via API
    try:
        response = requests.post(f"{API_URL}/api/v1/rooms/create", json={
            "title": "Test Room",
            "teacher_name": "Test Teacher"
        })
        if response.status_code == 200:
            room_data = response.json()
            room_code = room_data["room_code"]
            print(f"   Room created: {room_code}")
        else:
            print(f"   Failed to create room (Status {response.status_code}): {response.text}")
            return
    except Exception as e:
        print(f"   Failed to connect to API: {e}")
        return

    sample_rate = 16000
    # Read real audio file
    try:
        with open("test_audio.wav", "rb") as f:
            # Skip WAV header (44 bytes) for raw data simulation, or just read all
            # Assuming simple WAV 16kHz mono
            f.seek(44)
            raw_audio = f.read()
            # Convert to float32
            audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    except FileNotFoundError:
        print("   test_audio.wav not found, generating fallback audio")
        # Fallback to noise
        duration = 5
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = np.random.normal(0, 0.1, t.shape).astype(np.float32)

    chunk_size = 4096 # samples (approx 0.25s)
    
    ws_endpoint = f"{WS_URL}/ws/room/{room_code}/teacher"
    print(f"2. Connecting to WebSocket: {ws_endpoint}")

    async with websockets.connect(ws_endpoint) as ws:
        print("   Connected!")
        
        # Initial handshake
        try:
             msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
             print(f"   Received: {msg}")
        except asyncio.TimeoutError:
             print("   Timeout waiting for welcome message")

        # Split into chunks of 4096 samples
        chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
        
        print(f"3. Sending {len(chunks)} audio chunks...")
        
        caption_count = 0

        async def listen():
            nonlocal caption_count
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    print(f"   [RX] Type: {data.get('type')}")
                    if data.get('type') == 'caption_sent':
                        print(f"   [CAPTION] {data.get('text')}")
                        caption_count += 1
                    elif data.get('type') == 'error':
                        print(f"   [ERROR] {data.get('message')}")
            except websockets.exceptions.ConnectionClosed:
                print("   Connection closed.")
            except Exception as e:
                print(f"   Listen error: {e}")

        listener = asyncio.create_task(listen())

        for i, chunk in enumerate(chunks):
            # Encode chunk
            # float32 bytes
            chunk_bytes = chunk.tobytes()
            b64_data = base64.b64encode(chunk_bytes).decode('utf-8')
            
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "data": b64_data,
                "sample_rate": sample_rate
            }))
            
            # Sleep to simulate real-time
            await asyncio.sleep(chunk_size / sample_rate)
            
            if i % 5 == 0:
                print(f"   Sent chunk {i+1}/{len(chunks)}")

        # Send 1 second of silence to trigger VAD flush
        silence = np.zeros(16000, dtype=np.float32)
        silence_chunks = [silence[i:i + chunk_size] for i in range(0, len(silence), chunk_size)]
        print("   Sending silence to trigger flush...")
        for chunk in silence_chunks:
            chunk_bytes = chunk.tobytes()
            b64_data = base64.b64encode(chunk_bytes).decode('utf-8')
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "data": b64_data,
                "sample_rate": sample_rate
            }))
            await asyncio.sleep(chunk_size / sample_rate)

        print("   Finished sending audio. Waiting for processing...")
        await asyncio.sleep(5)
        listener.cancel()
        
        if caption_count > 0:
            print("SUCCESS: Captions received.")
        else:
            print("FAILURE: No captions received.")

if __name__ == "__main__":
    asyncio.run(test_live_captions())
