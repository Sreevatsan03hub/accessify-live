# test_microphone_stream.py
import websockets
import asyncio
import json

async def stream_microphone():
    async with websockets.connect("ws://127.0.0.1:8001/api/v1/audio/stream/live") as ws:
        while True:
            chunk = await ws.recv()
            data = json.loads(chunk)
            if data.get("type") == "audio_chunk":
                print(f"Received chunk: {len(data['data'])} bytes")

asyncio.get_event_loop().run_until_complete(stream_microphone())