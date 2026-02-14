import asyncio
import websockets
import json
import base64
import numpy as np

async def test_stream():
    uri = "ws://127.0.0.1:8001/api/v1/audio/transcribe/stream?language=en&sample_rate=16000"
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Generate 2 seconds of silence/tone
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create a 440Hz sine wave
            audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            
            # Convert to int16 PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            chunk_size = 4096
            
            print("Sending audio chunks...")
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                audio_b64 = base64.b64encode(chunk).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "audio_chunk",
                    "data": audio_b64
                }))
                await asyncio.sleep(0.01)
                
            print("Sending end message...")
            await websocket.send(json.dumps({"type": "end"}))
            
            print("Waiting for response...")
            response = await websocket.recv()
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
