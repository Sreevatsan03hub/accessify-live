# 🎙️ Accessify Live

<div align="center">
  <p>An AI-powered accessibility platform providing <strong>real-time live captions</strong>, multi-language translation, and video transcription for teachers and students — built specifically for Deaf & Hard-of-Hearing (DHH) users.</p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
  [![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## ✨ Key Features

- 🎤 **Live Real-time Captions:** Teacher speaks, students see captions instantly.
- 🌐 **Multi-language Translation:** Support for English, Hindi, Tamil, and Telugu.
- 🧠 **AI Enrichment:** Simplifies complex text, highlights keywords, and analyzes sentence tone.
- 📹 **Video Upload & Transcription:** Upload lectures and download generated VTT captions.
- 🔊 **Non-Speech Sound Detection:** Notifies users of ambient sounds (e.g., clapping, alarms).
- 🛡️ **Content Moderation:** Built-in profanity filtering and redaction for safe classrooms.
- ⏱️ **Screen Time Controls:** Tools to manage student screen time exposure.
- 🎥 **Peer-to-Peer Video:** Ultra-low latency WebRTC video streaming via PeerJS.
- 🌙 **Accessible UI:** Clean, responsive interface with Dark & Light modes.

---

## 🏗️ System Architecture

```text
accessify-live/
├── backend/          # FastAPI Python backend
│   ├── app.py        # Main API & WebSocket application
│   ├── routes/       # API endpoints definitions
│   └── services/     # AI/ML modules (Whisper STT, translation, etc.)
├── frontend/         # React SPA frontend
│   └── src/
│       ├── pages/    # TeacherRoom, StudentRoom, Dashboard
│       ├── components/ # Reusable UI components
│       ├── hooks/    # Custom React hooks
│       └── context/  # Global state management
└── peerserver.js     # Local PeerJS WebRTC signaling server
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your local machine:
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/en/download/)
- [FFmpeg](https://ffmpeg.org/download.html) (Required for audio/video processing)

### 1. Start the Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```
*Backend runs at: **http://127.0.0.1:8001***  
*Swagger API Docs: **http://127.0.0.1:8001/docs***

### 2. Start the Frontend

Open a new terminal window:

```bash
cd frontend

# Install Node modules
npm install

# Start the development server
npm run dev
```
*Frontend runs at: **http://localhost:3000***

### 3. Start the PeerJS Signaling Server (Video)

Open a third terminal window:

```bash
# From the project root directory
npm install
node peerserver.js
```
*PeerJS server runs at: **http://localhost:9000/peerjs***

---

## 🧠 AI Capabilities

| Component | Technology / Description |
|-----------|-------------|
| **Speech-to-Text (STT)** | OpenAI Whisper `small` model for accurate transcription |
| **Translation Engine** | Real-time translation (English → Hindi, Tamil, Telugu) |
| **Text Simplification** | NLP to simplify complex sentences for DHH comprehension |
| **Keyword Detection** | Identifies and highlights critical vocabulary dynamically |
| **Tone Analysis** | Classifies sentence intent (question, statement, exclamation) |
| **Acoustic Detection** | Identifies non-speech environmental sounds in the audio stream |
| **Safety Engine** | Proactive profanity filtering and content moderation |

---

## � API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/rooms/create` | Create a new virtual classroom |
| `POST` | `/api/v1/rooms/{code}/join` | Authenticate and join a room |
| `WS`   | `/ws/room/{code}/teacher`| WebSocket for teacher audio stream |
| `WS`   | `/ws/room/{code}/student/{id}` | WebSocket to receive live captions |
| `POST` | `/api/v1/video/upload` | Upload MP4/WebM for async transcription |
| `GET`  | `/api/v1/sessions/` | Retrieve past session history |

---

## ⚙️ Environment Configuration

Create a `.env` file in the `backend/` directory to configure your environment variables:

```env
# backend/.env
WHISPER_MODEL_SIZE=small
HOST=127.0.0.1
PORT=8001
# Add API Keys for external services (if applicable) here
```

---

## 🧪 Testing

To ensure everything is working correctly, you can run the built-in tests:

```bash
# 1. Test the WebSocket connection via browser
http://127.0.0.1:8001/test_client.html

# 2. Run Python Unit Tests
cd backend
python -m pytest tests/
```

---

## 📄 License

This project is licensed under the **MIT License**.  
*Proudly built by **Sreevatsan** to make education accessible for everyone.*
