# 🎙️ Accessify Live

<div align="center">
  <p>An AI-powered accessibility platform providing <strong>real-time live captions</strong>, multi-language translation, and video transcription for teachers and students — built specifically for Deaf &amp; Hard-of-Hearing (DHH) users.</p>

  [![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-accessify--live.vercel.app-brightgreen)](https://accessify-live.vercel.app)
  [![Backend](https://img.shields.io/badge/⚙️_Backend-Render-blue)](https://accessify-backend-4uq4.onrender.com)
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🌐 Live Deployment

| Service | URL |
|---------|-----|
| **Frontend** (Vercel) | [https://accessify-live.vercel.app](https://accessify-live.vercel.app) |
| **Backend API** (Render) | [https://accessify-backend-4uq4.onrender.com](https://accessify-backend-4uq4.onrender.com) |
| **API Docs** | [https://accessify-backend-4uq4.onrender.com/docs](https://accessify-backend-4uq4.onrender.com/docs) |

> ⚠️ The backend runs on Render's free tier — first request after inactivity may take ~30 seconds to wake up.

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
├── backend/          # FastAPI Python backend (deployed on Render)
│   ├── app.py        # Main API & WebSocket application
│   ├── routes/       # API endpoints definitions
│   └── services/     # AI/ML modules (Whisper STT, translation, etc.)
├── frontend/         # React SPA frontend (deployed on Vercel)
│   └── src/
│       ├── pages/    # TeacherRoom, StudentRoom, Dashboard
│       ├── components/ # Reusable UI components
│       ├── hooks/    # Custom React hooks
│       └── context/  # Global state management
└── peerserver.js     # PeerJS WebRTC signaling server
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/en/download/)
- [FFmpeg](https://ffmpeg.org/download.html) (for audio/video processing)

### 1. Start the Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```
*Backend API: **http://127.0.0.1:8001***  
*Swagger Docs: **http://127.0.0.1:8001/docs***

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```
*Frontend: **http://localhost:3000***

---

## 🧠 AI Capabilities

| Component | Technology |
|-----------|------------|
| **Speech-to-Text** | OpenAI Whisper `small` model |
| **Translation** | Real-time (English → Hindi, Tamil, Telugu) |
| **Text Simplification** | NLP for DHH-friendly comprehension |
| **Keyword Detection** | Highlights critical vocabulary dynamically |
| **Tone Analysis** | Classifies sentence intent (question/statement/exclamation) |
| **Acoustic Detection** | Identifies non-speech environmental sounds |
| **Safety Engine** | Profanity filtering and content moderation |

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/rooms/create` | Create a new virtual classroom |
| `POST` | `/api/v1/rooms/{code}/join` | Join a room |
| `WS`   | `/ws/room/{code}/teacher` | Teacher audio stream |
| `WS`   | `/ws/room/{code}/student/{id}` | Receive live captions |
| `POST` | `/api/v1/video/upload` | Upload MP4/WebM for transcription |
| `GET`  | `/api/v1/sessions/` | Retrieve session history |

---

## ⚙️ Environment Variables

### Backend (`backend/.env`)
```env
FIREBASE_SERVICE_ACCOUNT_JSON=<your-firebase-admin-sdk-json>
FIREBASE_STORAGE_BUCKET=accessify-live.firebasestorage.app
```

### Frontend (`frontend/.env.local`)
```env
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=...
VITE_FIREBASE_STORAGE_BUCKET=...
VITE_FIREBASE_MESSAGING_SENDER_ID=...
VITE_FIREBASE_APP_ID=...
VITE_API_URL=https://accessify-backend-4uq4.onrender.com
VITE_WS_URL=wss://accessify-backend-4uq4.onrender.com
```

---

## 📄 License

This project is licensed under the **MIT License**.
