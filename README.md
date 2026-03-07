# Accessify Live 🎙️

An AI-powered accessibility platform that provides **real-time live captions**, multi-language translation, and video transcription for teachers and students — built for Deaf & Hard-of-Hearing (DHH) users.

---

## ✨ Features

- 🎤 **Live captions** during class — teacher speaks, students see captions in real-time
- 🌐 **Multi-language translation** (English, Hindi, Tamil, Telugu)
- 🧠 **AI enrichment** — simplified text, keyword extraction, tone analysis
- 📹 **Video upload & transcription** with downloadable VTT captions
- 🔊 **Non-speech sound detection** (e.g., clapping, alarms)
- 🛡️ **Content moderation** & profanity filtering
- ⏱️ **Screen time controls** for students
- 🎥 **Peer-to-peer video** via PeerJS (teacher ↔ students)
- 🌙 **Dark/Light mode** with accessible UI

---

## 🏗️ Project Structure

```
accessify-live/
├── backend/          # FastAPI Python backend
│   ├── app.py        # Main application entry point
│   ├── routes/       # API & WebSocket endpoints
│   └── services/     # AI/ML services (STT, translation, etc.)
├── frontend/         # React frontend
│   └── src/
│       ├── pages/    # TeacherRoom, StudentRoom, Dashboard, etc.
│       ├── components/
│       ├── hooks/
│       └── context/
└── peerserver.js     # Local PeerJS signaling server
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [FFmpeg](https://ffmpeg.org/) (for audio/video processing)

---

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

Backend runs at: **http://127.0.0.1:8001**

API docs at: **http://127.0.0.1:8001/docs**

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

### 3. PeerJS Signaling Server (for video)

```bash
# From the root directory
npm install
node peerserver.js
```

PeerJS server runs at: **http://localhost:9000/peerjs**

---

## 🔌 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/rooms/create` | Create a new classroom room |
| POST | `/api/v1/rooms/{code}/join` | Join a room as teacher/student |
| WS | `/ws/room/{code}/teacher` | Teacher audio stream (live captions) |
| WS | `/ws/room/{code}/student/{id}` | Student caption receiver |
| POST | `/api/v1/video/upload` | Upload video for transcription |
| GET | `/api/v1/sessions/` | Session history |

---

## 🧠 AI Services

| Service | Description |
|---------|-------------|
| **Whisper STT** | Speech-to-text transcription (OpenAI Whisper `small` model) |
| **Translation** | Real-time translation (en → hi, ta, te) |
| **Text Simplification** | Simplifies complex sentences for DHH users |
| **Keyword Detection** | Highlights important words in captions |
| **Tone Analysis** | Detects sentence tone (question, statement, etc.) |
| **Sound Detection** | Identifies non-speech sounds in audio |
| **Content Moderation** | Profanity filtering & redaction |

---

## 🖥️ Live Classroom Flow

1. **Teacher** creates a room → shares the room code
2. **Students** join using the code → select preferred language
3. Teacher speaks → audio streams via WebSocket to backend
4. Backend: Whisper transcribes → AI enriches → broadcasts captions
5. Students receive captions in real-time in their chosen language
6. Video is shared peer-to-peer via PeerJS

---

## ⚙️ Environment

Create a `.env` file in the `backend/` directory:

```env
# Example — update as needed
WHISPER_MODEL_SIZE=small
HOST=127.0.0.1
PORT=8001
```

---

## 📋 Requirements Summary

**Backend (Python):**
- `fastapi`, `uvicorn`, `websockets`
- `openai-whisper`, `librosa`, `soundfile`, `numpy`
- `deep-translator`, `noisereduce`

**Frontend (Node):**
- React, React Router, TailwindCSS
- PeerJS (WebRTC)

---

## 🧪 Testing

```bash
# Backend — open the test client in browser
http://127.0.0.1:8001/test_client.html

# Run backend unit tests
cd backend
python -m pytest tests/
```

---

## 📄 License

MIT License — built by Sreevatsan for accessibility.
