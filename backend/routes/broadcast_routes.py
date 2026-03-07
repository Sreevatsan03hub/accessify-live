"""
Broadcast Routes — Live Caption Broadcasting API (Feature 10)
REST endpoints for room management + WebSocket endpoints for live streaming.

Architecture:
  Teacher → WebSocket /ws/room/{code}/teacher → AI pipeline → broadcast to all students
  Student → WebSocket /ws/room/{code}/student/{participant_id} → receives captions
"""
import time
import base64
import logging
import numpy as np
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from services.room_service import get_room_service
from services.speech_to_text import get_stt_service
from services.text_simplification_service import get_simplifier
from services.keyword_detection_service import get_keyword_detector
from services.tone_analysis_service import get_tone_service
from services.translation_service import get_translator
from services.audio_preprocessing import preprocess_for_transcription
from services.audio_service import is_silence
from services.sound_detection_service import get_sound_detector
from services.moderation_service import get_moderator
from services.screen_time_service import (
    load_safety_settings, start_session, end_session, check_screen_time
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["broadcast"])

SUPPORTED_LANGUAGES = ["en", "hi", "ta", "te"]


# ─── Request Models ────────────────────────────────────────────────────────────

class CreateRoomRequest(BaseModel):
    title: str
    teacher_name: str

class JoinRoomRequest(BaseModel):
    name: str
    role: str = "student"           # "teacher" or "student"
    language: str = "en"            # Preferred caption language


# ─── REST Endpoints ────────────────────────────────────────────────────────────

@router.post("/api/v1/rooms/create")
async def create_room(req: CreateRoomRequest):
    """Create a new classroom room. Returns a unique room code."""
    service = get_room_service()
    room = service.create_room(title=req.title, created_by=req.teacher_name)
    return {
        "room_code": room.room_code,
        "title": room.title,
        "created_at": room.created_at,
        "message": f"Share room code '{room.room_code}' with your students"
    }


@router.post("/api/v1/rooms/{room_code}/join")
async def join_room(room_code: str, req: JoinRoomRequest):
    """Join an existing room. Returns participant_id for WebSocket connection."""
    service = get_room_service()
    room = service.get_room(room_code)

    if not room:
        raise HTTPException(status_code=404, detail=f"Room '{room_code}' not found")
    if not room.is_active:
        raise HTTPException(status_code=400, detail="Room is no longer active")
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language must be one of: {SUPPORTED_LANGUAGES}")

    # Use service method so join is persisted to disk
    participant = service.add_participant(
        room_code=room_code,
        name=req.name,
        role=req.role,
        language=req.language
    )

    ws_path = (
        f"/ws/room/{room_code}/teacher"
        if req.role == "teacher"
        else f"/ws/room/{room_code}/student/{participant.participant_id}"
    )

    return {
        "participant_id": participant.participant_id,
        "room_code": room_code,
        "role": req.role,
        "language": req.language,
        "websocket_url": ws_path,
        "message": f"Joined room '{room_code}' as {req.role}"
    }


@router.post("/api/v1/rooms/{room_code}/leave/{participant_id}")
async def leave_room(room_code: str, participant_id: str):
    """Leave a room."""
    service = get_room_service()
    room = service.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    room.remove_participant(participant_id)
    return {"status": "left", "room_code": room_code}


@router.post("/api/v1/rooms/{room_code}/close")
async def close_room(room_code: str):
    """Close a room (teacher only action)."""
    service = get_room_service()
    room = service.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    service.close_room(room_code)
    return {"status": "closed", "room_code": room_code}


@router.get("/api/v1/rooms/")
async def list_rooms():
    """List all active rooms."""
    service = get_room_service()
    rooms = service.list_rooms()
    return {"rooms": rooms, "total": len(rooms)}


@router.get("/api/v1/rooms/{room_code}")
async def get_room(room_code: str):
    """Get room details and participant list."""
    service = get_room_service()
    room = service.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_dict()


@router.get("/api/v1/debug/translate/{room_code}")
async def debug_translate(room_code: str, text: str = "Hello how are you"):
    """Debug endpoint: shows room language state + tests translation."""
    service = get_room_service()
    room = service.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    students = room.get_connected_students()
    needed_langs = set(s.language for s in students if s.language != "en")
    all_participants = [{"name": p.name, "lang": p.language, "connected": pid in room.connections}
                        for pid, p in room.participants.items()]

    translations = {}
    for lang in needed_langs:
        try:
            from services.translation_service import get_translator
            t = get_translator().translate(text, lang)
            translations[lang] = t.translated_text
        except Exception as e:
            translations[lang] = f"ERROR: {e}"

    return {
        "room_code": room_code,
        "all_participants": all_participants,
        "connected_students": [{"name": s.name, "lang": s.language} for s in students],
        "needed_langs": list(needed_langs),
        "test_translations": translations,
    }


class TranslateRequest(BaseModel):
    text: str
    target_lang: str  # "hi", "ta", "te"
    source_lang: str = "en"


@router.post("/api/v1/translate")
async def translate_text(req: TranslateRequest):
    """
    On-demand translation endpoint.
    Uses Google Translate unofficial API directly via urllib (no extra deps).
    """
    if not req.text.strip() or req.target_lang == "en":
        return {"translated": req.text, "lang": req.target_lang}

    def _call():
        import urllib.request, urllib.parse, json as _json
        params = urllib.parse.urlencode({
            'client': 'gtx',
            'sl': 'en',
            'tl': req.target_lang,
            'dt': 't',
            'q': req.text,
        })
        url = f'https://translate.googleapis.com/translate_a/single?{params}'
        r = urllib.request.urlopen(url, timeout=6)
        data = _json.loads(r.read().decode('utf-8'))
        return ''.join(item[0] for item in data[0] if item[0])

    try:
        translated = await asyncio.to_thread(_call)
        logger.info(f"[/translate] {req.target_lang}: '{translated[:50]}'")
        return {"translated": translated, "lang": req.target_lang}
    except Exception as e:
        logger.warning(f"[/translate] Failed {req.target_lang}: {e}")
        return {"translated": req.text, "lang": req.target_lang}

# ─── Background caption processor ────────────────────────────────────────────
async def _process_caption(audio: np.ndarray, room, websocket) -> None:
    """
    Background task: STT → AI enrichment (parallel) → broadcast.
    Runs concurrently — the caller resets the audio buffer immediately and
    keeps accumulating new audio while this task is still running.
    """
    from starlette.websockets import WebSocketState
    try:
        # ── VAD: Skip processing if audio is just silence/noise ────────────────
        if is_silence(audio):
            return

        stt    = get_stt_service()
        result = await asyncio.to_thread(
            stt.transcribe_realtime_audio,
            audio_data=audio, sample_rate=16000, language="en"
        )

        text = result.text.strip()
        if not text:
            return

        # ── Hallucination Filter ──────────────────────────────────────────────
        # Reject common Whisper "ghost" phrases often generated from noise
        HALLUCINATIONS = {
            "Awesome.", "Thank you.", "Bye.", "Amara.", "Unara.", "You.", "MBC."
        }
        # Check specific phrases or repetitive loops
        if text in HALLUCINATIONS or (text.count("Awesome.") >= 1):
             logger.info(f"[Filter] Dropped hallucination: '{text}'")
             return

        logger.info(f"[Caption] {text[:100]}")

        # ── Moderation filter ─────────────────────────────────────────────────
        safety    = load_safety_settings(room.room_code)
        mod_result = None
        broadcast_text = result.text
        if safety.get("profanity_filter_enabled", True):
            mod_result  = get_moderator().moderate(result.text)
            if mod_result.flagged:
                action = safety.get("profanity_action", "redact")
                if action == "block":
                    logger.info(f"[Moderation] Caption blocked [{mod_result.category}]")
                    return   # silently drop this caption
                else:        # redact
                    broadcast_text = mod_result.redacted_text
                    logger.info(f"[Moderation] Caption redacted [{mod_result.category}]")

        # Run all enrichment steps in parallel (saves ~2s vs sequential)
        simplified, keywords, tone = await asyncio.gather(
            asyncio.to_thread(get_simplifier().simplify, broadcast_text),
            asyncio.to_thread(get_keyword_detector().extract_keywords, broadcast_text),
            asyncio.to_thread(get_tone_service().analyze_tone, broadcast_text),
        )

        # Per-student translations
        students     = room.get_connected_students()
        needed_langs = set(s.language for s in students if s.language != "en")
        translations = {}
        keyword_translations = {}
        logger.info(f"[Caption] Students: {[f'{s.name}={s.language}' for s in students]} | needed_langs={needed_langs}")

        if needed_langs:
            async def _do_translate(text: str, lang: str) -> str:
                """Translate a single text to target language, returns translated string."""
                def _call():
                    from services.translation_service import get_translator
                    return get_translator().translate(text, lang).translated_text
                try:
                    result_text = await asyncio.to_thread(_call)
                    logger.info(f"[Translation] {lang}: '{result_text[:60]}'")
                    return result_text
                except Exception as ex:
                    logger.warning(f"[Translation] {lang} FAILED: {ex}")
                    return text  # fallback to original

            # Translate message text for all needed languages in parallel
            lang_list = list(needed_langs)
            translated_results = await asyncio.gather(
                *[_do_translate(result.text, lang) for lang in lang_list]
            )
            translations = dict(zip(lang_list, translated_results))

            if keywords:
                def _translate_kws(kw_list, langs):
                    from services.translation_service import get_translator
                    t, out = get_translator(), {}
                    for lang in langs:
                        kws = []
                        for kw in kw_list:
                            try:
                                w  = kw.get("original", kw.get("keyword", ""))
                                tr = t.translate(w, lang).translated_text
                                kws.append({**kw, "keyword": tr, "original": tr})
                            except Exception as ex:
                                logger.warning(f"Keyword translation {lang} failed: {ex}")
                                kws.append(kw)
                        out[lang] = kws
                    return out

                keyword_translations = await asyncio.to_thread(
                    _translate_kws, keywords, lang_list
                )

        caption_data = {
            "text":                 broadcast_text,
            "simplified_text":      simplified,
            "keywords":             keywords,
            "keyword_translations": keyword_translations,
            "tone":                 tone,
            "translations":         translations,
            "timestamp":            time.time(),
            "sound_event":          None,
            "moderation": {
                "flagged":   mod_result.flagged   if mod_result else False,
                "category":  mod_result.category  if mod_result else None,
            } if mod_result else None,
        }

        await room.broadcast_caption(caption_data)

        # Echo to teacher (best-effort)
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.send_json({
                    "type":             "caption_sent",
                    "text":             result.text,
                    "simplified_text":  simplified,
                    "keywords":         keywords,
                    "tone":             tone,
                    "students_reached": len(room.get_connected_students()),
                })
        except Exception:
            pass

    except Exception as e:
        logger.error(f"[_process_caption] error: {e}", exc_info=True)


# ─── WebSocket: Teacher Audio Stream ──────────────────────────────────────────

@router.websocket("/ws/room/{room_code}/teacher")

async def teacher_audio_stream(websocket: WebSocket, room_code: str):
    """
    Teacher WebSocket endpoint.
    Teacher sends audio chunks → AI pipeline → captions broadcast to all students.
    """
    await websocket.accept()
    service = get_room_service()
    room = service.get_room(room_code)

    if not room or not room.is_active:
        await websocket.send_json({"type": "error", "message": "Room not found or inactive"})
        await websocket.close()
        return

    # Get teacher participant
    teacher = room.get_teacher()
    if teacher:
        room.register_websocket(teacher.participant_id, websocket)

    await websocket.send_json({
        "type": "connected",
        "room_code": room_code,
        "message": "Teacher connected. Start speaking to broadcast captions."
    })

    logger.info(f"Teacher connected to room {room_code}")

    # Audio buffering for VAD (STT pipeline)
    audio_buffer = []
    buffer_duration  = 0.0
    silence_duration = 0.0
    SILENCE_THRESHOLD  = 0.5    # seconds of silence before flushing a caption segment
    MAX_BUFFER_SECONDS = 4.0    # hard cap — no more than 4s per caption
    MIN_AUDIO_SECONDS  = 0.5    # ignore very short fragments
    SILENCE_DB         = -50.0  # Threshold for "silence" (speech ends)

    # Separate rolling window for real-time non-speech sound detection
    sound_buf: list = []
    sound_buf_dur: float = 0.0
    SOUND_WINDOW_SECS: float = 2.0

    try:
        while True:
            logger.warning(f"[WS-Trace] Waiting for raw message in room {room_code}")
            raw_msg = await websocket.receive()
            logger.warning(f"[WS-Trace] Raw message type: {raw_msg.get('type')}")
            
            if raw_msg.get("type") == "websocket.disconnect":
                logger.warning(f"[WS-Trace] Received disconnect event")
                break
            
            if "text" in raw_msg:
                try:
                    import json
                    message = json.loads(raw_msg["text"])
                    logger.warning(f"[WS-Trace] Decoded JSON type: {message.get('type')}")
                except Exception as je:
                    logger.warning(f"[WS-Trace] JSON load error: {je}. Raw text: {raw_msg['text'][:100]}")
                    continue
            elif "bytes" in raw_msg:
                logger.warning(f"[WS-Trace] Received binary data: {len(raw_msg['bytes'])} bytes")
                continue
            else:
                logger.warning(f"[WS-Trace] Unknown message format: {raw_msg}")
                continue

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if message.get("type") == "teacher_peer_id":
                peer_id = message.get("peer_id")
                if peer_id:
                    room.teacher_peer_id = peer_id
                    logger.info(f"Room {room_code} video peer set to: {peer_id}")
                    for student in room.get_connected_students():
                        ws = room.connections.get(student.participant_id)
                        if ws:
                            try:
                                await ws.send_json({"type": "teacher_peer_id", "peer_id": peer_id})
                            except Exception:
                                pass
                continue

            if message.get("type") != "audio_chunk":
                if message.get("type") != "ping":
                    logger.warning(f"[WS-Trace] Received non-audio message: {message.get('type')}")
                continue

            # Decode raw audio
            audio_b64   = message.get("data", "")
            sample_rate = int(message.get("sample_rate", 16000))
            try:
                audio_bytes = base64.b64decode(audio_b64)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                logger.warning(f"[WS-Trace] Decoded chunk: {len(audio_array)} samples")
            except Exception as e:
                logger.warning(f"[WS-Trace] Bad audio chunk: {e}")
                continue

            chunk_duration = len(audio_array) / sample_rate

            # ── Real-time sound detection (2s rolling window) ──────────────
            sound_buf.append(audio_array)
            sound_buf_dur += chunk_duration
            if sound_buf_dur >= SOUND_WINDOW_SECS:
                sound_audio   = np.concatenate(sound_buf)
                sound_buf     = []
                sound_buf_dur = 0.0
                if np.max(np.abs(sound_audio)) >= 0.003:
                    try:
                        snd_evt = get_sound_detector().detect_sound(
                            sound_audio, sample_rate=sample_rate
                        )
                        if snd_evt:
                            logger.info(f"[Realtime] {snd_evt['display']}")
                            await room.broadcast_sound_event(snd_evt)
                    except Exception as _se:
                        logger.warning(f"Sound detect error: {_se}")

            # ── VAD / caption buffering ────────────────────────────────────
            # Use RMS dB check to determine if this chunk is silence
            if is_silence(audio_array, threshold_db=SILENCE_DB):
                silence_duration += chunk_duration
            else:
                silence_duration = 0.0

            audio_buffer.append(audio_array)
            buffer_duration += chunk_duration

            # Transcribe if:
            # 1. Enough silence observed (speech ended) AND minimum segment length reached
            # 2. OR max buffer size reached (force flush so captions don't lag too much)
            should_transcribe = (
                (silence_duration >= SILENCE_THRESHOLD and buffer_duration >= MIN_AUDIO_SECONDS)
                or buffer_duration >= MAX_BUFFER_SECONDS
            )

            if not should_transcribe:
                continue

            logger.info(f"[VAD] Flushing buffer: length={buffer_duration:.2f}s, silence={silence_duration:.2f}s")

            # ── Flush buffer — reset IMMEDIATELY, process in background ────
            segment          = np.concatenate(audio_buffer)
            audio_buffer     = []
            buffer_duration  = 0.0
            silence_duration = 0.0

            # Preprocess (fast — runs in-line)
            # No secondary VAD gate here — trust the buffer logic
            segment = preprocess_for_transcription(
                segment, sample_rate=16000,
                apply_highpass=True, apply_noise_reduction=True,
                apply_normalization=True, target_rms=0.05
            )

            # Safety check: Whisper needs at least ~0.1s of audio to avoid tensor errors
            if len(segment) < 3200:  # 0.2s at 16kHz
                continue

            # STT + enrichment + broadcast run as a non-blocking background task
            # The loop continues immediately and accumulates fresh audio
            asyncio.create_task(_process_caption(segment, room, websocket))

    except WebSocketDisconnect:
        logger.info(f"Teacher disconnected from room {room_code}")
    except Exception as e:
        logger.error(f"Teacher stream error in room {room_code}: {e}", exc_info=True)
    finally:
        if teacher:
            room.unregister_websocket(teacher.participant_id)
        await websocket.close() if websocket.client_state != WebSocketState.DISCONNECTED else None


# ─── WebSocket: Student Caption Receiver ──────────────────────────────────────

@router.websocket("/ws/room/{room_code}/student/{participant_id}")
async def student_caption_receiver(websocket: WebSocket, room_code: str, participant_id: str):
    """
    Student WebSocket endpoint.
    Student connects and receives live captions in their preferred language.
    """
    await websocket.accept()
    service = get_room_service()
    room = service.get_room(room_code)

    if not room or not room.is_active:
        await websocket.send_json({"type": "error", "message": "Room not found or inactive"})
        await websocket.close()
        return

    participant = room.participants.get(participant_id)
    if not participant:
        # Participant not found — could be a reconnect after server restart.
        # Try to recover language from disk JSON so translation isn't reset to 'en'.
        logger.warning(f"Participant {participant_id} not in room {room_code} — auto-recovering")
        from services.room_service import Participant as P
        from datetime import datetime
        import json, os
        recovered_lang = "en"
        try:
            disk_path = f"data/rooms/{room_code}.json"
            if os.path.exists(disk_path):
                with open(disk_path, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                disk_p = disk_data.get("participants", {}).get(participant_id, {})
                recovered_lang = disk_p.get("language", "en")
                logger.info(f"[AutoRecover] Recovered language '{recovered_lang}' for {participant_id}")
        except Exception as ex:
            logger.warning(f"[AutoRecover] Could not read disk language: {ex}")

        participant = P(
            participant_id=participant_id,
            name="Student",
            role="student",
            language=recovered_lang,
            joined_at=datetime.now().isoformat(),
        )
        room.participants[participant_id] = participant
        service._save_to_disk(room)


    # Register WebSocket
    room.register_websocket(participant_id, websocket)

    # Start screen time session
    start_session(participant_id)

    try:
        # Send welcome — wrapped in try so a fast-disconnect doesn't crash ASGI
        await websocket.send_json({
            "type": "connected",
            "room_code": room_code,
            "participant_id": participant_id,
            "name": participant.name,
            "language": participant.language,
            "message": f"Connected to room '{room_code}'. Waiting for teacher to speak..."
        })

        # Initial screen time check
        status = check_screen_time(participant_id, room_code)
        if not status.allowed:
            await websocket.send_json({
                "type": "error",
                "message": f"Global Access Limit: {status.reason}"
            })
            await websocket.close()
            return

        if status.remaining_minutes < 1000:
            await websocket.send_json({
                "type": "screen_time_info",
                "remaining_minutes": status.remaining_minutes
            })

        logger.info(f"Student '{participant.name}' connected to room {room_code} (lang={participant.language})")

        # If teacher already has a video peer active, send it immediately
        if room.teacher_peer_id:
            await websocket.send_json({
                "type": "teacher_peer_id",
                "peer_id": room.teacher_peer_id
            })

        # Keep connection alive — captions are pushed by teacher stream
        while True:
            try:
                # Wait for message with timeout to periodically check screen time
                message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "change_language":
                    new_lang = message.get("language", "en")
                    if new_lang in SUPPORTED_LANGUAGES:
                        participant.language = new_lang
                        # Persist so server restarts retain language preference
                        get_room_service()._save_to_disk(room)
                        logger.info(f"[Language] {participant.name} switched to '{new_lang}'")
                        await websocket.send_json({
                            "type": "language_changed",
                            "language": new_lang
                        })

            except asyncio.TimeoutError:
                # Timeout just means no input from student (normal) — check screen time
                pass

            # ── Check screen time limits ──────────────────────────────────────
            status = check_screen_time(participant_id, room_code)
            if not status.allowed:
                logger.info(f"Screen time limit reached for {participant.name}: {status.reason}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Session Ended: {status.reason}"
                })
                end_session(participant_id)
                await websocket.close()
                return

            # Warn if running low (e.g. 5 minutes left)
            if 0 < status.remaining_minutes <= 5.0:
                 await websocket.send_json({
                    "type": "screen_time_warning",
                    "message": f"⚠️ {int(status.remaining_minutes)} minutes remaining",
                    "remaining_minutes": status.remaining_minutes
                })

    except WebSocketDisconnect:
        logger.info(f"Student '{participant.name}' disconnected from room {room_code}")
    except Exception as e:
        logger.warning(f"Student WS error (room {room_code}): {e}")
    finally:
        end_session(participant_id)
        room.unregister_websocket(participant_id)
        try:
            await websocket.close()
        except Exception:
            pass
