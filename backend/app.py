"""
Accessify Backend Application
AI-powered accessibility platform for Deaf & Hard-of-Hearing users.
"""

# ── Load .env FIRST — before any service imports read os.getenv() ──────────
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
# ────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.audio_routes import router as audio_router
from routes.video_routes import router as video_router
from routes.session_routes import router as session_router
from routes.export_routes import router as export_router
from routes.buffer_routes import router as buffer_router
from routes.broadcast_routes import router as broadcast_router
from routes.safety_routes import router as safety_router
from services.unified_audio_pipeline import init_pipeline, shutdown_pipeline
from services.firebase_service import init_firebase
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting Accessify Backend...")

    # ── 1. Firebase (optional — only runs if credentials configured) ──
    try:
        firebase_ok = init_firebase()
        if firebase_ok:
            logger.info("🔥 Firebase connected (Firestore + Storage active)")
        else:
            logger.info("📁 Firebase not configured — using local JSON file storage")
    except Exception as e:
        logger.warning(f"Firebase init skipped: {e}")

    # ── 2. Audio pipeline ──────────────────────────────────────────────
    try:
        init_pipeline()
        logger.info("Audio pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize audio pipeline: {e}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Accessify Backend...")
    try:
        shutdown_pipeline()
    except Exception:
        pass



app = FastAPI(
    title="Accessify API",
    description="AI-powered accessibility platform for Deaf & Hard-of-Hearing users",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_router, prefix="/api/v1/audio")
app.include_router(video_router, prefix="/api/v1/video")
app.include_router(session_router)   # Already has /api/v1/sessions prefix
app.include_router(export_router)    # Already has /api/v1/export prefix
app.include_router(buffer_router)    # Already has /api/v1/buffer prefix
app.include_router(broadcast_router) # /api/v1/rooms + /ws/room WebSocket
app.include_router(safety_router)    # /api/v1/rooms/{code}/safety


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Accessify AI Caption Backend Running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/test_client.html")
def serve_test_client():
    """Serve the audio test client."""
    return FileResponse("test_client.html")


@app.get("/test_video_upload.html")
def serve_video_test():
    """Serve the video upload test page."""
    return FileResponse("test_video_upload.html")


if __name__ == "__main__":
    import uvicorn

    # 🔥 Force localhost for development
    uvicorn.run(
        "app:app",
        host="127.0.0.1",   # <-- FIXED HERE
        port=8001,
        reload=True
    )