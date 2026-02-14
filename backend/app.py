"""
Accessify Backend Application
AI-powered accessibility platform for Deaf & Hard-of-Hearing users.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.audio_routes import router as audio_router
from services.unified_audio_pipeline import init_pipeline, shutdown_pipeline
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

    # Initialize the audio pipeline
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


if __name__ == "__main__":
    import uvicorn

    # 🔥 Force localhost for development
    uvicorn.run(
        "app:app",
        host="127.0.0.1",   # <-- FIXED HERE
        port=8001,
        reload=True
    )