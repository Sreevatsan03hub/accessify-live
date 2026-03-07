"""
Firebase Admin SDK initialization.
Reads credentials from environment variables or a service account JSON file.
Set FIREBASE_SERVICE_ACCOUNT_PATH in .env to point to your serviceAccountKey.json
"""
import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)

_firebase_app = None
_db = None
_bucket = None

def init_firebase():
    """Initialize Firebase Admin SDK. Safe to call multiple times."""
    global _firebase_app, _db, _bucket

    if _firebase_app is not None:
        return True   # already initialised

    # Option 1: path to serviceAccountKey.json
    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
    # Option 2: inline JSON string (for cloud deployments)
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
    storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "")

    try:
        if sa_path and os.path.exists(sa_path):
            cred = credentials.Certificate(sa_path)
            logger.info(f"Firebase: loading credentials from file: {sa_path}")
        elif sa_json:
            cred = credentials.Certificate(json.loads(sa_json))
            logger.info("Firebase: loading credentials from environment JSON string")
        else:
            logger.warning(
                "Firebase credentials not configured. "
                "Set FIREBASE_SERVICE_ACCOUNT_PATH in backend/.env\n"
                "App will continue using local JSON file storage."
            )
            return False

        opts = {"storageBucket": storage_bucket} if storage_bucket else {}
        _firebase_app = firebase_admin.initialize_app(cred, opts)
        _db = firestore.client()
        if storage_bucket:
            _bucket = storage.bucket()
        logger.info("✅ Firebase Admin SDK initialised successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Firebase init failed: {e}")
        return False


def get_db():
    """Return the Firestore client (or None if Firebase not configured)."""
    return _db


def get_bucket():
    """Return the Firebase Storage bucket (or None if not configured)."""
    return _bucket


def is_configured():
    """True when Firebase was successfully initialised."""
    return _firebase_app is not None
