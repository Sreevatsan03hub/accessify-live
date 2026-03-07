"""
Buffer Service — Low-Internet Resilience (Feature 11)
Handles audio buffering, batch processing, and reconnection support.
Ensures captions continue working even with unstable internet.
"""
import os
import json
import time
import uuid
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Storage for offline buffer data
BUFFER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "buffers")


class ClientBuffer:
    """Tracks buffered audio state for a single client."""
    
    def __init__(self, client_id: str, session_id: str = None):
        self.client_id = client_id
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.last_active = time.time()
        self.chunks_received = 0
        self.chunks_processed = 0
        self.pending_chunks: List[Dict] = []
        self.is_connected = True
        self.reconnect_count = 0
    
    def add_chunk(self, chunk_data: dict):
        """Add a buffered audio chunk."""
        self.pending_chunks.append({
            "index": self.chunks_received,
            "timestamp": time.time(),
            "data": chunk_data
        })
        self.chunks_received += 1
        self.last_active = time.time()
    
    def mark_processed(self, count: int = 1):
        """Mark chunks as processed and remove from pending."""
        self.chunks_processed += count
        self.pending_chunks = self.pending_chunks[count:]
    
    def mark_reconnected(self):
        """Handle client reconnection."""
        self.is_connected = True
        self.reconnect_count += 1
        self.last_active = time.time()
        logger.info(f"Client {self.client_id} reconnected (attempt #{self.reconnect_count})")
    
    def mark_disconnected(self):
        """Handle client disconnection."""
        self.is_connected = False
        logger.info(f"Client {self.client_id} disconnected. Pending chunks: {len(self.pending_chunks)}")
    
    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "is_connected": self.is_connected,
            "reconnect_count": self.reconnect_count,
            "chunks_received": self.chunks_received,
            "chunks_processed": self.chunks_processed,
            "pending_count": len(self.pending_chunks),
            "last_active": self.last_active,
        }


class BufferService:
    """
    Manages client buffers for low-internet resilience.
    
    How it works:
    1. Client connects via WebSocket and gets a client_id
    2. If connection drops, client buffers audio locally (in browser IndexedDB)
    3. When connection resumes, client sends buffered chunks via batch API
    4. Backend processes the batch and returns all captions at once
    5. Client continues with live streaming
    """
    
    def __init__(self):
        os.makedirs(BUFFER_DIR, exist_ok=True)
        self.clients: Dict[str, ClientBuffer] = {}
        # Session timeout: auto-clean clients inactive for 30 min
        self.timeout_seconds = 30 * 60
        logger.info("Buffer service initialized (low-internet resilience)")
    
    def register_client(self, session_id: str = None) -> str:
        """Register a new client and return client_id."""
        client_id = str(uuid.uuid4())[:8]
        self.clients[client_id] = ClientBuffer(client_id, session_id)
        logger.info(f"Client registered: {client_id}")
        return client_id
    
    def get_client(self, client_id: str) -> Optional[ClientBuffer]:
        """Get client buffer state."""
        return self.clients.get(client_id)
    
    def reconnect_client(self, client_id: str) -> Optional[ClientBuffer]:
        """Handle client reconnection. Returns the client buffer or None."""
        client = self.clients.get(client_id)
        if client:
            client.mark_reconnected()
            return client
        
        # Client not found — create new one with same ID
        logger.warning(f"Client {client_id} not found during reconnect, creating new buffer")
        self.clients[client_id] = ClientBuffer(client_id)
        self.clients[client_id].reconnect_count = 1
        return self.clients[client_id]
    
    def disconnect_client(self, client_id: str):
        """Mark client as disconnected (but keep buffer)."""
        client = self.clients.get(client_id)
        if client:
            client.mark_disconnected()
            # Save pending state to disk in case server restarts
            self._save_buffer_to_disk(client)
    
    def add_buffered_chunks(self, client_id: str, chunks: List[dict]):
        """Accept batch of buffered chunks from a reconnecting client."""
        client = self.clients.get(client_id)
        if not client:
            client = ClientBuffer(client_id)
            self.clients[client_id] = client
        
        for chunk in chunks:
            client.add_chunk(chunk)
        
        logger.info(f"Client {client_id}: received {len(chunks)} buffered chunks")
        return len(chunks)
    
    def get_pending_chunks(self, client_id: str) -> List[dict]:
        """Get all pending (unprocessed) chunks for a client."""
        client = self.clients.get(client_id)
        if not client:
            return []
        return [c["data"] for c in client.pending_chunks]
    
    def mark_chunks_processed(self, client_id: str, count: int):
        """Mark chunks as processed after successful AI processing."""
        client = self.clients.get(client_id)
        if client:
            client.mark_processed(count)
    
    def get_status(self, client_id: str) -> dict:
        """Get buffer status for a client."""
        client = self.clients.get(client_id)
        if not client:
            return {"error": "Client not found"}
        return client.to_dict()
    
    def get_all_clients(self) -> List[dict]:
        """Get all active client buffers."""
        self._cleanup_stale()
        return [c.to_dict() for c in self.clients.values()]
    
    def cleanup_client(self, client_id: str):
        """Remove client buffer entirely."""
        if client_id in self.clients:
            del self.clients[client_id]
            # Also remove from disk
            filepath = os.path.join(BUFFER_DIR, f"{client_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def _save_buffer_to_disk(self, client: ClientBuffer):
        """Save client buffer to disk for persistence."""
        filepath = os.path.join(BUFFER_DIR, f"{client.client_id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(client.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save buffer {client.client_id}: {e}")
    
    def _cleanup_stale(self):
        """Remove clients that have been inactive beyond timeout."""
        now = time.time()
        stale = [cid for cid, c in self.clients.items() 
                 if (now - c.last_active) > self.timeout_seconds and not c.is_connected]
        for cid in stale:
            self.cleanup_client(cid)
            logger.info(f"Cleaned up stale client: {cid}")


# Singleton
_buffer_service = None

def get_buffer_service() -> BufferService:
    global _buffer_service
    if _buffer_service is None:
        _buffer_service = BufferService()
    return _buffer_service
