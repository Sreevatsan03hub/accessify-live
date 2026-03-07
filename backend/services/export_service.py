"""
Export Service — Post-Processed Downloads (Feature 10)
Generates downloadable caption files from saved sessions.
Supports: SRT, VTT, TXT, PDF formats.
"""
import os
import logging
from typing import Optional
from datetime import timedelta

logger = logging.getLogger(__name__)


class ExportService:
    """Generate downloadable caption files from session data."""
    
    def __init__(self):
        self.export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "exports")
        os.makedirs(self.export_dir, exist_ok=True)
        logger.info("Export service initialized")
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format seconds to VTT timestamp: HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _build_segments(self, session_data: dict) -> list:
        """
        Build timed segments from session data.
        If session has VTT segments (video), use those.
        Otherwise, create segments from captions (live session).
        """
        segments = []
        
        # Check for VTT segments in metadata (video sessions)
        metadata = session_data.get("metadata", {})
        if metadata.get("segments"):
            for seg in metadata["segments"]:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })
            return segments
        
        # For live sessions: create segments from captions
        captions = session_data.get("captions", [])
        duration_per_caption = 5.0  # Default 5 seconds per caption
        
        for i, cap in enumerate(captions):
            text = cap.get("text", "").strip()
            if not text:
                continue
            start = i * duration_per_caption
            end = start + duration_per_caption
            segments.append({"start": start, "end": end, "text": text})
        
        return segments

    def generate_srt(self, session_data: dict) -> str:
        """Generate SRT subtitle content."""
        segments = self._build_segments(session_data)
        if not segments:
            return ""
        
        lines = []
        for i, seg in enumerate(segments, 1):
            start_ts = self._format_timestamp_srt(seg["start"])
            end_ts = self._format_timestamp_srt(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg["text"])
            lines.append("")  # blank line between entries
        
        return "\n".join(lines)
    
    def generate_vtt(self, session_data: dict) -> str:
        """Generate WebVTT subtitle content."""
        # Prefer pre-generated VTT from metadata (video uploads)
        metadata = session_data.get("metadata", {})
        if metadata.get("vtt"):
            return metadata["vtt"]

        segments = self._build_segments(session_data)
        if not segments:
            return "WEBVTT\n"
        
        lines = ["WEBVTT", f"Title: {session_data.get('title', 'Captions')}", ""]
        
        for i, seg in enumerate(segments, 1):
            start_ts = self._format_timestamp_vtt(seg["start"])
            end_ts = self._format_timestamp_vtt(seg["end"])
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg["text"])
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_txt(self, session_data: dict, include_metadata: bool = True) -> str:
        """Generate plain text transcript."""
        lines = []
        
        if include_metadata:
            lines.append(f"TRANSCRIPT: {session_data.get('title', 'Session')}")
            lines.append(f"Date: {session_data.get('created_at', 'Unknown')}")
            lines.append(f"Language: {session_data.get('language', 'en')}")
            lines.append(f"Type: {session_data.get('session_type', 'unknown')}")
            lines.append("=" * 60)
            lines.append("")
        
        # Check for segments (video) or captions (live)
        segments = self._build_segments(session_data)
        if segments:
            for seg in segments:
                lines.append(seg["text"])
        else:
            # Fallback: raw captions
            for cap in session_data.get("captions", []):
                text = cap.get("text", "")
                if text:
                    lines.append(text)
        
        # Add enrichment summary if available
        captions = session_data.get("captions", [])
        if captions and include_metadata:
            lines.append("")
            lines.append("=" * 60)
            lines.append("ENRICHMENT SUMMARY")
            lines.append("=" * 60)
            
            for cap in captions:
                # Keywords
                keywords = cap.get("keywords", [])
                if keywords:
                    kw_list = ", ".join([f"{k['keyword']} {k.get('emoji', '')}" for k in keywords])
                    lines.append(f"Keywords: {kw_list}")
                
                # Tone
                tone = cap.get("tone", {})
                if tone:
                    lines.append(f"Tone: {tone.get('emotion', 'N/A')} | Intent: {tone.get('intent', 'N/A')}")
                
                # Translation
                translation = cap.get("translation", {})
                if translation and translation.get("text"):
                    lines.append(f"Translation ({translation.get('target_language', '?')}): {translation['text']}")
        
        return "\n".join(lines)
    
    def generate_summary(self, session_data: dict) -> str:
        """Generate a concise summary of the session for note-making."""
        lines = []
        lines.append(f"📋 SESSION SUMMARY")
        lines.append(f"Title: {session_data.get('title', 'Session')}")
        lines.append(f"Date: {session_data.get('created_at', 'Unknown')}")
        
        duration = session_data.get("metadata", {}).get("duration", 0)
        if duration:
            mins = int(duration // 60)
            secs = int(duration % 60)
            lines.append(f"Duration: {mins}m {secs}s")
        
        lines.append(f"Captions: {session_data.get('caption_count', 0)}")
        lines.append("")
        
        # Full text
        lines.append("📝 FULL TRANSCRIPT")
        lines.append("-" * 40)
        segments = self._build_segments(session_data)
        full_text = " ".join([seg["text"] for seg in segments]) if segments else ""
        if not full_text:
            for cap in session_data.get("captions", []):
                if cap.get("text"):
                    full_text += cap["text"] + " "
        lines.append(full_text.strip())
        lines.append("")
        
        # Keywords
        all_keywords = []
        for cap in session_data.get("captions", []):
            for kw in cap.get("keywords", []):
                kw_str = f"{kw.get('emoji', '🔑')} {kw['keyword']}"
                if kw_str not in all_keywords:
                    all_keywords.append(kw_str)
        
        if all_keywords:
            lines.append("⭐ KEY TOPICS")
            lines.append("-" * 40)
            lines.append(", ".join(all_keywords))
            lines.append("")
        
        # Tone
        for cap in session_data.get("captions", []):
            tone = cap.get("tone", {})
            if tone:
                lines.append("🎭 TONE & INTENT")
                lines.append("-" * 40)
                lines.append(f"Emotion: {tone.get('emotion', 'N/A')} {tone.get('emoji', '')}")
                lines.append(f"Intent: {tone.get('intent', 'N/A')}")
                break
        
        return "\n".join(lines)


# Singleton
_export_service = None

def get_export_service() -> ExportService:
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
