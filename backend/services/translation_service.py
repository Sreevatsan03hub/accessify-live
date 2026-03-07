"""
Translation Service — Google Translate via direct urllib call.
No model downloads, no heavy dependencies — just a lightweight HTTP call.
Previously used deep_translator which timed out inside uvicorn's thread pool.
"""
import urllib.request
import urllib.parse
import json
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of a translation call."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str


SUPPORTED_LANGUAGES = ["en", "hi", "ta", "te"]

# Google Translate language codes (these match the standard BCP 47 codes)
LANG_CODES = {
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "en": "en",   # English
}


def _google_translate(text: str, target_lang: str, source_lang: str = "en") -> str:
    """
    Translate text using Google Translate's unofficial API via urllib.
    This is the same endpoint deep_translator uses internally.
    No extra dependencies — stdlib urllib only.
    """
    params = urllib.parse.urlencode({
        'client': 'gtx',
        'sl': source_lang,
        'tl': LANG_CODES.get(target_lang, target_lang),
        'dt': 't',
        'q': text,
    })
    url = f'https://translate.googleapis.com/translate_a/single?{params}'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0'
    })
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode('utf-8'))
    # data[0] is a list of [translated_chunk, original_chunk, ...]
    return ''.join(item[0] for item in data[0] if item[0])


class IndicTranslator:
    """
    Lightweight translation service using Google Translate via HTTP.
    Zero memory overhead — no model files, no downloads.
    """

    SUPPORTED_LANGUAGES = SUPPORTED_LANGUAGES

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en"
    ) -> TranslationResult:
        """Translate text from English to target Indian language."""
        if not text or not text.strip():
            return TranslationResult(
                source_text=text, translated_text="",
                source_lang=source_lang, target_lang=target_lang
            )

        if target_lang == "en" or target_lang not in LANG_CODES:
            return TranslationResult(
                source_text=text, translated_text=text,
                source_lang=source_lang, target_lang=target_lang
            )

        try:
            translated = _google_translate(text, target_lang, source_lang)
            logger.info(f"[Translation] {target_lang}: '{text[:40]}' → '{translated[:40]}'")
            return TranslationResult(
                source_text=text, translated_text=translated,
                source_lang=source_lang, target_lang=target_lang
            )
        except Exception as e:
            logger.error(f"[Translation] {target_lang} FAILED: {e}")
            # Always fall back — never crash the caption pipeline
            return TranslationResult(
                source_text=text, translated_text=text,
                source_lang=source_lang, target_lang=target_lang
            )

    def batch_translate(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple texts."""
        return [self.translate(t, target_lang, source_lang) for t in texts]


# Singleton
_translator = None


def get_translator() -> IndicTranslator:
    global _translator
    if _translator is None:
        _translator = IndicTranslator()
    return _translator
