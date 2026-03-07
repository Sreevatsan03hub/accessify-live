"""
Moderation Service — Keyword-Based Content Filter
Fast, fully offline, zero extra dependencies.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Keyword lists (curated, commonly offensive/harmful terms) ─────────────────
# These are internal patterns — kept sanitised in comments for clarity.

_PATTERNS: dict[str, list[str]] = {
    "abuse": [
        r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\bimbecile\b",
        r"\bpathetic\b", r"\bloser\b", r"\bworthless\b", r"\bscum\b",
        r"\bdumba(ss|rse)\b", r"\ba+s+h+o+l+e\b", r"\bbast+ard\b",
        r"\bfuck(ing)?\b", r"\bs+h+i+t\b", r"\bcr+ap\b",
    ],
    "bullying": [
        r"\bbully\b", r"\bpick on\b", r"\blaugh at\b",
        r"\bno one likes you\b", r"\bkill yourself\b", r"\bkys\b",
        r"\byou('re| are) ugly\b", r"\byou('re| are) fat\b",
        r"\bnobody wants you\b", r"\bgo away\b",
    ],
    "sexual": [
        r"\bporn(o)?\b", r"\bsex(ual)?\b", r"\bnaked\b", r"\bnude(s)?\b",
        r"\bboobs?\b", r"\bdick\b", r"\bpenis\b", r"\bvagina\b",
        r"\bhooker\b", r"\bprostit\w+\b", r"\bsexting\b",
    ],
    "threat": [
        r"\bi('ll| will) kill\b", r"\bi('ll| will) hurt\b",
        r"\bi('ll| will) beat\b", r"\bwatch your back\b",
        r"\byou('re| are) dead\b", r"\bthreaten\b",
        r"\bbomb\b", r"\bshoot(ing)?\b", r"\bweapon\b",
        r"\bi know where you live\b",
    ],
}

# Pre-compile all patterns for speed
_COMPILED: dict[str, list[re.Pattern]] = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in _PATTERNS.items()
}

PRIORITY = ["threat", "sexual", "bullying", "abuse"]   # highest first


@dataclass
class ModerationResult:
    flagged: bool
    category: Optional[str]        # e.g. "threat"
    matched_word: Optional[str]    # the specific match
    redacted_text: str             # original text with bad words replaced by ****


def _redact(text: str, pattern: re.Pattern) -> str:
    """Replace all matches with asterisks of equal length."""
    def replacer(m: re.Match) -> str:
        return "*" * len(m.group())
    return pattern.sub(replacer, text)


def moderate(text: str) -> ModerationResult:
    """
    Scan text for harmful content.

    Returns a ModerationResult with:
      - flagged: True if content was detected
      - category: highest-priority category matched
      - matched_word: the literal match
      - redacted_text: text with ALL matches across ALL categories replaced
    """
    if not text or not text.strip():
        return ModerationResult(flagged=False, category=None, matched_word=None, redacted_text=text)

    redacted = text
    first_category: Optional[str] = None
    first_match: Optional[str] = None

    # Redact ALL categories (not just the first)
    for category in PRIORITY:
        for pattern in _COMPILED[category]:
            m = pattern.search(redacted)
            if m and first_category is None:
                first_category = category
                first_match = m.group()
            redacted = _redact(redacted, pattern)

    flagged = first_category is not None
    if flagged:
        logger.info(f"[Moderation] Flagged [{first_category}] — '{first_match}' in: {text[:60]}")

    return ModerationResult(
        flagged=flagged,
        category=first_category,
        matched_word=first_match,
        redacted_text=redacted,
    )


# Singleton
_moderator: Optional["ModerationService"] = None


class ModerationService:
    def moderate(self, text: str) -> ModerationResult:
        return moderate(text)


def get_moderator() -> ModerationService:
    global _moderator
    if _moderator is None:
        _moderator = ModerationService()
    return _moderator
