import { KeywordBadge } from './KeywordBadge';
import { ToneIndicator } from './ToneIndicator';
import { SoundEventBanner } from './SoundEventBanner';
import { useCaptions } from '../../context/CaptionContext';

/**
 * CaptionCard — Displays a single caption entry.
 *
 * Language logic:
 *  - When language === 'en' (or no translation):  shows English text + English keywords
 *  - When a non-English language is selected:  shows translation.text as the PRIMARY caption
 *    and falls back to English text only if no translation is available.
 *    Keywords are already sent from the backend in the correct language.
 */
export function CaptionCard({
  text,
  simplifiedText,
  keywords = [],
  tone = {},
  translation = null,
  soundEvent = null,
  showEmojis = true,
  showTranslation = false,
  language = 'en',
}) {
  const { getCaptionSizeClass } = useCaptions();

  if (soundEvent) {
    return <SoundEventBanner event={soundEvent} />;
  }

  // ── Determine the primary display text ──────────────────────────────────────
  // If a non-English language is selected AND the backend provided a translation,
  // show that as the MAIN caption. Show English as a small subtitle below.
  const isTranslated = language !== 'en' && translation && translation.text;
  const primaryText = isTranslated ? translation.text : text;
  const secondaryText = isTranslated ? text : null; // English original shown small

  // Language label map for subtitle hint
  const LANG_LABELS = { hi: 'हिंदी', ta: 'தமிழ்', te: 'తెలుగు', en: 'English' };

  return (
    <div className="mb-4 p-4 bg-caption-bg rounded-xl border-l-4 border-primary animate-slideIn">

      {/* ── Primary caption (translated or English) ─────────────────────── */}
      <p className={`${getCaptionSizeClass()} font-bold text-white mb-2 leading-relaxed`}>
        {primaryText}
      </p>

      {/* ── English original shown as small secondary when translated ────── */}
      {isTranslated && secondaryText && (
        <p className="text-xs text-gray-500 italic mb-2 leading-relaxed">
          🇬🇧 {secondaryText}
        </p>
      )}

      {/* ── Simplified plain-English note (only in English mode) ────────── */}
      {!isTranslated && simplifiedText && simplifiedText !== text && (
        <p className="text-sm text-gray-300 italic mb-3 opacity-75">
          {simplifiedText}
        </p>
      )}

      {/* ── Keywords (backend sends them already in the student's language) ─ */}
      {showEmojis && keywords && keywords.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1">
          {keywords.map((kw, idx) => (
            <KeywordBadge
              key={idx}
              keyword={typeof kw === 'string' ? kw : kw.keyword}
              emoji={typeof kw === 'string' ? null : kw.emoji}
            />
          ))}
        </div>
      )}

      {/* ── Translation toggle block (only shown when showTranslation AND English mode) */}
      {showTranslation && language === 'en' && translation && translation.text && (
        <div className="mb-3 p-2 bg-white/10 rounded text-sm text-gray-200 border border-white/10">
          <span className="text-xs text-gray-400 block mb-1">
            🌐 {LANG_LABELS[translation.target_language] || translation.target_language}:
          </span>
          {translation.text}
        </div>
      )}

      {/* ── Tone indicator ──────────────────────────────────────────────── */}
      {tone && tone.emoji && (
        <ToneIndicator emotion={tone.emotion} intent={tone.intent} emoji={tone.emoji} />
      )}
    </div>
  );
}
