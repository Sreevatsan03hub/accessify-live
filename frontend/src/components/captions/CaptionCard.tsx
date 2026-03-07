import { Caption } from '../../context/CaptionContext'
import { KeywordBadge } from './KeywordBadge'
import { ToneIndicator } from './ToneIndicator'
import { SoundEventBanner } from './SoundEventBanner'

interface CaptionCardProps {
  caption: Caption
  fontSize: string
  showEmojis: boolean
  showTranslation: boolean
  isActive?: boolean
}

export function CaptionCard({
  caption,
  fontSize,
  showEmojis,
  showTranslation,
  isActive,
}: CaptionCardProps) {
  return (
    <div
      className={`w-full p-4 mb-3 rounded-lg border-2 transition-all duration-200 ${
        isActive
          ? 'border-accent bg-accent/10 shadow-lg shadow-accent/50'
          : 'border-border bg-black/40 hover:border-accent/50'
      }`}
      role="region"
      aria-live="polite"
      aria-label="Caption"
    >
      {caption.sound_event && <SoundEventBanner event={caption.sound_event} />}

      <div
        className="caption-text font-semibold leading-relaxed"
        style={{ fontSize }}
      >
        {caption.text}
      </div>

      {caption.simplified_text && (
        <div className="mt-2 text-sm text-foreground/70 leading-relaxed pl-2 border-l-2 border-muted/40">
          {caption.simplified_text}
        </div>
      )}

      <div className="flex items-center gap-2 mt-3 flex-wrap">
        <ToneIndicator
          emotion={caption.tone?.emotion}
          intent={caption.tone?.intent}
          emoji={caption.tone?.emoji}
        />

        {showEmojis && caption.keywords?.length > 0 && (
          <div className="flex gap-1 flex-wrap">
            {caption.keywords.map((kw, idx) => (
              <KeywordBadge key={idx} keyword={kw.keyword} emoji={kw.emoji} />
            ))}
          </div>
        )}
      </div>

      {showTranslation && caption.translation && (
        <div className="mt-3 p-2 bg-primary/10 rounded border border-primary/30 text-sm">
          <div className="text-xs text-muted mb-1">Translation ({caption.translation.target_language})</div>
          <div>{caption.translation.text}</div>
        </div>
      )}
    </div>
  )
}
