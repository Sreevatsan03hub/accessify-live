import { useEffect, useRef } from 'react'
import { Caption } from '../../context/CaptionContext'
import { CaptionCard } from './CaptionCard'

interface CaptionPanelProps {
  captions: Caption[]
  fontSize: string
  showEmojis: boolean
  showTranslation: boolean
  autoScroll?: boolean
  maxHeight?: string
}

export function CaptionPanel({
  captions,
  fontSize,
  showEmojis,
  showTranslation,
  autoScroll = true,
  maxHeight = 'h-96',
}: CaptionPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll && panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight
    }
  }, [captions, autoScroll])

  return (
    <div
      ref={panelRef}
      className={`caption-panel ${maxHeight} overflow-y-auto rounded-xl p-4 space-y-2 border border-border`}
      role="log"
      aria-label="Live captions"
    >
      {captions.length === 0 ? (
        <div className="flex items-center justify-center h-full text-center text-foreground/50">
          <div>
            <p className="text-lg font-semibold">No captions yet</p>
            <p className="text-sm">Captions will appear here as they are generated</p>
          </div>
        </div>
      ) : (
        captions.map((caption, idx) => (
          <CaptionCard
            key={caption.id || idx}
            caption={caption}
            fontSize={fontSize}
            showEmojis={showEmojis}
            showTranslation={showTranslation && !!caption.translation}
            isActive={idx === captions.length - 1}
          />
        ))
      )}
    </div>
  )
}
