import { useEffect, useRef } from 'react';
import { CaptionCard } from './CaptionCard';
import { useCaptions } from '../../context/CaptionContext';

export function CaptionPanel({
  captions = [],
  showEmojis = true,
  showTranslation = false,
  language = 'en',
  maxHeight = 'max-h-96',
}) {
  const { autoScroll } = useCaptions();
  const panelRef = useRef(null);
  const lastCaptionRef = useRef(null);

  useEffect(() => {
    if (autoScroll && lastCaptionRef.current) {
      lastCaptionRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [captions, autoScroll]);

  if (!captions || captions.length === 0) {
    return (
      <div
        className={`
          ${maxHeight}
          p-6
          bg-caption-bg
          rounded-xl
          border-2 border-gray-600
          flex items-center justify-center
          text-gray-400
          text-center
        `}
      >
        <div>
          <p className="text-lg font-semibold mb-2">Waiting for captions...</p>
          <p className="text-sm">Captions will appear here as they're transcribed</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={panelRef}
      className={`
        ${maxHeight}
        overflow-y-auto
        p-4
        bg-gradient-to-b from-black/70 to-black/85
        rounded-xl
        border-2 border-gray-700
        space-y-2
      `}
    >
      {captions.map((caption, index) => (
        <div
          key={caption.id || index}
          ref={index === captions.length - 1 ? lastCaptionRef : null}
        >
          <CaptionCard
            text={caption.text}
            simplifiedText={caption.simplified_text}
            keywords={caption.keywords}
            tone={caption.tone}
            translation={caption.translation}
            soundEvent={caption.sound_event}
            showEmojis={showEmojis}
            showTranslation={showTranslation}
            language={language}
          />
        </div>
      ))}
    </div>
  );
}
