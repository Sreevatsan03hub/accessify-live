import { TONE_COLORS } from '../../utils/constants'

interface ToneIndicatorProps {
  emotion?: string
  intent?: string
  emoji?: string
}

export function ToneIndicator({ emotion, intent, emoji }: ToneIndicatorProps) {
  const getToneKey = (): keyof typeof TONE_COLORS => {
    if (intent === 'urgent') return 'urgent'
    if (emotion === 'positive') return 'positive'
    if (intent === 'question') return 'question'
    return 'neutral'
  }

  const key = getToneKey()
  const toneConfig = TONE_COLORS[key]

  return (
    <div
      className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold"
      style={{
        backgroundColor: toneConfig.bg,
        color: toneConfig.color,
        borderLeft: `3px solid ${toneConfig.color}`,
      }}
    >
      <span className="text-lg leading-none">{emoji || toneConfig.emoji}</span>
      <span className="capitalize">{intent || emotion || 'Neutral'}</span>
    </div>
  )
}
