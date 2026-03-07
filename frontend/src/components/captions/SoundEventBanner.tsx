import { useEffect, useState } from 'react'

interface SoundEventBannerProps {
  event?: string
  duration?: number
}

export function SoundEventBanner({ event, duration = 2500 }: SoundEventBannerProps) {
  const [isVisible, setIsVisible] = useState(!!event)

  useEffect(() => {
    if (event) {
      setIsVisible(true)
      const timer = setTimeout(() => setIsVisible(false), duration)
      return () => clearTimeout(timer)
    }
  }, [event, duration])

  if (!isVisible || !event) return null

  return (
    <div
      className="w-full mb-4 px-4 py-3 bg-accent/20 border-2 border-accent text-accent rounded-lg font-bold text-center animate-pulse"
      role="alert"
      aria-live="polite"
      aria-label={`Sound event: ${event}`}
    >
      {event}
    </div>
  )
}
