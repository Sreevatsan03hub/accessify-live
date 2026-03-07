import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { CaptionPanel } from '../components/captions/CaptionPanel'
import { Caption } from '../context/CaptionContext'
import { MOCK_CAPTIONS } from '../utils/constants'
import { Download, Play, Pause } from 'lucide-react'

export default function Player() {
  const { state } = useLocation() as {
    state?: {
      sessionId: string
      filename: string
      duration: number
      captions: Caption[]
      vtt: string
    }
  }
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [captions, setCaptions] = useState<Caption[]>(state?.captions || MOCK_CAPTIONS)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large' | 'xl'>('medium')
  const [showEmojis, setShowEmojis] = useState(true)

  useEffect(() => {
    if (!state?.sessionId) {
      navigate('/upload')
      return
    }
  }, [state, navigate])

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  const handleDownload = async (format: 'srt' | 'vtt' | 'txt' | 'summary') => {
    try {
      // Simulate download
      const content = state?.vtt || 'Sample captions'
      const filename = `captions_${Date.now()}.${format}`
      const blob = new Blob([content], { type: 'text/plain' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed:', error)
    }
  }

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    return `${hours > 0 ? hours + ':' : ''}${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  if (!state) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12">
        <Card>
          <p className="text-center text-muted py-8">No video data available</p>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-2">{state.filename}</h1>
      <p className="text-muted mb-8">Duration: {formatTime(state.duration)}</p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="aspect-video bg-black/60 flex items-center justify-center overflow-hidden">
            <video
              ref={videoRef}
              className="w-full h-full"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
            >
              <source src="/sample-video.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </Card>

          {/* Player Controls */}
          <Card>
            <div className="flex items-center gap-4 mb-4">
              <Button
                variant="secondary"
                onClick={handlePlayPause}
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </Button>

              <div className="flex-1">
                <input
                  type="range"
                  min="0"
                  max={duration}
                  value={currentTime}
                  onChange={(e) => {
                    if (videoRef.current) {
                      videoRef.current.currentTime = parseFloat(e.target.value)
                      setCurrentTime(parseFloat(e.target.value))
                    }
                  }}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="text-sm font-mono">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
          </Card>

          {/* Download Options */}
          <Card>
            <h3 className="font-bold text-sm mb-3">Download Captions</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {(['srt', 'vtt', 'txt', 'summary'] as const).map((format) => (
                <Button
                  key={format}
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDownload(format)}
                  className="gap-2"
                >
                  <Download size={16} />
                  {format.toUpperCase()}
                </Button>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-4">
          {/* Caption Panel */}
          <Card>
            <h3 className="font-bold text-sm mb-3">Captions</h3>
            <CaptionPanel
              captions={captions}
              fontSize={fontSize}
              showEmojis={showEmojis}
              showTranslation={false}
              language="en"
              maxHeight="max-h-96"
            />
          </Card>

          {/* Caption Settings */}
          <Card>
            <h3 className="font-bold text-sm mb-3">Caption Size</h3>
            <div className="space-y-2">
              {(['small', 'medium', 'large', 'xl'] as const).map((size) => (
                <button
                  key={size}
                  onClick={() => setFontSize(size)}
                  className={`w-full px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                    fontSize === size
                      ? 'bg-accent text-black'
                      : 'bg-black/40 border border-border hover:border-accent/50'
                  }`}
                >
                  {size === 'small' && 'A-'}
                  {size === 'medium' && 'A'}
                  {size === 'large' && 'A+'}
                  {size === 'xl' && 'A++'}
                </button>
              ))}
            </div>
          </Card>

          {/* Options */}
          <Card>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showEmojis}
                  onChange={(e) => setShowEmojis(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">Show emojis</span>
              </label>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
