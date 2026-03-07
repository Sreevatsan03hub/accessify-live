import { useEffect, useState } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { CaptionPanel } from '../components/captions/CaptionPanel'
import { Caption } from '../context/CaptionContext'
import { useUser } from '../context/UserContext'
import { MOCK_CAPTIONS, CAPTION_SIZES } from '../utils/constants'
import { Phone, Settings as SettingsIcon, Volume2, VolumeX } from 'lucide-react'

export default function StudentRoom() {
  const { code, participantId } = useParams<{ code: string; participantId: string }>()
  const { state } = useLocation() as { state?: { name: string; language: string } }
  const navigate = useNavigate()
  const { user, updateSettings } = useUser()
  const [captions, setCaptions] = useState<Caption[]>([])
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large' | 'xl'>('medium')
  const [language, setLanguage] = useState<'en' | 'hi' | 'ta' | 'te'>(
    (state?.language as any) || user?.language || 'en'
  )
  const [showEmojis, setShowEmojis] = useState(true)
  const [soundsEnabled, setSoundsEnabled] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)

  useEffect(() => {
    if (!code || !participantId) {
      navigate('/room/join')
      return
    }

    // Simulate WebSocket connection
    setIsConnected(true)

    // Simulate receiving captions
    let captionIndex = 0
    const interval = setInterval(() => {
      if (isConnected && captionIndex < MOCK_CAPTIONS.length) {
        setCaptions(prev => [...prev, MOCK_CAPTIONS[captionIndex]])
        captionIndex++
      }
    }, 3000)

    return () => {
      clearInterval(interval)
    }
  }, [code, participantId, isConnected, navigate])

  const handleLeaveClass = () => {
    navigate('/dashboard')
  }

  const handleLanguageChange = (newLanguage: 'en' | 'hi' | 'ta' | 'te') => {
    setLanguage(newLanguage)
    updateSettings?.({ language: newLanguage })
  }

  const handleFontSizeChange = (newSize: 'small' | 'medium' | 'large' | 'xl') => {
    setFontSize(newSize)
    updateSettings?.({ captionSize: newSize })
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-black/50">
      <div className="max-w-6xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold">🎓 Live Class</h1>
            <p className="text-muted">Room Code: {code}</p>
          </div>
          <Button
            variant="danger"
            size="lg"
            onClick={handleLeaveClass}
            className="gap-2"
          >
            <Phone size={20} />
            Leave Class
          </Button>
        </div>

        {reconnecting && (
          <div className="mb-4 p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm flex items-center gap-2">
            <div className="w-2 h-2 bg-warning rounded-full animate-pulse" />
            Reconnecting...
          </div>
        )}

        {!isConnected && !reconnecting && (
          <div className="mb-4 p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
            Connection lost. Attempting to reconnect...
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Caption Area */}
          <div className="lg:col-span-3">
            <Card className="min-h-96">
              <h2 className="text-xl font-bold mb-4">Live Captions</h2>
              <CaptionPanel
                captions={captions}
                fontSize={fontSize}
                showEmojis={showEmojis}
                showTranslation={language !== 'en'}
                language={language}
                maxHeight="max-h-96"
                autoScroll={true}
              />
            </Card>
          </div>

          {/* Right Sidebar - Controls */}
          <div className="space-y-4">
            {/* Language Selector */}
            <Card>
              <h3 className="font-bold text-sm mb-3">Language</h3>
              <div className="space-y-2">
                {(['en', 'hi', 'ta', 'te'] as const).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => handleLanguageChange(lang)}
                    className={`w-full px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                      language === lang
                        ? 'bg-accent text-black'
                        : 'bg-black/40 border border-border hover:border-accent/50'
                    }`}
                  >
                    {lang === 'en' && '🇬🇧 EN'}
                    {lang === 'hi' && '🇮🇳 हिं'}
                    {lang === 'ta' && '🇮🇳 தமி'}
                    {lang === 'te' && '🇮🇳 తెలు'}
                  </button>
                ))}
              </div>
            </Card>

            {/* Caption Size */}
            <Card>
              <h3 className="font-bold text-sm mb-3">Caption Size</h3>
              <div className="space-y-2">
                {(['small', 'medium', 'large', 'xl'] as const).map((size) => (
                  <button
                    key={size}
                    onClick={() => handleFontSizeChange(size)}
                    className={`w-full px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                      fontSize === size
                        ? 'bg-accent text-black'
                        : 'bg-black/40 border border-border hover:border-accent/50'
                    }`}
                  >
                    {CAPTION_SIZES[size].label}
                  </button>
                ))}
              </div>
            </Card>

            {/* Options */}
            <Card>
              <h3 className="font-bold text-sm mb-3">Options</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setShowEmojis(!showEmojis)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-black/40 border border-border hover:border-accent/50 transition-all text-sm"
                >
                  <input
                    type="checkbox"
                    checked={showEmojis}
                    onChange={() => {}}
                    className="rounded"
                  />
                  <span>Emojis</span>
                </button>
                <button
                  onClick={() => setSoundsEnabled(!soundsEnabled)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-black/40 border border-border hover:border-accent/50 transition-all text-sm"
                >
                  {soundsEnabled ? (
                    <>
                      <Volume2 size={16} />
                      <span>Sound On</span>
                    </>
                  ) : (
                    <>
                      <VolumeX size={16} />
                      <span>Sound Off</span>
                    </>
                  )}
                </button>
              </div>
            </Card>

            {/* Connection Status */}
            <Card className={isConnected ? 'border-accent' : 'border-warning'}>
              <div className="flex items-center gap-2 mb-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isConnected ? 'bg-accent' : 'bg-warning'
                  } ${!isConnected && 'animate-pulse'}`}
                />
                <span className="font-semibold text-sm">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <p className="text-xs text-muted">
                {isConnected
                  ? 'Receiving live captions'
                  : 'Attempting to reconnect...'}
              </p>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
