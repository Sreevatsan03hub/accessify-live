import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { CaptionPanel } from '../components/captions/CaptionPanel'
import { Caption } from '../context/CaptionContext'
import { useMicrophone } from '../hooks/useMicrophone'
import { copyToClipboard, formatTime } from '../utils/helpers'
import { MOCK_CAPTIONS } from '../utils/constants'
import { Mic, MicOff, Video, VideoOff, Phone, Copy, Check } from 'lucide-react'

export default function TeacherRoom() {
  const { code } = useParams<{ code: string }>()
  const navigate = useNavigate()
  const [captions, setCaptions] = useState<Caption[]>([])
  const [studentCount, setStudentCount] = useState(5)
  const [isBroadcasting, setIsBroadcasting] = useState(false)
  const [broadcastTime, setBroadcastTime] = useState(0)
  const [copied, setCopied] = useState(false)
  const { isRecording, startRecording, stopRecording, error: micError } = useMicrophone()

  useEffect(() => {
    if (!code) {
      navigate('/')
      return
    }

    // Simulate adding mock captions
    let captionIndex = 0
    const interval = setInterval(() => {
      if (isBroadcasting && captionIndex < MOCK_CAPTIONS.length) {
        setCaptions(prev => [...prev, MOCK_CAPTIONS[captionIndex]])
        captionIndex++
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [isBroadcasting, code, navigate])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isBroadcasting) {
      interval = setInterval(() => {
        setBroadcastTime(prev => prev + 1)
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isBroadcasting])

  const handleStartBroadcasting = async () => {
    if (!isBroadcasting) {
      try {
        await startRecording((audioData) => {
          console.log('[v0] Audio chunk received:', audioData.substring(0, 50) + '...')
        })
        setIsBroadcasting(true)
        setBroadcastTime(0)
      } catch (err) {
        console.error('Failed to start recording:', err)
      }
    }
  }

  const handleStopBroadcasting = () => {
    stopRecording()
    setIsBroadcasting(false)
  }

  const handleCopyCode = async () => {
    const success = await copyToClipboard(code || '')
    if (success) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleLeaveRoom = () => {
    handleStopBroadcasting()
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-black/50">
      <div className="max-w-6xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold">🎤 Live Classroom</h1>
            <p className="text-muted">Teacher View</p>
          </div>
          <div className="text-right">
            {broadcastTime > 0 && (
              <p className="text-2xl font-bold text-accent">{formatTime(broadcastTime * 1000)}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Video Section */}
          <div className="lg:col-span-2">
            <Card className="aspect-video bg-black/60 flex items-center justify-center relative overflow-hidden">
              <div className="text-center">
                <div className="text-6xl mb-4">📹</div>
                <p className="text-muted">Camera preview (mock)</p>
                <p className="text-xs text-muted/60 mt-2">WebRTC video would appear here</p>
              </div>
              {isBroadcasting && (
                <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-warning rounded-full">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  <span className="text-xs font-semibold">LIVE</span>
                </div>
              )}
            </Card>

            {/* Controls */}
            <div className="mt-4 flex gap-2 flex-wrap">
              <Button
                variant={isRecording ? 'danger' : 'primary'}
                size="lg"
                onClick={isRecording ? handleStopBroadcasting : handleStartBroadcasting}
                className="flex-1"
              >
                {isRecording ? (
                  <>
                    <MicOff size={20} />
                    Stop Broadcast
                  </>
                ) : (
                  <>
                    <Mic size={20} />
                    Start Broadcasting
                  </>
                )}
              </Button>
              <Button
                variant="secondary"
                size="lg"
                className="flex-1"
              >
                <Video size={20} />
                Camera
              </Button>
              <Button
                variant="ghost"
                size="lg"
                onClick={handleLeaveRoom}
              >
                <Phone size={20} />
                Leave
              </Button>
            </div>

            {micError && (
              <div className="mt-4 p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
                {micError}
              </div>
            )}
          </div>

          {/* Right Sidebar */}
          <div className="space-y-4">
            {/* Room Info */}
            <Card>
              <p className="text-sm text-muted mb-2">Broadcasting to</p>
              <p className="text-3xl font-bold">{studentCount}</p>
              <p className="text-xs text-muted">students</p>
            </Card>

            {/* Room Code */}
            <Card>
              <p className="text-sm text-muted mb-2">Room Code</p>
              <p className="text-2xl font-bold tracking-widest mb-2">{code}</p>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyCode}
                className="w-full gap-2"
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
                {copied ? 'Copied!' : 'Copy'}
              </Button>
            </Card>

            {/* Broadcast Status */}
            <Card className={isBroadcasting ? 'border-accent' : ''}>
              <div className="flex items-center gap-2 mb-2">
                {isBroadcasting ? (
                  <>
                    <div className="w-3 h-3 bg-accent rounded-full animate-pulse" />
                    <span className="font-semibold text-accent">Broadcasting</span>
                  </>
                ) : (
                  <>
                    <div className="w-3 h-3 bg-muted rounded-full" />
                    <span className="font-semibold">Ready to broadcast</span>
                  </>
                )}
              </div>
              <p className="text-xs text-muted">
                {isBroadcasting
                  ? 'Your audio is being sent to students'
                  : 'Click "Start Broadcasting" to begin'}
              </p>
            </Card>
          </div>
        </div>

        {/* Caption Panel */}
        <div>
          <h2 className="text-xl font-bold mb-4">Live Captions</h2>
          <CaptionPanel
            captions={captions}
            fontSize="medium"
            showEmojis={true}
            showTranslation={false}
            language="en"
            maxHeight="max-h-80"
          />
        </div>
      </div>
    </div>
  )
}
