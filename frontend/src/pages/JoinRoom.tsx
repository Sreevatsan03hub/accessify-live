import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Card } from '../components/ui/Card'
import { useUser } from '../context/UserContext'
import { generateParticipantId } from '../utils/helpers'

export default function JoinRoom() {
  const [roomCode, setRoomCode] = useState('')
  const [name, setName] = useState('')
  const [language, setLanguage] = useState('en')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const { user } = useUser()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!roomCode.trim() || !name.trim()) {
      setError('Please fill in all fields')
      return
    }

    if (roomCode.length !== 6) {
      setError('Room code must be 6 characters')
      return
    }

    try {
      setLoading(true)
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 800))

      const participantId = generateParticipantId()
      navigate(`/room/${roomCode.toUpperCase()}/student/${participantId}`, {
        state: { name, language },
      })
    } catch (err) {
      setError('Failed to join room. Please check the room code.')
      console.error('Join room error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-2">Join a Live Class</h1>
      <p className="text-muted mb-8">Enter the room code provided by your teacher</p>

      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          <Input
            label="Room Code"
            placeholder="e.g., ABC123"
            value={roomCode.toUpperCase()}
            onChange={(e) => setRoomCode(e.target.value.toUpperCase())}
            maxLength={6}
            required
            helperText="Ask your teacher for the 6-character room code"
          />

          <Input
            label="Your Name"
            placeholder="John Doe"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />

          <Select
            label="Caption Language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            options={[
              { value: 'en', label: '🇬🇧 English' },
              { value: 'hi', label: '🇮🇳 हिंदी (Hindi)' },
              { value: 'ta', label: '🇮🇳 தமிழ் (Tamil)' },
              { value: 'te', label: '🇮🇳 తెలుగు (Telugu)' },
            ]}
          />

          {error && (
            <div className="p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
              {error}
            </div>
          )}

          <Button type="submit" size="lg" className="w-full" loading={loading}>
            Join Class
          </Button>
        </form>
      </Card>

      <Card className="mt-6 border-accent/50 bg-accent/5">
        <h3 className="font-bold mb-3">What to expect:</h3>
        <ul className="space-y-2 text-sm text-muted">
          <li className="flex gap-2">
            <span className="text-accent">✓</span>
            <span>Real-time captions of what your teacher is saying</span>
          </li>
          <li className="flex gap-2">
            <span className="text-accent">✓</span>
            <span>Automatic translation to your preferred language</span>
          </li>
          <li className="flex gap-2">
            <span className="text-accent">✓</span>
            <span>Highlighted keywords and important concepts</span>
          </li>
          <li className="flex gap-2">
            <span className="text-accent">✓</span>
            <span>Sound event notifications (applause, laughter, etc.)</span>
          </li>
        </ul>
      </Card>
    </div>
  )
}
