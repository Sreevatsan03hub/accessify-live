import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Card } from '../components/ui/Card'
import { useUser } from '../context/UserContext'
import { generateRoomCode, copyToClipboard } from '../utils/helpers'
import { Copy, Check } from 'lucide-react'

export default function CreateRoom() {
  const [title, setTitle] = useState('')
  const [teacherName, setTeacherName] = useState('')
  const [roomCode, setRoomCode] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const { user } = useUser()

  const handleCreateRoom = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!title.trim() || !teacherName.trim()) {
      setError('Please fill in all fields')
      return
    }

    try {
      setLoading(true)
      // In demo mode, generate a mock room code
      const code = generateRoomCode()
      setRoomCode(code)

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800))
    } catch (err) {
      setError('Failed to create room. Please try again.')
      console.error('Room creation error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleCopyCode = async () => {
    const success = await copyToClipboard(roomCode)
    if (success) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleStartBroadcasting = () => {
    navigate(`/room/${roomCode}/teacher`)
  }

  if (!user || user.role !== 'teacher') {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12">
        <Card>
          <p className="text-center text-muted">Only teachers can create rooms</p>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-2">Create a Live Class</h1>
      <p className="text-muted mb-8">Set up your classroom and invite students</p>

      {!roomCode ? (
        <Card>
          <form onSubmit={handleCreateRoom} className="space-y-6">
            <Input
              label="Class Title"
              placeholder="e.g., Machine Learning Fundamentals"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />

            <Input
              label="Teacher Name"
              placeholder="Your name"
              value={teacherName}
              onChange={(e) => setTeacherName(e.target.value)}
              required
            />

            {error && (
              <div className="p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
                {error}
              </div>
            )}

            <Button type="submit" size="lg" className="w-full" loading={loading}>
              Create Room
            </Button>
          </form>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card className="border-accent/50 bg-accent/5">
            <div className="text-center mb-6">
              <p className="text-sm text-muted mb-2">Your Room is Ready!</p>
              <h2 className="text-3xl font-bold">{title}</h2>
            </div>

            <div className="bg-black/40 rounded-lg p-6 mb-6 text-center">
              <p className="text-sm text-muted mb-2">Room Code</p>
              <p className="text-5xl font-bold tracking-widest mb-4">{roomCode}</p>
              <Button
                variant="secondary"
                size="lg"
                onClick={handleCopyCode}
                className="gap-2 mb-4"
              >
                {copied ? (
                  <>
                    <Check size={18} />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy size={18} />
                    Copy Code
                  </>
                )}
              </Button>
            </div>

            <div className="bg-primary/10 border border-primary/30 rounded-lg p-4 mb-6">
              <p className="text-sm font-semibold mb-2">Shareable Link</p>
              <p className="text-xs text-muted break-all">
                {window.location.origin}/room/join?code={roomCode}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Button
                variant="secondary"
                size="lg"
                onClick={handleStartBroadcasting}
              >
                🎤 Start Broadcasting
              </Button>
              <Button
                variant="ghost"
                size="lg"
                onClick={() => {
                  setRoomCode('')
                  setTitle('')
                  setTeacherName('')
                }}
              >
                Create Another Room
              </Button>
            </div>
          </Card>

          <Card className="border-border/50">
            <h3 className="font-bold mb-3">How to share with students:</h3>
            <ol className="space-y-2 text-sm text-muted list-decimal list-inside">
              <li>Copy the room code above</li>
              <li>Share it with your students verbally or via email</li>
              <li>Students use the code on the "Join Class" page</li>
              <li>Their captions will appear in real-time as you speak</li>
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
