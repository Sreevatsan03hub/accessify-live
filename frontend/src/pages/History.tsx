import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { formatDate } from '../utils/helpers'
import { Eye, Download, Trash2 } from 'lucide-react'

interface Session {
  id: string
  title: string
  type: 'video' | 'live'
  created_at: string
  caption_count: number
  duration?: number
}

export default function History() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const loadSessions = async () => {
      try {
        setLoading(true)
        // Simulate loading sessions
        await new Promise(resolve => setTimeout(resolve, 500))
        setSessions([
          {
            id: '1',
            title: 'Machine Learning Fundamentals',
            type: 'live',
            created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            caption_count: 245,
            duration: 3600,
          },
          {
            id: '2',
            title: 'Web Development Best Practices',
            type: 'video',
            created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
            caption_count: 312,
            duration: 2700,
          },
          {
            id: '3',
            title: 'Advanced Python Techniques',
            type: 'video',
            created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
            caption_count: 428,
            duration: 5400,
          },
        ])
      } catch (error) {
        console.error('Failed to load sessions:', error)
      } finally {
        setLoading(false)
      }
    }

    loadSessions()
  }, [])

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
      return
    }

    setDeletingId(id)
    try {
      await new Promise(resolve => setTimeout(resolve, 500))
      setSessions(prev => prev.filter(session => session.id !== id))
    } catch (error) {
      console.error('Failed to delete session:', error)
    } finally {
      setDeletingId(null)
    }
  }

  const handleDownload = async (id: string, format: 'srt' | 'vtt' | 'txt' | 'summary') => {
    try {
      // Simulate download
      const content = `Sample ${format.toUpperCase()} captions`
      const filename = `captions_${id}.${format}`
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

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    }
    return `${minutes}m`
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold">Caption History</h1>
          <p className="text-muted">View and manage your captured captions</p>
        </div>
        <Link to="/upload">
          <Button size="lg">Upload New Video</Button>
        </Link>
      </div>

      {loading ? (
        <Card>
          <p className="text-center text-muted py-8">Loading sessions...</p>
        </Card>
      ) : sessions.length === 0 ? (
        <Card>
          <div className="text-center py-12">
            <p className="text-muted mb-4">No caption sessions yet</p>
            <Link to="/upload">
              <Button>Upload Your First Video</Button>
            </Link>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {sessions.map((session) => (
            <Card key={session.id} className="hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-bold text-lg truncate">{session.title}</h3>
                    <span className={`text-xs px-2 py-1 rounded-full font-semibold ${
                      session.type === 'live'
                        ? 'bg-warning/20 text-warning'
                        : 'bg-accent/20 text-accent'
                    }`}>
                      {session.type === 'live' ? '🔴 Live' : '📹 Video'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm text-muted">
                    <div>
                      <p className="text-xs">Date</p>
                      <p className="font-semibold text-foreground">{formatDate(session.created_at)}</p>
                    </div>
                    <div>
                      <p className="text-xs">Captions</p>
                      <p className="font-semibold text-foreground">{session.caption_count}</p>
                    </div>
                    <div>
                      <p className="text-xs">Duration</p>
                      <p className="font-semibold text-foreground">{formatDuration(session.duration)}</p>
                    </div>
                    <div>
                      <p className="text-xs">Status</p>
                      <p className="font-semibold text-accent">Ready</p>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 flex-wrap">
                  <Link to="/player" state={{ sessionId: session.id }}>
                    <Button variant="secondary" size="sm" className="gap-2">
                      <Eye size={16} />
                      View
                    </Button>
                  </Link>
                  <div className="relative group">
                    <Button variant="ghost" size="sm">
                      <Download size={16} />
                    </Button>
                    <div className="absolute right-0 top-full mt-1 hidden group-hover:flex flex-col gap-1 bg-black/80 border border-border rounded-lg p-2 z-10 min-w-max">
                      {(['srt', 'vtt', 'txt', 'summary'] as const).map((format) => (
                        <button
                          key={format}
                          onClick={() => handleDownload(session.id, format)}
                          className="px-3 py-1.5 text-xs font-semibold hover:bg-primary/20 rounded transition-colors text-left"
                        >
                          {format.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => handleDelete(session.id)}
                    disabled={deletingId === session.id}
                    className="gap-2"
                  >
                    <Trash2 size={16} />
                    Delete
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
