import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useUser } from '../context/UserContext'
import { getSessions } from '../services/sessionService'
import { formatDate } from '../utils/helpers'

interface Session {
  id: string
  title: string
  type: 'video' | 'live'
  created_at: string
  caption_count: number
}

export default function Dashboard() {
  const { user, logout } = useUser()
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!user) {
      navigate('/login')
      return
    }

    const loadSessions = async () => {
      try {
        setLoading(true)
        // In demo mode, we'll show mock sessions instead of calling the API
        setSessions([
          {
            id: '1',
            title: 'Machine Learning Fundamentals',
            type: 'live',
            created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            caption_count: 245,
          },
          {
            id: '2',
            title: 'Web Development Best Practices',
            type: 'video',
            created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
            caption_count: 312,
          },
        ])
      } catch (error) {
        console.error('Failed to load sessions:', error)
      } finally {
        setLoading(false)
      }
    }

    loadSessions()
  }, [user, navigate])

  if (!user) {
    return null
  }

  const dashboardItems = [
    {
      icon: '🎥',
      title: 'Start Live Class',
      description: 'Create a new live classroom and start teaching',
      href: '/room/create',
    },
    {
      icon: '🔗',
      title: 'Join a Class',
      description: 'Enter a room code to join a live session',
      href: '/room/join',
    },
    {
      icon: '📁',
      title: 'Upload Video',
      description: 'Upload a video file for automatic captioning',
      href: '/upload',
    },
    {
      icon: '📚',
      title: 'My Videos',
      description: 'View and manage your uploaded videos',
      href: '/history',
    },
    {
      icon: '📝',
      title: 'Caption History',
      description: 'Review and download your captions',
      href: '/history',
    },
    {
      icon: '⚙️',
      title: 'Settings',
      description: 'Customize your preferences and accessibility',
      href: '/settings',
    },
  ]

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      {/* Welcome Section */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold mb-2">
          Welcome back, {user.name}! 👋
        </h1>
        <p className="text-muted">
          You're logged in as <span className="font-semibold">{user.role}</span> • Language: <span className="font-semibold">{user.language.toUpperCase()}</span>
        </p>
      </div>

      {/* Dashboard Actions Grid */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboardItems.map((item) => (
            <Link key={item.href} to={item.href}>
              <Card icon={item.icon} title={item.title} className="h-full hover:shadow-lg">
                <p className="text-muted text-sm">{item.description}</p>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* Recent Sessions */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Recent Sessions</h2>
        {loading ? (
          <Card>
            <p className="text-center text-muted py-8">Loading sessions...</p>
          </Card>
        ) : sessions.length === 0 ? (
          <Card>
            <p className="text-center text-muted py-8">No sessions yet. Start your first class or upload a video!</p>
          </Card>
        ) : (
          <div className="space-y-4">
            {sessions.map((session) => (
              <Card key={session.id} className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="font-bold text-lg mb-1">{session.title}</h3>
                  <p className="text-sm text-muted">
                    {session.caption_count} captions • {formatDate(session.created_at)} • {session.type === 'live' ? '🔴 Live' : '📹 Video'}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Link to="/player">
                    <Button variant="ghost" size="sm">
                      View
                    </Button>
                  </Link>
                  <Button variant="ghost" size="sm">Download</Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </section>

      {/* Demo Banner */}
      <div className="mt-12 p-6 bg-primary/10 border border-primary/30 rounded-xl">
        <h3 className="font-bold mb-2">Demo Mode</h3>
        <p className="text-sm text-muted mb-4">
          You're in demo mode. Live classroom and video upload features are coming soon as the backend is configured.
        </p>
        <Button variant="ghost" size="sm" onClick={logout}>
          Logout
        </Button>
      </div>
    </div>
  )
}
