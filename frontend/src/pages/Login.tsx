import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { useUser } from '../context/UserContext'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const { setUser } = useUser()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!email || !password) {
      setError('Please fill in all fields')
      return
    }

    // Mock authentication - show coming soon message
    setError('Coming soon: Backend authentication not yet implemented')

    // In a real app, this would call the API
    // For now, just navigate to dashboard with mock data
    setTimeout(() => {
      const mockUser = {
        name: email.split('@')[0],
        email,
        role: 'student' as const,
        language: 'en' as const,
        captionSize: 'medium' as const,
        showEmojis: true,
      }
      setUser(mockUser)
      navigate('/dashboard')
    }, 2000)
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-background to-black/50">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="text-5xl mb-4">🎓</div>
          <h1 className="text-3xl font-bold">Sign In to Accessify</h1>
          <p className="text-muted mt-2">Access your classes and captions</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Email Address"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <Input
            label="Password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <div className="flex items-center">
            <input type="checkbox" id="remember" className="rounded" />
            <label htmlFor="remember" className="ml-2 text-sm">
              Remember me
            </label>
          </div>

          {error && (
            <div className="p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
              {error}
            </div>
          )}

          <Button type="submit" size="lg" className="w-full">
            Sign In
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-muted">
            Don't have an account?{' '}
            <Link to="/register" className="text-accent hover:underline font-semibold">
              Sign up
            </Link>
          </p>
        </div>

        <div className="mt-8 p-4 bg-primary/10 border border-primary/30 rounded-lg text-sm">
          <p className="font-semibold mb-2">Demo Mode</p>
          <p className="text-muted">
            Authentication backend is coming soon. Use any email to continue to the dashboard.
          </p>
        </div>
      </div>
    </div>
  )
}
