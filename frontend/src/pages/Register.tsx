import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { useUser } from '../context/UserContext'

export default function Register() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'student',
    language: 'en',
  })
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const { setUser } = useUser()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!formData.name || !formData.email || !formData.password || !formData.confirmPassword) {
      setError('Please fill in all fields')
      return
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    // Mock registration
    setError('Coming soon: Backend registration not yet implemented')

    setTimeout(() => {
      const newUser = {
        name: formData.name,
        email: formData.email,
        role: formData.role as 'teacher' | 'student',
        language: formData.language as 'en' | 'hi' | 'ta' | 'te',
        captionSize: 'medium' as const,
        showEmojis: true,
      }
      setUser(newUser)
      navigate('/dashboard')
    }, 2000)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-background to-black/50">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="text-5xl mb-4">🎓</div>
          <h1 className="text-3xl font-bold">Create Your Account</h1>
          <p className="text-muted mt-2">Join Accessify to start learning</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Full Name"
            name="name"
            placeholder="John Doe"
            value={formData.name}
            onChange={handleChange}
            required
          />

          <Input
            label="Email Address"
            name="email"
            type="email"
            placeholder="you@example.com"
            value={formData.email}
            onChange={handleChange}
            required
          />

          <Select
            label="I am a"
            name="role"
            value={formData.role}
            onChange={handleChange}
            options={[
              { value: 'student', label: 'Student' },
              { value: 'teacher', label: 'Teacher' },
            ]}
          />

          <Select
            label="Preferred Language"
            name="language"
            value={formData.language}
            onChange={handleChange}
            options={[
              { value: 'en', label: 'English' },
              { value: 'hi', label: 'हिंदी (Hindi)' },
              { value: 'ta', label: 'தமிழ் (Tamil)' },
              { value: 'te', label: 'తెలుగు (Telugu)' },
            ]}
          />

          <Input
            label="Password"
            name="password"
            type="password"
            placeholder="••••••••"
            value={formData.password}
            onChange={handleChange}
            helperText="At least 6 characters"
            required
          />

          <Input
            label="Confirm Password"
            name="confirmPassword"
            type="password"
            placeholder="••••••••"
            value={formData.confirmPassword}
            onChange={handleChange}
            required
          />

          {error && (
            <div className="p-3 bg-warning/20 border border-warning rounded-lg text-warning text-sm">
              {error}
            </div>
          )}

          <Button type="submit" size="lg" className="w-full">
            Create Account
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-muted">
            Already have an account?{' '}
            <Link to="/login" className="text-accent hover:underline font-semibold">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
