import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { VideoIcon, Globe, Sparkles, Volume2 } from 'lucide-react'

export default function Landing() {
  const features = [
    {
      icon: '🎙️',
      title: 'Real-Time Captions',
      description: 'Instant speech-to-text as teacher speaks',
    },
    {
      icon: '🌐',
      title: 'Multilingual',
      description: 'English, Hindi, Tamil, Telugu',
    },
    {
      icon: '⭐',
      title: 'Smart Keywords',
      description: 'Important words highlighted automatically',
    },
    {
      icon: '🔔',
      title: 'Sound Awareness',
      description: '[APPLAUSE 👏] [LAUGHTER 😂] [ALARM 🚨]',
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-black/50">
      {/* Hero Section */}
      <section className="max-w-6xl mx-auto px-4 py-20 text-center">
        <div className="mb-8 text-6xl md:text-7xl">🎓</div>
        <h1 className="text-4xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          Accessify
        </h1>
        <p className="text-xl md:text-2xl text-muted mb-4 max-w-2xl mx-auto">
          AI-Powered Accessible Learning
        </p>
        <p className="text-lg text-muted/80 mb-12 max-w-2xl mx-auto">
          Real-time captions, translations & emoji context for every classroom
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
          <Link to="/room/join">
            <Button size="lg" className="w-full sm:w-auto">
              <VideoIcon size={20} />
              Join Live Class
            </Button>
          </Link>
          <Link to="/upload">
            <Button variant="secondary" size="lg" className="w-full sm:w-auto">
              <span className="text-xl">📁</span>
              Upload Video
            </Button>
          </Link>
        </div>

        {/* Caption Demo */}
        <div className="bg-black/60 border border-border rounded-xl p-8 mb-16 max-w-2xl mx-auto">
          <p className="text-sm text-muted mb-4">Live Caption Demo</p>
          <div className="space-y-3">
            <div className="p-3 bg-primary/20 rounded-lg">
              <p className="font-semibold">Welcome everyone to today's class</p>
              <p className="text-sm text-muted mt-1">😊 Positive</p>
            </div>
            <div className="p-3 bg-accent/20 rounded-lg">
              <p className="font-semibold">The assignment is due on Friday</p>
              <p className="text-sm text-muted mt-1">
                <span className="inline-block mr-2">⚠️ Urgent</span>
                <span className="inline-block px-2 py-1 bg-accent/30 rounded text-xs">📅 Friday</span>
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-6xl mx-auto px-4 py-20">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">
          Powerful Features for Accessible Learning
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, idx) => (
            <Card key={idx} icon={feature.icon} title={feature.title}>
              <p className="text-muted">{feature.description}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border mt-20">
        <div className="max-w-6xl mx-auto px-4 py-8 text-center text-muted">
          <p>Built with accessibility in mind for Deaf and Hard-of-Hearing students</p>
          <p className="text-sm mt-2">© 2024 Accessify. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
