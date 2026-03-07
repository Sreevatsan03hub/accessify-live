import { Link } from 'react-router-dom';
import { CaptionPanel } from '../components/captions/CaptionPanel';
import { MOCK_CAPTIONS } from '../utils/mockData';
import { useUser } from '../context/UserContext';
import { Mic, Globe, Star, Bell, ArrowRight, Zap, Shield, Users } from 'lucide-react';

const FEATURES = [
  {
    id: 'stt',
    Icon: Mic,
    iconBg: '#DBEAFE',
    iconColor: '#2563EB',
    title: 'Real-Time Captions',
    description: 'Instant speech-to-text as your teacher speaks — zero delay.',
  },
  {
    id: 'translate',
    Icon: Globe,
    iconBg: '#D1FAE5',
    iconColor: '#059669',
    title: 'Multilingual',
    description: 'English, Hindi, Tamil, Telugu — captions in your language.',
  },
  {
    id: 'keywords',
    Icon: Star,
    iconBg: '#FEF3C7',
    iconColor: '#D97706',
    title: 'Smart Keywords',
    description: 'Important words highlighted automatically for quick focus.',
  },
  {
    id: 'sounds',
    Icon: Bell,
    iconBg: '#FCE7F3',
    iconColor: '#DB2777',
    title: 'Sound Events',
    description: 'Applause, laughter, alarms — every sound gets a label.',
  },
];

const STATS = [
  { Icon: Zap, value: '< 1s', label: 'Latency' },
  { Icon: Shield, value: '97%', label: 'Accuracy' },
  { Icon: Users, value: '1000+', label: 'Students' },
  { Icon: Globe, value: '3', label: 'Languages' },
];

export function Landing() {
  const { user } = useUser();

  return (
    <div className="min-h-screen" style={{ background: '#0F172A' }}>

      {/* ── HERO ─────────────────────────────────── */}
      <section
        className="relative overflow-hidden"
        style={{ background: 'linear-gradient(135deg,#0F172A 0%,#1E3A8A 100%)' }}
      >
        {/* Subtle grid overlay */}
        <div className="absolute inset-0 opacity-10"
          style={{ backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)', backgroundSize: '48px 48px' }} />

        <div className="relative max-w-5xl mx-auto px-6 py-28 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-7 text-sm font-medium text-blue-300"
            style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}>
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            AI-Powered Classroom Platform
          </div>

          <h1 className="font-extrabold text-white mb-6 leading-none"
            style={{ fontSize: '64px', letterSpacing: '-0.03em' }}>
            Accessify
          </h1>

          <p className="text-2xl font-semibold mb-3" style={{ color: '#94A3B8' }}>
            AI-Powered Accessible Learning
          </p>
          <p className="text-base mb-10 max-w-xl mx-auto" style={{ color: '#64748B' }}>
            Real-time captions, translations &amp; emoji context for every classroom —
            so no student is ever left behind.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap gap-3 justify-center mb-16">
            {user ? (
              <Link to="/dashboard">
                <button
                  className="flex items-center gap-2 font-semibold rounded-xl transition-all"
                  style={{
                    padding: '14px 28px', fontSize: '16px', background: '#2563EB',
                    color: '#fff', boxShadow: '0 10px 25px rgba(37,99,235,0.35)'
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = '#1D4ED8'}
                  onMouseLeave={e => e.currentTarget.style.background = '#2563EB'}
                >
                  🎓 Go to Dashboard <ArrowRight size={16} />
                </button>
              </Link>
            ) : (
              <>
                <Link to="/room/join">
                  <button
                    className="flex items-center gap-2 font-semibold rounded-xl transition-all"
                    style={{
                      padding: '14px 28px', fontSize: '16px', background: '#2563EB',
                      color: '#fff', boxShadow: '0 10px 25px rgba(37,99,235,0.35)'
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = '#1D4ED8'}
                    onMouseLeave={e => e.currentTarget.style.background = '#2563EB'}
                  >
                    🎥 Join Live Class <ArrowRight size={16} />
                  </button>
                </Link>
                <Link to="/register">
                  <button
                    className="flex items-center gap-2 font-semibold rounded-xl border transition-all"
                    style={{
                      padding: '14px 28px', fontSize: '16px', color: '#e2e8f0',
                      background: 'rgba(255,255,255,0.07)', borderColor: 'rgba(255,255,255,0.2)'
                    }}
                  >
                    Sign Up Free
                  </button>
                </Link>
              </>
            )}
          </div>

          {/* Stats strip */}
          <div className="flex flex-wrap justify-center gap-10">
            {STATS.map(({ Icon, value, label }) => (
              <div key={label} className="text-center">
                <Icon size={20} className="mx-auto mb-1" style={{ color: '#60A5FA' }} />
                <p className="text-2xl font-bold text-white">{value}</p>
                <p className="text-xs" style={{ color: '#64748B' }}>{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── LIVE DEMO ─────────────────────────────── */}
      <section className="py-20" style={{ background: '#0F172A' }}>
        <div className="max-w-3xl mx-auto px-6">
          <p className="text-xs font-bold uppercase tracking-widest text-blue-400 text-center mb-3">Live Preview</p>
          <h2 className="text-3xl font-bold text-white text-center mb-10" style={{ letterSpacing: '-0.02em' }}>
            See captions in action
          </h2>
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.08)', background: '#1E293B' }}>
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <span className="w-3 h-3 rounded-full bg-red-500" />
              <span className="w-3 h-3 rounded-full bg-yellow-400" />
              <span className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-xs text-slate-400 ml-2">Live Caption Demo</span>
            </div>
            <div className="p-6">
              <CaptionPanel
                captions={MOCK_CAPTIONS.slice(0, 2)}
                showEmojis={true}
                showTranslation={false}
                maxHeight="max-h-56"
              />
            </div>
          </div>
        </div>
      </section>

      {/* ── FEATURES ─────────────────────────────── */}
      <section className="py-20" style={{ background: '#1E293B' }}>
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-xs font-bold uppercase tracking-widest text-blue-400 text-center mb-3">Why Accessify?</p>
          <h2 className="text-3xl font-bold text-white text-center mb-12" style={{ letterSpacing: '-0.02em' }}>
            Everything your classroom needs
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            {FEATURES.map(({ id, Icon, iconBg, iconColor, title, description }) => (
              <div key={id}
                className="rounded-2xl p-6 transition-all duration-200 hover:-translate-y-1 cursor-default"
                style={{
                  background: '#0F172A', border: '1px solid rgba(255,255,255,0.07)',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
                }}>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-5"
                  style={{ background: iconBg }}>
                  <Icon size={22} style={{ color: iconColor }} />
                </div>
                <h3 className="text-base font-bold text-white mb-2">{title}</h3>
                <p className="text-sm" style={{ color: '#64748B', lineHeight: '1.6' }}>{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── BOTTOM CTA ───────────────────────────── */}
      <section className="py-24" style={{ background: 'linear-gradient(135deg,#0F172A,#1E3A8A)' }}>
        <div className="max-w-2xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-extrabold text-white mb-4" style={{ letterSpacing: '-0.02em' }}>
            Ready to Get Started?
          </h2>
          <p className="mb-10" style={{ color: '#94A3B8', fontSize: '17px' }}>
            Join thousands of students receiving accessible, engaging captions in real-time.
          </p>

          {user ? (
            <Link to="/dashboard">
              <button
                className="font-semibold rounded-xl transition-all"
                style={{
                  padding: '14px 32px', fontSize: '16px', background: '#2563EB',
                  color: '#fff', boxShadow: '0 10px 25px rgba(37,99,235,0.35)'
                }}
              >
                Go to Dashboard →
              </button>
            </Link>
          ) : (
            <div className="flex flex-wrap gap-3 justify-center">
              <Link to="/register">
                <button className="font-semibold rounded-xl transition-all"
                  style={{
                    padding: '14px 28px', fontSize: '16px', background: '#2563EB',
                    color: '#fff', boxShadow: '0 10px 25px rgba(37,99,235,0.35)'
                  }}>
                  Sign Up Free
                </button>
              </Link>
              <Link to="/login">
                <button className="font-semibold rounded-xl border transition-all"
                  style={{
                    padding: '14px 28px', fontSize: '16px', color: '#e2e8f0',
                    background: 'rgba(255,255,255,0.07)', borderColor: 'rgba(255,255,255,0.2)'
                  }}>
                  Already have an account? Sign In
                </button>
              </Link>
            </div>
          )}
        </div>
      </section>

    </div>
  );
}
