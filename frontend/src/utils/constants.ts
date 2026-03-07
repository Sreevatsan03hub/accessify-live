export const API_BASE = 'http://localhost:8001/api/v1'
export const WS_BASE = 'ws://localhost:8001/ws'

export const LANGUAGES = {
  en: { label: 'English', flag: '🇺🇸' },
  hi: { label: 'हिंदी', flag: '🇮🇳' },
  ta: { label: 'தமிழ்', flag: '🇮🇳' },
  te: { label: 'తెలుగు', flag: '🇮🇳' },
}

export const CAPTION_SIZES = {
  small: { value: 'small', label: 'A-', size: '14px' },
  medium: { value: 'medium', label: 'A', size: '18px' },
  large: { value: 'large', label: 'A+', size: '22px' },
  xl: { value: 'xl', label: 'A++', size: '26px' },
}

export const EMOJI_KEYWORDS = {
  exam: '📘',
  assignment: '📝',
  deadline: '⏰',
  urgent: '⚠️',
  important: '🔑',
  question: '❓',
  answer: '✅',
  homework: '📚',
  test: '📊',
  project: '🎯',
  lecture: '🎓',
  class: '🏫',
}

export const TONE_COLORS = {
  positive: { emoji: '😊', color: '#00D4AA', bg: 'rgba(0, 212, 170, 0.15)' },
  urgent: { emoji: '⚠️', color: '#FF6B6B', bg: 'rgba(255, 107, 107, 0.15)' },
  neutral: { emoji: '😐', color: '#A0AEC0', bg: 'rgba(160, 174, 192, 0.15)' },
  question: { emoji: '❓', color: '#6C63FF', bg: 'rgba(108, 99, 255, 0.15)' },
}

export const SOUND_EVENTS = {
  APPLAUSE: '👏 APPLAUSE',
  LAUGHTER: '😂 LAUGHTER',
  ALARM: '🚨 ALARM',
  BELL: '🔔 BELL',
  DOOR: '🚪 DOOR KNOCK',
  PHONE: '📞 PHONE RING',
  NOTIFICATION: '📬 NOTIFICATION',
}

export const MOCK_CAPTIONS = [
  {
    id: 1,
    text: 'Welcome to today\'s lecture on machine learning.',
    simplified_text: 'Welcome to machine learning class.',
    keywords: [{ keyword: 'machine learning', emoji: '🔑', score: 0.95 }],
    tone: { emotion: 'positive', intent: 'statement', emoji: '😊' },
    translation: null,
    sound_event: null,
    timestamp: Date.now(),
  },
  {
    id: 2,
    text: 'The exam will be held next Friday. Please submit your assignment before that.',
    simplified_text: 'Exam is next Friday. Submit assignment before then.',
    keywords: [
      { keyword: 'exam', emoji: '📘', score: 0.98 },
      { keyword: 'assignment', emoji: '📝', score: 0.97 },
      { keyword: 'Friday', emoji: '📅', score: 0.85 },
    ],
    tone: { emotion: 'neutral', intent: 'urgent', emoji: '⚠️' },
    translation: { text: 'தேர்வு அடுத்த வெள்ளி.', target_language: 'ta' },
    sound_event: null,
    timestamp: Date.now() + 5000,
  },
  {
    id: 3,
    text: '[APPLAUSE]',
    simplified_text: '',
    keywords: [],
    tone: { emotion: 'positive', intent: 'event', emoji: '👏' },
    translation: null,
    sound_event: 'APPLAUSE 👏',
    timestamp: Date.now() + 10000,
  },
]
