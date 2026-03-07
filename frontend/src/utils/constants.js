export const LANGUAGES = {
  en: { name: 'English', flag: '🇬🇧' },
  hi: { name: 'हिंदी', flag: '🇮🇳' },
  ta: { name: 'தமிழ்', flag: '🇮🇳' },
  te: { name: 'తెలుగు', flag: '🇮🇳' },
};

export const CAPTION_SIZES = {
  small: 'A-',
  medium: 'A',
  large: 'A+',
  xl: 'A++',
};

export const COLOR_PALETTE = {
  primary: '#6C63FF',
  accent: '#00D4AA',
  warning: '#FF6B6B',
  success: '#00D4AA',
  background: {
    dark: '#0F0F1A',
    light: '#F8F9FF',
  },
  caption: 'rgba(0,0,0,0.85)',
};

export const API_BASE = 'http://localhost:8001';

export const TONE_COLORS = {
  positive: '#00D4AA',
  negative: '#FF6B6B',
  neutral: '#A0AEC0',
  urgent: '#FF6B6B',
  question: '#6C63FF',
};

export const TONE_EMOJIS = {
  positive: '😊',
  negative: '😞',
  neutral: '😐',
  urgent: '⚠️',
  question: '❓',
};

export const SOUND_EVENTS = {
  APPLAUSE: '👏',
  LAUGHTER: '😂',
  ALARM: '🚨',
  SILENCE: '🤐',
  BACKGROUND_NOISE: '📢',
};

export const FEATURE_CARDS = [
  {
    id: 1,
    icon: '🎙️',
    title: 'Real-Time Captions',
    description: 'Instant speech-to-text as teacher speaks',
  },
  {
    id: 2,
    icon: '🌐',
    title: 'Multilingual',
    description: 'English, Hindi, Tamil, Telugu',
  },
  {
    id: 3,
    icon: '⭐',
    title: 'Smart Keywords',
    description: 'Important words highlighted automatically',
  },
  {
    id: 4,
    icon: '🔔',
    title: 'Sound Awareness',
    description: '[APPLAUSE 👏] [LAUGHTER 😂] [ALARM 🚨]',
  },
];

export const DASHBOARD_CARDS = [
  {
    id: 1,
    icon: '🎥',
    title: 'Start Live Class',
    description: 'Create a new classroom',
    link: '/room/create',
  },
  {
    id: 2,
    icon: '🔗',
    title: 'Join a Class',
    description: 'Enter room code',
    link: '/room/join',
  },
  {
    id: 3,
    icon: '📁',
    title: 'Upload Video',
    description: 'Add captions to recordings',
    link: '/upload',
  },
  {
    id: 4,
    icon: '📼',
    title: 'My Videos',
    description: 'Replay uploaded videos with AI captions',
    link: '/my-videos',
  },
  {
    id: 5,
    icon: '🎥',
    title: 'Live Session History',
    description: 'Transcripts from past live classes',
    link: '/history',
  },
  {
    id: 6,
    icon: '⚙️',
    title: 'Settings',
    description: 'Customize preferences',
    link: '/settings',
  },
];
