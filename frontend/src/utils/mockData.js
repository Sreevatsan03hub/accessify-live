export const MOCK_CAPTIONS = [
  {
    id: 1,
    text: 'Welcome to today\'s lecture on machine learning.',
    simplified_text: 'Welcome to machine learning class.',
    keywords: [
      { keyword: 'machine learning', emoji: '🔑' },
      { keyword: 'lecture', emoji: '📚' },
    ],
    tone: { emotion: 'positive', intent: 'statement', emoji: '😊' },
    translation: null,
    sound_event: null,
    timestamp: Date.now(),
  },
  {
    id: 2,
    text: 'The exam will be held next Friday. Please submit your assignment before that.',
    simplified_text: 'Exam next Friday. Submit assignment before then.',
    keywords: [
      { keyword: 'exam', emoji: '📘' },
      { keyword: 'assignment', emoji: '📝' },
      { keyword: 'Friday', emoji: '📅' },
    ],
    tone: { emotion: 'neutral', intent: 'urgent', emoji: '⚠️' },
    translation: {
      text: 'தேர்வு அடுத்த வெள்ளி. அதற்கு முன் வேலை சமர்ப்பிக்கவும்.',
      target_language: 'ta',
    },
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
  {
    id: 4,
    text: 'Please review chapters 3 through 7 for the midterm exam.',
    simplified_text: 'Review chapters 3-7 for the midterm.',
    keywords: [
      { keyword: 'chapters', emoji: '📖' },
      { keyword: 'midterm exam', emoji: '📋' },
      { keyword: 'review', emoji: '👀' },
    ],
    tone: { emotion: 'neutral', intent: 'instruction', emoji: '📢' },
    translation: null,
    sound_event: null,
    timestamp: Date.now() + 15000,
  },
  {
    id: 5,
    text: '[LAUGHTER]',
    simplified_text: '',
    keywords: [],
    tone: { emotion: 'positive', intent: 'event', emoji: '😂' },
    translation: null,
    sound_event: 'LAUGHTER 😂',
    timestamp: Date.now() + 20000,
  },
  {
    id: 6,
    text: 'Are there any questions about the assignment requirements?',
    simplified_text: 'Any questions about the assignment?',
    keywords: [
      { keyword: 'questions', emoji: '❓' },
      { keyword: 'assignment', emoji: '📝' },
    ],
    tone: { emotion: 'neutral', intent: 'question', emoji: '❓' },
    translation: null,
    sound_event: null,
    timestamp: Date.now() + 25000,
  },
];

export const MOCK_SESSIONS = [
  {
    id: 'session_001',
    title: 'Introduction to Machine Learning',
    type: 'video',
    duration: 3600,
    created_at: new Date(Date.now() - 86400000).toISOString(),
    caption_count: 125,
    language: 'en',
  },
  {
    id: 'session_002',
    title: 'Data Structures Class - Week 5',
    type: 'live',
    duration: 5400,
    created_at: new Date(Date.now() - 172800000).toISOString(),
    caption_count: 203,
    language: 'en',
  },
  {
    id: 'session_003',
    title: 'Advanced Python Programming',
    type: 'video',
    duration: 7200,
    created_at: new Date(Date.now() - 259200000).toISOString(),
    caption_count: 456,
    language: 'en',
  },
];

export const MOCK_ROOM_DATA = {
  room_code: 'ABC123',
  title: 'Machine Learning 101',
  teacher_name: 'Dr. Sarah Johnson',
  created_at: new Date().toISOString(),
  participants: [
    { id: 'p1', name: 'John Doe', role: 'student', language: 'en' },
    { id: 'p2', name: 'Jane Smith', role: 'student', language: 'hi' },
    { id: 'p3', name: 'Raj Kumar', role: 'student', language: 'ta' },
  ],
};

export const MOCK_UPLOAD_RESPONSE = {
  success: true,
  filename: 'lecture_001.mp4',
  duration: 45.2,
  session_id: 'session_001',
  transcription: {
    text: 'Welcome to today\'s lecture. Today we will be discussing machine learning fundamentals. The exam will be held next Friday. Please submit your assignments before that.',
    language: 'en',
    processing_time: 12.3,
    vtt: `WEBVTT

00:00:01.000 --> 00:00:05.000
Welcome to today's lecture.

00:00:05.000 --> 00:00:10.000
Today we will be discussing machine learning fundamentals.

00:00:10.000 --> 00:00:15.000
The exam will be held next Friday.

00:00:15.000 --> 00:00:20.000
Please submit your assignments before that.`,
  },
  enrichment: {
    keywords: [
      { keyword: 'exam', emoji: '📘', score: 0.9 },
      { keyword: 'assignment', emoji: '📝', score: 0.85 },
      { keyword: 'machine learning', emoji: '🔑', score: 0.95 },
    ],
  },
  tone: { emotion: 'neutral', intent: 'statement', emoji: '😐' },
  translation: null,
};

export const generateMockCaption = (index = 0) => {
  const templates = [
    {
      text: 'The next topic we\'ll cover is very important for your understanding.',
      simplified_text: 'Next topic is very important.',
      keywords: [
        { keyword: 'important', emoji: '⭐' },
        { keyword: 'topic', emoji: '📚' },
      ],
      tone: { emotion: 'positive', intent: 'statement', emoji: '😊' },
    },
    {
      text: 'This concept might seem complex at first, but with practice it becomes clear.',
      simplified_text: 'This concept is complex. Practice helps.',
      keywords: [
        { keyword: 'concept', emoji: '💡' },
        { keyword: 'practice', emoji: '🎯' },
      ],
      tone: { emotion: 'neutral', intent: 'statement', emoji: '😐' },
    },
    {
      text: 'Don\'t forget the deadline for the final project.',
      simplified_text: 'Final project deadline coming.',
      keywords: [
        { keyword: 'deadline', emoji: '⏰' },
        { keyword: 'project', emoji: '🚀' },
      ],
      tone: { emotion: 'neutral', intent: 'urgent', emoji: '⚠️' },
    },
  ];

  const template = templates[index % templates.length];
  return {
    ...template,
    id: Date.now(),
    sound_event: null,
    translation: null,
    timestamp: Date.now(),
  };
};
