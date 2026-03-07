# Accessify - AI-Powered Accessible Learning Platform

A production-ready React 18 + Vite web application for Deaf and Hard-of-Hearing students featuring AI-powered real-time captions, translations, and smart accessibility features.

## Features

- 🎙️ **Real-Time Captions** - Instant speech-to-text as teachers speak
- 🌐 **Multilingual Support** - English, Hindi, Tamil, Telugu
- ⭐ **Smart Keywords** - Important words highlighted with emojis automatically
- 🔔 **Sound Awareness** - [APPLAUSE 👏], [LAUGHTER 😂], [ALARM 🚨]
- 🎥 **Live Classrooms** - Real-time streaming with synchronized captions
- 📁 **Video Upload** - Upload videos with automatic caption generation
- 📊 **Caption History** - View and download past sessions
- 🎨 **Dark/Light Mode** - Customizable theme with high contrast option
- ♿ **WCAG 2.1 AA** - Full accessibility compliance

## Tech Stack

- **Frontend**: React 18, React Router v6
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **HTTP Client**: Axios
- **State Management**: Context API
- **Language**: JavaScript/JSX

## Project Structure

```
src/
├── components/
│   ├── captions/           # Caption display components
│   │   ├── CaptionCard.jsx
│   │   ├── CaptionPanel.jsx
│   │   ├── KeywordBadge.jsx
│   │   ├── SoundEventBanner.jsx
│   │   └── ToneIndicator.jsx
│   ├── classroom/          # Classroom-specific components
│   │   └── RoomCodeDisplay.jsx
│   ├── settings/           # Settings components
│   │   ├── CaptionSizeControl.jsx
│   │   └── LanguageSelector.jsx
│   ├── ui/                 # Reusable UI components
│   │   ├── Button.jsx
│   │   ├── Card.jsx
│   │   ├── Footer.jsx
│   │   ├── Modal.jsx
│   │   ├── Navbar.jsx
│   │   └── ThemeToggle.jsx
│   └── upload/             # Upload components
│       └── FileUploader.jsx
├── pages/                  # Page components (routes)
│   ├── CreateRoom.jsx
│   ├── Dashboard.jsx
│   ├── History.jsx
│   ├── JoinRoom.jsx
│   ├── Landing.jsx
│   ├── Login.jsx
│   ├── Player.jsx
│   ├── Register.jsx
│   ├── Settings.jsx
│   ├── StudentRoom.jsx
│   ├── TeacherRoom.jsx
│   └── Upload.jsx
├── context/                # React Context providers
│   ├── CaptionContext.jsx
│   ├── ThemeContext.jsx
│   └── UserContext.jsx
├── hooks/                  # Custom React hooks
│   ├── useMicrophone.js   # Audio capture (mock)
│   └── useWebSocket.js    # WebSocket connection (mock)
├── services/               # API services
│   └── (Prepared for future backend integration)
├── utils/                  # Utility functions
│   ├── audioUtils.js
│   ├── captionUtils.js
│   ├── constants.js
│   └── mockData.js         # Mock data for development
├── App.jsx                 # Main app component with routing
├── main.jsx                # Entry point
└── index.css               # Global styles
```

## Getting Started

### Prerequisites

- Node.js 16+
- pnpm (or npm/yarn)

### Installation

```bash
# Install dependencies
pnpm install

# Start development server
pnpm run dev

# Build for production
pnpm run build

# Preview production build
pnpm run preview
```

The app will be available at `http://localhost:5173`

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Landing | `/` | Hero section with features and CTA buttons |
| Login | `/login` | User login (mock auth) |
| Register | `/register` | User registration (mock auth) |
| Dashboard | `/dashboard` | Main hub with quick action cards |
| Create Room | `/room/create` | Teacher creates a live classroom |
| Join Room | `/room/join` | Student joins a live classroom |
| Teacher View | `/room/:code/teacher` | Live broadcast interface for teachers |
| Student View | `/room/:code/student/:id` | Live caption viewing for students |
| Upload | `/upload` | Video upload with caption generation |
| Player | `/player` | Video playback with synchronized captions |
| History | `/history` | Past sessions with download options |
| Settings | `/settings` | User preferences and accessibility options |

## Key Features Explained

### Mock Data & Development

Currently, the app uses mock data and simulated WebSockets for development:

- **Caption Generation**: Mock captions appear at regular intervals
- **WebSocket**: Returns simulated caption streams
- **Microphone**: Simulated audio capture (no actual recording)
- **Video Upload**: Shows progress simulation without actual upload

### Authentication

User authentication is localStorage-based (UI-only):
- No backend API calls
- Data persists in browser storage
- Resets on browser clear

### Caption System

Captions display with:
- Original text (large, readable)
- Simplified version (smaller, plain language)
- Keyword emoji badges
- Tone indicators (😊 Positive, ⚠️ Urgent, ❓ Question)
- Optional translations
- Sound event banners ([APPLAUSE 👏])

### Settings & Preferences

Customizable through the Settings page:
- Caption size (A-, A, A+, A++)
- Language preference
- Dark/Light mode
- High contrast mode
- Emoji assistance
- Caption opacity
- Auto-scroll toggle

## Future Backend Integration

The code is prepared for real backend integration:

1. **WebSocket Connections**
   - Replace mock data with real ws:// streams
   - Real audio streaming from teacher to students
   - Actual caption broadcasting

2. **Authentication**
   - Real login/register API calls
   - JWT token management
   - Session handling

3. **File Upload**
   - Replace progress simulation with real FormData upload
   - Real backend video processing

4. **API Calls**
   - Room creation/joining via API
   - Session management
   - Export downloads

All API endpoints are pre-configured in the constants and services files.

## Accessibility Features

- **ARIA Labels**: All interactive elements have proper labels
- **Keyboard Navigation**: Full keyboard support (Tab, Enter, Space, Esc)
- **High Contrast Mode**: Enhanced contrast option
- **Reduced Motion**: Respects `prefers-reduced-motion`
- **Screen Reader Support**: Semantic HTML and ARIA attributes
- **Font Options**: Dyslexia-friendly fonts available
- **Large Captions**: Minimum 18px, adjustable to 2rem+

## Styling System

Uses Tailwind CSS with custom color variables:
- **Primary**: #6C63FF (purple)
- **Accent**: #00D4AA (teal)
- **Warning**: #FF6B6B (red)
- **Background Dark**: #0F0F1A
- **Background Light**: #F8F9FF
- **Caption BG**: rgba(0,0,0,0.85)

All colors are customizable in `tailwind.config.js`

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Vite for fast development builds
- React 18 with Suspense ready
- Optimized CSS with Tailwind's purge
- Lazy loading for routes
- Efficient re-renders with Context API

## Development Tips

1. **Mock Data**: Edit `src/utils/mockData.js` to test different caption scenarios
2. **Colors**: Update `src/index.css` variables or `tailwind.config.js` for theming
3. **Components**: Import from `src/components/` - they're designed for reusability
4. **Contexts**: Use `useTheme()`, `useUser()`, `useCaptions()` throughout the app
5. **Utilities**: Check `src/utils/` for caption parsing, audio processing stubs

## License

MIT - Available for educational and commercial use.

## Support

For issues or questions about the UI implementation, check:
- Component documentation in `src/components/README.md` (future)
- Example usage in the page files
- Tailwind CSS docs: https://tailwindcss.com
- React Router docs: https://reactrouter.com

---

Built with ❤️ for accessibility
