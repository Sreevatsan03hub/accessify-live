# Accessify Component Guide

## UI Components

### Button
Reusable button with multiple variants and sizes.

```jsx
import { Button } from './components/ui/Button'

<Button variant="primary" size="md">Click me</Button>
<Button variant="secondary" size="sm">Secondary</Button>
<Button variant="ghost">Ghost button</Button>
<Button variant="danger">Delete</Button>
```

**Props:**
- `variant`: 'primary' | 'secondary' | 'ghost' | 'danger' (default: 'primary')
- `size`: 'sm' | 'md' | 'lg' (default: 'md')
- `disabled`: boolean
- `type`: 'button' | 'submit' | 'reset'
- `className`: string (for additional Tailwind classes)

---

### Card
Container component for content sections.

```jsx
import { Card } from './components/ui/Card'

<Card>Content here</Card>
<Card hover>Clickable card</Card>
<Card className="max-w-md">Custom width</Card>
```

**Props:**
- `hover`: boolean (adds hover animation)
- `className`: string (additional classes)
- `children`: ReactNode

---

### Modal
Dialog component with keyboard support.

```jsx
import { Modal } from './components/ui/Modal'
import { Button } from './components/ui/Button'

const [isOpen, setIsOpen] = useState(false)

<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Confirm Action"
  size="md"
  footer={
    <>
      <Button onClick={() => setIsOpen(false)}>Cancel</Button>
      <Button variant="danger">Delete</Button>
    </>
  }
>
  Are you sure?
</Modal>
```

**Props:**
- `isOpen`: boolean (required)
- `onClose`: function (required)
- `title`: string (optional)
- `size`: 'sm' | 'md' | 'lg' | 'xl'
- `footer`: ReactNode (optional)
- `children`: ReactNode

---

### ThemeToggle
Dark/light mode toggle button.

```jsx
import { ThemeToggle } from './components/ui/ThemeToggle'

<ThemeToggle />
```

No props required. Uses ThemeContext internally.

---

### Navbar
Global navigation bar (sticky).

```jsx
import { Navbar } from './components/ui/Navbar'

<Navbar />
```

Shows logo, nav links, theme toggle, and user menu. Uses UserContext.

---

### Footer
Global footer with links.

```jsx
import { Footer } from './components/ui/Footer'

<Footer />
```

Contains platform links, resources, and copyright.

---

## Caption Components

### CaptionPanel
Container for displaying multiple captions with auto-scroll.

```jsx
import { CaptionPanel } from './components/captions/CaptionPanel'
import { MOCK_CAPTIONS } from './utils/mockData'

<CaptionPanel
  captions={MOCK_CAPTIONS}
  showEmojis={true}
  showTranslation={false}
  language="en"
  maxHeight="max-h-96"
/>
```

**Props:**
- `captions`: Caption[] (required)
- `showEmojis`: boolean (default: true)
- `showTranslation`: boolean (default: false)
- `language`: string (default: 'en')
- `maxHeight`: string (Tailwind height class)

---

### CaptionCard
Individual caption display with all metadata.

```jsx
import { CaptionCard } from './components/captions/CaptionCard'

<CaptionCard
  text="Welcome to class"
  simplifiedText="Welcome"
  keywords={[{ keyword: "class", emoji: "📚" }]}
  tone={{ emotion: "positive", intent: "statement", emoji: "😊" }}
  translation={{ text: "வகுப்பிற்கு வருங்கள்", target_language: "ta" }}
  soundEvent={null}
  showEmojis={true}
  showTranslation={true}
  language="ta"
/>
```

**Props:**
- `text`: string (required)
- `simplifiedText`: string
- `keywords`: { keyword: string, emoji: string }[]
- `tone`: { emotion: string, intent: string, emoji: string }
- `translation`: { text: string, target_language: string }
- `soundEvent`: string (e.g., "APPLAUSE 👏")
- `showEmojis`: boolean
- `showTranslation`: boolean
- `language`: string

---

### KeywordBadge
Emoji + text badge for keywords.

```jsx
import { KeywordBadge } from './components/captions/KeywordBadge'

<KeywordBadge keyword="exam" emoji="📘" />
<KeywordBadge keyword="deadline" emoji="⏰" />
```

**Props:**
- `keyword`: string (required)
- `emoji`: string (default: '🔑')

---

### ToneIndicator
Color-coded tone/intent indicator.

```jsx
import { ToneIndicator } from './components/captions/ToneIndicator'

<ToneIndicator emotion="positive" intent="urgent" emoji="⚠️" />
<ToneIndicator emotion="neutral" intent="question" emoji="❓" />
```

**Props:**
- `emotion`: string
- `intent`: string (required - determines color)
- `emoji`: string

---

### SoundEventBanner
Full-width banner for sound events.

```jsx
import { SoundEventBanner } from './components/captions/SoundEventBanner'

<SoundEventBanner event="APPLAUSE 👏" duration={2000} />
```

**Props:**
- `event`: string (e.g., "APPLAUSE 👏")
- `duration`: number (milliseconds, default: 2000)

Auto-fades after duration.

---

## Settings Components

### LanguageSelector
Dropdown or pills for language selection.

```jsx
import { LanguageSelector } from './components/settings/LanguageSelector'

// Dropdown (default)
<LanguageSelector value="en" onChange={(lang) => setLanguage(lang)} />

// Pills
<LanguageSelector
  value="en"
  onChange={(lang) => setLanguage(lang)}
  variant="pills"
/>
```

**Props:**
- `value`: string (language code: 'en', 'hi', 'ta', 'te')
- `onChange`: (value: string) => void (required)
- `variant`: 'dropdown' | 'pills' (default: 'dropdown')

---

### CaptionSizeControl
Four-button size control.

```jsx
import { CaptionSizeControl } from './components/settings/CaptionSizeControl'

<CaptionSizeControl
  size="medium"
  onChange={(size) => setCaptionSize(size)}
/>
```

**Props:**
- `size`: 'small' | 'medium' | 'large' | 'xl'
- `onChange`: (size: string) => void (required)

---

## Classroom Components

### RoomCodeDisplay
Display room code with copy and share buttons.

```jsx
import { RoomCodeDisplay } from './components/classroom/RoomCodeDisplay'

<RoomCodeDisplay
  code="ABC123"
  teacherName="Dr. Smith"
  title="Introduction to Machine Learning"
/>
```

**Props:**
- `code`: string (required)
- `teacherName`: string (required)
- `title`: string (required)

---

## Upload Components

### FileUploader
Drag-and-drop file uploader with validation.

```jsx
import { FileUploader } from './components/upload/FileUploader'

const [file, setFile] = useState(null)
const [isLoading, setIsLoading] = useState(false)

<FileUploader
  onFileSelect={(file) => setFile(file)}
  isLoading={isLoading}
/>
```

**Props:**
- `onFileSelect`: (file: File) => void (required)
- `isLoading`: boolean (disables input when true)

Accepts: .mp4, .mkv, .avi, .mov, .webm

---

## Context Hooks

### useTheme
Access and control theme settings.

```jsx
import { useTheme } from './context/ThemeContext'

const { isDark, toggleTheme, highContrast, toggleHighContrast, fontFamily, setFontFamily } = useTheme()

<button onClick={toggleTheme}>
  {isDark ? '☀️' : '🌙'}
</button>
```

---

### useUser
Access and manage user state.

```jsx
import { useUser } from './context/UserContext'

const { user, login, logout, updateLanguage } = useUser()

// Check if logged in
if (!user) {
  navigate('/login')
}

// Login user
login('John Doe', 'student', 'en')

// Logout
logout()
```

---

### useCaptions
Access and manage captions.

```jsx
import { useCaptions } from './context/CaptionContext'

const {
  captions,
  addCaption,
  clearCaptions,
  captionSize,
  setCaptionSize,
  showEmojis,
  setShowEmojis,
  autoScroll,
  setAutoScroll,
  getCaptionSizeClass
} = useCaptions()
```

---

## Custom Hooks

### useWebSocket
Mock WebSocket connection for live captions.

```jsx
import { useWebSocket } from './hooks/useWebSocket'

const { isConnected, isReconnecting, send, disconnect, reconnect } = useWebSocket(
  'ws://localhost:8001/ws/room/ABC123/teacher',
  (message) => {
    if (message.type === 'caption') {
      addCaption(message.data)
    }
  },
  true // enabled
)
```

---

### useMicrophone
Mock microphone capture.

```jsx
import { useMicrophone } from './hooks/useMicrophone'

const { isActive, isLoading, error, volume, toggleMicrophone } = useMicrophone(true)

<button onClick={toggleMicrophone}>
  {isActive ? 'Stop Recording' : 'Start Recording'}
</button>
```

---

## Utility Functions

### Caption Utilities
```jsx
import {
  parseVTT,
  generateSRT,
  generateVTT,
  generatePlainText,
  generateSummary,
  formatTimestamp,
  estimateCaptionDuration
} from './utils/captionUtils'

const vttContent = generateVTT(captions)
const srtContent = generateSRT(captions)
const summary = generateSummary(captions)
```

---

### Audio Utilities
```jsx
import {
  downloadAudioBlob,
  downloadTextFile,
  downloadJSON
} from './utils/audioUtils'

downloadTextFile('Hello World', 'hello.txt')
downloadJSON(data, 'export.json')
```

---

### Constants
```jsx
import {
  LANGUAGES,
  CAPTION_SIZES,
  COLOR_PALETTE,
  TONE_COLORS,
  SOUND_EVENTS,
  FEATURE_CARDS,
  DASHBOARD_CARDS
} from './utils/constants'

Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
  <option key={code} value={code}>{flag} {name}</option>
))
```

---

## Layout Patterns

### Full Page Layout
```jsx
<div className="min-h-screen bg-bg-dark">
  <Navbar />
  <main className="max-w-7xl mx-auto px-4 py-12">
    {/* Content */}
  </main>
  <Footer />
</div>
```

### Two Column Layout
```jsx
<div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
  <div className="lg:col-span-2">
    {/* Main content */}
  </div>
  <div>
    {/* Sidebar */}
  </div>
</div>
```

### Card Grid
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
  {items.map(item => (
    <Card key={item.id} hover>
      {/* Card content */}
    </Card>
  ))}
</div>
```

---

## Common Patterns

### Form Submission
```jsx
const handleSubmit = async (e) => {
  e.preventDefault()
  setError('')
  
  try {
    setIsLoading(true)
    // API call here
    navigate('/success')
  } catch (err) {
    setError(err.message)
  } finally {
    setIsLoading(false)
  }
}
```

### Conditional Rendering with Auth
```jsx
const { user } = useUser()
const navigate = useNavigate()

if (!user) {
  navigate('/login')
  return null
}

return <Dashboard />
```

### Settings State Management
```jsx
const [preference, setPreference] = useState(() => {
  return localStorage.getItem('preference') || 'default'
})

useEffect(() => {
  localStorage.setItem('preference', preference)
}, [preference])
```

---

**For more examples, check the `/pages` directory for real-world component usage.**
