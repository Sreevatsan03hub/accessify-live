import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

export type Language = 'en' | 'hi' | 'ta' | 'te'

export interface Keyword {
  keyword: string
  emoji: string
  score?: number
}

export interface Tone {
  emotion: string
  intent: string
  emoji: string
}

export interface Translation {
  text: string
  target_language: string
}

export interface Caption {
  id: number
  text: string
  simplified_text: string
  keywords: Keyword[]
  tone: Tone
  translation?: Translation
  sound_event?: string
  timestamp: number
}

interface CaptionContextType {
  captions: Caption[]
  addCaption: (caption: Caption) => void
  clearCaptions: () => void
  setLanguage: (lang: Language) => void
  currentLanguage: Language
}

const CaptionContext = createContext<CaptionContextType | undefined>(undefined)

export function CaptionProvider({ children }: { children: ReactNode }) {
  const [captions, setCaptions] = useState<Caption[]>([])
  const [currentLanguage, setLanguageState] = useState<Language>('en')

  const addCaption = useCallback((caption: Caption) => {
    setCaptions(prev => [...prev, caption])
  }, [])

  const clearCaptions = useCallback(() => {
    setCaptions([])
  }, [])

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang)
  }, [])

  return (
    <CaptionContext.Provider value={{ captions, addCaption, clearCaptions, setLanguage, currentLanguage }}>
      {children}
    </CaptionContext.Provider>
  )
}

export function useCaption() {
  const context = useContext(CaptionContext)
  if (!context) {
    throw new Error('useCaption must be used within CaptionProvider')
  }
  return context
}
