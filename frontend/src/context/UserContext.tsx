import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

export type Language = 'en' | 'hi' | 'ta' | 'te'
export type UserRole = 'teacher' | 'student'
export type CaptionSize = 'small' | 'medium' | 'large' | 'xl'

export interface User {
  name: string
  role: UserRole
  language: Language
  captionSize: CaptionSize
  showEmojis: boolean
}

interface UserContextType {
  user: User | null
  setUser: (user: User) => void
  logout: () => void
  updateSettings: (settings: Partial<User>) => void
}

const UserContext = createContext<UserContextType | undefined>(undefined)

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<User | null>(null)

  useEffect(() => {
    const savedUser = localStorage.getItem('user')
    if (savedUser) {
      setUserState(JSON.parse(savedUser))
    }
  }, [])

  const setUser = (newUser: User) => {
    setUserState(newUser)
    localStorage.setItem('user', JSON.stringify(newUser))
  }

  const logout = () => {
    setUserState(null)
    localStorage.removeItem('user')
  }

  const updateSettings = (settings: Partial<User>) => {
    if (user) {
      const updated = { ...user, ...settings }
      setUser(updated)
    }
  }

  return (
    <UserContext.Provider value={{ user, setUser, logout, updateSettings }}>
      {children}
    </UserContext.Provider>
  )
}

export function useUser() {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error('useUser must be used within UserProvider')
  }
  return context
}
