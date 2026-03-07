import { useState, useEffect } from 'react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Select } from '../components/ui/Select'
import { useUser } from '../context/UserContext'
import { useTheme } from '../context/ThemeContext'
import { Save, RotateCcw } from 'lucide-react'

export default function Settings() {
  const { user, updateSettings } = useUser()
  const { theme, toggleTheme, highContrast, setHighContrast } = useTheme()
  const [saved, setSaved] = useState(false)
  const [settings, setSettings] = useState({
    captionSize: user?.captionSize || 'medium',
    language: user?.language || 'en',
    showEmojis: user?.showEmojis !== false,
    autoScroll: true,
    captionOpacity: 85,
    fontSize: 'default',
  })

  const handleChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    setSaved(false)
  }

  const handleSave = () => {
    updateSettings?.({
      captionSize: settings.captionSize as any,
      language: settings.language as any,
      showEmojis: settings.showEmojis,
    })
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const handleReset = () => {
    setSettings({
      captionSize: user?.captionSize || 'medium',
      language: user?.language || 'en',
      showEmojis: user?.showEmojis !== false,
      autoScroll: true,
      captionOpacity: 85,
      fontSize: 'default',
    })
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-2">Settings & Preferences</h1>
      <p className="text-muted mb-8">Customize your experience and accessibility options</p>

      {saved && (
        <div className="mb-6 p-4 bg-accent/20 border border-accent rounded-lg text-accent text-sm">
          Settings saved successfully!
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Caption Preferences */}
          <Card>
            <h2 className="text-2xl font-bold mb-6">Caption Preferences</h2>
            <div className="space-y-4">
              <Select
                label="Default Caption Size"
                value={settings.captionSize}
                onChange={(e) => handleChange('captionSize', e.target.value)}
                options={[
                  { value: 'small', label: 'Small (A-)' },
                  { value: 'medium', label: 'Medium (A)' },
                  { value: 'large', label: 'Large (A+)' },
                  { value: 'xl', label: 'Extra Large (A++)' },
                ]}
              />

              <div>
                <label className="block text-sm font-semibold mb-2">Caption Background Opacity</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.captionOpacity}
                  onChange={(e) => handleChange('captionOpacity', parseInt(e.target.value))}
                  className="w-full cursor-pointer"
                />
                <p className="text-xs text-muted mt-1">{settings.captionOpacity}%</p>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.showEmojis}
                    onChange={(e) => handleChange('showEmojis', e.target.checked)}
                    className="rounded"
                  />
                  <span>Show emoji indicators for keywords</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.autoScroll}
                    onChange={(e) => handleChange('autoScroll', e.target.checked)}
                    className="rounded"
                  />
                  <span>Auto-scroll captions to latest</span>
                </label>
              </div>
            </div>
          </Card>

          {/* Language & Localization */}
          <Card>
            <h2 className="text-2xl font-bold mb-6">Language & Localization</h2>
            <div className="space-y-4">
              <Select
                label="Default Caption Language"
                value={settings.language}
                onChange={(e) => handleChange('language', e.target.value)}
                options={[
                  { value: 'en', label: '🇬🇧 English' },
                  { value: 'hi', label: '🇮🇳 हिंदी (Hindi)' },
                  { value: 'ta', label: '🇮🇳 தமிழ் (Tamil)' },
                  { value: 'te', label: '🇮🇳 తెలుగు (Telugu)' },
                ]}
              />
            </div>
          </Card>

          {/* Appearance */}
          <Card>
            <h2 className="text-2xl font-bold mb-6">Appearance</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-black/40 rounded-lg border border-border">
                <div>
                  <p className="font-semibold">Dark Mode</p>
                  <p className="text-sm text-muted">Currently {theme === 'dark' ? 'enabled' : 'disabled'}</p>
                </div>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={toggleTheme}
                >
                  {theme === 'dark' ? '🌙' : '☀️'}
                </Button>
              </div>

              <div className="flex items-center justify-between p-4 bg-black/40 rounded-lg border border-border">
                <div>
                  <p className="font-semibold">High Contrast Mode</p>
                  <p className="text-sm text-muted">Increases readability</p>
                </div>
                <Button
                  variant={highContrast ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setHighContrast(!highContrast)}
                >
                  {highContrast ? 'ON' : 'OFF'}
                </Button>
              </div>

              <Select
                label="Font Family"
                value={settings.fontSize}
                onChange={(e) => handleChange('fontSize', e.target.value)}
                options={[
                  { value: 'default', label: 'Default' },
                  { value: 'dyslexia', label: 'Dyslexia-Friendly (OpenDyslexic)' },
                  { value: 'mono', label: 'Monospace' },
                ]}
              />
            </div>
          </Card>

          {/* Accessibility */}
          <Card>
            <h2 className="text-2xl font-bold mb-6">Accessibility</h2>
            <div className="space-y-3">
              <label className="flex items-center gap-2 cursor-pointer p-3 bg-black/40 rounded-lg border border-border hover:border-accent/50 transition-colors">
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded"
                  disabled
                />
                <span>Keyboard Navigation</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer p-3 bg-black/40 rounded-lg border border-border hover:border-accent/50 transition-colors">
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded"
                  disabled
                />
                <span>Screen Reader Compatible</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer p-3 bg-black/40 rounded-lg border border-border hover:border-accent/50 transition-colors">
                <input
                  type="checkbox"
                  className="rounded"
                />
                <span>Reduce Motion</span>
              </label>
            </div>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* User Info */}
          <Card>
            <h3 className="font-bold mb-4">Account</h3>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-muted">Name</p>
                <p className="font-semibold">{user?.name}</p>
              </div>
              <div>
                <p className="text-muted">Email</p>
                <p className="font-semibold text-xs break-all">{user?.email}</p>
              </div>
              <div>
                <p className="text-muted">Role</p>
                <p className="font-semibold capitalize">{user?.role}</p>
              </div>
            </div>
          </Card>

          {/* Save Actions */}
          <Card>
            <div className="flex flex-col gap-2">
              <Button
                size="lg"
                onClick={handleSave}
                className="gap-2"
              >
                <Save size={16} />
                Save Changes
              </Button>
              <Button
                variant="ghost"
                size="lg"
                onClick={handleReset}
                className="gap-2"
              >
                <RotateCcw size={16} />
                Reset
              </Button>
            </div>
          </Card>

          {/* Help */}
          <Card className="border-accent/50 bg-accent/5">
            <h3 className="font-bold mb-2">Need Help?</h3>
            <p className="text-sm text-muted mb-3">
              Check our accessibility guide for more information about available features.
            </p>
            <Button variant="ghost" size="sm" className="w-full">
              View Guide
            </Button>
          </Card>
        </div>
      </div>
    </div>
  )
}
