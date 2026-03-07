import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useCaptions } from '../context/CaptionContext';
import { useUser } from '../context/UserContext';
import { User, Type, Palette, Accessibility } from 'lucide-react';

const S = {
  page: { background: '#F1F5F9', minHeight: '100vh', padding: '48px 0' },
  container: { maxWidth: 720, margin: '0 auto', padding: '0 24px' },
  heading: { fontSize: 34, fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em', marginBottom: 4 },
  sub: { fontSize: 15, color: '#64748B', marginBottom: 40 },
  card: {
    background: '#fff', border: '1px solid #E2E8F0', borderRadius: 16,
    padding: 32, marginBottom: 20, boxShadow: '0 2px 12px rgba(0,0,0,0.05)'
  },
  secTitle: {
    fontSize: 18, fontWeight: 700, color: '#0F172A', marginBottom: 24,
    display: 'flex', alignItems: 'center', gap: 10
  },
  label: {
    display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
    marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em'
  },
  hint: { fontSize: 12, color: '#64748B', marginTop: 4 },
  divider: { border: 'none', borderTop: '1px solid #F1F5F9', margin: '20px 0' },
};

const Toggle = ({ checked, onChange }) => (
  <button
    onClick={() => onChange(!checked)}
    style={{
      width: 44, height: 24, borderRadius: 12, border: 'none', cursor: 'pointer',
      background: checked ? '#2563EB' : '#CBD5E1', position: 'relative', transition: 'background 0.2s',
      flexShrink: 0,
    }}
  >
    <span style={{
      display: 'block', width: 18, height: 18, borderRadius: '50%', background: '#fff',
      position: 'absolute', top: 3, left: checked ? 23 : 3, transition: 'left 0.2s',
      boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
    }} />
  </button>
);

const Row = ({ label, hint, children }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
    <div style={{ flex: 1 }}>
      <p style={{ fontSize: 14, fontWeight: 600, color: '#0F172A', marginBottom: 2 }}>{label}</p>
      {hint && <p style={{ fontSize: 12, color: '#64748B' }}>{hint}</p>}
    </div>
    {children}
  </div>
);

export function Settings() {
  const navigate = useNavigate();
  const [saved, setSaved] = useState(false);
  const { isDark, toggleTheme, highContrast, toggleHighContrast, fontFamily, setFontFamily } = useTheme();
  const { captionSize, setCaptionSize, showEmojis, setShowEmojis,
    showTranslations, setShowTranslations, captionOpacity, setCaptionOpacity,
    autoScroll, setAutoScroll } = useCaptions();
  const { user, updateLanguage } = useUser();
  // Define the valid sizes mapped to display names
  const CAPTION_SIZES = [
    { id: 'small', label: 'SM' },
    { id: 'medium', label: 'MD' },
    { id: 'large', label: 'LG' },
    { id: 'xl', label: 'XL' }
  ];

  const handleSubmit = () => {
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      navigate('/'); // Go back to dashboard after saving
    }, 1000);
  };

  return (
    <div style={S.page}>
      <div style={S.container}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 4 }}>
          <button onClick={() => navigate('/')} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 24 }}>←</button>
          <h1 style={{ ...S.heading, marginBottom: 0 }}>Settings</h1>
        </div>
        <p style={S.sub}>Customise your Accessify experience</p>

        {saved && (
          <div style={{
            marginBottom: 20, padding: '12px 16px', borderRadius: 10, background: '#D1FAE5',
            border: '1px solid #6EE7B7', color: '#065F46', fontWeight: 600, fontSize: 14
          }}>
            ✓ Settings saved successfully!
          </div>
        )}

        {/* ── User Settings ── */}
        <div style={S.card}>
          <h2 style={S.secTitle}>
            <div style={{
              width: 32, height: 32, borderRadius: 8, background: '#DBEAFE',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <User size={17} color="#2563EB" />
            </div>
            User Settings
          </h2>

          {user && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div>
                <label style={S.label}>Display Name</label>
                <input type="text" value={user.name} disabled className="input-field"
                  style={{ opacity: 0.7, cursor: 'not-allowed' }} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={S.label}>Role</label>
                  <input type="text" value={user.role} disabled className="input-field"
                    style={{ opacity: 0.7, cursor: 'not-allowed', textTransform: 'capitalize' }} />
                </div>
                <div>
                  <label style={S.label}>Preferred Language</label>
                  <select value={user.language} onChange={e => updateLanguage(e.target.value)}
                    className="input-field">
                    <option value="en">🇬🇧 English</option>
                    <option value="hi">🇮🇳 हिंदी (Hindi)</option>
                    <option value="ta">🇮🇳 தமிழ் (Tamil)</option>
                    <option value="te">🇮🇳 తెలుగు (Telugu)</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Caption Settings ── */}
        <div style={S.card}>
          <h2 style={S.secTitle}>
            <div style={{
              width: 32, height: 32, borderRadius: 8, background: '#D1FAE5',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <Type size={17} color="#059669" />
            </div>
            Caption Settings
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div>
              <label style={S.label}>Caption Size</label>
              <div style={{ display: 'flex', gap: 8 }}>
                {CAPTION_SIZES.map(s => (
                  <button key={s.id} onClick={() => setCaptionSize(s.id)} style={{
                    flex: 1, padding: '8px 0', borderRadius: 8, border: '2px solid',
                    borderColor: captionSize === s.id ? '#2563EB' : '#E2E8F0',
                    background: captionSize === s.id ? '#EFF6FF' : '#fff',
                    color: captionSize === s.id ? '#2563EB' : '#64748B',
                    fontWeight: 700, fontSize: 13, cursor: 'pointer', textTransform: 'uppercase',
                    transition: 'all 0.15s',
                  }}>
                    {s.label}
                  </button>
                ))}
              </div>
              <p style={S.hint}>Adjusts caption text size in the student room</p>
            </div>

            <hr style={S.divider} />
            <Row label="Show Emoji Keywords" hint="Highlight important words with emoji badges">
              <Toggle checked={showEmojis} onChange={setShowEmojis} />
            </Row>
            <hr style={S.divider} />
            <Row label="Show Translations" hint="Display multilingual captions alongside English">
              <Toggle checked={showTranslations} onChange={setShowTranslations} />
            </Row>
            <hr style={S.divider} />
            <Row label="Auto-Scroll Captions" hint="Automatically follow the latest caption">
              <Toggle checked={autoScroll} onChange={setAutoScroll} />
            </Row>
            <hr style={S.divider} />
            <div>
              <label style={S.label}>Caption Opacity — {Math.round(captionOpacity * 100)}%</label>
              <input type="range" min="0.3" max="1" step="0.1" value={captionOpacity}
                onChange={e => setCaptionOpacity(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#2563EB' }} />
              <p style={S.hint}>Adjust caption background transparency</p>
            </div>
          </div>
        </div>

        {/* ── Appearance ── */}
        <div style={S.card}>
          <h2 style={S.secTitle}>
            <div style={{
              width: 32, height: 32, borderRadius: 8, background: '#FEF3C7',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <Palette size={17} color="#D97706" />
            </div>
            Appearance
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <Row label="Dark Mode" hint={isDark ? 'Dark mode is ON' : 'Dark mode is OFF'}>
              <Toggle checked={isDark} onChange={toggleTheme} />
            </Row>
            <hr style={S.divider} />
            <Row label="High Contrast Mode" hint="Increase contrast for better accessibility">
              <Toggle checked={highContrast} onChange={toggleHighContrast} />
            </Row>
            <hr style={S.divider} />
            <div>
              <label style={S.label}>Font Family</label>
              <select value={fontFamily} onChange={e => setFontFamily(e.target.value)} className="input-field">
                <option value="default">Inter (Default)</option>
                <option value="dyslexia">OpenDyslexic (Dyslexia-friendly)</option>
                <option value="mono">Monospace</option>
              </select>
            </div>
          </div>
        </div>

        {/* ── Accessibility ── */}
        <div style={S.card}>
          <h2 style={S.secTitle}>
            <div style={{
              width: 32, height: 32, borderRadius: 8, background: '#FCE7F3',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <Accessibility size={17} color="#DB2777" />
            </div>
            Accessibility
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ background: '#EFF6FF', border: '1px solid #BFDBFE', borderRadius: 10, padding: 16 }}>
              <p style={{ fontSize: 13, fontWeight: 700, color: '#1D4ED8', marginBottom: 8 }}>⌨️ Keyboard Navigation</p>
              <ul style={{ fontSize: 13, color: '#374151', lineHeight: 2 }}>
                <li><code style={{ background: '#DBEAFE', color: '#1D4ED8', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>Tab</code> — Navigate between elements</li>
                <li><code style={{ background: '#DBEAFE', color: '#1D4ED8', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>Enter</code> — Activate buttons</li>
                <li><code style={{ background: '#DBEAFE', color: '#1D4ED8', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>Esc</code> — Close modals</li>
              </ul>
            </div>
            <div style={{ background: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: 10, padding: 16 }}>
              <p style={{ fontSize: 13, fontWeight: 700, color: '#15803D', marginBottom: 4 }}>♿ Screen Reader Support</p>
              <p style={{ fontSize: 13, color: '#374151' }}>All interactive elements have proper ARIA labels for full screen reader compatibility.</p>
            </div>
          </div>
        </div>

        {/* Save */}
        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', marginTop: 8 }}>
          <button onClick={() => navigate('/')}
            style={{
              padding: '10px 20px', borderRadius: 10, border: '1.5px solid #E2E8F0',
              background: '#fff', color: '#374151', fontWeight: 600, fontSize: 14, cursor: 'pointer'
            }}>
            Cancel
          </button>
          <button onClick={handleSubmit}
            style={{
              padding: '10px 24px', borderRadius: 10, border: 'none', background: '#2563EB',
              color: '#fff', fontWeight: 700, fontSize: 14, cursor: 'pointer',
              boxShadow: '0 4px 14px rgba(37,99,235,0.3)'
            }}
            onMouseEnter={e => e.currentTarget.style.background = '#1D4ED8'}
            onMouseLeave={e => e.currentTarget.style.background = '#2563EB'}>
            ✓ Save Settings
          </button>
        </div>

      </div>
    </div>
  );
}
