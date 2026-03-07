import { useEffect, useState } from 'react';

const EVENT_STYLES = {
  'APPLAUSE': { bg: 'bg-green-500', text: 'text-white', ring: 'ring-green-300' },
  'LAUGHTER': { bg: 'bg-amber-400', text: 'text-black', ring: 'ring-amber-200' },
  'BACKGROUND NOISE': { bg: 'bg-blue-500', text: 'text-white', ring: 'ring-blue-300' },
  'DOOR OPENS': { bg: 'bg-orange-500', text: 'text-white', ring: 'ring-orange-300' },
  'DOOR': { bg: 'bg-orange-500', text: 'text-white', ring: 'ring-orange-300' },
  'DEFAULT': { bg: 'bg-gray-700', text: 'text-white', ring: 'ring-gray-500' },
};

function getStyle(display) {
  if (!display) return EVENT_STYLES.DEFAULT;
  const up = display.toUpperCase();
  for (const key of Object.keys(EVENT_STYLES)) {
    if (up.includes(key)) return EVENT_STYLES[key];
  }
  return EVENT_STYLES.DEFAULT;
}

export function SoundEventBanner({ event, duration = 4000 }) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    setVisible(true);
    const t = setTimeout(() => setVisible(false), duration);
    return () => clearTimeout(t);
  }, [event, duration]);

  if (!visible || !event) return null;

  const style = getStyle(event);
  const parts = event.split(' ');
  const emoji = parts[0];
  const label = parts.slice(1).join(' ');

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={[
        'w-full flex items-center justify-center gap-3',
        'px-5 py-3 mb-3 rounded-xl shadow-lg',
        'font-bold text-lg tracking-wide',
        'ring-2 animate-pulse',
        style.bg, style.text, style.ring,
      ].join(' ')}
    >
      <span className="text-2xl">{emoji}</span>
      <span>{label || event}</span>
    </div>
  );
}
