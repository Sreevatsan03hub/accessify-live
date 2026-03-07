/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        // Brand
        primary: '#1C64F2',
        'primary-lt': '#EBF2FF',
        accent: '#0EA5E9',
        success: '#10B981',
        warning: '#F59E0B',
        danger: '#EF4444',
        // Backgrounds
        'bg-app': '#F0F2F5',
        'bg-card': '#FFFFFF',
        'bg-dark': '#1B2A4A',
        'bg-darker': '#111E35',
        // Text
        'text-primary': '#111827',
        'text-secondary': '#6B7280',
        'text-light': '#9CA3AF',
        // Legacy (keep for student room which is still dark)
        'bg-light': '#F8F9FF',
        'caption-bg': 'rgba(0,0,0,0.85)',
      },
      borderRadius: {
        DEFAULT: '10px',
        lg: '16px',
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.08)',
        'card-hover': '0 4px 16px rgba(0,0,0,0.10)',
        'card-lg': '0 10px 40px rgba(0,0,0,0.14)',
        nav: '0 2px 12px rgba(0,0,0,0.20)',
      },
      fontSize: {
        caption: ['18px', { lineHeight: '1.6' }],
      },
    },
  },
  plugins: [],
}
