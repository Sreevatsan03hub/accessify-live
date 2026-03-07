import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        primary: 'var(--primary)',
        accent: 'var(--accent)',
        warning: 'var(--warning)',
        success: 'var(--success)',
        muted: 'var(--muted)',
      },
      fontSize: {
        base: '16px',
      },
      borderRadius: {
        lg: '12px',
      },
    },
  },
  plugins: [],
}

export default config
