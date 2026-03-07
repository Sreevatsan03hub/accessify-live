import { LANGUAGES } from '../../utils/constants';

export function LanguageSelector({ value, onChange, variant = 'dropdown' }) {
  if (variant === 'pills') {
    return (
      <div className="flex flex-wrap gap-2">
        {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
          <button
            key={code}
            onClick={() => onChange(code)}
            className={`
              px-4 py-2 rounded-full font-semibold transition-all
              ${value === code
                ? 'bg-primary text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white hover:bg-primary hover:text-white'
              }
            `}
          >
            <span className="mr-1">{flag}</span>
            {name}
          </button>
        ))}
      </div>
    );
  }

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="input-field"
    >
      {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
        <option key={code} value={code}>
          {flag} {name}
        </option>
      ))}
    </select>
  );
}
