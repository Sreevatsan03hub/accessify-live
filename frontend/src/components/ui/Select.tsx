import { cn } from '../../utils/helpers'

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  options: { value: string; label: string }[]
}

export function Select({ label, error, options, className, ...props }: SelectProps) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-semibold mb-2 text-foreground">
          {label}
        </label>
      )}
      <select
        className={cn(
          'w-full px-4 py-2 rounded-lg border border-border bg-black/40 text-foreground',
          'focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary',
          'transition-all duration-200 cursor-pointer',
          error && 'border-warning focus:ring-warning',
          className
        )}
        {...props}
      >
        <option value="">Select an option</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {error && <p className="mt-1 text-sm text-warning">{error}</p>}
    </div>
  )
}
