import { cn } from '../../utils/helpers'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helperText?: string
}

export function Input({ label, error, helperText, className, ...props }: InputProps) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-semibold mb-2 text-foreground">
          {label}
        </label>
      )}
      <input
        className={cn(
          'w-full px-4 py-2 rounded-lg border border-border bg-black/40 text-foreground placeholder-muted/50',
          'focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary',
          'transition-all duration-200',
          error && 'border-warning focus:ring-warning',
          className
        )}
        {...props}
      />
      {error && <p className="mt-1 text-sm text-warning">{error}</p>}
      {helperText && <p className="mt-1 text-sm text-muted">{helperText}</p>}
    </div>
  )
}
