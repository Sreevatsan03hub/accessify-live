import { ReactNode } from 'react'
import { cn } from '../../utils/helpers'

interface CardProps {
  children: ReactNode
  className?: string
  onClick?: () => void
  title?: string
  icon?: ReactNode
}

export function Card({ children, className, onClick, title, icon }: CardProps) {
  return (
    <div
      className={cn(
        'p-6 rounded-xl border border-border bg-black/40 hover:border-accent/50 transition-all duration-200',
        onClick && 'cursor-pointer hover:shadow-lg hover:shadow-accent/20',
        className
      )}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          onClick()
        }
      }}
    >
      {(title || icon) && (
        <div className="flex items-center gap-3 mb-4">
          {icon && <div className="text-3xl">{icon}</div>}
          {title && <h3 className="text-lg font-bold">{title}</h3>}
        </div>
      )}
      {children}
    </div>
  )
}
