export function Button({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  disabled = false,
  type = 'button',
  ...props
}) {
  const base = 'inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-50 disabled:cursor-not-allowed';

  const variants = {
    primary: 'bg-primary text-white hover:bg-blue-700 shadow-sm',
    secondary: 'bg-white text-text-primary border border-gray-200 hover:border-primary hover:text-primary shadow-sm',
    ghost: 'text-primary border border-primary/40 hover:bg-primary hover:text-white',
    danger: 'bg-danger text-white hover:bg-red-600 shadow-sm',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-sm',
    lg: 'px-7 py-3 text-base',
  };

  return (
    <button
      type={type}
      disabled={disabled}
      className={`${base} ${variants[variant] ?? ''} ${sizes[size] ?? ''} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
