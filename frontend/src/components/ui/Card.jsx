export function Card({ children, className = '', hover = false }) {
  return (
    <div
      className={`
        bg-white
        rounded-xl
        border border-gray-200
        shadow-card
        p-6
        transition-all duration-200
        ${hover ? 'card-hover' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}
