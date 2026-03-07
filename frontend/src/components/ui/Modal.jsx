import { useEffect } from 'react';
import { Button } from './Button';

export function Modal({
  isOpen,
  onClose,
  title,
  children,
  footer = null,
  size = 'md',
}) {
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-2xl',
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={onClose}
      role="presentation"
    >
      <div
        className={`
          bg-white dark:bg-gray-900
          rounded-2xl
          shadow-2xl
          p-6
          ${sizeClasses[size]}
          max-h-[90vh]
          overflow-y-auto
          animate-in fade-in zoom-in-95 duration-200
        `}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
            {title}
          </h2>
        )}

        <div className="mb-6">
          {children}
        </div>

        {footer && (
          <div className="flex gap-3 justify-end border-t border-gray-200 dark:border-gray-800 pt-4">
            {footer}
          </div>
        )}

        {!footer && (
          <div className="flex gap-3 justify-end">
            <Button variant="secondary" onClick={onClose}>
              Close
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
