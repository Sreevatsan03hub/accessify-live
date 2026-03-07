export function ToneIndicator({ emotion, intent, emoji = '😐' }) {
  const getColorClass = () => {
    switch (intent) {
      case 'urgent':
        return 'bg-warning/20 text-warning';
      case 'positive':
        return 'bg-accent/20 text-accent';
      case 'question':
        return 'bg-primary/20 text-primary';
      default:
        return 'bg-gray-400/20 text-gray-400';
    }
  };

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${getColorClass()}`}>
      <span>{emoji}</span>
      <span className="capitalize">{intent || emotion}</span>
    </div>
  );
}
