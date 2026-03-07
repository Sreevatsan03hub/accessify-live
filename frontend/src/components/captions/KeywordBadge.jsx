export function KeywordBadge({ keyword, emoji = '🔑' }) {
  return (
    <span className="inline-block px-3 py-1 bg-accent text-bg-dark rounded-full text-sm font-semibold mr-2 mb-2 whitespace-nowrap">
      <span className="mr-1">{emoji}</span>
      {keyword}
    </span>
  );
}
