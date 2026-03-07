interface KeywordBadgeProps {
  keyword: string
  emoji: string
}

export function KeywordBadge({ keyword, emoji }: KeywordBadgeProps) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 mx-1 text-sm font-semibold rounded-full bg-primary/15 text-primary border border-primary/40 hover:bg-primary/25 transition-colors">
      <span className="text-lg leading-none">{emoji}</span>
      <span>{keyword}</span>
    </span>
  )
}
