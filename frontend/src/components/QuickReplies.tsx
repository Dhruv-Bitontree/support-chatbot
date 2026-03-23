"use client";

interface Props {
  suggestions: string[];
  onSelect: (text: string) => void;
}

export default function QuickReplies({ suggestions, onSelect }: Props) {
  if (!suggestions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 px-4 pb-2">
      {suggestions.map((s) => (
        <button
          key={s}
          onClick={() => onSelect(s)}
          className="text-xs px-3 py-1.5 rounded-full border border-primary-300 text-primary-600
                     hover:bg-primary-50 transition-colors whitespace-nowrap"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
