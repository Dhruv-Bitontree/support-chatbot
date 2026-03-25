"use client";

interface Props {
  suggestions: string[];
  onSelect: (text: string) => void;
}

export default function QuickReplies({ suggestions, onSelect }: Props) {
  if (!suggestions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 overflow-x-hidden border-t border-slate-200 bg-white/80 px-4 pb-2 pt-3">
      {suggestions.map((s) => (
        <button
          key={s}
          onClick={() => onSelect(s)}
          className="max-w-full min-h-11 rounded-full border border-primary-300 bg-white px-3.5 py-1.5 text-left text-xs font-medium text-primary-700 shadow-sm transition hover:-translate-y-0.5 hover:bg-primary-50 hover:shadow focus:outline-none focus:ring-2 focus:ring-primary-200 sm:min-h-0 sm:whitespace-nowrap"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
