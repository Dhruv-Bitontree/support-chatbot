"use client";

export default function TypingIndicator() {
  return (
    <div className="mb-4 flex max-w-full items-end gap-2 overflow-hidden">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gray-200 text-sm font-medium text-gray-600">
        AI
      </div>
      <div className="max-w-[75%] rounded-2xl rounded-bl-sm bg-gray-100 px-4 py-3">
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 bg-gray-400 rounded-full animate-bounce-dot"
              style={{ animationDelay: `${i * 0.16}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
