"use client";

export default function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 mb-4">
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium bg-gray-200 text-gray-600">
        AI
      </div>
      <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3">
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
