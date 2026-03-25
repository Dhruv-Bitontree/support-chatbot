"use client";

import { useState, useRef, useCallback } from "react";

interface Props {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Type your message...",
}: Props) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [input, disabled, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <div className="sticky bottom-0 z-20 border-t border-slate-200 bg-white/95 px-3 pt-3 pb-[calc(env(safe-area-inset-bottom)+0.75rem)] backdrop-blur">
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="max-h-[120px] flex-1 resize-none rounded-2xl border border-slate-300 bg-white
                     px-4 py-2.5 text-[15px] leading-6 text-slate-900 placeholder:text-slate-400 shadow-sm
                     transition focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-200
                     disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          aria-label="Send message"
          className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl bg-primary-600 text-white
                     shadow-md shadow-primary-600/20 transition hover:bg-primary-700 active:scale-95
                     disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="h-5 w-5"
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
