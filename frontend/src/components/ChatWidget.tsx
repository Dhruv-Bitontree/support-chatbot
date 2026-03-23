"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import ChatWindow from "./ChatWindow";

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      {/* Chat panel */}
      <div
        className={cn(
          "transition-all duration-300 ease-in-out origin-bottom-right",
          isOpen
            ? "opacity-100 scale-100 pointer-events-auto"
            : "opacity-0 scale-95 pointer-events-none",
        )}
      >
        <div className="w-[380px] h-[560px] shadow-2xl rounded-2xl overflow-hidden">
          <ChatWindow embedded />
        </div>
      </div>

      {/* Floating button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all duration-200",
          isOpen
            ? "bg-gray-600 hover:bg-gray-700 rotate-0"
            : "bg-primary-600 hover:bg-primary-700",
        )}
      >
        {isOpen ? (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="white"
            className="w-6 h-6"
          >
            <path
              fillRule="evenodd"
              d="M5.47 5.47a.75.75 0 011.06 0L12 10.94l5.47-5.47a.75.75 0 111.06 1.06L13.06 12l5.47 5.47a.75.75 0 11-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 01-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 010-1.06z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="white"
            className="w-6 h-6"
          >
            <path
              fillRule="evenodd"
              d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z"
              clipRule="evenodd"
            />
          </svg>
        )}
      </button>
    </div>
  );
}
