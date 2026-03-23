"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatMessage as ChatMessageType } from "@/lib/types";
import { sendMessage } from "@/lib/api";
import { generateId } from "@/lib/utils";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import TypingIndicator from "./TypingIndicator";
import QuickReplies from "./QuickReplies";

const INITIAL_SUGGESTIONS = [
  "What is your return policy?",
  "Track my order",
  "I have a complaint",
  "What payment methods do you accept?",
];

interface Props {
  embedded?: boolean;
}

export default function ChatWindow({ embedded = false }: Props) {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState(INITIAL_SUGGESTIONS);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  const handleSend = useCallback(
    async (text: string) => {
      setError(null);
      const userMsg: ChatMessageType = {
        id: generateId(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setSuggestions([]);

      try {
        const response = await sendMessage({
          message: text,
          session_id: sessionId || undefined,
          channel: embedded ? "widget" : "web",
        });

        setSessionId(response.session_id);

        const assistantMsg: ChatMessageType = {
          id: generateId(),
          role: "assistant",
          content: response.message,
          timestamp: new Date(response.timestamp),
          intent: response.intent,
          metadata: response.metadata,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        // Context-aware suggestions
        if (response.intent === "greeting") {
          setSuggestions(INITIAL_SUGGESTIONS);
        } else if (response.intent === "order_tracking") {
          setSuggestions(["ORD-1001", "ORD-1002", "Check another order"]);
        } else {
          setSuggestions([]);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Something went wrong. Please try again.",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, embedded],
  );

  return (
    <div className="flex flex-col h-full bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-primary-600 text-white px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-6 h-6"
            >
              <path
                fillRule="evenodd"
                d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div>
            <h2 className="font-semibold text-base">Support Assistant</h2>
            <p className="text-primary-200 text-xs">
              {isLoading ? "Typing..." : "Online"}
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto chat-scroll p-4"
      >
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary-50 flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-8 h-8 text-primary-400"
              >
                <path
                  fillRule="evenodd"
                  d="M12 2.25c-2.429 0-4.817.178-7.152.521C2.87 3.061 1.5 4.795 1.5 6.741v6.018c0 1.946 1.37 3.68 3.348 3.97.877.129 1.761.234 2.652.316V21a.75.75 0 001.28.53l4.184-4.183A48.96 48.96 0 0012 17.25c2.429 0 4.817-.178 7.152-.521 1.978-.29 3.348-2.024 3.348-3.97V6.741c0-1.946-1.37-3.68-3.348-3.97A49.145 49.145 0 0012 2.25z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-500">
              Welcome! How can we help you today?
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Ask about FAQs, track an order, or file a complaint.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && <TypingIndicator />}

        {error && (
          <div className="text-center text-red-500 text-sm py-2 px-4 bg-red-50 rounded-lg mb-4">
            {error}
          </div>
        )}
      </div>

      {/* Quick replies */}
      <QuickReplies suggestions={suggestions} onSelect={handleSend} />

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
