"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatMessage as ChatMessageType } from "@/lib/types";
import { sendMessage } from "@/lib/api";
import { generateId, parseServerTimestamp } from "@/lib/utils";
import { toast } from "sonner";
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
  const [showResetDialog, setShowResetDialog] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  const handleSend = useCallback(
    async (text: string) => {
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
          timestamp: parseServerTimestamp(response.timestamp),
          intent: response.intent,
          metadata: response.metadata,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        if (hasTicketCreated(response.metadata)) {
          toast.success("Support ticket created successfully.");
        }

        if (response.intent === "greeting") {
          setSuggestions(INITIAL_SUGGESTIONS);
        } else if (response.intent === "order_tracking") {
          setSuggestions(["ORD-1001", "ORD-1002", "Check another order"]);
        } else {
          setSuggestions([]);
        }
      } catch (err) {
        toast.error(resolveSendErrorMessage(err));
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, embedded],
  );

  const clearSession = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setSuggestions(INITIAL_SUGGESTIONS);
    setShowResetDialog(false);
    toast.success("New chat session started.");
  }, []);

  const handleNewSession = useCallback(() => {
    if (isLoading) return;

    if (messages.length > 0) {
      setShowResetDialog(true);
      return;
    }

    clearSession();
  }, [isLoading, messages.length, clearSession]);

  return (
    <div
      className={`relative flex h-full flex-col overflow-hidden rounded-none border-0 bg-white/95 shadow-none backdrop-blur sm:rounded-3xl sm:border sm:border-slate-200/80 sm:shadow-[0_18px_48px_rgba(15,23,42,0.14)] ${
        embedded ? "max-[479px]:rounded-none max-[479px]:border-0 max-[479px]:shadow-none" : ""
      }`}
    >
      <div className="relative overflow-hidden bg-gradient-to-r from-primary-700 via-primary-600 to-blue-500 px-5 py-4 text-white">
        <div className="pointer-events-none absolute -right-8 -top-10 h-24 w-24 rounded-full bg-white/10" />
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/20 ring-1 ring-white/30">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-6 w-6"
              >
                <path
                  fillRule="evenodd"
                  d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-semibold tracking-wide">Support Assistant</h2>
              <p className="flex items-center gap-1.5 text-xs text-blue-100">
                <span className="inline-flex h-2 w-2 rounded-full bg-emerald-300" />
                {isLoading ? "Typing..." : "Online"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleNewSession}
              type="button"
              disabled={isLoading}
              className="min-h-11 rounded-lg bg-white/20 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-white/30 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-white/40"
            >
              New chat
            </button>
          </div>
        </div>
      </div>

      <div ref={scrollRef} className="chat-scroll flex-1 overflow-y-auto bg-gradient-to-b from-slate-50 to-slate-100/80 px-4 py-5">
        {messages.length === 0 && (
          <div className="mx-auto mt-8 max-w-sm rounded-2xl border border-slate-200 bg-white/90 px-5 py-6 text-center shadow-sm">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-primary-50 text-primary-500">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-7 w-7"
              >
                <path
                  fillRule="evenodd"
                  d="M12 2.25c-2.429 0-4.817.178-7.152.521C2.87 3.061 1.5 4.795 1.5 6.741v6.018c0 1.946 1.37 3.68 3.348 3.97.877.129 1.761.234 2.652.316V21a.75.75 0 001.28.53l4.184-4.183A48.96 48.96 0 0012 17.25c2.429 0 4.817-.178 7.152-.521 1.978-.29 3.348-2.024 3.348-3.97V6.741c0-1.946-1.37-3.68-3.348-3.97A49.145 49.145 0 0012 2.25z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <p className="text-sm font-semibold text-slate-700">Welcome! How can we help you today?</p>
            <p className="mt-1 text-xs text-slate-500">
              Ask about FAQs, track an order, or file a complaint.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && <TypingIndicator />}
      </div>

      <QuickReplies suggestions={suggestions} onSelect={handleSend} />

      <ChatInput onSend={handleSend} disabled={isLoading} />

      {showResetDialog && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/45 p-4 backdrop-blur-[2px]"
          role="dialog"
          aria-modal="true"
          aria-label="Confirm new chat session"
        >
          <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-5 shadow-2xl">
            <p className="text-sm font-medium text-slate-700">
              Start a new chat session? Your current conversation will be cleared.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setShowResetDialog(false)}
                type="button"
                className="min-h-11 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-slate-200"
              >
                Cancel
              </button>
              <button
                onClick={clearSession}
                type="button"
                className="min-h-11 rounded-xl bg-primary-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-300"
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function hasTicketCreated(metadata: Record<string, unknown> | undefined): boolean {
  return typeof metadata?.ticket_id === "string" && metadata.ticket_id.length > 0;
}

function resolveSendErrorMessage(error: unknown): string {
  if (isConnectionError(error)) {
    return "Connection lost. Please check your internet.";
  }

  if (error instanceof Error) {
    const message = error.message.trim();
    if (!message || message === "Something went wrong.") {
      return "Failed to send message. Please try again.";
    }
    return message;
  }

  return "Something went wrong. Please try again.";
}

function isConnectionError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = error.message.toLowerCase();
  return (
    message.includes("network") ||
    message.includes("failed to fetch") ||
    message.includes("connection") ||
    message.includes("load failed")
  );
}
