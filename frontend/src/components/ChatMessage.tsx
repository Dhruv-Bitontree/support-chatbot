"use client";

import { cn, formatTime } from "@/lib/utils";
import { ChatMessage as ChatMessageType } from "@/lib/types";

interface Props {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn("flex w-full mb-4", isUser ? "justify-end" : "justify-start")}
    >
      <div className={cn("flex items-end gap-2 max-w-[80%]", isUser && "flex-row-reverse")}>
        {/* Avatar */}
        <div
          className={cn(
            "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium",
            isUser
              ? "bg-primary-600 text-white"
              : "bg-gray-200 text-gray-600",
          )}
        >
          {isUser ? "U" : "AI"}
        </div>

        {/* Bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "bg-primary-600 text-white rounded-br-sm"
              : "bg-gray-100 text-gray-800 rounded-bl-sm",
          )}
        >
          <div
            className="chat-message"
            dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
          />
          <div
            className={cn(
              "text-[10px] mt-1",
              isUser ? "text-primary-200" : "text-gray-400",
            )}
          >
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
}

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n- /g, "\n• ")
    .replace(/\n/g, "<br />");
}
