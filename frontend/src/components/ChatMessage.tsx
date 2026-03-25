"use client";

import { cn, formatTime } from "@/lib/utils";
import { ChatMessage as ChatMessageType } from "@/lib/types";
import DOMPurify from "dompurify";

interface Props {
  message: ChatMessageType;
}

let purifyHookInitialized = false;

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";
  const sanitizedContent = sanitizeMessageHtml(formatMarkdown(message.content));

  return (
    <div className={cn("mb-4 flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div className={cn("flex max-w-[85%] items-end gap-2 sm:max-w-[84%]", isUser && "flex-row-reverse")}>
        <div
          className={cn(
            "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-sm font-semibold",
            isUser
              ? "bg-primary-600 text-white shadow-sm"
              : "bg-slate-200 text-slate-700",
          )}
        >
          {isUser ? "U" : "AI"}
        </div>

        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm",
            isUser
              ? "rounded-br-sm bg-gradient-to-br from-primary-600 to-primary-700 text-white"
              : "rounded-bl-sm border border-slate-200 bg-white text-slate-800",
          )}
        >
          <div
            className="chat-message"
            dangerouslySetInnerHTML={{ __html: sanitizedContent }}
          />
          <div
            className={cn(
              "mt-1 text-[10px]",
              isUser ? "text-blue-100" : "text-slate-400",
            )}
          >
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
}

function sanitizeMessageHtml(content: string): string {
  if (!purifyHookInitialized) {
    DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      if ("tagName" in node && (node as Element).tagName === "A") {
        const anchor = node as HTMLAnchorElement;
        if (anchor.getAttribute("href")) {
          anchor.setAttribute("target", "_blank");
          anchor.setAttribute("rel", "noopener noreferrer");
        }
      }
    });
    purifyHookInitialized = true;
  }

  return DOMPurify.sanitize(content, {
    ALLOWED_TAGS: ["b", "i", "em", "strong", "a", "ul", "ol", "li", "p", "br", "code", "pre"],
    ALLOWED_ATTR: ["href", "target", "rel"],
  });
}

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n- /g, "\n&bull; ")
    .replace(/\n/g, "<br />");
}
