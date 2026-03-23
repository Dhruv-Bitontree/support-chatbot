export type MessageRole = "user" | "assistant" | "system";

export type Intent =
  | "faq"
  | "order_tracking"
  | "complaint"
  | "general"
  | "greeting";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  intent?: Intent;
  metadata?: Record<string, unknown>;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  channel?: string;
}

export interface ChatResponse {
  message: string;
  session_id: string;
  intent?: Intent;
  metadata?: Record<string, unknown>;
  timestamp: string;
}

export interface WSMessage {
  message: string;
  session_id?: string;
  channel?: string;
}

export interface WSResponse {
  message: string;
  session_id: string;
  intent?: Intent;
  metadata?: Record<string, unknown>;
  error?: string;
}
