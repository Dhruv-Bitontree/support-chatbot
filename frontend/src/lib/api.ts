import { ChatRequest, ChatResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

class APIError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const data = await res
      .json()
      .catch(() => ({} as { error?: string; detail?: string }));
    const errorMessage = data.error || data.detail || "Something went wrong.";
    throw new APIError(res.status, errorMessage);
  }

  return res.json();
}

export async function sendMessage(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function healthCheck(): Promise<{ status: string }> {
  return request("/health");
}
