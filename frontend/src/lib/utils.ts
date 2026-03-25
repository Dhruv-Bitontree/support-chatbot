import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

export function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function parseServerTimestamp(value: string): Date {
  const raw = value.trim();
  if (!raw) {
    return new Date();
  }

  // Backward compatibility: old backend rows may be UTC without an explicit timezone.
  const hasTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(raw);
  const normalized = hasTimezone ? raw : `${raw}Z`;
  const parsed = new Date(normalized);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed;
  }

  // Final fallback for unexpected formats.
  const fallback = new Date(raw);
  return Number.isNaN(fallback.getTime()) ? new Date() : fallback;
}
