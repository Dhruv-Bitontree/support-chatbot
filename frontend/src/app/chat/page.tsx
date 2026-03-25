"use client";

import { useRouter, useSearchParams } from "next/navigation";
import ChatWindow from "@/components/ChatWindow";

export default function ChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const isEmbed = searchParams.get("embed") === "true";

  const handleBack = () => {
    if (window.history.length > 1) {
      router.back();
      return;
    }
    router.push("/");
  };

  return (
    <div
      className={
        isEmbed
          ? "h-[100dvh] w-screen bg-gradient-to-br from-slate-50 to-blue-50"
          : "flex h-[100dvh] flex-col bg-gradient-to-br from-slate-50 to-blue-50 sm:items-center sm:justify-center sm:p-4"
      }
    >
      {!isEmbed && (
        <button
          type="button"
          onClick={handleBack}
          className="m-3 inline-flex min-h-11 items-center rounded-xl bg-primary-600 px-4 py-2.5 text-white shadow-md shadow-primary-200 transition-colors hover:bg-primary-700 sm:fixed sm:left-4 sm:top-4 sm:z-30 sm:m-0 sm:px-6 sm:py-3"
        >
          {"Back"}
        </button>
      )}
      <div
        className={
          isEmbed
            ? "h-full w-full"
            : "h-[calc(100dvh-4.75rem)] w-full sm:h-[85vh] sm:max-w-2xl"
        }
      >
        <ChatWindow />
      </div>
    </div>
  );
}
