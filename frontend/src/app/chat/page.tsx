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
          ? "h-screen w-screen bg-gradient-to-br from-slate-50 to-blue-50"
          : "h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-4"
      }
    >
      {!isEmbed && (
        <button
          type="button"
          onClick={handleBack}
          className="fixed left-4 top-4 z-30 px-6 py-3 bg-primary-600 text-white rounded-xl font-medium
                         hover:bg-primary-700 transition-colors shadow-md shadow-primary-200"
        >
          {"Back"}
        </button>
      )}
      <div className={isEmbed ? "h-full w-full" : "w-full max-w-2xl h-[85vh]"}>
        <ChatWindow />
      </div>
    </div>
  );
}
