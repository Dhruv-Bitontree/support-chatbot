import ChatWindow from "@/components/ChatWindow";

export default function ChatPage() {
  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl h-[85vh]">
        <ChatWindow />
      </div>
    </div>
  );
}
