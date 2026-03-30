import ChatWidget from "@/components/ChatWidget";

export default function HomePage() {
  const frontendUrl =
    process.env.NEXT_PUBLIC_FRONTEND_URL || "http://localhost:3000";
  const backendApiUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Hero Section */}
      <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 sm:py-20">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 bg-primary-100 text-primary-700 text-sm font-medium px-4 py-1.5 rounded-full mb-6">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            AI-Powered Support
          </div>
          <h1 className="mb-4 text-[clamp(2rem,9vw,3rem)] font-bold text-gray-900 md:text-5xl">
            Customer Support
            <span className="text-primary-600"> Chatbot</span>
          </h1>
          <p className="mx-auto mb-8 max-w-2xl text-base text-gray-600 sm:text-lg">
            Get instant help with FAQs, order tracking, and support tickets.
            Our AI assistant is available 24/7 to answer your questions.
          </p>
          <div className="flex justify-center gap-4 max-[479px]:flex-col">
            <a
              href="/chat"
              className="flex min-h-11 items-center justify-center rounded-xl bg-primary-600 px-6 py-3 font-medium text-white
                         max-[479px]:w-full
                         hover:bg-primary-700 transition-colors shadow-md shadow-primary-200"
            >
              Open Full Chat
            </a>
            <a
              href="#features"
              className="flex min-h-11 items-center justify-center rounded-xl border border-gray-200 bg-white px-6 py-3 font-medium text-gray-700
                         max-[479px]:w-full
                         hover:bg-gray-50 transition-colors"
            >
              Learn More
            </a>
          </div>
        </div>

        {/* Features */}
        <div id="features" className="mt-12 grid grid-cols-1 gap-4 sm:mt-20 sm:gap-6 md:grid-cols-3">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-blue-600">
                <path fillRule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm11.378-3.917c-.89-.777-2.366-.777-3.255 0a.75.75 0 01-.988-1.129c1.454-1.272 3.776-1.272 5.23 0 1.513 1.324 1.513 3.518 0 4.842a3.75 3.75 0 01-.837.552c-.676.328-1.028.774-1.028 1.152v.75a.75.75 0 01-1.5 0v-.75c0-1.279 1.06-2.107 1.875-2.502.182-.088.351-.199.503-.331.83-.727.83-1.857 0-2.584zM12 18a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Smart FAQs</h3>
            <p className="text-sm text-gray-600">
              Powered by vector search (FAISS) for instant, relevant answers to
              your questions.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-green-600">
                <path d="M3.375 4.5C2.339 4.5 1.5 5.34 1.5 6.375V13.5h12V6.375c0-1.036-.84-1.875-1.875-1.875h-8.25zM13.5 15h-12v2.625c0 1.035.84 1.875 1.875 1.875h8.25c1.035 0 1.875-.84 1.875-1.875V15z" />
                <path d="M8.25 19.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0zM15.75 6.75a.75.75 0 00-.75.75v11.25c0 .087.015.17.042.248a3 3 0 015.958.464c.853-.175 1.522-.935 1.5-1.849V8.625a1.875 1.875 0 00-1.875-1.875h-4.875z" />
                <path d="M19.5 19.5a1.5 1.5 0 10-3 0 1.5 1.5 0 003 0z" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Order Tracking</h3>
            <p className="text-sm text-gray-600">
              Track your orders in real-time. Just provide your order ID and
              get instant status updates.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-orange-600">
                <path fillRule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.75 6a.75.75 0 00-1.5 0v6c0 .414.336.75.75.75h4.5a.75.75 0 000-1.5h-3.75V6z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">
              Complaint Escalation
            </h3>
            <p className="text-sm text-gray-600">
              Smart sentiment analysis automatically prioritizes and escalates
              urgent issues.
            </p>
          </div>
        </div>

        {/* API Section */}
        <div className="mt-16 rounded-2xl border border-gray-100 bg-white p-5 shadow-sm sm:p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Multi-Channel API
          </h2>
          <p className="text-gray-600 mb-6">
            Integrate our chatbot anywhere with a simple REST API or
            embeddable widget.
          </p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-xl text-sm font-mono overflow-x-auto">
            <pre>{`# REST API
curl -X POST ${backendApiUrl}/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is your return policy?"}'

# Embeddable Widget
<script src="${frontendUrl}/widget.js"
        data-api-url="${backendApiUrl}">
</script>`}</pre>
          </div>
        </div>
      </div>

      {/* Chat Widget */}
      <ChatWidget />
    </main>
  );
}
