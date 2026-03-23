import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Customer Support Chatbot",
  description:
    "AI-powered customer support with FAQ, order tracking, and complaint management.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
