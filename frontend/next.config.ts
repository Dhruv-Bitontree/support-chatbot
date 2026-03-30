import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

const allowedDevOrigins =
  process.env.NEXT_ALLOWED_DEV_ORIGINS
    ?.split(",")
    .map((origin) => origin.trim())
    .filter(Boolean) ?? [
      "localhost",
      "127.0.0.1",
      "192.168.1.25",
      "multifoliate-hilaria-uncynical.ngrok-free.dev",
    ];

const baseConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"}/:path*`,
      },
    ];
  },
};

export default function nextConfig(phase: string): NextConfig {
  return {
    ...baseConfig,
    ...(phase === PHASE_DEVELOPMENT_SERVER ? { allowedDevOrigins } : {}),
    // Keep development artifacts separate from production build output.
    // This avoids corrupted manifests/files when switching between `dev` and `build`.
    distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
  };
}
