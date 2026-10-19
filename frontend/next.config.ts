import type { NextConfig } from "next";

// On Vercel, the top-level vercel.json routes /api/* to the Python service.
// Locally, proxy /api/* to uvicorn so the browser talks to one origin in both places.
const backendDevUrl = process.env.BACKEND_DEV_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // Gzip would buffer the Server-Sent Events from /api; Vercel's CDN compresses static assets anyway.
  compress: false,
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [{ source: "/api/:path*", destination: `${backendDevUrl}/api/:path*` }];
  },
};

export default nextConfig;
