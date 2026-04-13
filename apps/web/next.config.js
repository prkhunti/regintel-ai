/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        // API_URL is a server-side-only variable set in the Dockerfile (Docker)
        // or .env.local (local dev). Never use NEXT_PUBLIC_ here — that prefix
        // bakes the value at build time, so the runtime env is ignored.
        destination: `${process.env.API_URL || "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
