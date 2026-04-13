import { useState, useRef, type FormEvent } from "react";
import Head from "next/head";
import { askQuery, ApiClientError, type AnswerPayload } from "@/lib/api";
import { AnswerCard } from "@/components/AnswerCard";
import { Nav } from "@/components/Nav";

const EXAMPLE_QUERIES = [
  "What is the intended use of the device?",
  "What are the contraindications?",
  "Summarise the risk management approach per ISO 14971.",
  "What clinical evidence supports the safety claims?",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnswerPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await askQuery({ user_query: q, top_k: 10 });
      setResult(data);
    } catch (err) {
      if (err instanceof ApiClientError) {
        setError(err.error.message);
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleExample(q: string) {
    setQuery(q);
    inputRef.current?.focus();
  }

  return (
    <>
      <Head>
        <title>RegIntel AI</title>
        <meta name="description" content="Clinical document intelligence for regulated medical device environments." />
      </Head>

      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Nav />

        {/* Main */}
        <main className="flex-1 max-w-2xl w-full mx-auto px-4 py-10 flex flex-col gap-6">
          {/* Hero */}
          <div className="text-center space-y-1">
            <h1 className="text-xl font-bold text-gray-900">Ask your documents</h1>
            <p className="text-sm text-gray-500">
              Grounded answers from clinical evidence — every claim cited, every response scored.
            </p>
          </div>

          {/* Query form */}
          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="relative">
              <textarea
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e as unknown as FormEvent);
                  }
                }}
                placeholder="Ask a question about your clinical documents…"
                rows={3}
                className="w-full resize-none rounded-xl border border-gray-300 bg-white px-4 py-3 pr-12 text-sm text-gray-900 placeholder-gray-400 shadow-sm focus:border-brand-600 focus:outline-none focus:ring-1 focus:ring-brand-600"
              />
              <button
                type="submit"
                disabled={!query.trim() || loading}
                className="absolute right-3 bottom-3 p-1.5 rounded-lg bg-brand-600 text-white hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                aria-label="Submit query"
              >
                {loading ? (
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.269 20.876L5.999 12zm0 0h7.5" />
                  </svg>
                )}
              </button>
            </div>
            <p className="text-xs text-gray-400 text-right">Enter to send · Shift+Enter for new line</p>
          </form>

          {/* Example queries */}
          {!result && !loading && (
            <div>
              <p className="text-xs font-medium text-gray-500 mb-2">Try an example</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLE_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleExample(q)}
                    className="text-xs px-3 py-1.5 rounded-full border border-gray-200 bg-white text-gray-600 hover:border-brand-600 hover:text-brand-600 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex gap-3 items-start text-sm text-red-800 bg-red-50 border border-red-200 rounded-xl p-4">
              <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <p>{error}</p>
            </div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 space-y-3 animate-pulse">
              <div className="flex gap-3">
                <div className="h-4 w-20 bg-gray-200 rounded-full" />
                <div className="h-4 w-28 bg-gray-200 rounded-full" />
              </div>
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 rounded w-full" />
                <div className="h-3 bg-gray-200 rounded w-5/6" />
                <div className="h-3 bg-gray-200 rounded w-4/6" />
              </div>
            </div>
          )}

          {/* Result */}
          {result && !loading && <AnswerCard result={result} />}
        </main>

        <footer className="text-center py-4 text-xs text-gray-400 border-t border-gray-100">
          RegIntel AI · Responses are grounded in uploaded documents only
        </footer>
      </div>
    </>
  );
}
