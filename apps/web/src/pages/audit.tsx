import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import { Nav } from "@/components/Nav";
import {
  getAuditEvents,
  ApiClientError,
  type AuditEvent,
  type AuditEventType,
  type AuditEventsParams,
} from "@/lib/api";

const PAGE_SIZE = 50;

const EVENT_TYPE_LABELS: Record<AuditEventType, string> = {
  document_uploaded:    "Document Uploaded",
  document_processed:   "Document Processed",
  query_submitted:      "Query Submitted",
  answer_generated:     "Answer Generated",
  answer_refused:       "Answer Refused",
  pii_detected:         "PII Detected",
  injection_detected:   "Injection Detected",
  eval_run_started:     "Eval Run Started",
  eval_run_completed:   "Eval Run Completed",
};

const EVENT_TYPE_STYLE: Record<AuditEventType, string> = {
  document_uploaded:    "bg-blue-50 text-blue-700 border-blue-200",
  document_processed:   "bg-blue-50 text-blue-700 border-blue-200",
  query_submitted:      "bg-purple-50 text-purple-700 border-purple-200",
  answer_generated:     "bg-green-50 text-green-700 border-green-200",
  answer_refused:       "bg-yellow-50 text-yellow-800 border-yellow-200",
  pii_detected:         "bg-red-50 text-red-700 border-red-200",
  injection_detected:   "bg-red-50 text-red-700 border-red-200",
  eval_run_started:     "bg-gray-50 text-gray-600 border-gray-200",
  eval_run_completed:   "bg-gray-50 text-gray-600 border-gray-200",
};

function EventTypeBadge({ type }: { type: string }) {
  const label = EVENT_TYPE_LABELS[type as AuditEventType] ?? type;
  const style = EVENT_TYPE_STYLE[type as AuditEventType] ?? "bg-gray-50 text-gray-600 border-gray-200";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-medium ${style}`}>
      {label}
    </span>
  );
}

function formatTs(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }),
    time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
  };
}

const ALL_EVENT_TYPES = Object.keys(EVENT_TYPE_LABELS) as AuditEventType[];

export default function AuditPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  // Filters
  const [eventType, setEventType] = useState<AuditEventType | "">("");
  const [resourceType, setResourceType] = useState("");

  // Expanded row
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = useCallback(
    async (newOffset: number, append = false) => {
      setLoading(true);
      setError(null);
      try {
        const params: AuditEventsParams = {
          limit: PAGE_SIZE + 1,
          offset: newOffset,
        };
        if (eventType) params.event_type = eventType;
        if (resourceType.trim()) params.resource_type = resourceType.trim();

        const data = await getAuditEvents(params);
        const more = data.length > PAGE_SIZE;
        const page = more ? data.slice(0, PAGE_SIZE) : data;

        setEvents((prev) => (append ? [...prev, ...page] : page));
        setHasMore(more);
        setOffset(newOffset);
      } catch (err) {
        setError(err instanceof ApiClientError ? err.error.message : "Failed to load audit events.");
      } finally {
        setLoading(false);
      }
    },
    [eventType, resourceType]
  );

  // Reload when filters change
  useEffect(() => {
    setOffset(0);
    setEvents([]);
    load(0, false);
  }, [eventType, resourceType]); // load intentionally excluded — it's stable per filter state

  function handleFilterSubmit(e: React.FormEvent) {
    e.preventDefault();
    load(0, false);
  }

  return (
    <>
      <Head>
        <title>Audit Log · RegIntel AI</title>
      </Head>

      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Nav />

        <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8 space-y-5">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Audit Log</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Immutable record of all system events — newest first.
            </p>
          </div>

          {/* Filters */}
          <form
            onSubmit={handleFilterSubmit}
            className="flex flex-wrap gap-3 items-end bg-white border border-gray-200 rounded-xl p-4"
          >
            <div className="space-y-1">
              <label className="block text-xs font-medium text-gray-600">Event type</label>
              <select
                value={eventType}
                onChange={(e) => setEventType(e.target.value as AuditEventType | "")}
                className="rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600"
              >
                <option value="">All types</option>
                {ALL_EVENT_TYPES.map((t) => (
                  <option key={t} value={t}>{EVENT_TYPE_LABELS[t]}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="block text-xs font-medium text-gray-600">Resource type</label>
              <input
                type="text"
                value={resourceType}
                onChange={(e) => setResourceType(e.target.value)}
                placeholder="e.g. document, response"
                className="rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600 w-44"
              />
            </div>

            <button
              type="submit"
              className="px-4 py-1.5 rounded-lg bg-brand-600 text-white text-sm hover:bg-brand-700 transition-colors"
            >
              Filter
            </button>
            {(eventType || resourceType) && (
              <button
                type="button"
                onClick={() => { setEventType(""); setResourceType(""); }}
                className="px-3 py-1.5 rounded-lg border border-gray-300 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
              >
                Clear
              </button>
            )}
          </form>

          {/* Table */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {error && (
              <div className="px-5 py-4 text-sm text-red-700 bg-red-50 border-b border-red-100">
                {error}
              </div>
            )}

            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                  <th className="px-4 py-2.5 text-left font-medium w-40">Timestamp</th>
                  <th className="px-4 py-2.5 text-left font-medium">Event</th>
                  <th className="px-4 py-2.5 text-left font-medium hidden sm:table-cell">Resource</th>
                  <th className="px-4 py-2.5 text-left font-medium hidden md:table-cell w-32">Actor</th>
                  <th className="px-4 py-2.5 text-right font-medium w-16"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {events.map((ev) => {
                  const { date, time } = formatTs(ev.created_at);
                  const isOpen = expanded === ev.id;
                  return (
                    <>
                      <tr
                        key={ev.id}
                        className="hover:bg-gray-50 cursor-pointer"
                        onClick={() => setExpanded(isOpen ? null : ev.id)}
                      >
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="text-gray-800 tabular-nums">{time}</div>
                          <div className="text-gray-400 text-xs">{date}</div>
                        </td>
                        <td className="px-4 py-3">
                          <EventTypeBadge type={ev.event_type} />
                        </td>
                        <td className="px-4 py-3 hidden sm:table-cell">
                          {ev.resource_type && (
                            <span className="text-gray-600">
                              {ev.resource_type}
                              {ev.resource_id && (
                                <span className="text-gray-400 font-mono text-xs ml-1">
                                  {ev.resource_id.slice(0, 8)}
                                </span>
                              )}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 hidden md:table-cell text-gray-500 text-xs">
                          {ev.actor ?? <span className="text-gray-300">—</span>}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-400">
                          <svg
                            className={`w-4 h-4 inline-block transition-transform ${isOpen ? "rotate-180" : ""}`}
                            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                          </svg>
                        </td>
                      </tr>
                      {isOpen && (
                        <tr key={`${ev.id}-detail`} className="bg-gray-50">
                          <td colSpan={5} className="px-4 pb-4 pt-1">
                            <div className="rounded-lg bg-white border border-gray-200 p-3">
                              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                                Detail
                              </p>
                              <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                                {JSON.stringify(ev.detail, null, 2)}
                              </pre>
                              <p className="text-xs text-gray-400 mt-2 font-mono">id: {ev.id}</p>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}

                {!loading && events.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-10 text-center text-sm text-gray-400">
                      No audit events found.
                    </td>
                  </tr>
                )}

                {loading && (
                  <tr>
                    <td colSpan={5} className="px-4 py-6">
                      <div className="space-y-2 animate-pulse">
                        {[...Array(5)].map((_, i) => (
                          <div key={i} className="flex gap-4">
                            <div className="h-4 w-24 bg-gray-200 rounded" />
                            <div className="h-4 w-32 bg-gray-200 rounded" />
                            <div className="h-4 w-20 bg-gray-200 rounded" />
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>

            {hasMore && !loading && (
              <div className="px-4 py-3 border-t border-gray-100 text-center">
                <button
                  onClick={() => load(offset + PAGE_SIZE, true)}
                  className="text-sm text-brand-600 hover:underline"
                >
                  Load more
                </button>
              </div>
            )}
          </div>
        </main>

        <footer className="text-center py-4 text-xs text-gray-400 border-t border-gray-100">
          RegIntel AI · Audit log is append-only and immutable
        </footer>
      </div>
    </>
  );
}
