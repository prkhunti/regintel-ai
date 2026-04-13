import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import { Nav } from "@/components/Nav";
import {
  listEvalCases,
  createEvalCase,
  deleteEvalCase,
  triggerEvalRun,
  listEvalRuns,
  ApiClientError,
  type EvalCaseRead,
  type EvalRunRead,
} from "@/lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

function MetricCell({ value, thresholds }: { value: number | null; thresholds?: [number, number] }) {
  if (value == null) return <span className="text-gray-300">—</span>;
  const pct = Math.round(value * 100);
  let color = "text-gray-700";
  if (thresholds) {
    color = pct >= thresholds[0] ? "text-green-700" : pct >= thresholds[1] ? "text-yellow-700" : "text-red-700";
  }
  return <span className={`tabular-nums font-medium ${color}`}>{pct}%</span>;
}

function formatTs(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// ── Add case form ─────────────────────────────────────────────────────────────

function AddCaseForm({ onAdded }: { onAdded: () => void }) {
  const [query, setQuery] = useState("");
  const [chunkIds, setChunkIds] = useState("");
  const [isInsufficient, setIsInsufficient] = useState(false);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const ids = chunkIds.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      await createEvalCase({
        query: query.trim(),
        expected_chunk_ids: ids,
        is_insufficient: isInsufficient,
        notes: notes.trim() || undefined,
      });
      setQuery("");
      setChunkIds("");
      setIsInsufficient(false);
      setNotes("");
      onAdded();
    } catch (err) {
      setError(err instanceof ApiClientError ? err.error.message : "Failed to create case.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-800">Add eval case</h3>
      {error && <p className="text-xs text-red-600">{error}</p>}
      <div className="space-y-1">
        <label className="block text-xs font-medium text-gray-600">Query</label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter the evaluation query…"
          className="w-full rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600"
          required
        />
      </div>
      <div className="flex gap-3 flex-wrap">
        <div className="space-y-1 flex-1 min-w-48">
          <label className="block text-xs font-medium text-gray-600">
            Expected chunk IDs <span className="text-gray-400">(comma-separated)</span>
          </label>
          <input
            value={chunkIds}
            onChange={(e) => setChunkIds(e.target.value)}
            placeholder="uuid-1, uuid-2"
            className="w-full rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600"
          />
        </div>
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Notes</label>
          <input
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Optional"
            className="rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600 w-40"
          />
        </div>
      </div>
      <div className="flex items-center gap-2">
        <input
          id="insufficient"
          type="checkbox"
          checked={isInsufficient}
          onChange={(e) => setIsInsufficient(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="insufficient" className="text-xs text-gray-600">
          Expected answer: insufficient context (should be refused)
        </label>
      </div>
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="px-4 py-1.5 rounded-lg bg-brand-600 text-white text-sm hover:bg-brand-700 disabled:opacity-40 transition-colors"
      >
        {loading ? "Adding…" : "Add case"}
      </button>
    </form>
  );
}

// ── Trigger run form ──────────────────────────────────────────────────────────

function TriggerRunForm({ caseCount, onComplete }: { caseCount: number; onComplete: () => void }) {
  const [label, setLabel] = useState("");
  const [model, setModel] = useState("gpt-4o");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!label.trim()) return;
    setLoading(true);
    setError(null);
    try {
      await triggerEvalRun({
        label: label.trim(),
        model_name: model,
        retriever_config: { fusion: "rrf" },
      });
      setLabel("");
      onComplete();
    } catch (err) {
      setError(err instanceof ApiClientError ? err.error.message : "Run failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-800">
        Trigger eval run
        <span className="ml-2 text-xs font-normal text-gray-500">{caseCount} case{caseCount !== 1 ? "s" : ""} available</span>
      </h3>
      {error && <p className="text-xs text-red-600">{error}</p>}
      <div className="flex gap-3 flex-wrap items-end">
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Run label</label>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. baseline-gpt4o"
            className="rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600 w-48"
            required
          />
        </div>
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded-lg border border-gray-300 text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-600"
          >
            <option value="gpt-4o">gpt-4o</option>
            <option value="gpt-4o-mini">gpt-4o-mini</option>
            <option value="claude-sonnet-4-6">claude-sonnet-4-6</option>
            <option value="claude-haiku-4-5-20251001">claude-haiku-4-5</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={loading || caseCount === 0 || !label.trim()}
          className="px-4 py-1.5 rounded-lg bg-brand-600 text-white text-sm hover:bg-brand-700 disabled:opacity-40 transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running…
            </span>
          ) : "Run eval"}
        </button>
      </div>
    </form>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function EvalPage() {
  const [cases, setCases] = useState<EvalCaseRead[]>([]);
  const [runs, setRuns] = useState<EvalRunRead[]>([]);
  const [loadingCases, setLoadingCases] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [tab, setTab] = useState<"runs" | "cases">("runs");

  const refreshCases = useCallback(async () => {
    setLoadingCases(true);
    try { setCases(await listEvalCases()); } catch { /* ignore */ }
    finally { setLoadingCases(false); }
  }, []);

  const refreshRuns = useCallback(async () => {
    setLoadingRuns(true);
    try { setRuns(await listEvalRuns()); } catch { /* ignore */ }
    finally { setLoadingRuns(false); }
  }, []);

  useEffect(() => { refreshCases(); refreshRuns(); }, [refreshCases, refreshRuns]);

  async function handleDelete(id: string) {
    try {
      await deleteEvalCase(id);
      setCases((prev) => prev.filter((c) => c.id !== id));
    } catch { /* ignore */ }
  }

  return (
    <>
      <Head><title>Evaluation Harness · RegIntel AI</title></Head>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Nav />

        <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8 space-y-6">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Evaluation Harness</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Measure retrieval and answer quality against labelled ground-truth cases.
            </p>
          </div>

          <AddCaseForm onAdded={refreshCases} />
          <TriggerRunForm caseCount={cases.length} onComplete={refreshRuns} />

          {/* Tab bar */}
          <div className="border-b border-gray-200 flex gap-1">
            {(["runs", "cases"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  tab === t
                    ? "border-brand-600 text-brand-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                {t === "runs" ? `Runs (${runs.length})` : `Cases (${cases.length})`}
              </button>
            ))}
          </div>

          {/* Runs table */}
          {tab === "runs" && (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-100 bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                    <th className="px-4 py-2.5 text-left font-medium">Label</th>
                    <th className="px-4 py-2.5 text-left font-medium hidden sm:table-cell">Model</th>
                    <th className="px-4 py-2.5 text-center font-medium">Cases</th>
                    <th className="px-4 py-2.5 text-center font-medium">Recall@10</th>
                    <th className="px-4 py-2.5 text-center font-medium hidden md:table-cell">MRR</th>
                    <th className="px-4 py-2.5 text-center font-medium hidden md:table-cell">Citation Recall</th>
                    <th className="px-4 py-2.5 text-center font-medium hidden lg:table-cell">Refusal Acc</th>
                    <th className="px-4 py-2.5 text-right font-medium hidden lg:table-cell">Latency</th>
                    <th className="px-4 py-2.5 text-right font-medium">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {loadingRuns ? (
                    <tr>
                      <td colSpan={9} className="px-4 py-6">
                        <div className="space-y-2 animate-pulse">
                          {[...Array(3)].map((_, i) => (
                            <div key={i} className="flex gap-4">
                              <div className="h-4 w-32 bg-gray-200 rounded" />
                              <div className="h-4 w-20 bg-gray-200 rounded" />
                              <div className="h-4 w-12 bg-gray-200 rounded" />
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ) : runs.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-4 py-10 text-center text-sm text-gray-400">
                        No runs yet. Add cases and trigger a run above.
                      </td>
                    </tr>
                  ) : (
                    runs.map((run) => (
                      <tr key={run.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3 font-medium text-gray-800">{run.label}</td>
                        <td className="px-4 py-3 text-gray-500 text-xs hidden sm:table-cell font-mono">
                          {run.model_name}
                        </td>
                        <td className="px-4 py-3 text-center text-gray-600">{run.total_cases}</td>
                        <td className="px-4 py-3 text-center">
                          <MetricCell value={run.recall_at_10} thresholds={[70, 50]} />
                        </td>
                        <td className="px-4 py-3 text-center hidden md:table-cell">
                          <MetricCell value={run.mrr} thresholds={[60, 40]} />
                        </td>
                        <td className="px-4 py-3 text-center hidden md:table-cell">
                          <MetricCell value={run.faithfulness_score} thresholds={[70, 50]} />
                        </td>
                        <td className="px-4 py-3 text-center hidden lg:table-cell">
                          <MetricCell value={run.refusal_accuracy} thresholds={[90, 70]} />
                        </td>
                        <td className="px-4 py-3 text-right text-gray-500 tabular-nums hidden lg:table-cell">
                          {run.mean_latency_ms != null ? `${run.mean_latency_ms}ms` : "—"}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-400 text-xs whitespace-nowrap">
                          {formatTs(run.created_at)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Cases table */}
          {tab === "cases" && (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-100 bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                    <th className="px-4 py-2.5 text-left font-medium">Query</th>
                    <th className="px-4 py-2.5 text-center font-medium hidden sm:table-cell">Expected chunks</th>
                    <th className="px-4 py-2.5 text-center font-medium hidden md:table-cell">Insufficient</th>
                    <th className="px-4 py-2.5 text-left font-medium hidden lg:table-cell">Notes</th>
                    <th className="px-4 py-2.5 text-right font-medium w-12"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {loadingCases ? (
                    <tr>
                      <td colSpan={5} className="px-4 py-6">
                        <div className="space-y-2 animate-pulse">
                          {[...Array(3)].map((_, i) => (
                            <div key={i} className="h-4 w-full bg-gray-200 rounded" />
                          ))}
                        </div>
                      </td>
                    </tr>
                  ) : cases.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-4 py-10 text-center text-sm text-gray-400">
                        No eval cases yet. Add one above.
                      </td>
                    </tr>
                  ) : (
                    cases.map((c) => (
                      <tr key={c.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-gray-800 max-w-xs truncate" title={c.query}>
                          {c.query}
                        </td>
                        <td className="px-4 py-3 text-center text-gray-500 hidden sm:table-cell">
                          {c.expected_chunk_ids.length}
                        </td>
                        <td className="px-4 py-3 text-center hidden md:table-cell">
                          {c.is_insufficient ? (
                            <span className="text-yellow-700 text-xs font-medium">Yes</span>
                          ) : (
                            <span className="text-gray-300 text-xs">No</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-gray-400 text-xs hidden lg:table-cell truncate max-w-xs">
                          {c.notes ?? "—"}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            onClick={() => handleDelete(c.id)}
                            className="text-gray-400 hover:text-red-500 transition-colors"
                            title="Delete case"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </main>

        <footer className="text-center py-4 text-xs text-gray-400 border-t border-gray-100">
          RegIntel AI · Evaluation metrics computed end-to-end against ground-truth cases
        </footer>
      </div>
    </>
  );
}
