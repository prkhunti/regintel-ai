import type { AnswerPayload } from "@/lib/api";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationList } from "./CitationList";

interface Props {
  result: AnswerPayload;
}

function renderAnswerWithCitations(text: string): React.ReactNode {
  // Render [N] markers as superscript badges
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return (
        <sup key={i} className="ml-0.5 text-brand-600 font-semibold text-xs">
          [{match[1]}]
        </sup>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function AnswerCard({ result }: Props) {
  const latency = result.latency_ms != null ? `${(result.latency_ms / 1000).toFixed(2)}s` : null;

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between gap-4 flex-wrap">
        <ConfidenceBadge confidence={result.confidence} riskLevel={result.risk_level} />
        {latency && (
          <span className="text-xs text-gray-400 tabular-nums">{latency}</span>
        )}
      </div>

      {/* Body */}
      <div className="px-5 py-4">
        {result.refused ? (
          <div className="flex gap-3 items-start text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
            <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
            <p>{result.refusal_reason ?? "Insufficient context to answer this query."}</p>
          </div>
        ) : (
          <p className="text-gray-800 text-sm leading-relaxed">
            {renderAnswerWithCitations(result.answer)}
          </p>
        )}

        <CitationList citations={result.citations} />

        {result.warnings.length > 0 && (
          <ul className="mt-3 space-y-1">
            {result.warnings.map((w, i) => (
              <li key={i} className="text-xs text-yellow-700 flex gap-1.5 items-center">
                <span>⚠</span> {w}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="px-5 py-2 bg-gray-50 border-t border-gray-100 text-xs text-gray-400 flex gap-3">
        <span>Query {result.query_id.slice(0, 8)}</span>
        <span>·</span>
        <span>{result.citations.length} citation{result.citations.length !== 1 ? "s" : ""}</span>
      </div>
    </div>
  );
}
