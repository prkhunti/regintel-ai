import type { CitationRead } from "@/lib/api";

interface Props {
  citations: CitationRead[];
}

export function CitationList({ citations }: Props) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
        Sources
      </h3>
      <ol className="space-y-2">
        {citations.map((c, i) => (
          <li key={c.id} className="flex gap-3 text-sm">
            <span className="mt-0.5 flex-shrink-0 w-5 h-5 rounded-full bg-brand-100 text-brand-600 text-xs font-bold flex items-center justify-center">
              {i + 1}
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
                <span className="font-medium text-gray-800 truncate">{c.document_title}</span>
                {c.section_title && (
                  <span className="text-gray-500 text-xs truncate">› {c.section_title}</span>
                )}
                {c.page_start != null && (
                  <span className="text-gray-400 text-xs">
                    p.{c.page_start}{c.page_end && c.page_end !== c.page_start ? `–${c.page_end}` : ""}
                  </span>
                )}
                <span className="text-gray-400 text-xs tabular-nums ml-auto">
                  {Math.round(c.relevance_score * 100)}% relevant
                </span>
              </div>
              <blockquote className="mt-0.5 pl-2 border-l-2 border-gray-200 text-gray-600 italic text-xs leading-relaxed">
                {c.quote}
              </blockquote>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
