/**
 * Typed API client for the RegIntel backend.
 * All requests go through Next.js rewrites → /api/v1/*
 */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type DocumentType =
  | "clinical_evaluation_report"
  | "risk_management_file"
  | "software_requirements"
  | "ifu"
  | "cybersecurity"
  | "standard_guidance"
  | "other";

export type ProcessingStatus = "pending" | "processing" | "complete" | "failed";

export interface DocumentSummary {
  id: string;
  title: string;
  document_type: DocumentType;
  version: string;
  status: ProcessingStatus;
  uploaded_at: string;
  chunk_count: number;
}

export interface CitationRead {
  id: string;
  chunk_id: string;
  document_title: string;
  section_title: string | null;
  page_start: number | null;
  page_end: number | null;
  quote: string;
  relevance_score: number;
}

export interface AnswerPayload {
  query_id: string;
  response_id: string;
  query_text: string;
  answer: string;
  confidence: number;
  risk_level: RiskLevel;
  citations: CitationRead[];
  evidence_snippets: string[];
  warnings: string[];
  refused: boolean;
  refusal_reason: string | null;
  latency_ms: number | null;
}

export interface QueryCreate {
  user_query: string;
  top_k?: number;
  document_ids?: string[];
}

export type AuditEventType =
  | "document_uploaded"
  | "document_processed"
  | "query_submitted"
  | "answer_generated"
  | "answer_refused"
  | "pii_detected"
  | "injection_detected"
  | "eval_run_started"
  | "eval_run_completed";

export interface AuditEvent {
  id: string;
  event_type: AuditEventType;
  actor: string | null;
  resource_type: string | null;
  resource_id: string | null;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface AuditEventsParams {
  event_type?: AuditEventType;
  resource_type?: string;
  resource_id?: string;
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}

export interface ApiError {
  code: number;
  message: string;
  detail?: unknown;
}

class ApiClientError extends Error {
  constructor(public readonly error: ApiError) {
    super(error.message);
    this.name = "ApiClientError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const json = await res.json();

  if (!res.ok) {
    const err: ApiError = json?.error ?? { code: res.status, message: "Unknown error" };
    throw new ApiClientError(err);
  }

  return json as T;
}

export async function askQuery(payload: QueryCreate): Promise<AnswerPayload> {
  return post<AnswerPayload>("/api/v1/query", payload);
}

export async function uploadDocument(
  file: File,
  title: string,
  documentType: DocumentType,
  version = "1.0",
): Promise<DocumentSummary> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("title", title);
  fd.append("document_type", documentType);
  fd.append("version", version);

  const res = await fetch("/api/v1/documents/upload", { method: "POST", body: fd });
  const json = await res.json();
  if (!res.ok) {
    const err: ApiError = json?.error ?? { code: res.status, message: "Upload failed" };
    throw new ApiClientError(err);
  }
  return json as DocumentSummary;
}

export async function listDocuments(limit = 100, skip = 0): Promise<DocumentSummary[]> {
  const res = await fetch(`/api/v1/documents?limit=${limit}&skip=${skip}`);
  const json = await res.json();
  if (!res.ok) {
    const err: ApiError = json?.error ?? { code: res.status, message: "Unknown error" };
    throw new ApiClientError(err);
  }
  return json as DocumentSummary[];
}

export async function getDocument(id: string): Promise<DocumentSummary> {
  const res = await fetch(`/api/v1/documents/${id}`);
  const json = await res.json();
  if (!res.ok) {
    const err: ApiError = json?.error ?? { code: res.status, message: "Unknown error" };
    throw new ApiClientError(err);
  }
  return json as DocumentSummary;
}

// ── Health ────────────────────────────────────────────────────────────────────

export type HealthStatus = "ok" | "degraded" | "error" | "unconfigured" | "quota_exceeded";

export interface ComponentHealth {
  status: HealthStatus;
  latency_ms?: number;
  detail?: string;
  provider?: string;
}

export interface HealthReport {
  status: "ok" | "degraded";
  components: {
    database: ComponentHealth;
    redis: ComponentHealth;
    llm: ComponentHealth;
  };
}

export async function getHealth(): Promise<HealthReport> {
  const res = await fetch("/api/v1/health");
  const json = await res.json();
  if (!res.ok) throw new ApiClientError(json?.error ?? { code: res.status, message: "Health check failed" });
  return json as HealthReport;
}

export async function getAuditEvents(params: AuditEventsParams = {}): Promise<AuditEvent[]> {
  const qs = new URLSearchParams();
  if (params.event_type) qs.set("event_type", params.event_type);
  if (params.resource_type) qs.set("resource_type", params.resource_type);
  if (params.resource_id) qs.set("resource_id", params.resource_id);
  if (params.from) qs.set("from", params.from);
  if (params.to) qs.set("to", params.to);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));

  const res = await fetch(`/api/v1/audit/events?${qs}`);
  const json = await res.json();
  if (!res.ok) {
    const err: ApiError = json?.error ?? { code: res.status, message: "Unknown error" };
    throw new ApiClientError(err);
  }
  return json as AuditEvent[];
}

// ── Eval ─────────────────────────────────────────────────────────────────────

export interface EvalCaseCreate {
  query: string;
  expected_chunk_ids: string[];
  is_insufficient?: boolean;
  notes?: string;
}

export interface EvalCaseRead {
  id: string;
  query: string;
  expected_chunk_ids: string[];
  expected_answer_pattern: string | null;
  is_insufficient: boolean;
  notes: string | null;
  created_at: string;
}

export interface EvalRunCreate {
  label: string;
  model_name: string;
  retriever_config: Record<string, string>;
  case_ids?: string[] | null;
}

export interface EvalRunRead {
  id: string;
  label: string;
  model_name: string;
  retriever_config: Record<string, unknown>;
  total_cases: number;
  recall_at_10: number | null;
  precision_at_10: number | null;
  mrr: number | null;
  faithfulness_score: number | null;
  hallucination_rate: number | null;
  refusal_accuracy: number | null;
  mean_latency_ms: number | null;
  created_at: string;
}

export async function listEvalCases(): Promise<EvalCaseRead[]> {
  const res = await fetch("/api/v1/eval/cases?limit=200");
  const json = await res.json();
  if (!res.ok) throw new ApiClientError(json?.error ?? { code: res.status, message: "Unknown error" });
  return json as EvalCaseRead[];
}

export async function createEvalCase(payload: EvalCaseCreate): Promise<EvalCaseRead> {
  return post<EvalCaseRead>("/api/v1/eval/cases", payload);
}

export async function deleteEvalCase(id: string): Promise<void> {
  const res = await fetch(`/api/v1/eval/cases/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const json = await res.json().catch(() => ({}));
    throw new ApiClientError(json?.error ?? { code: res.status, message: "Delete failed" });
  }
}

export async function triggerEvalRun(payload: EvalRunCreate): Promise<EvalRunRead> {
  return post<EvalRunRead>("/api/v1/eval/runs", payload);
}

export async function listEvalRuns(): Promise<EvalRunRead[]> {
  const res = await fetch("/api/v1/eval/runs?limit=50");
  const json = await res.json();
  if (!res.ok) throw new ApiClientError(json?.error ?? { code: res.status, message: "Unknown error" });
  return json as EvalRunRead[];
}

export { ApiClientError };
