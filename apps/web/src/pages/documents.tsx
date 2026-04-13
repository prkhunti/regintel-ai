import { useState, useRef, useEffect, useCallback } from "react";
import Head from "next/head";
import {
  uploadDocument,
  listDocuments,
  getDocument,
  getHealth,
  ApiClientError,
  type DocumentSummary,
  type DocumentType,
  type HealthReport,
} from "@/lib/api";
import { Nav } from "@/components/Nav";

// ── Types ─────────────────────────────────────────────────────────────────────

type UploadStatus = "queued" | "uploading" | "ingesting" | "complete" | "failed";

interface UploadItem {
  id: string; // local key
  file: File;
  relativePath: string;
  status: UploadStatus;
  docId?: string;
  error?: string;
}

const DOC_TYPE_OPTIONS: { value: DocumentType; label: string }[] = [
  { value: "clinical_evaluation_report", label: "Clinical Evaluation Report" },
  { value: "risk_management_file", label: "Risk Management File" },
  { value: "software_requirements", label: "Software Requirements" },
  { value: "ifu", label: "Instructions for Use (IFU)" },
  { value: "cybersecurity", label: "Cybersecurity" },
  { value: "standard_guidance", label: "Standard / Guidance" },
  { value: "other", label: "Other" },
];

const STATUS_STYLE: Record<UploadStatus | "pending" | "processing", string> = {
  queued:     "bg-gray-100 text-gray-600",
  uploading:  "bg-blue-100 text-blue-700",
  ingesting:  "bg-yellow-100 text-yellow-700",
  complete:   "bg-green-100 text-green-700",
  failed:     "bg-red-100 text-red-700",
  pending:    "bg-yellow-100 text-yellow-700",
  processing: "bg-blue-100 text-blue-700",
};

const STATUS_LABEL: Record<string, string> = {
  queued:     "Queued",
  uploading:  "Uploading…",
  ingesting:  "Ingesting…",
  complete:   "Complete",
  failed:     "Failed",
  pending:    "Pending",
  processing: "Processing",
};

// Max simultaneous HTTP upload requests
const UPLOAD_CONCURRENCY = 2;

// ── Component ─────────────────────────────────────────────────────────────────

export default function DocumentsPage() {
  const [docType, setDocType] = useState<DocumentType>("other");
  const [queue, setQueue] = useState<UploadItem[]>([]);
  const [library, setLibrary] = useState<DocumentSummary[]>([]);
  const [libraryLoading, setLibraryLoading] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  // Tick counter drives the CSS shimmer on ingesting rows without extra deps
  const [tick, setTick] = useState(0);

  const folderInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Track which docIds are still polling
  const pollingRef = useRef<Set<string>>(new Set());

  // ── Library fetch ───────────────────────────────────────────────────────────
  const refreshLibrary = useCallback(async () => {
    try {
      const docs = await listDocuments(200);
      setLibrary(docs);
    } catch {
      // non-fatal
    } finally {
      setLibraryLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshLibrary();
    const interval = setInterval(refreshLibrary, 5000);
    return () => clearInterval(interval);
  }, [refreshLibrary]);

  // Health probe — runs once on mount, then every 30 s
  useEffect(() => {
    const check = async () => {
      try {
        const report = await getHealth();
        setHealth(report);
      } catch {
        setHealth(null);
      } finally {
        setHealthLoading(false);
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  // Advance the shimmer tick every 400 ms while anything is ingesting
  useEffect(() => {
    const ingesting = queue.some((q) => q.status === "uploading" || q.status === "ingesting");
    if (!ingesting) return;
    const id = setInterval(() => setTick((t) => (t + 1) % 100), 400);
    return () => clearInterval(id);
  }, [queue]);

  // ── Poll a single document until terminal ──────────────────────────────────
  const pollDocument = useCallback((localId: string, docId: string) => {
    if (pollingRef.current.has(docId)) return;
    pollingRef.current.add(docId);

    const tick = async () => {
      try {
        const doc = await getDocument(docId);
        if (doc.status === "complete" || doc.status === "failed") {
          pollingRef.current.delete(docId);
          setQueue((prev) =>
            prev.map((item) =>
              item.id === localId
                ? { ...item, status: doc.status === "complete" ? "complete" : "failed" }
                : item
            )
          );
          refreshLibrary();
        } else {
          setTimeout(tick, 3000);
        }
      } catch {
        pollingRef.current.delete(docId);
        setQueue((prev) =>
          prev.map((item) =>
            item.id === localId ? { ...item, status: "failed", error: "Polling failed" } : item
          )
        );
      }
    };

    setTimeout(tick, 2000);
  }, [refreshLibrary]);

  // ── Upload all items, then start ingestion polling for each ─────────────────
  //
  // Phase 1 (upload): run all HTTP POSTs with a concurrency cap. Every item
  //   stays in "uploading" until its own request completes — the others don't
  //   flip to "ingesting" until their own upload is done either.
  // Phase 2 (ingest): once an upload finishes we immediately start polling
  //   that document, so ingestion tracks run in parallel across all files.
  const runQueue = useCallback(
    async (items: UploadItem[]) => {
      // Mark all as uploading up-front so the user sees the full list immediately
      setQueue((prev) =>
        prev.map((q) =>
          items.some((it) => it.id === q.id) ? { ...q, status: "uploading" } : q
        )
      );

      // Semaphore: at most UPLOAD_CONCURRENCY requests in flight at once
      const pending = [...items];
      const worker = async () => {
        while (pending.length > 0) {
          const item = pending.shift()!;
          try {
            const title = item.relativePath
              .replace(/\.pdf$/i, "")
              .replace(/[/_-]+/g, " ")
              .trim();
            const doc = await uploadDocument(item.file, title, docType);
            // Upload done — move to ingesting and start polling immediately
            setQueue((prev) =>
              prev.map((q) =>
                q.id === item.id ? { ...q, status: "ingesting", docId: doc.id } : q
              )
            );
            pollDocument(item.id, doc.id);
          } catch (err) {
            const msg = err instanceof ApiClientError ? err.error.message : "Upload failed";
            setQueue((prev) =>
              prev.map((q) => (q.id === item.id ? { ...q, status: "failed", error: msg } : q))
            );
          }
        }
      };

      // Launch N workers that drain the shared pending array
      await Promise.all(
        Array.from({ length: Math.min(UPLOAD_CONCURRENCY, items.length) }, worker)
      );
    },
    [docType, pollDocument]
  );

  // ── File selection helpers ─────────────────────────────────────────────────
  const enqueueFiles = useCallback(
    (files: File[]) => {
      const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
      if (pdfs.length === 0) return;

      const newItems: UploadItem[] = pdfs.map((f) => ({
        id: `${Date.now()}-${Math.random()}`,
        file: f,
        // webkitRelativePath gives subfolder path; fall back to name
        relativePath: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name,
        status: "queued",
      }));

      setQueue((prev) => [...prev, ...newItems]);
      runQueue(newItems);
    },
    [runQueue]
  );

  function handleFolderChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    enqueueFiles(Array.from(e.target.files));
    e.target.value = "";
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    enqueueFiles(Array.from(e.target.files));
    e.target.value = "";
  }

  // ── Drag-and-drop ──────────────────────────────────────────────────────────
  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }
  function handleDragLeave() {
    setIsDragging(false);
  }
  async function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);

    // IMPORTANT: DataTransferItem references are only valid synchronously during
    // the event. Collect all entries/files before any await, then process async.
    type Collected = { entry: FileSystemEntry } | { file: File };
    const collected: Collected[] = [];

    for (const item of Array.from(e.dataTransfer.items)) {
      if (item.kind !== "file") continue;
      const entry = item.webkitGetAsEntry?.();
      if (entry) {
        collected.push({ entry });
      } else {
        // Fallback for browsers without FileSystem API
        const f = item.getAsFile();
        if (f) collected.push({ file: f });
      }
    }

    const files: File[] = [];
    await Promise.all(
      collected.map(async (c) => {
        if ("file" in c) {
          files.push(c.file);
        } else if (c.entry.isDirectory) {
          await collectFromEntry(c.entry as FileSystemDirectoryEntry, files);
        } else {
          const file = await new Promise<File>((resolve, reject) =>
            (c.entry as FileSystemFileEntry).file(resolve, reject)
          );
          files.push(file);
        }
      })
    );

    enqueueFiles(files);
  }

  // ── Stats ──────────────────────────────────────────────────────────────────
  const total = queue.length;
  const done = queue.filter((q) => q.status === "complete").length;
  const failed = queue.filter((q) => q.status === "failed").length;
  const inProgress = total - done - failed;

  return (
    <>
      <Head>
        <title>Documents · RegIntel AI</title>
      </Head>

      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Nav />

        <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-8 flex flex-col gap-8">
          {/* ── System health banner ─────────────────────────────────────── */}
          <HealthBanner health={health} loading={healthLoading} />

          {/* ── Upload zone ──────────────────────────────────────────────── */}
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h1 className="text-base font-semibold text-gray-900">Upload documents</h1>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-500">Document type</label>
                <select
                  value={docType}
                  onChange={(e) => setDocType(e.target.value as DocumentType)}
                  className="text-xs border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-brand-600"
                >
                  {DOC_TYPE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Drop zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative rounded-2xl border-2 border-dashed transition-colors p-10 flex flex-col items-center gap-4 text-center ${
                isDragging
                  ? "border-brand-600 bg-brand-50"
                  : "border-gray-200 bg-white hover:border-brand-400"
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center">
                <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-700">
                  Drop a folder or PDF files here
                </p>
                <p className="text-xs text-gray-400 mt-0.5">
                  Sub-folders are scanned recursively · PDFs only
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => folderInputRef.current?.click()}
                  className="text-xs px-4 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-700 transition-colors font-medium"
                >
                  Select folder
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-xs px-4 py-2 rounded-lg border border-gray-200 bg-white text-gray-600 hover:border-brand-600 hover:text-brand-600 transition-colors"
                >
                  Select files
                </button>
              </div>

              {/* Hidden inputs */}
              <input
                ref={folderInputRef}
                type="file"
                // @ts-expect-error webkitdirectory is non-standard
                webkitdirectory="true"
                multiple
                accept=".pdf"
                className="hidden"
                onChange={handleFolderChange}
              />
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            {/* Upload queue */}
            {queue.length > 0 && (
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                {/* Summary bar */}
                <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-700">
                    {total} file{total !== 1 ? "s" : ""}
                    {inProgress > 0 && ` · ${inProgress} in progress`}
                    {done > 0 && ` · ${done} complete`}
                    {failed > 0 && ` · ${failed} failed`}
                  </span>
                  {total > 0 && (
                    <div className="w-32 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-brand-600 rounded-full transition-all"
                        style={{ width: `${Math.round(((done + failed) / total) * 100)}%` }}
                      />
                    </div>
                  )}
                </div>

                {/* File rows (max 8 visible, scrollable) */}
                <div className="max-h-64 overflow-y-auto divide-y divide-gray-50">
                  {queue.map((item) => (
                    <div key={item.id} className="px-4 py-2.5 space-y-1.5">
                      <div className="flex items-center gap-3">
                        <PdfIcon />
                        <span className="flex-1 text-xs text-gray-700 truncate" title={item.relativePath}>
                          {item.relativePath}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${STATUS_STYLE[item.status]}`}>
                          {STATUS_LABEL[item.status]}
                        </span>
                      </div>
                      {/* Progress bar — shown while active, filled when done */}
                      {item.status !== "queued" && (
                        <ProgressBar status={item.status} tick={tick} error={item.error} />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* ── Document library ─────────────────────────────────────────── */}
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-gray-900">
                Document library
                {library.length > 0 && (
                  <span className="ml-2 text-xs font-normal text-gray-400">{library.length} doc{library.length !== 1 ? "s" : ""}</span>
                )}
              </h2>
              <button
                onClick={refreshLibrary}
                className="text-xs text-gray-400 hover:text-brand-600 transition-colors"
              >
                Refresh
              </button>
            </div>

            {libraryLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-12 bg-white border border-gray-100 rounded-xl animate-pulse" />
                ))}
              </div>
            ) : library.length === 0 ? (
              <div className="text-center py-12 text-sm text-gray-400 bg-white border border-dashed border-gray-200 rounded-xl">
                No documents yet. Upload a folder to get started.
              </div>
            ) : (
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                      <th className="text-left px-4 py-2.5 font-medium text-gray-500">Title</th>
                      <th className="text-left px-4 py-2.5 font-medium text-gray-500 hidden sm:table-cell">Type</th>
                      <th className="text-left px-4 py-2.5 font-medium text-gray-500 hidden md:table-cell">Chunks</th>
                      <th className="text-left px-4 py-2.5 font-medium text-gray-500">Status</th>
                      <th className="text-left px-4 py-2.5 font-medium text-gray-500 hidden lg:table-cell">Uploaded</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {library.map((doc) => (
                      <tr key={doc.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 text-gray-800 font-medium max-w-xs truncate" title={doc.title}>
                          {doc.title}
                        </td>
                        <td className="px-4 py-3 text-gray-500 hidden sm:table-cell">
                          {DOC_TYPE_OPTIONS.find((o) => o.value === doc.document_type)?.label ?? doc.document_type}
                        </td>
                        <td className="px-4 py-3 text-gray-500 hidden md:table-cell">
                          {doc.chunk_count > 0 ? doc.chunk_count : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full font-medium ${STATUS_STYLE[doc.status as keyof typeof STATUS_STYLE] ?? "bg-gray-100 text-gray-600"}`}>
                            {STATUS_LABEL[doc.status] ?? doc.status}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-400 hidden lg:table-cell">
                          {new Date(doc.uploaded_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </main>
      </div>
    </>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function HealthBanner({ health, loading }: { health: HealthReport | null; loading: boolean }) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-50 border border-gray-100 animate-pulse">
        <div className="w-2 h-2 rounded-full bg-gray-300" />
        <span className="text-xs text-gray-400">Checking system status…</span>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-50 border border-red-200">
        <div className="w-2 h-2 rounded-full bg-red-500" />
        <span className="text-xs text-red-700 font-medium">API unreachable — check that all containers are running</span>
      </div>
    );
  }

  const llm = health.components.llm;
  const db  = health.components.database;
  const redis = health.components.redis;

  const allOk = health.status === "ok";

  return (
    <div className={`px-4 py-3 rounded-xl border text-xs space-y-2 ${allOk ? "bg-green-50 border-green-200" : "bg-yellow-50 border-yellow-200"}`}>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${allOk ? "bg-green-500" : "bg-yellow-500"}`} />
        <span className={`font-medium ${allOk ? "text-green-800" : "text-yellow-800"}`}>
          {allOk ? "All systems operational — ingestion ready" : "System degraded — ingestion may fail"}
        </span>
      </div>
      <div className="flex flex-wrap gap-4 pl-4">
        <ComponentPill label="Database" c={db} />
        <ComponentPill label="Redis" c={redis} />
        <ComponentPill label={`LLM (${llm.provider ?? "?"})`} c={llm} />
      </div>
      {llm.status !== "ok" && llm.detail && (
        <p className="pl-4 text-yellow-700 leading-snug">{llm.detail}</p>
      )}
    </div>
  );
}

function ComponentPill({ label, c }: { label: string; c: { status: string; latency_ms?: number } }) {
  const ok = c.status === "ok";
  const warn = c.status === "quota_exceeded" || c.status === "unconfigured";
  return (
    <span className={`flex items-center gap-1 ${ok ? "text-green-700" : warn ? "text-yellow-700" : "text-red-600"}`}>
      <span>{ok ? "✓" : warn ? "⚠" : "✗"}</span>
      <span>{label}{warn ? ` (${c.status.replace("_", " ")})` : ""}</span>
      {ok && c.latency_ms != null && <span className="text-gray-400">({c.latency_ms}ms)</span>}
    </span>
  );
}

function ProgressBar({ status, tick, error }: { status: UploadStatus; tick: number; error?: string }) {
  if (status === "failed") {
    return (
      <div className="space-y-0.5">
        <div className="h-1 w-full rounded-full bg-red-200">
          <div className="h-full w-full rounded-full bg-red-400" />
        </div>
        {error && <p className="text-xs text-red-500 truncate" title={error}>{error}</p>}
      </div>
    );
  }

  if (status === "complete") {
    return (
      <div className="h-1 w-full rounded-full bg-green-100">
        <div className="h-full w-full rounded-full bg-green-500 transition-all duration-500" />
      </div>
    );
  }

  // Uploading or ingesting — animated shimmer
  const isIngesting = status === "ingesting";
  // Shimmer position: cycles 0→100→0 smoothly
  const pct = Math.abs(((tick * 2) % 200) - 100);

  return (
    <div className="space-y-0.5">
      <div className="h-1 w-full rounded-full bg-gray-100 overflow-hidden relative">
        <div
          className={`h-full rounded-full transition-all duration-300 ${isIngesting ? "bg-yellow-400" : "bg-blue-400"}`}
          style={{ width: `${20 + pct * 0.6}%`, marginLeft: `${pct * 0.2}%` }}
        />
      </div>
      <p className="text-xs text-gray-400">
        {isIngesting ? "Parsing, chunking & embedding…" : "Uploading to server…"}
      </p>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function PdfIcon() {
  return (
    <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  );
}

async function collectFromEntry(
  entry: FileSystemDirectoryEntry,
  out: File[],
): Promise<void> {
  const reader = entry.createReader();
  // readEntries returns at most 100 items per call — loop until empty.
  const allEntries: FileSystemEntry[] = [];
  while (true) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject)
    );
    if (batch.length === 0) break;
    allEntries.push(...batch);
  }
  await Promise.all(
    allEntries.map(async (e) => {
      if (e.isFile) {
        const file = await new Promise<File>((resolve, reject) =>
          (e as FileSystemFileEntry).file(resolve, reject)
        );
        if (file.name.toLowerCase().endsWith(".pdf")) out.push(file);
      } else if (e.isDirectory) {
        await collectFromEntry(e as FileSystemDirectoryEntry, out);
      }
    })
  );
}
