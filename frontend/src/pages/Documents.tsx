import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, apiUpload } from "../api/client";
import { useToast } from "../components/Toast";
import WorkspaceSwitch, { type Workspace } from "../components/WorkspaceSwitch";

interface DocItem {
  id: number;
  filename: string;
  status: "processing" | "ready" | "failed";
  created_at: string;
}
interface Usage {
  documents: { used: number; limit: number };
}
interface UploadResponse {
  id: number;
  filename: string;
  status: string;
}

const POLL_MS = 2000;
const POLL_MAX_MS = 120_000;

export default function Documents() {
  const { toast } = useToast();
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [workspace, setWorkspace] = useState<Workspace>("academic");
  const [file, setFile] = useState<File | null>(null);
  const [metaOpen, setMetaOpen] = useState(false);
  const [docDate, setDocDate] = useState("");
  const [docVersion, setDocVersion] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [uploading, setUploading] = useState(false);
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const pollTimers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const refresh = useCallback(async () => {
    try {
      const [list, use] = await Promise.all([
        api<DocItem[]>("/documents"),
        api<Usage>("/usage"),
      ]);
      setDocs(list);
      setUsage(use);
      for (const doc of list) {
        if (doc.status === "processing") startPolling(doc.id);
      }
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Failed to load documents", "error");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast]);

  useEffect(() => {
    void refresh();
    const timers = pollTimers.current;
    return () => {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function startPolling(id: number, waited = 0) {
    const timers = pollTimers.current;
    if (timers.has(id) || waited > POLL_MAX_MS) return;
    const delay = waited > 30_000 ? 5000 : POLL_MS;
    const timer = setTimeout(async () => {
      timers.delete(id);
      try {
        const doc = await api<DocItem>(`/documents/${id}`);
        if (doc.status === "processing") {
          startPolling(id, waited + delay);
        } else {
          if (doc.status === "ready") {
            toast(`${doc.filename} indexed.`, "success");
          } else {
            toast(`${doc.filename} failed to process.`, "error");
          }
          void refresh();
        }
      } catch {
        startPolling(id, waited + delay);
      }
    }, delay);
    timers.set(id, timer);
  }

  async function upload() {
    if (!file || uploading) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("workspace", workspace);
      if (docDate) form.append("doc_date", docDate);
      if (docVersion) form.append("doc_version", docVersion);
      if (jurisdiction) form.append("jurisdiction", jurisdiction);
      const created = await apiUpload<UploadResponse>("/documents", form);
      toast(`${created.filename} uploaded. Indexing…`, "info");
      setFile(null);
      setDocDate("");
      setDocVersion("");
      setJurisdiction("");
      const input = document.getElementById("pdf-input") as HTMLInputElement | null;
      if (input) input.value = "";
      startPolling(created.id);
      await refresh();
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Upload failed", "error");
    } finally {
      setUploading(false);
    }
  }

  async function remove(doc: DocItem) {
    try {
      const result = await api<{ deleted: string }>(`/documents/${doc.id}`, {
        method: "DELETE",
      });
      toast(`${result.deleted} deleted.`, "success");
      setConfirmId(null);
      await refresh();
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Delete failed", "error");
    }
  }

  const quotaFull = usage ? usage.documents.used >= usage.documents.limit : false;

  return (
    <main className="panel-page shell">
      <header className="panel-head">
        <div>
          <span className="chip">Library</span>
          <h1>Documents</h1>
          <p className="panel-lede">
            Upload PDFs into a workspace. Indexing runs in the background; the
            list refreshes itself until every file is ready.
          </p>
        </div>
        <div className="panel-head-right">
          <WorkspaceSwitch value={workspace} onChange={setWorkspace} />
          {usage && (
            <span className="quota-pill">
              {usage.documents.used} / {usage.documents.limit} documents
            </span>
          )}
        </div>
      </header>

      <section className="card upload-card">
        <span className="demo-label">Upload</span>
        <label className={`dropzone${file ? " has-file" : ""}`} htmlFor="pdf-input">
          <input
            id="pdf-input"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          {file ? (
            <span className="dropzone-file">{file.name}</span>
          ) : (
            <span className="dropzone-hint">
              Drop a PDF here, or click to browse
              <em>Research papers, contracts, reports. Max 100 MB per plan.</em>
            </span>
          )}
        </label>

        <button className="link-btn" onClick={() => setMetaOpen((open) => !open)}>
          {metaOpen ? "− Hide optional metadata" : "+ Optional metadata (date, version, jurisdiction)"}
        </button>

        {metaOpen && (
          <div className="meta-grid">
            <label>
              Document date
              <input type="date" value={docDate} onChange={(e) => setDocDate(e.target.value)} />
            </label>
            <label>
              Version
              <input
                type="text"
                placeholder="v2.1"
                value={docVersion}
                onChange={(e) => setDocVersion(e.target.value)}
              />
            </label>
            <label>
              Jurisdiction
              <input
                type="text"
                placeholder="Delaware"
                value={jurisdiction}
                onChange={(e) => setJurisdiction(e.target.value)}
              />
            </label>
          </div>
        )}

        <div className="upload-actions">
          <button
            className="btn btn-primary"
            disabled={!file || uploading || quotaFull}
            onClick={() => void upload()}
          >
            {uploading ? "Uploading…" : "Index document"}
          </button>
          {quotaFull && (
            <span className="inline-warn">
              Document limit reached. Upgrade to Pro for 10x quota.
            </span>
          )}
        </div>
      </section>

      <section className="doc-list-section">
        <span className="demo-label">Indexed documents</span>
        {docs.length === 0 ? (
          <div className="panel-empty">
            <p>No documents yet. Upload a PDF above.</p>
          </div>
        ) : (
          <ul className="doc-list">
            {docs.map((doc) => (
              <li key={doc.id} className="doc-row">
                <span className="doc-name">{doc.filename}</span>
                <span className={`doc-status doc-${doc.status}`}>
                  {doc.status === "processing" && <i className="pulse-dot" />}
                  {doc.status}
                </span>
                <span className="doc-date">
                  {new Date(doc.created_at).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
                {confirmId === doc.id ? (
                  <span className="doc-confirm">
                    <button className="danger" onClick={() => void remove(doc)}>
                      Confirm delete
                    </button>
                    <button className="link-btn" onClick={() => setConfirmId(null)}>
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    className="link-btn danger"
                    onClick={() => setConfirmId(doc.id)}
                  >
                    Delete
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
