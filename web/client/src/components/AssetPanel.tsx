import { createSHA256, sha256 } from "hash-wasm";
import { useEffect, useRef, useState } from "react";
import type { FileKind, FileRecord, UploadSession } from "../api";
import { controlPlane } from "../api";
import { Icon } from "./Icon";

const kinds: { value: FileKind; label: string }[] = [
  { value: "inputs", label: "Inputs" },
  { value: "projects", label: "Projects" },
  { value: "outputs", label: "Outputs" },
  { value: "diffusion_models", label: "Diffusion models" },
  { value: "text_encoders", label: "Text encoders" },
  { value: "vae", label: "VAE" },
  { value: "loras", label: "LoRAs" },
  { value: "upscale_models", label: "Upscalers" },
  { value: "face_detection", label: "Face detection" },
];

interface Props {
  workspaceId: string;
  onClose: () => void;
  onSelect?: (file: FileRecord) => void;
  onEvent?: (message: string, kind: "info" | "error" | "worker") => void;
  initialKind?: FileKind;
}

export function AssetPanel({ workspaceId, onClose, onSelect, onEvent, initialKind = "inputs" }: Props) {
  const [kind, setKind] = useState<FileKind>(initialKind);
  const [files, setFiles] = useState<FileRecord[]>([]);
  const [selected, setSelected] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadSession | null>(null);
  const [uploadHistory, setUploadHistory] = useState<UploadSession[]>([]);
  const [phase, setPhase] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(0);
  const [eta, setEta] = useState(0);
  const [error, setError] = useState("");
  const paused = useRef(false);

  function rememberUpload(next: UploadSession) {
    setUpload(next);
    setUploadHistory((current) => [next, ...current.filter((item) => item.id !== next.id)]);
  }

  async function refresh(nextKind = kind) {
    try {
      const page = await controlPlane.files(workspaceId, nextKind);
      setFiles(page.items);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load cloud files");
    }
  }

  useEffect(() => { void refresh(kind); }, [kind, workspaceId]);

  useEffect(() => {
    let cancelled = false;
    void controlPlane.uploads(workspaceId).then((items) => {
      if (cancelled) return;
      setUploadHistory(items);
      const restored = items.find((item) => item.state === "uploading") ?? items[0];
      if (restored) {
        setUpload(restored);
        setKind(restored.destination_kind);
        setPhase(uploadPhase(restored));
        setProgress(uploadProgress(restored));
      }
    }).catch((caught) => { if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not restore upload history"); });
    return () => { cancelled = true; };
  }, [workspaceId]);

  async function beginUpload() {
    if (!selected) return;
    try {
      paused.current = false;
      setError("");
      setPhase("Hashing");
      onEvent?.(`Preparing ${selected.name} for upload to ${kind}.`, "info");
      const hasher = await createSHA256();
      hasher.init();
      const hashChunk = 8 * 1024 * 1024;
      for (let offset = 0; offset < selected.size; offset += hashChunk) {
        hasher.update(new Uint8Array(await selected.slice(offset, offset + hashChunk).arrayBuffer()));
        setProgress(Math.min(0.1, ((offset + hashChunk) / selected.size) * 0.1));
      }
      const digest = hasher.digest("hex");
      let session: UploadSession;
      if (upload?.state === "uploading") {
        if (upload.display_name !== selected.name || upload.size_bytes !== selected.size || upload.sha256 !== digest || upload.destination_kind !== kind) {
          throw new Error(`Choose the original ${upload.display_name} file to resume this upload, or cancel it first.`);
        }
        session = await controlPlane.uploadStatus(workspaceId, upload.id);
      } else {
        session = await controlPlane.createUpload(workspaceId, {
          filename: selected.name,
          destination_kind: kind,
          size_bytes: selected.size,
          sha256: digest,
          chunk_size_bytes: 8 * 1024 * 1024,
        });
      }
      rememberUpload(session);
      await transfer(selected, session);
    } catch (caught) {
      setPhase("Failed");
      setError(caught instanceof Error ? caught.message : "Could not start upload");
    }
  }

  async function transfer(file: File, current: UploadSession) {
    paused.current = false;
    setPhase("Uploading");
    setError("");
    const started = performance.now();
    let sent = current.completed_chunks.reduce((total, index) => {
      const start = index * current.chunk_size_bytes;
      return total + Math.min(current.chunk_size_bytes, current.size_bytes - start);
    }, 0);
    const completed = new Set(current.completed_chunks);
    try {
      for (let index = 0; index < current.chunk_count; index += 1) {
        if (paused.current) { setPhase("Paused"); return; }
        if (completed.has(index)) continue;
        const start = index * current.chunk_size_bytes;
        const buffer = await file.slice(start, start + current.chunk_size_bytes).arrayBuffer();
        await controlPlane.uploadChunk(workspaceId, current.id, index, buffer, await sha256(new Uint8Array(buffer)));
        completed.add(index);
        rememberUpload({ ...current, completed_chunks: [...completed].sort((left, right) => left - right), updated_at: new Date().toISOString() });
        sent += buffer.byteLength;
        const elapsed = Math.max(0.001, (performance.now() - started) / 1000);
        const bytesPerSecond = sent / elapsed;
        setSpeed(bytesPerSecond);
        setEta((current.size_bytes - sent) / bytesPerSecond);
        setProgress(0.1 + (sent / current.size_bytes) * 0.9);
      }
      const result = await controlPlane.completeUpload(workspaceId, current.id);
      setPhase(result.duplicate ? "Already present" : "Complete");
      setProgress(1);
      rememberUpload({ ...current, state: "completed", completed_chunks: Array.from({ length: current.chunk_count }, (_value, index) => index), updated_at: new Date().toISOString() });
      onEvent?.(`${result.duplicate ? "Verified existing" : "Uploaded"} ${file.name} in ${kind}.`, "info");
      await refresh();
    } catch (caught) {
      setPhase("Retry available");
      const detail = caught instanceof Error ? caught.message : "Upload failed";
      setError(detail);
      onEvent?.(detail, "error");
    }
  }

  async function cancel() {
    paused.current = true;
    if (upload) await controlPlane.cancelUpload(workspaceId, upload.id);
    if (upload) rememberUpload({ ...upload, state: "cancelled", updated_at: new Date().toISOString() });
    setSelected(null);
    setPhase("Cancelled");
    onEvent?.("Upload cancelled; resumable staging data was removed.", "info");
  }

  return (
    <div className="asset-backdrop">
      <section className="asset-panel glass-card" aria-label="Cloud files">
        <header><div><p className="kicker">Persistent workspace</p><h2>Cloud files</h2></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <div className="asset-kind-tabs">{kinds.map((item) => <button key={item.value} className={kind === item.value ? "active" : ""} onClick={() => setKind(item.value)}>{item.label}</button>)}</div>
        <div className="asset-upload">
          <label className="quiet-button">Choose local file<input type="file" hidden onChange={(event) => { setSelected(event.target.files?.[0] ?? null); setPhase("Ready"); setProgress(0); }} /></label>
          <span>{selected?.name ?? "No file selected"}</span>
          {upload?.state !== "uploading" && <button className="primary-button" disabled={!selected} onClick={() => void beginUpload()}>Upload to {kinds.find((item) => item.value === kind)?.label}</button>}
          {upload?.state === "uploading" && phase === "Uploading" && <button className="quiet-button" onClick={() => { paused.current = true; }}>Pause</button>}
          {upload?.state === "uploading" && selected && phase !== "Uploading" && <button className="primary-button" onClick={() => void beginUpload()}>Verify and resume</button>}
          {upload?.state === "uploading" && <button className="danger-text-button" onClick={() => void cancel()}>Cancel</button>}
        </div>
        {(selected || upload) && <div className="transfer-progress"><div><i style={{ width: `${progress * 100}%` }} /></div><span>{phase} · {(progress * 100).toFixed(0)}%{speed > 0 ? ` · ${formatBytes(speed)}/s · ${Math.ceil(eta)}s remaining` : ""}{upload?.state === "uploading" && !selected ? ` · reselect ${upload.display_name} to resume` : ""}</span></div>}
        {error && <div className="error-banner">{error}</div>}
        {uploadHistory.length > 0 && <div className="transfer-history"><strong>Recent local uploads</strong>{uploadHistory.map((item) => <button key={item.id} className={item.id === upload?.id ? "selected" : ""} onClick={() => { setUpload(item); setKind(item.destination_kind); setPhase(uploadPhase(item)); setProgress(uploadProgress(item)); }}><span><b>{item.display_name}</b><small>{item.destination_kind.replaceAll("_", " ")} · {formatBytes(item.size_bytes)}</small></span><em className={item.state}>{item.state}</em></button>)}</div>}
        <div className="asset-list">{files.length === 0 ? <p className="field-help">No files in this category.</p> : files.map((file) => <div key={file.id}><Icon name="folder" /><span><strong>{file.display_name}</strong><small>{formatBytes(file.size_bytes)} · {file.sha256.slice(0, 12)}…</small></span>{["inputs", "outputs", "projects"].includes(file.kind) && <a className="quiet-button asset-download" href={controlPlane.fileUrl(workspaceId, file.id)} download={file.display_name}>Download</a>}{onSelect && <button className="quiet-button" onClick={() => { onSelect(file); onClose(); }}>{file.kind === "projects" ? "Open project" : "Use in studio"}</button>}</div>)}</div>
      </section>
    </div>
  );
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

function uploadProgress(upload: UploadSession) {
  const completedBytes = upload.completed_chunks.reduce((total, index) => {
    const start = index * upload.chunk_size_bytes;
    return total + Math.min(upload.chunk_size_bytes, upload.size_bytes - start);
  }, 0);
  return upload.state === "completed" ? 1 : completedBytes / Math.max(1, upload.size_bytes);
}

function uploadPhase(upload: UploadSession) {
  if (upload.state === "completed") return "Complete";
  if (upload.state === "cancelled") return "Cancelled";
  return "Paused";
}
