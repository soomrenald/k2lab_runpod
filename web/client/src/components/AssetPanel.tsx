import { useEffect, useState } from "react";
import type { FileKind, FileRecord, UploadSession } from "../api";
import { controlPlane } from "../api";
import type { LocalUploadItem, UploadQueueController } from "../useUploadQueue";
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
  uploadQueue: UploadQueueController;
}

export function AssetPanel({
  workspaceId,
  onClose,
  onSelect,
  onEvent,
  uploadQueue,
  initialKind = "inputs",
}: Props) {
  const [kind, setKind] = useState<FileKind>(initialKind);
  const [files, setFiles] = useState<FileRecord[]>([]);
  const [selected, setSelected] = useState<File[]>([]);
  const [uploadHistory, setUploadHistory] = useState<UploadSession[]>([]);
  const [error, setError] = useState("");
  const completedCount = uploadQueue.items.filter((item) => item.state === "completed").length;
  const activeCount = uploadQueue.items.filter((item) => (
    ["hashing", "uploading", "pausing", "cancelling"].includes(item.state)
  )).length;
  const queuedCount = uploadQueue.items.filter((item) => item.state === "queued").length;

  async function refresh(nextKind = kind) {
    try {
      const page = await controlPlane.files(workspaceId, nextKind);
      setFiles(page.items);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load cloud files");
    }
  }

  useEffect(() => { void refresh(kind); }, [kind, workspaceId, completedCount]);

  useEffect(() => {
    let cancelled = false;
    void controlPlane.uploads(workspaceId).then((items) => {
      if (cancelled) return;
      setUploadHistory(items);
    }).catch((caught) => { if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not restore upload history"); });
    return () => { cancelled = true; };
  }, [workspaceId]);

  function enqueueSelected() {
    if (!selected.length) return;
    setError("");
    uploadQueue.enqueue(selected, kind);
    setSelected([]);
    onEvent?.("Uploads continue in the background when this panel is closed.", "info");
  }

  return (
    <div className="asset-backdrop">
      <section className="asset-panel glass-card" aria-label="Cloud files">
        <header><div><p className="kicker">Persistent workspace</p><h2>Cloud files</h2></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <div className="asset-kind-tabs">{kinds.map((item) => <button key={item.value} className={kind === item.value ? "active" : ""} onClick={() => setKind(item.value)}>{item.label}</button>)}</div>
        <div className="asset-upload">
          <label className="quiet-button">Choose local files<input type="file" multiple hidden onClick={(event) => { event.currentTarget.value = ""; }} onChange={(event) => { setSelected(Array.from(event.target.files ?? [])); }} /></label>
          <span>{selected.length ? `${selected.length} file${selected.length === 1 ? "" : "s"} selected` : "No files selected"}</span>
          <button className="primary-button" disabled={!selected.length} onClick={enqueueSelected}>Queue for {kinds.find((item) => item.value === kind)?.label}</button>
        </div>
        <p className="field-help">Uploads run one at a time in queue order and continue when you close Assets or open another panel.</p>
        {error && <div className="error-banner">{error}</div>}
        {uploadQueue.items.length > 0 && (
          <div className="upload-queue">
            <div className="upload-queue-head">
              <strong>Background upload queue</strong>
              <span>{activeCount} active · {queuedCount} queued</span>
              <button className="quiet-button" onClick={uploadQueue.clearFinished}>Clear finished</button>
            </div>
            {uploadQueue.items.map((item, index) => (
              <UploadQueueRow
                key={item.id}
                item={item}
                position={index + 1}
                active={item.id === uploadQueue.activeId}
                onPause={() => uploadQueue.pause(item.id)}
                onResume={() => uploadQueue.resume(item.id)}
                onCancel={() => uploadQueue.cancel(item.id)}
              />
            ))}
          </div>
        )}
        {uploadHistory.length > 0 && <div className="transfer-history"><strong>Uploads retained by the workspace</strong>{uploadHistory.map((item) => <button key={item.id} onClick={() => setKind(item.destination_kind)}><span><b>{item.display_name}</b><small>{item.destination_kind.replaceAll("_", " ")} · {formatBytes(item.size_bytes)}{item.state === "uploading" ? " · reselect this file to resume after a browser restart" : ""}</small></span><em className={item.state}>{item.state}</em></button>)}</div>}
        <div className="asset-list">{files.length === 0 ? <p className="field-help">No files in this category.</p> : files.map((file) => <div key={file.id}><Icon name="folder" /><span><strong>{file.display_name}</strong><small>{formatBytes(file.size_bytes)} · {file.sha256.slice(0, 12)}…</small></span>{["inputs", "outputs", "projects"].includes(file.kind) && <a className="quiet-button asset-download" href={controlPlane.fileUrl(workspaceId, file.id)} download={file.display_name}>Download</a>}{onSelect && <button className="quiet-button" onClick={() => { onSelect(file); onClose(); }}>{file.kind === "projects" ? "Open project" : "Use in studio"}</button>}</div>)}</div>
      </section>
    </div>
  );
}

function UploadQueueRow({
  item,
  position,
  active,
  onPause,
  onResume,
  onCancel,
}: {
  item: LocalUploadItem;
  position: number;
  active: boolean;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const canPause = ["queued", "hashing", "uploading"].includes(item.state);
  const canResume = ["paused", "failed"].includes(item.state);
  const canCancel = !["completed", "cancelled", "cancelling"].includes(item.state);
  return (
    <article className={`upload-queue-row ${active ? "active" : ""}`}>
      <div className="upload-queue-copy">
        <span>#{position}</span>
        <div>
          <strong>{item.file.name}</strong>
          <small>{item.destinationKind.replaceAll("_", " ")} · {formatBytes(item.file.size)}</small>
        </div>
        <em className={item.state}>{queueStateLabel(item.state, active)}</em>
      </div>
      <div className="transfer-progress">
        <div><i style={{ width: `${item.progress * 100}%` }} /></div>
        <span>{(item.progress * 100).toFixed(0)}%{item.speed > 0 ? ` · ${formatBytes(item.speed)}/s · ${Math.ceil(item.eta)}s remaining` : ""}</span>
      </div>
      {item.error && <small className="upload-queue-error">{item.error}</small>}
      <div className="upload-queue-actions">
        {canPause && <button className="quiet-button" onClick={onPause}>Pause</button>}
        {canResume && <button className="primary-button" onClick={onResume}>{item.session ? "Resume" : "Retry"}</button>}
        {canCancel && <button className="danger-text-button" onClick={onCancel}>Cancel</button>}
      </div>
    </article>
  );
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

function queueStateLabel(state: LocalUploadItem["state"], active: boolean) {
  if (state === "queued") return active ? "starting" : "queued";
  return state;
}
