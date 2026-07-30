import { useEffect, useMemo, useState } from "react";
import type { FileKind, FileRecord, UploadSession } from "../api";
import { controlPlane, queuedFileBlob } from "../api";
import { sortOutputFiles, type OutputSort } from "../outputSort";
import type { LocalUploadItem, UploadQueueController } from "../useUploadQueue";
import { Icon } from "./Icon";

const kinds: { value: FileKind; label: string }[] = [
  { value: "inputs", label: "Inputs" },
  { value: "projects", label: "Projects" },
  { value: "outputs", label: "Outputs" },
  { value: "diffusion_models", label: "Diffusion models" },
  { value: "text_encoders", label: "Text encoders" },
  { value: "tokenizers", label: "Tokenizers" },
  { value: "vae", label: "VAE" },
  { value: "loras", label: "LoRAs" },
  { value: "krea_control_loras", label: "Krea pose adapters" },
  { value: "upscale_models", label: "Upscalers" },
  { value: "controlnet_models", label: "ControlNet models" },
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
  const [selectedUploads, setSelectedUploads] = useState<File[]>([]);
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [uploadHistory, setUploadHistory] = useState<UploadSession[]>([]);
  const [previewed, setPreviewed] = useState<FileRecord | null>(null);
  const [outputSort, setOutputSort] = useState<OutputSort>("newest");
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const completedCount = uploadQueue.items.filter((item) => item.state === "completed").length;
  const activeCount = uploadQueue.items.filter((item) => (
    ["hashing", "uploading", "pausing", "cancelling"].includes(item.state)
  )).length;
  const queuedCount = uploadQueue.items.filter((item) => item.state === "queued").length;
  const displayedFiles = useMemo(
    () => kind === "outputs" ? sortOutputFiles(files, outputSort) : files,
    [files, kind, outputSort],
  );

  async function refresh(nextKind = kind) {
    try {
      const items: FileRecord[] = [];
      let cursor: string | undefined;
      do {
        const page = await controlPlane.files(workspaceId, nextKind, cursor);
        items.push(...page.items);
        cursor = page.next_cursor ?? undefined;
      } while (cursor);
      setFiles(items);
      const availableIds = new Set(items.map((file) => file.id));
      setSelectedFileIds((current) => new Set(
        [...current].filter((fileId) => availableIds.has(fileId)),
      ));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load cloud files");
    }
  }

  useEffect(() => {
    setPreviewed(null);
    setSelectedFileIds(new Set());
  }, [kind, workspaceId]);

  useEffect(() => {
    void refresh(kind);
  }, [kind, workspaceId, completedCount]);

  useEffect(() => {
    let cancelled = false;
    void controlPlane.uploads(workspaceId).then((items) => {
      if (cancelled) return;
      setUploadHistory(items);
    }).catch((caught) => { if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not restore upload history"); });
    return () => { cancelled = true; };
  }, [workspaceId]);

  function enqueueSelected() {
    if (!selectedUploads.length) return;
    setError("");
    uploadQueue.enqueue(selectedUploads, kind);
    setSelectedUploads([]);
    onEvent?.("Uploads continue in the background when this panel is closed.", "info");
  }

  function toggleFile(fileId: string) {
    setSelectedFileIds((current) => {
      const next = new Set(current);
      if (next.has(fileId)) next.delete(fileId);
      else next.add(fileId);
      return next;
    });
  }

  async function deleteFiles(targets: FileRecord[]) {
    if (!targets.length || deleting) return;
    const description = targets.length === 1
      ? `"${targets[0].display_name}"`
      : `${targets.length} selected files`;
    if (!window.confirm(`Permanently delete ${description} from the workspace? This cannot be undone.`)) return;
    setDeleting(true);
    setError("");
    const deletedIds = new Set<string>();
    const failures: string[] = [];
    for (const file of targets) {
      try {
        await controlPlane.deleteFile(workspaceId, file.id);
        deletedIds.add(file.id);
      } catch (caught) {
        failures.push(`${file.display_name}: ${caught instanceof Error ? caught.message : "delete failed"}`);
      }
    }
    if (deletedIds.size) {
      setFiles((current) => current.filter((file) => !deletedIds.has(file.id)));
      setSelectedFileIds((current) => new Set(
        [...current].filter((fileId) => !deletedIds.has(fileId)),
      ));
      setPreviewed((current) => current && deletedIds.has(current.id) ? null : current);
      onEvent?.(`Deleted ${deletedIds.size} workspace file${deletedIds.size === 1 ? "" : "s"}.`, "info");
    }
    if (failures.length) {
      setError(`Could not delete ${failures.length} file${failures.length === 1 ? "" : "s"}. ${failures.join(" · ")}`);
    }
    setDeleting(false);
  }

  const selectedFiles = displayedFiles.filter((file) => selectedFileIds.has(file.id));

  return (
    <div className="asset-backdrop">
      <section className="asset-panel glass-card" aria-label="Cloud files">
        <header><div><p className="kicker">Persistent workspace</p><h2>Cloud files</h2></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <div className="asset-kind-tabs">{kinds.map((item) => <button key={item.value} className={kind === item.value ? "active" : ""} onClick={() => setKind(item.value)}>{item.label}</button>)}</div>
        <div className="asset-upload">
          <label className="quiet-button">Choose local files<input type="file" multiple hidden onClick={(event) => { event.currentTarget.value = ""; }} onChange={(event) => { setSelectedUploads(Array.from(event.target.files ?? [])); }} /></label>
          <span>{selectedUploads.length ? `${selectedUploads.length} file${selectedUploads.length === 1 ? "" : "s"} selected` : "No files selected"}</span>
          <button className="primary-button" disabled={!selectedUploads.length} onClick={enqueueSelected}>Queue for {kinds.find((item) => item.value === kind)?.label}</button>
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
        <div className="asset-file-toolbar">
          <span>{files.length} file{files.length === 1 ? "" : "s"} · {selectedFileIds.size} selected</span>
          <div>
            {kind === "outputs" && (
              <label>
                <span>Sort</span>
                <select className="select-input" value={outputSort} onChange={(event) => setOutputSort(event.target.value as OutputSort)}>
                  <option value="newest">Newest first</option>
                  <option value="oldest">Oldest first</option>
                  <option value="name-asc">Name A–Z</option>
                  <option value="name-desc">Name Z–A</option>
                  <option value="size-desc">Largest first</option>
                  <option value="size-asc">Smallest first</option>
                </select>
              </label>
            )}
            <button className="quiet-button" disabled={!displayedFiles.length || selectedFileIds.size === displayedFiles.length} onClick={() => setSelectedFileIds(new Set(displayedFiles.map((file) => file.id)))}>Select all</button>
            <button className="quiet-button" disabled={!selectedFileIds.size} onClick={() => setSelectedFileIds(new Set())}>Clear selection</button>
            <button className="quiet-button asset-delete" disabled={!selectedFiles.length || deleting} onClick={() => void deleteFiles(selectedFiles)}>
              {deleting ? "Deleting…" : `Delete selected${selectedFiles.length ? ` (${selectedFiles.length})` : ""}`}
            </button>
          </div>
        </div>
        {kind === "outputs" && displayedFiles.length > 0 ? (
          <div className="asset-thumbnail-grid">
            {displayedFiles.map((file) => (
              <article className="asset-thumbnail-card" key={file.id}>
                <label className="asset-thumbnail-selection">
                  <input type="checkbox" checked={selectedFileIds.has(file.id)} onChange={() => toggleFile(file.id)} />
                  <span>Select</span>
                </label>
                <button className="asset-thumbnail" onClick={() => setPreviewed(file)} aria-label={`Preview ${file.display_name}`}>
                  <QueuedFileImage
                    src={controlPlane.fileUrl(workspaceId, file.id)}
                    alt={file.display_name}
                    priority={20}
                  />
                </button>
                <div className="asset-thumbnail-copy">
                  <strong title={file.display_name}>{file.display_name}</strong>
                  <small>{formatModified(file.modified_at)} · {formatBytes(file.size_bytes)}</small>
                </div>
                <div className="asset-thumbnail-actions">
                  <button className="quiet-button" onClick={() => setPreviewed(file)}>Preview</button>
                  <a className="quiet-button asset-download" href={controlPlane.fileUrl(workspaceId, file.id)} download={file.display_name}>Download</a>
                  <button className="quiet-button asset-delete" disabled={deleting} onClick={() => void deleteFiles([file])}>Delete</button>
                  {onSelect && <button className="quiet-button" onClick={() => { onSelect(file); onClose(); }}>Use in studio</button>}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="asset-list">{displayedFiles.length === 0 ? <p className="field-help">No files in this category.</p> : displayedFiles.map((file) => <div key={file.id}><input className="asset-file-checkbox" type="checkbox" aria-label={`Select ${file.display_name}`} checked={selectedFileIds.has(file.id)} onChange={() => toggleFile(file.id)} /><Icon name="folder" /><span><strong>{file.display_name}</strong><small>{formatBytes(file.size_bytes)} · {file.sha256.slice(0, 12)}…</small></span>{["inputs", "projects"].includes(file.kind) && <a className="quiet-button asset-download" href={controlPlane.fileUrl(workspaceId, file.id)} download={file.display_name}>Download</a>}<button className="quiet-button asset-delete" disabled={deleting} onClick={() => void deleteFiles([file])}>Delete</button>{onSelect && <button className="quiet-button" onClick={() => { onSelect(file); onClose(); }}>{file.kind === "projects" ? "Open project" : "Use in studio"}</button>}</div>)}</div>
        )}
        {previewed && (
          <div className="asset-image-preview" role="presentation" onClick={(event) => {
            if (event.target === event.currentTarget) setPreviewed(null);
          }}>
            <div className="asset-image-preview-toolbar">
              <span><strong>{previewed.display_name}</strong><small>{formatBytes(previewed.size_bytes)}</small></span>
              <button className="quiet-button" onClick={() => setPreviewed(null)}>Back to thumbnails</button>
            </div>
            <div className="asset-image-preview-stage" onClick={() => setPreviewed(null)}>
              <QueuedFileImage
                src={controlPlane.fileUrl(workspaceId, previewed.id)}
                alt={previewed.display_name}
                priority={0}
                onClick={(event) => event.stopPropagation()}
              />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function QueuedFileImage({
  src,
  alt,
  priority,
  onClick,
}: {
  src: string;
  alt: string;
  priority: number;
  onClick?: React.MouseEventHandler<HTMLImageElement>;
}) {
  const [objectUrl, setObjectUrl] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let loadedUrl = "";
    void queuedFileBlob(src, { priority, signal: controller.signal })
      .then((blob) => {
        if (controller.signal.aborted) return;
        loadedUrl = URL.createObjectURL(blob);
        setObjectUrl(loadedUrl);
      })
      .catch((caught) => {
        if (!(caught instanceof DOMException && caught.name === "AbortError")) {
          setObjectUrl("");
        }
      });
    return () => {
      controller.abort();
      if (loadedUrl) URL.revokeObjectURL(loadedUrl);
    };
  }, [priority, src]);

  return objectUrl
    ? <img src={objectUrl} alt={alt} onClick={onClick} />
    : <span className="asset-thumbnail-loading" role="status" aria-label={`Loading ${alt}`} />;
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

function formatModified(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "Unknown date" : parsed.toLocaleString();
}

function queueStateLabel(state: LocalUploadItem["state"], active: boolean) {
  if (state === "queued") return active ? "starting" : "queued";
  return state;
}
