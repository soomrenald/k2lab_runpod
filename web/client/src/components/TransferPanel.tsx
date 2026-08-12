import { useEffect, useMemo, useRef, useState } from "react";
import type { CivitaiPreview, CredentialStatus, FileKind, HuggingFacePreview, RemoteProvider, RemoteTransfer } from "../api";
import { controlPlane } from "../api";

const destinations: { value: FileKind; label: string }[] = [
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
  onEvent?: (message: string, kind: "info" | "error" | "worker") => void;
}

export function TransferPanel({ workspaceId, onClose, onEvent }: Props) {
  const [provider, setProvider] = useState<RemoteProvider>("civitai");
  const [credential, setCredential] = useState<CredentialStatus | null>(null);
  const [token, setToken] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [destination, setDestination] = useState<FileKind>("loras");
  const [patterns, setPatterns] = useState("*.safetensors");
  const [civitaiPreview, setCivitaiPreview] = useState<CivitaiPreview | null>(null);
  const [huggingFacePreview, setHuggingFacePreview] = useState<HuggingFacePreview | null>(null);
  const [fileId, setFileId] = useState("");
  const [allowUnsafe, setAllowUnsafe] = useState(false);
  const [transfer, setTransfer] = useState<RemoteTransfer | null>(null);
  const [history, setHistory] = useState<RemoteTransfer[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState(0);
  const sample = useRef<{ bytes: number; time: number } | null>(null);
  const lastReportedStates = useRef(new Map<string, string>());
  const pollFailures = useRef(0);
  const pollError = useRef<string | null>(null);
  const batchFileInput = useRef<HTMLInputElement>(null);
  const batchCancelRequested = useRef(false);
  const [batchMode, setBatchMode] = useState(false);
  const [batchText, setBatchText] = useState("");
  const [batchBusy, setBatchBusy] = useState(false);

  function remember(next: RemoteTransfer) {
    setTransfer(next);
    setHistory((current) => [next, ...current.filter((item) => item.id !== next.id)]);
  }

  useEffect(() => {
    let cancelled = false;
    void controlPlane.transfers(workspaceId).then((items) => {
      if (cancelled) return;
      setHistory(items);
      if (items[0]) {
        setTransfer(items[0]);
        setProvider(items[0].provider);
        setDestination(items[0].destination_kind);
        setSourceUrl(items[0].source_url);
      }
    }).catch((caught) => { if (!cancelled) setError(message(caught)); });
    return () => { cancelled = true; };
  }, [workspaceId]);

  useEffect(() => {
    void controlPlane.downloadCredential(provider).then(setCredential).catch(() => setCredential(null));
    setCivitaiPreview(null); setHuggingFacePreview(null); setFileId(""); setAllowUnsafe(false);
  }, [provider]);

  const queueHeadId = useMemo(() => history
    .filter((item) => !terminal(item.state))
    .sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at))[0]?.id ?? null,
  [history]);

  useEffect(() => {
    if (!queueHeadId) return undefined;
    let cancelled = false;
    let timer: number | undefined;
    const refresh = async () => {
      try {
        const next = await controlPlane.transfer(workspaceId, queueHeadId);
        if (cancelled) return;
        pollFailures.current = 0;
        if (pollError.current) {
          const resolved = pollError.current;
          pollError.current = null;
          setError((current) => current === resolved ? "" : current);
        }
        setHistory((current) => current.map((item) => item.id === next.id ? next : item));
        setTransfer((current) => current?.id === next.id ? next : current);
        if (transfer?.id === next.id) {
          const now = performance.now();
          const previous = sample.current;
          if (previous && now > previous.time) {
            setSpeed(Math.max(0, (next.bytes_complete - previous.bytes) / ((now - previous.time) / 1000)));
          }
          sample.current = { bytes: next.bytes_complete, time: now };
        }
        const reportedState = lastReportedStates.current.get(next.id);
        if (next.state !== reportedState && terminal(next.state)) {
          lastReportedStates.current.set(next.id, next.state);
          onEvent?.(next.state === "completed"
            ? `Provider transfer completed with ${next.files.length} verified file(s).`
            : `Provider transfer ${next.state}${next.error_message ? `: ${next.error_message}` : "."}`,
          next.state === "failed" ? "error" : "info");
        }
      } catch (caught) {
        if (cancelled) return;
        pollFailures.current += 1;
        if (pollFailures.current >= 3) {
          const detail = message(caught);
          pollError.current = detail;
          setError(detail);
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void refresh(), 1000);
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [queueHeadId, transfer?.id, workspaceId]);

  const selectedFile = useMemo(() => civitaiPreview?.files.find((file) => file.id === fileId) ?? null, [civitaiPreview, fileId]);

  async function saveToken() {
    setBusy(true); setError("");
    try { setCredential(await controlPlane.storeDownloadCredential(provider, token)); setToken(""); }
    catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function disconnectToken() {
    setBusy(true); setError("");
    try { setCredential(await controlPlane.clearDownloadCredential(provider)); }
    catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function preview() {
    setBusy(true); setError(""); setTransfer(null);
    try {
      if (provider === "civitai") {
        const result = await controlPlane.previewCivitai(workspaceId, sourceUrl);
        setCivitaiPreview(result); setHuggingFacePreview(null);
        setFileId((result.files.find((file) => file.preferred) ?? result.files[0]).id);
      } else {
        setHuggingFacePreview(await controlPlane.previewHuggingFace(workspaceId, sourceUrl, patternList(patterns)));
        setCivitaiPreview(null);
      }
      setAllowUnsafe(false);
    } catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function start(resume = false) {
    setBusy(true); setError(""); setSpeed(0); sample.current = null;
    try {
      const resume_transfer_id = resume && transfer ? transfer.id : undefined;
      const next = provider === "civitai"
        ? await controlPlane.startCivitai(workspaceId, { source_url: sourceUrl, file_id: fileId, destination_kind: destination, allow_unsafe_format: allowUnsafe, resume_transfer_id })
        : await controlPlane.startHuggingFace(workspaceId, { source_url: sourceUrl, destination_kind: destination, allow_patterns: patternList(patterns), allow_unsafe_format: allowUnsafe, resume_transfer_id });
      remember(next);
      lastReportedStates.current.set(next.id, next.state);
      onEvent?.(`${resume ? "Resumed" : "Started"} ${provider} transfer into ${destination}.`, "info");
    } catch (caught) { const detail = message(caught); setError(detail); onEvent?.(detail, "error"); }
    finally { setBusy(false); }
  }

  function batchUrls() {
    return batchText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  }

  function saveBatch() {
    const payload = JSON.stringify({
      version: 1,
      provider,
      urls: batchUrls(),
      destination,
      patterns,
    }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = `k2lab-${provider}-batch.json`;
    link.click();
    URL.revokeObjectURL(href);
  }

  function loadBatch(file: File) {
    void file.text().then((text) => {
      const parsed = JSON.parse(text) as {
        version?: number;
        provider?: RemoteProvider;
        urls?: unknown;
        destination?: FileKind;
        patterns?: string;
      };
      if (parsed.version !== 1 || !Array.isArray(parsed.urls) || !parsed.urls.every((item) => typeof item === "string")) {
        throw new Error("This is not a K2 Lab batch file.");
      }
      if (parsed.provider && parsed.provider !== "civitai" && parsed.provider !== "huggingface") {
        throw new Error("The batch file has an unsupported provider.");
      }
      setProvider(parsed.provider ?? provider);
      setBatchText(parsed.urls.join("\n"));
      if (parsed.destination && destinations.some((item) => item.value === parsed.destination)) setDestination(parsed.destination);
      if (typeof parsed.patterns === "string") setPatterns(parsed.patterns);
      setError("");
    }).catch((caught) => setError(message(caught)));
  }

  async function waitForTransfer(id: string) {
    for (;;) {
      if (batchCancelRequested.current) return;
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      const next = await controlPlane.transfer(workspaceId, id);
      remember(next);
      if (terminal(next.state)) {
        if (next.state !== "completed") throw new Error(next.error_message || `Transfer ${next.state}.`);
        return;
      }
    }
  }

  async function runBatch() {
    const urls = batchUrls();
    if (!urls.length) return;
    setBatchBusy(true); setBusy(true); setError(""); batchCancelRequested.current = false;
    try {
      for (let index = 0; index < urls.length; index += 1) {
        if (batchCancelRequested.current) return;
        const url = urls[index];
        setSourceUrl(url);
        onEvent?.(`Inspecting batch item ${index + 1} of ${urls.length}.`, "info");
        if (provider === "civitai") {
          const previewResult = await controlPlane.previewCivitai(workspaceId, url);
          const selected = previewResult.files.find((file) => file.preferred) ?? previewResult.files[0];
          if (!selected) throw new Error(`Civitai returned no downloadable files for ${url}.`);
          if (selected.requires_unsafe_confirmation && !allowUnsafe) {
            throw new Error(`Batch item ${index + 1} requires unsafe-format confirmation.`);
          }
          const batchDestination = inferDestination(selected.filename, previewResult.model_type);
          const next = await controlPlane.startCivitai(workspaceId, {
            source_url: url,
            file_id: selected.id,
            destination_kind: batchDestination,
            allow_unsafe_format: allowUnsafe,
          });
          remember(next); onEvent?.(`Batch item ${index + 1} routed to ${batchDestination.replaceAll("_", " ")}.`, "info"); await waitForTransfer(next.id);
        } else {
          const previewResult = await controlPlane.previewHuggingFace(workspaceId, url, patternList(patterns));
          if (previewResult.files.some((file) => unsafeName(file.filename)) && !allowUnsafe) {
            throw new Error(`Batch item ${index + 1} requires unsafe-format confirmation.`);
          }
          const batchDestination = inferDestination(
            previewResult.files.map((file) => file.filename).join(" "),
            previewResult.repo_id,
          );
          const next = await controlPlane.startHuggingFace(workspaceId, {
            source_url: url,
            destination_kind: batchDestination,
            allow_patterns: patternList(patterns),
            allow_unsafe_format: allowUnsafe,
          });
          remember(next); onEvent?.(`Batch item ${index + 1} routed to ${batchDestination.replaceAll("_", " ")}.`, "info"); await waitForTransfer(next.id);
        }
      }
      onEvent?.(`Completed ${urls.length} ${provider} batch download${urls.length === 1 ? "" : "s"}.`, "info");
    } catch (caught) {
      const detail = message(caught); setError(detail); onEvent?.(detail, "error");
    } finally { setBatchBusy(false); setBusy(false); }
  }

  async function cancelBatch() {
    batchCancelRequested.current = true;
    if (transfer && !terminal(transfer.state)) {
      try { remember(await controlPlane.cancelTransfer(workspaceId, transfer.id)); } catch (caught) { setError(message(caught)); }
    }
    setBatchBusy(false); setBusy(false);
  }

  function exitBatch() {
    if (batchBusy) void cancelBatch();
    setBatchMode(false); setBatchText("");
  }

  async function cancel() {
    if (!transfer) return;
    setBusy(true);
    try { remember(await controlPlane.cancelTransfer(workspaceId, transfer.id)); onEvent?.("Provider transfer cancelled; resumable data was retained.", "info"); }
    catch (caught) { const detail = message(caught); setError(detail); onEvent?.(detail, "error"); }
    finally { setBusy(false); }
  }

  const active = Boolean(transfer && !terminal(transfer.state));
  const progress = transfer?.bytes_total ? Math.min(1, transfer.bytes_complete / transfer.bytes_total) : 0;
  const eta = speed > 0 && transfer?.bytes_total ? Math.max(0, (transfer.bytes_total - transfer.bytes_complete) / speed) : null;
  const canStart = provider === "civitai" ? Boolean(civitaiPreview && fileId) : Boolean(huggingFacePreview);
  const requiresUnsafe = Boolean(selectedFile?.requires_unsafe_confirmation || huggingFacePreview?.files.some((file) => unsafeName(file.filename)));

  return (
    <div className="asset-backdrop">
      <section className="asset-panel transfer-panel glass-card" aria-label="Provider downloads">
        <header><div><p className="kicker">Provider-side transfer</p><h2>Download models</h2></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <div className="asset-kind-tabs"><button className={provider === "civitai" ? "active" : ""} onClick={() => { setProvider("civitai"); setTransfer(history.find((item) => item.provider === "civitai") ?? null); }}>Civitai</button><button className={provider === "huggingface" ? "active" : ""} onClick={() => { setProvider("huggingface"); setTransfer(history.find((item) => item.provider === "huggingface") ?? null); }}>Hugging Face</button><button className="quiet-button" onClick={() => { setBatchMode(true); setBatchText(""); setError(""); }}>Batch load</button></div>
        <div className="provider-credential">
          <span>{credential?.configured ? `Token connected ${credential.key_hint ?? ""}` : "Public files work without a token. Add one for private or gated files."}</span>
          {credential?.configured ? <button className="danger-text-button" disabled={busy} onClick={() => void disconnectToken()}>Remove token</button> : <><input className="text-input secret-input" type="password" autoComplete="off" placeholder={provider === "civitai" ? "Download-only token" : "Read-only token"} value={token} onChange={(event) => setToken(event.target.value)} /><button className="quiet-button" disabled={busy || token.length < 8} onClick={() => void saveToken()}>Save encrypted token</button></>}
        </div>
        {batchMode ? <div className="batch-download-editor">
          <div><p className="kicker">{provider === "civitai" ? "Civitai" : "Hugging Face"} batch</p><h3>Sequential model downloads</h3><p className="field-help">Enter one URL per line. Each item is inspected and downloaded completely before the next begins.</p></div>
          <textarea className="text-input batch-url-input" value={batchText} onChange={(event) => setBatchText(event.target.value)} placeholder={provider === "civitai" ? "https://civitai.com/api/download/models/..." : "https://huggingface.co/owner/repository"} rows={8} />
          {provider === "huggingface" && <label><span>Repository file filters</span><input className="text-input" value={patterns} onChange={(event) => setPatterns(event.target.value)} placeholder="*.safetensors, *.json" /></label>}
          {(provider === "civitai" || provider === "huggingface") && <UnsafeConfirmation checked={allowUnsafe} onChange={setAllowUnsafe} />}
          <div className="batch-actions">
            <button className="quiet-button" disabled={batchBusy} onClick={saveBatch}>Save batch</button>
            <button className="quiet-button" disabled={batchBusy} onClick={() => batchFileInput.current?.click()}>Load batch</button>
            <input ref={batchFileInput} type="file" accept="application/json,.json" hidden onChange={(event) => { const file = event.target.files?.[0]; event.currentTarget.value = ""; if (file) loadBatch(file); }} />
            <button className="quiet-button" onClick={exitBatch}>Cancel</button>
            <button className="primary-button" disabled={batchBusy || !batchUrls().length} onClick={() => void runBatch()}>{batchBusy ? "Downloading…" : "Download"}</button>
          </div>
        </div> : <div className="download-form">
          <label><span>{provider === "civitai" ? "Civitai model download" : "Canonical Hugging Face repository or file"} URL</span><input className="text-input" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder={provider === "civitai" ? "https://civitai.red/api/download/models/...?fileId=..." : "https://huggingface.co/owner/repo/..."} /></label>
          <label><span>Install into</span><select className="select-input" value={destination} onChange={(event) => setDestination(event.target.value as FileKind)}>{destinations.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          {provider === "huggingface" && <label><span>Repository file filters</span><input className="text-input" value={patterns} onChange={(event) => setPatterns(event.target.value)} placeholder="*.safetensors, *.json" /><small>Comma-separated allow patterns. File URLs ignore this field.</small></label>}
          <button className="primary-button" disabled={busy || !sourceUrl} onClick={() => void preview()}>Inspect before download</button>
        </div>}
        {civitaiPreview && <div className="download-preview"><strong>{civitaiPreview.model_name} · {civitaiPreview.version_name}</strong><small>{civitaiPreview.model_type ?? "Model"} · {civitaiPreview.base_model ?? "Unknown base"}</small><select className="select-input" value={fileId} onChange={(event) => { setFileId(event.target.value); setAllowUnsafe(false); }}>{civitaiPreview.files.map((file) => <option key={file.id} value={file.id}>{file.filename} · {formatBytes(file.size_bytes ?? 0)}{file.preferred ? " · preferred" : ""}</option>)}</select>{selectedFile?.requires_unsafe_confirmation && <UnsafeConfirmation checked={allowUnsafe} onChange={setAllowUnsafe} />}</div>}
        {huggingFacePreview && <div className="download-preview"><strong>{huggingFacePreview.repo_id}</strong><small>{huggingFacePreview.mirror_repository ? `Repository mirror · ${huggingFacePreview.files.length} files` : huggingFacePreview.filename} · {formatBytes(huggingFacePreview.required_bytes)} required</small>{requiresUnsafe && <UnsafeConfirmation checked={allowUnsafe} onChange={setAllowUnsafe} />}</div>}
        {canStart && !active && transfer?.state !== "completed" && <button className="primary-button" disabled={busy || (requiresUnsafe && !allowUnsafe)} onClick={() => void start(Boolean(transfer))}>{transfer ? "Retry / resume transfer" : "Start provider download"}</button>}
        {transfer && <div className="remote-transfer"><div className="transfer-progress"><div><i style={{ width: `${progress * 100}%` }} /></div><span>{transferStateLabel(transfer.state)} · {formatBytes(transfer.bytes_complete)}{transfer.bytes_total !== null ? ` / ${formatBytes(transfer.bytes_total)}` : ""}{speed > 0 ? ` · ${formatBytes(speed)}/s${eta !== null ? ` · ${Math.ceil(eta)}s remaining` : ""}` : ""}</span></div>{active && <button className="danger-text-button" disabled={busy} onClick={() => void cancel()}>Cancel and keep resumable data</button>}{transfer.state === "completed" && <p className="success-line">Installed {transfer.files.length} verified file{transfer.files.length === 1 ? "" : "s"}.</p>}{transfer.error_message && <div className="error-banner">{transfer.error_message} <small>{transfer.error_code}</small></div>}</div>}
        {history.length > 0 && <div className="transfer-history"><strong>Recent provider downloads</strong>{history.map((item) => <button key={item.id} className={item.id === transfer?.id ? "selected" : ""} onClick={() => { setTransfer(item); setProvider(item.provider); setDestination(item.destination_kind); setSourceUrl(item.source_url); }}><span><b>{item.filename ?? sourceLabel(item.source_url)}</b><small>{item.provider} · {item.destination_kind.replaceAll("_", " ")}</small></span><em className={item.state}>{transferStateLabel(item.state)}</em></button>)}</div>}
        {error && <div className="error-banner">{error}</div>}
        <p className="field-help">Tokens are encrypted by the control plane and sent to the agent only for one operation. URLs with embedded credentials and redirects to unapproved hosts are rejected.</p>
      </section>
    </div>
  );
}

function UnsafeConfirmation({ checked, onChange }: { checked: boolean; onChange: (value: boolean) => void }) { return <label className="check-row warning-check"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span><strong>Allow pickle-based model format</strong><small>Only continue if you trust the publisher. Safetensors is preferred.</small></span></label>; }
function terminal(state: string) { return ["completed", "failed", "cancelled", "paused"].includes(state); }
function patternList(value: string) { return value.split(",").map((item) => item.trim()).filter(Boolean); }
function unsafeName(value: string) { return [".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"].some((suffix) => value.toLowerCase().endsWith(suffix)); }
function inferDestination(filename: string, hint?: string | null): FileKind {
  const value = `${hint ?? ""} ${filename}`.toLowerCase();
  if (/\b(lora|locon|loha|lycoris|adapter)\b/.test(value)) return "loras";
  if (/\b(vae|variational autoencoder)\b/.test(value) || /(^|[._-])vae([._-]|$)/.test(value)) return "vae";
  if (/\b(upscale|upscaler|esrgan|realesrgan|4x|8x)\b/.test(value)) return "upscale_models";
  if (/\b(clip|text[-_ ]?encoder|t5|llama|qwen|gemma)\b/.test(value)) return "text_encoders";
  if (/\b(face[-_ ]?det|insightface|yolo)\b/.test(value) || /\.(onnx|pb)(?:$|[?#])/.test(value)) return "face_detection";
  return "diffusion_models";
}
function message(caught: unknown) { return caught instanceof Error ? caught.message : "Provider transfer failed"; }
function transferStateLabel(state: string) { return state === "pending" ? "queued" : state; }
function sourceLabel(value: string) { try { return decodeURIComponent(new URL(value).pathname.split("/").filter(Boolean).at(-1) ?? value); } catch { return value; } }
function formatBytes(value: number) { if (value < 1024) return `${value} B`; if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`; if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`; return `${(value / 1024 ** 3).toFixed(1)} GiB`; }
