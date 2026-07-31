import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { committedNumber } from "../src/numericDraft.ts";
import { sortOutputFiles } from "../src/outputSort.ts";
import { SerialRequestLane } from "../src/requestQueue.ts";
import { COMFYUI_SCHEDULERS } from "../src/studioProject.ts";

assert.equal(committedNumber("", 1, 100), null, "an empty editing draft must remain transient");
assert.equal(committedNumber("-", -10, 10), null, "an incomplete signed draft must remain transient");
assert.equal(committedNumber("27", 1, 100), 27);
assert.equal(committedNumber("200", 1, 100), 100);
assert.ok(
  COMFYUI_SCHEDULERS.includes("bong_tangent"),
  "The scheduler selector must expose the worker's bong_tangent implementation",
);

const requestLane = new SerialRequestLane();
let activeRequests = 0;
let peakRequests = 0;
const requestOrder = [];
const queuedRequest = (name, delay, priority) => requestLane.run(async () => {
  activeRequests += 1;
  peakRequests = Math.max(peakRequests, activeRequests);
  requestOrder.push(name);
  await new Promise((resolve) => setTimeout(resolve, delay));
  activeRequests -= 1;
  return name;
}, priority);
const firstRequest = queuedRequest("first", 5, 10);
const lowPriorityRequest = queuedRequest("low", 0, 20);
const urgentRequest = queuedRequest("urgent", 0, 0);
assert.deepEqual(
  await Promise.all([firstRequest, lowPriorityRequest, urgentRequest]),
  ["first", "low", "urgent"],
);
assert.equal(peakRequests, 1, "Each browser request lane must remain strictly serialized");
assert.deepEqual(
  requestOrder,
  ["first", "urgent", "low"],
  "Urgent cancellation/release requests must move ahead of queued background polling",
);

const inspector = await readFile(new URL("../src/components/Inspector.tsx", import.meta.url), "utf8");
const promptSection = inspector.split('{tab === "prompt"', 2)[1].split('{tab === "regions"', 1)[0];
const regionsSection = inspector.split('{tab === "regions" && mode !== "face"', 2)[1].split('{tab === "loras"', 1)[0];

assert.ok(!promptSection.includes("moveSelected("), "Prompt tab must not own region depth controls");
assert.ok(regionsSection.includes("moveSelected(-1)"), "Regions tab must move the selected region forward");
assert.ok(regionsSection.includes("moveSelected(1)"), "Regions tab must move the selected region backward");
assert.ok(
  inspector.includes('label="Execution mode"')
    && inspector.includes('label="VRAM reserve · GiB"')
    && inspector.includes('updateRuntime({ vramMode')
    && inspector.includes('updateRuntime({ reserveVramGb')
    && inspector.includes('label="Keep baseline model loaded between runs"')
    && inspector.includes('updateRuntime({ keepModelLoaded')
    && inspector.includes('label="Enable system RAM safeguard"')
    && inspector.includes("Actual non-cache use")
    && inspector.includes("Clean reclaimable cache")
    && inspector.includes("Release worker memory"),
  "Advanced settings must expose GPU controls, RAM safeguards, live cache diagnostics, and recovery",
);
assert.ok(
  inspector.includes("<SeedField")
    && inspector.includes('ariaLabel="Seed value"')
    && inspector.includes('aria-label="Seed behavior"')
    && !inspector.includes('<Choice label="Seed behavior"'),
  "Seed behavior must be a dropdown directly attached to the generation seed field",
);
assert.ok(
  inspector.includes("toggleRegion(lora, region.id")
    && inspector.includes("loraBindingKey(mode, activeLayer)")
    && inspector.includes("defaultLoraTrigger(lora.name)")
    && inspector.includes("setLoraBindingEnabled(item, bindingKey")
    && inspector.includes('role="switch"')
    && inspector.includes('binding.enabled ? <div className="lora-binding-controls">')
    && inspector.includes("duplicateStudioLora(lora, bindingKey)")
    && inspector.includes("onCheckLoras"),
  "LoRA controls must safely collapse disabled assignments while preserving per-mode routing, duplicates, and compatibility checks",
);
assert.ok(
  promptSection.includes('className="prompt-region-pane"')
    && promptSection.includes("onSelect(region.id)"),
  "Prompt editing must expose direct global and region selection without changing tabs",
);
assert.ok(
  inspector.includes("promptEmphasisFromSelection")
    && inspector.includes("reconcilePromptEmphases")
    && inspector.includes("emphasisAvailable && patch.prompt")
    && inspector.includes("if (emphasisAvailable)")
    && inspector.includes('reconcileEmphases("__global__", event.target.value)')
    && inspector.includes("reconcileEmphases(selected.id, patch.prompt)"),
  "Prompt edits must repair or remove stale phrase emphasis metadata",
);

for (const relativePath of [
  "../src/components/Inspector.tsx",
  "../src/components/CloudOnboarding.tsx",
  "../src/components/WorkspaceStudio.tsx",
]) {
  const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  assert.ok(!/type="number"[^>]*onChange=/.test(source), `${relativePath} bypasses draft-safe numeric input`);
}

const workspaceStudio = await readFile(new URL("../src/components/WorkspaceStudio.tsx", import.meta.url), "utf8");
const regionCanvas = await readFile(new URL("../src/components/RegionCanvas.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
const main = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
assert.ok(
  workspaceStudio.includes("isolatedLorasForMode(loras, mode)")
    && workspaceStudio.includes('kind: "validate_loras"')
    && workspaceStudio.includes("const cloudSource = cloudSources[mode]"),
  "Remote jobs, inputs, and compatibility checks must use only the active mode's bindings",
);
assert.ok(
  regionCanvas.includes("Hide regions")
    && regionCanvas.includes("data-region-overlay-hidden")
    && regionCanvas.includes("!regionOverlayHidden && orderedRegions.map"),
  "The canvas must expose a non-destructive region visibility toggle",
);
assert.ok(
  styles.includes("height: 100dvh")
    && styles.includes("max-height: 100dvh")
    && styles.includes(".studio-main { grid-row: 3; overflow-y: auto; }")
    && styles.includes(".lora-binding-controls")
    && styles.includes(".lora-disabled-note"),
  "Studio layout and stable LoRA controls must prevent body growth and toggle-induced blank space",
);
assert.ok(
  main.includes("<AppErrorBoundary>")
    && main.includes("</AppErrorBoundary>"),
  "The application must render a recoverable diagnostic instead of a blank screen after render failures",
);
const setupPanel = await readFile(new URL("../src/components/SetupPanel.tsx", import.meta.url), "utf8");
assert.ok(
  workspaceStudio.includes("const activeJobId")
    && workspaceStudio.includes("window.setTimeout(() => void refresh(), 1_000)")
    && workspaceStudio.includes("if (cancelled) return;")
    && !workspaceStudio.includes("setInterval(async"),
  "Workspace, memory, and generation polling must remain single-flight",
);
assert.ok(
  workspaceStudio.includes('loraFiles = await allFiles("loras")')
    && workspaceStudio.includes('outputFiles = await allFiles("outputs")')
    && !workspaceStudio.includes('allFiles("loras"), allFiles("upscale_models")'),
  "Project restoration must not burst all Pod inventory requests concurrently",
);
assert.ok(
  setupPanel.includes("for (const { kind } of modelKinds)")
    && !setupPanel.includes("Promise.all(modelKinds"),
  "Setup must load Pod model inventories sequentially",
);
const api = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
assert.ok(
  api.includes("controlRequestLane")
    && api.includes("bulkRequestLane")
    && api.includes('options.lane === "bulk"')
    && api.includes("queuedFileBlob"),
  "Browser requests must use serialized control and bulk lanes",
);
const assetPanel = await readFile(new URL("../src/components/AssetPanel.tsx", import.meta.url), "utf8");
assert.ok(
  assetPanel.includes("onSelectMany?: (files: FileRecord[]) => void")
    && assetPanel.includes('kind === "loras" && onSelectMany')
    && assetPanel.includes("onSelectMany(selectedFiles)")
    && assetPanel.includes("Add selected LoRAs to project"),
  "The Cloud LoRA tab must add all checked files to the project in one action",
);
assert.ok(
  workspaceStudio.includes("addStudioLoraFiles(current, selectedLoras, bindingKey)")
    && workspaceStudio.includes("addStudioLoraFiles(current, [file]")
    && workspaceStudio.includes("Added ${selectedLoras.length} cloud LoRA"),
  "Single and bulk Cloud LoRA selection must share the state-safe project binding path",
);
assert.ok(
  assetPanel.includes('useState<OutputSort>("newest")')
    && assetPanel.includes("Newest first")
    && assetPanel.includes("Oldest first")
    && assetPanel.includes("Name A–Z")
    && assetPanel.includes("Largest first"),
  "Outputs must default to newest-first and expose the standard sort choices",
);
const outputFixture = [
  { id: "old", display_name: "image-2.png", modified_at: "2026-01-01T00:00:00Z", size_bytes: 20 },
  { id: "new", display_name: "image-10.png", modified_at: "2026-02-01T00:00:00Z", size_bytes: 10 },
  { id: "large", display_name: "alpha.png", modified_at: "2026-01-15T00:00:00Z", size_bytes: 30 },
];
assert.deepEqual(sortOutputFiles(outputFixture, "newest").map((item) => item.id), ["new", "large", "old"]);
assert.deepEqual(sortOutputFiles(outputFixture, "oldest").map((item) => item.id), ["old", "large", "new"]);
assert.deepEqual(sortOutputFiles(outputFixture, "name-asc").map((item) => item.id), ["large", "old", "new"]);
assert.deepEqual(sortOutputFiles(outputFixture, "size-desc").map((item) => item.id), ["large", "old", "new"]);
const app = await readFile(new URL("../src/App.tsx", import.meta.url), "utf8");
assert.ok(
  app.includes("provider_resource_not_found")
    && app.includes("This Pod no longer exists")
    && app.includes("Create a new workspace")
    && app.includes("controlPlane.connectMigratedPod")
    && app.includes("controlPlane.terminateWorkspace"),
  "A missing provider Pod must open migration-or-cleanup recovery instead of the studio",
);
assert.ok(
  app.includes('.filter((item) => item.state !== "deleted").at(-1)'),
  "Startup must prefer the newest active workspace record",
);
assert.ok(
  workspaceStudio.includes('href="https://console.runpod.io/pods"'),
  "Cloud workspace status must link to RunPod's Docker startup progress",
);
assert.ok(
  workspaceStudio.includes("workspace.provider_resource_id"),
  "RunPod startup progress must identify the provider Pod",
);
assert.ok(
  workspaceStudio.includes("startWithoutTimeLimit")
    && workspaceStudio.includes("continue running and billing until you manually stop it"),
  "Restarting a Pod must offer an explicitly warned unlimited lease",
);
assert.ok(
  workspaceStudio.includes("Connect migrated Pod")
    && workspaceStudio.includes("controlPlane.connectMigratedPod")
    && workspaceStudio.includes("Verify and connect"),
  "Stopped persistent workspaces must support verified RunPod console migration reassociation",
);

const onboarding = await readFile(new URL("../src/components/CloudOnboarding.tsx", import.meta.url), "utf8");
assert.ok(
  onboarding.includes("request.lease_unlimited")
    && onboarding.includes("keep running and billing until you manually stop it"),
  "Initial Pod creation must offer an explicitly warned unlimited lease",
);
assert.ok(
  workspaceStudio.includes('setUtilityPanel("assets")')
    && workspaceStudio.includes('setUtilityPanel("transfers")')
    && workspaceStudio.includes('setUtilityPanel("setup")'),
  "Utility rail panels must replace one another instead of stacking",
);
assert.ok(
  workspaceStudio.includes("eventDockHeight")
    && workspaceStudio.includes("event-dock-resize")
    && workspaceStudio.includes("Resize event log")
    && workspaceStudio.includes("eventListRef"),
  "The event history must be a resizable bottom dock that retains multiple visible lines",
);
assert.ok(
  workspaceStudio.includes("const [eventDockOpen, setEventDockOpen] = useState(false)"),
  "The event history must start closed so it does not reserve empty workspace height",
);
assert.ok(
  workspaceStudio.includes("setResultUrl(controlPlane.outputUrl")
    && workspaceStudio.includes('setComparePosition(next.kind === "generate" ? 1 : 0.5)'),
  "Completed generation outputs must be selected for immediate full-canvas display",
);
assert.ok(
  workspaceStudio.includes('controlPlane.files(workspace.id, "outputs")')
    && workspaceStudio.includes("candidate.modified_at")
    && workspaceStudio.includes("current ?? controlPlane.outputUrl"),
  "Reopening the studio must restore the newest verified output after a lost completion connection",
);
const runRemoteJob = workspaceStudio.split("async function runRemoteJob()", 2)[1].split("async function cancelRemoteJob", 1)[0];
assert.ok(
  !runRemoteJob.includes("setResultUrl(null)"),
  "Starting a new job must retain the current output until its replacement completes",
);
assert.ok(
  runRemoteJob.includes('...(mode === "face"')
    && runRemoteJob.includes('face_detector_file_id: mode === "face"'),
  "Generation and image editing must not require or submit a face detector",
);
assert.ok(
  !workspaceStudio.includes("anchor.download = name")
    && workspaceStudio.includes("Assets › Projects")
    && workspaceStudio.includes("controlPlane.saveProject"),
  "Saving must persist projects to workspace assets without forcing a browser download",
);
assert.ok(
  workspaceStudio.includes("FACE_DETECTOR_SOURCE")
    && workspaceStudio.includes("showFaceDetectorInstall")
    && workspaceStudio.includes("installFaceDetector")
    && workspaceStudio.includes('destination_kind: "face_detection"'),
  "Missing face detection must offer a pinned, managed detector installation",
);

const transferPanel = await readFile(new URL("../src/components/TransferPanel.tsx", import.meta.url), "utf8");
assert.ok(transferPanel.includes("controlPlane.transfers(workspaceId)"), "Provider transfer history must restore when the panel opens");

assert.ok(assetPanel.includes("controlPlane.uploads(workspaceId)"), "Local upload history must restore when the panel opens");
assert.ok(
  assetPanel.includes("asset-thumbnail-grid")
    && assetPanel.includes("asset-image-preview")
    && assetPanel.includes("Back to thumbnails")
    && assetPanel.includes("queuedFileBlob"),
  "Output assets must render thumbnails with an in-pane enlarged preview",
);
assert.ok(
  regionCanvas.includes("(sourceUrl || resultUrl)")
    && regionCanvas.includes("Clear canvas"),
  "A visible source or generated result must expose an explicit canvas clear control",
);
assert.ok(
  regionCanvas.includes("draftLasso")
    && regionCanvas.includes("manual-face-path-draft")
    && regionCanvas.includes('alt="Generated result"'),
  "The canvas must render both a live lasso draft and a generated result without a source image",
);
assert.ok(
  regionCanvas.includes("RetryingResultImage")
    && regionCanvas.includes("onError={retry}")
    && regionCanvas.includes("retryCount.current >= 6"),
  "Transient output delivery failures must retry instead of leaving the canvas permanently blank",
);
assert.ok(
  regionCanvas.includes("orderedRegions")
    && regionCanvas.includes("visibleRegions.filter((region) => region.id !== selectedId)")
    && regionCanvas.includes("onPointerDown={region.id === selectedId")
    && styles.includes(".region-group:not(.selected) { pointer-events: none; }"),
  "Only the region selected in the inspector may receive move or resize pointer input",
);
assert.ok(
  regionCanvas.includes("new ResizeObserver")
    && regionCanvas.includes("availableWidth / canvasWidth")
    && regionCanvas.includes("availableHeight / canvasHeight")
    && regionCanvas.includes('aspectRatio: `${canvasWidth} / ${canvasHeight}`'),
  "The canvas frame must scale uniformly against both available axes when the event dock is resized",
);
const uploadQueue = await readFile(new URL("../src/useUploadQueue.ts", import.meta.url), "utf8");
assert.ok(
  workspaceStudio.includes("useUploadQueue(workspace.id")
    && assetPanel.includes("uploadQueue.enqueue")
    && assetPanel.includes("Background upload queue"),
  "Local uploads must be owned by the workspace so closing Assets does not stop them",
);
assert.ok(
  uploadQueue.includes('item.state === "queued"')
    && uploadQueue.includes("activeRef.current")
    && uploadQueue.includes("controlPlane.cancelUpload")
    && uploadQueue.includes("candidate.sha256 === digest"),
  "Local uploads must run through one resumable FIFO worker with explicit cancellation",
);

console.log("Studio UI contract passed");
