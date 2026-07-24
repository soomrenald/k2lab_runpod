import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { committedNumber } from "../src/numericDraft.ts";

assert.equal(committedNumber("", 1, 100), null, "an empty editing draft must remain transient");
assert.equal(committedNumber("-", -10, 10), null, "an incomplete signed draft must remain transient");
assert.equal(committedNumber("27", 1, 100), 27);
assert.equal(committedNumber("200", 1, 100), 100);

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
    && inspector.includes('updateRuntime({ keepModelLoaded'),
  "Generation Advanced settings must expose working GPU execution-mode, reserve, and residency controls",
);
assert.ok(
  inspector.includes("toggleRegion(lora, region.id")
    && inspector.includes('activeLayer === "generation"')
    && inspector.includes("defaultLoraTrigger(lora.name)")
    && inspector.includes("binding.regionIds.length === 0"),
  "LoRA routing controls must preserve desktop scope and identity-trigger semantics",
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

const assetPanel = await readFile(new URL("../src/components/AssetPanel.tsx", import.meta.url), "utf8");
assert.ok(assetPanel.includes("controlPlane.uploads(workspaceId)"), "Local upload history must restore when the panel opens");
assert.ok(
  assetPanel.includes("asset-thumbnail-grid")
    && assetPanel.includes("asset-image-preview")
    && assetPanel.includes("Back to thumbnails"),
  "Output assets must render thumbnails with an in-pane enlarged preview",
);
const regionCanvas = await readFile(new URL("../src/components/RegionCanvas.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
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
