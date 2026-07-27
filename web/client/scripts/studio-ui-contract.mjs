import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { committedNumber } from "../src/numericDraft.ts";
import { sortOutputFiles } from "../src/outputSort.ts";
import { SerialRequestLane } from "../src/requestQueue.ts";
import {
  clearPendingJobSubmission,
  isAmbiguousJobSubmissionError,
  loadPendingJobSubmission,
  savePendingJobSubmission,
  uniqueJobs,
} from "../src/submissionRecovery.ts";
import { COMFYUI_SCHEDULERS } from "../src/studioProject.ts";
import {
  POSE_LIMB_GROUPS,
  POSE_JOINT_NAMES,
  POSE_TORSO_JOINTS,
  poseGroupCenter,
  rotatePoseGroup,
  standingPose,
  translatePoseGroup,
} from "../src/pose.ts";

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

const recoveryStorage = new Map();
const storageAdapter = {
  getItem: (key) => recoveryStorage.get(key) ?? null,
  setItem: (key, value) => recoveryStorage.set(key, value),
  removeItem: (key) => recoveryStorage.delete(key),
};
const pendingRecovery = {
  version: 1,
  workspaceId: "workspace-1",
  payloads: [{
    command_id: "stable-command-id",
    kind: "generate",
    project_id: "studio-workspace-1",
    project: { schema: "k2-region-lab-project" },
    filename_prefix: "k2lab",
  }],
  acknowledgedJobs: [],
  createdAt: "2026-07-27T00:00:00Z",
  lastError: "The workspace agent did not finish before its network timeout.",
};
savePendingJobSubmission(storageAdapter, pendingRecovery);
assert.deepEqual(
  loadPendingJobSubmission(storageAdapter, "workspace-1"),
  pendingRecovery,
  "A timed-out POST must retain its exact idempotent payload across a page refresh",
);
assert.equal(isAmbiguousJobSubmissionError({ status: 504, code: "agent_read_timeout" }), true);
assert.equal(isAmbiguousJobSubmissionError({ status: 400, code: "invalid_job" }), false);
assert.deepEqual(
  uniqueJobs([
    { id: "same", command_id: "one", state: "queued" },
    { id: "same", command_id: "one", state: "running" },
  ]).map((item) => item.state),
  ["running"],
  "Recovered job receipts must replace, rather than duplicate, an existing tracked job",
);
clearPendingJobSubmission(storageAdapter, "workspace-1");
assert.equal(loadPendingJobSubmission(storageAdapter, "workspace-1"), null);

const standing = standingPose();
const movedArm = translatePoseGroup(standing, POSE_LIMB_GROUPS.left_arm, 0.1, -0.05);
for (const name of POSE_LIMB_GROUPS.left_arm) {
  const before = standing.joints.find((joint) => joint.name === name);
  const after = movedArm.joints.find((joint) => joint.name === name);
  assert.ok(Math.abs(after.x - before.x - 0.1) < 1e-9);
  assert.ok(Math.abs(after.y - before.y + 0.05) < 1e-9);
}
assert.deepEqual(
  movedArm.joints.find((joint) => joint.name === "right_elbow"),
  standing.joints.find((joint) => joint.name === "right_elbow"),
  "A limb group handle must not move the opposite limb",
);
const movedTorso = translatePoseGroup(standing, POSE_TORSO_JOINTS, 0.08, 0.03, true);
assert.ok(Math.abs(movedTorso.head.cx - standing.head.cx - 0.08) < 1e-9);
assert.ok(Math.abs(movedTorso.head.cy - standing.head.cy - 0.03) < 1e-9);
const torsoCenter = poseGroupCenter(standing, [
  "left_shoulder", "right_shoulder", "left_hip", "right_hip",
]);
const rotatedTorso = rotatePoseGroup(
  standing,
  POSE_JOINT_NAMES,
  torsoCenter,
  Math.PI / 3,
  400,
  900,
  true,
);
const beforeShoulder = standing.joints.find((joint) => joint.name === "left_shoulder");
const afterShoulder = rotatedTorso.joints.find((joint) => joint.name === "left_shoulder");
assert.ok(Math.abs(
  Math.hypot(
    (beforeShoulder.x - torsoCenter.x) * 400,
    (beforeShoulder.y - torsoCenter.y) * 900,
  ) - Math.hypot(
    (afterShoulder.x - torsoCenter.x) * 400,
    (afterShoulder.y - torsoCenter.y) * 900,
  ),
) < 1e-8, "Torso rotation must preserve joint distance in canvas space");
assert.notDeepEqual(
  rotatedTorso.joints.find((joint) => joint.name === "right_ankle"),
  standing.joints.find((joint) => joint.name === "right_ankle"),
  "The torso rotation handle must rotate the entire figure, including distal limbs",
);
assert.equal(peakRequests, 1, "Each browser request lane must remain strictly serialized");
assert.deepEqual(
  requestOrder,
  ["first", "urgent", "low"],
  "Urgent cancellation/release requests must move ahead of queued background polling",
);

const inspector = await readFile(new URL("../src/components/Inspector.tsx", import.meta.url), "utf8");
const poseGatingControls = await readFile(new URL("../src/components/PoseGatingControls.tsx", import.meta.url), "utf8");
const poseSource = await readFile(new URL("../src/pose.ts", import.meta.url), "utf8");
const canonicalPoseJoints = poseSource.split("POSE_JOINT_NAMES = [", 2)[1].split("] as const", 1)[0];
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
assert.ok(
  inspector.includes("Subject box · mannequin")
    && inspector.includes("Region box · no mannequin")
    && inspector.includes("Volumetric pose gating")
    && poseGatingControls.includes("Constrain generation to subject mannequins")
    && poseGatingControls.includes("Hard gate steps")
    && poseGatingControls.includes("Sigma schedule"),
  "The inspector must distinguish volumetric subject boxes and expose pose-gating controls",
);
assert.ok(
  canonicalPoseJoints.includes('"left_ankle"')
    && poseSource.includes("PoseHeadState")
    && !canonicalPoseJoints.includes('"nose"')
    && !canonicalPoseJoints.includes('"left_eye"'),
  "The canonical browser mannequin must use 13 body joints plus a head ellipse and no face handles",
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
    && !workspaceStudio.includes('pose_controlnet_file_id')
    && !workspaceStudio.includes('allFiles("loras"), allFiles("upscale_models")'),
  "Project restoration must remain sequential and pose submission must not require ControlNet",
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
  app.includes("workspace: activeWorkspaces.at(-1) ?? null"),
  "Startup must prefer the newest active workspace record",
);
assert.ok(
  app.includes("existingWorkspaces={state.workspaces}")
    && app.includes("onWorkspaceMenu")
    && app.includes("workspace: null"),
  "The studio must be able to return to the workspace menu without deleting a Pod",
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
  onboarding.includes("Existing cloud workspaces")
    && onboarding.includes("Reconnect without recreating")
    && onboarding.includes("controlPlane.workspace(workspace.id)")
    && onboarding.includes("Connect"),
  "The creation menu must list and reconnect existing managed Pods without starting them",
);
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
  runRemoteJob.includes("isAmbiguousJobSubmissionError")
    && runRemoteJob.includes("rememberSubmissionRecovery(recovery)")
    && workspaceStudio.includes("recoverTimedOutSubmission")
    && workspaceStudio.includes("Recover submission")
    && workspaceStudio.includes("Cancel all remote work")
    && workspaceStudio.includes("submissionRecovery.payloads[0].command_id"),
  "An ambiguous job POST must retain its command ID and present safe recovery and cancel-all remedies",
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
assert.ok(
  regionCanvas.includes("pose-torso-move")
    && regionCanvas.includes("pose-rotate-handle")
    && regionCanvas.includes("POSE_LIMB_GROUPS")
    && regionCanvas.includes("translatePoseGroup")
    && regionCanvas.includes("rotatePoseGroup")
    && regionCanvas.includes("pose-volume pose-neck"),
  "The volumetric mannequin must expose attached torso translation/rotation and whole-limb controls",
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
