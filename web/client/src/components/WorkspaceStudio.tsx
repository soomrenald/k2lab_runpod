import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { DatacenterOption, DetectedFaceRecord, FileKind, FileRecord, GenerationJob, JobKind, JobSubmitPayload, KreaControlCheckpointInspection, NetworkVolumeOption, RemoteTransfer, UnifiedPromptPreview, VolumetricControlPreview, WorkerMemoryStatus, WorkspaceMigrationRecord, WorkspaceRecord } from "../api";
import { controlPlane } from "../api";
import { Icon, type IconName } from "./Icon";
import { Inspector } from "./Inspector";
import { AssetPanel } from "./AssetPanel";
import { TransferPanel } from "./TransferPanel";
import { SetupPanel } from "./SetupPanel";
import { DraftNumberInput } from "./DraftNumberInput";
import { uploadWorkspaceFile } from "../uploads";
import { useUploadQueue } from "../useUploadQueue";
import { appendBoundedEvents, EVENT_LOG_LIMIT } from "../eventLog";
import {
  isPassiveProviderTransient,
  providerFreshnessEvent,
  scheduleProviderPoll,
} from "../providerFreshness";
import {
  clearPendingJobSubmission,
  isAmbiguousJobSubmissionError,
  loadPendingJobSubmission,
  type PendingJobSubmission,
  savePendingJobSubmission,
  uniqueJobs,
} from "../submissionRecovery";
import {
  buildProjectDocument,
  bindStudioLoraFiles,
  createStudioLora,
  createStudioSettings,
  loadStudioProjectDocument,
  projectDocumentFromPng,
  type StudioLora,
} from "../studioProject";
import {
  RegionCanvas,
  type DrawMode,
  type RegionBox,
  type RegionLayer,
  type StudioMode,
} from "./RegionCanvas";

interface Props {
  workspace: WorkspaceRecord;
  developmentBackend: boolean;
  poseSemanticRoutingAvailable: boolean;
  poseControlLoraAvailable: boolean;
  depthControlAvailable: boolean;
  depthRegionsAvailable: boolean;
  datacenters: DatacenterOption[];
  networkVolumes: NetworkVolumeOption[];
  onWorkspace: (workspace: WorkspaceRecord) => void;
  onWorkspaceMenu: () => void;
  onDelete: () => void;
}

const starterRegions: RegionBox[] = [];
const FACE_DETECTOR_SOURCE = "https://huggingface.co/acvlab/FantasyPortrait/resolve/14df15cac6721a1cabdb9ecbdc0fbd6d3e49154b/face_det.onnx";
type StudioEventKind = "info" | "warning" | "error" | "worker";
type StudioEventSource = "job" | "workspace" | "provider" | "transfer";

interface StudioEvent {
  id: string;
  createdAt: string;
  kind: StudioEventKind;
  source?: StudioEventSource;
  message: string;
}

export function WorkspaceStudio({ workspace, developmentBackend, poseSemanticRoutingAvailable, poseControlLoraAvailable, depthControlAvailable, depthRegionsAvailable, datacenters, networkVolumes, onWorkspace, onWorkspaceMenu, onDelete }: Props) {
  const [mode, setMode] = useState<StudioMode>("generation");
  const [activeLayer, setActiveLayer] = useState<RegionLayer>("generation");
  const [regions, setRegions] = useState<RegionBox[]>(starterRegions);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState<DrawMode>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [cloudSource, setCloudSource] = useState<FileRecord | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultName, setResultName] = useState("");
  const [comparePosition, setComparePosition] = useState(0.5);
  const [globalPrompts, setGlobalPrompts] = useState<Record<RegionLayer, string>>({
    generation: "",
    reference: "",
    targets: "",
  });
  const [studioSettings, setStudioSettings] = useState(createStudioSettings);
  const [loras, setLoras] = useState<StudioLora[]>([]);
  const [assetPurpose, setAssetPurpose] = useState<"source" | "lora" | "pose-control" | "depth-checkpoint" | "depth-image" | "upscale">("source");
  const [showCloud, setShowCloud] = useState(false);
  const [startWithoutTimeLimit, setStartWithoutTimeLimit] = useState(false);
  const [showConnectPod, setShowConnectPod] = useState(false);
  const [connectPodId, setConnectPodId] = useState("");
  const [connectWithoutTimeLimit, setConnectWithoutTimeLimit] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [utilityPanel, setUtilityPanel] = useState<"assets" | "transfers" | "setup" | null>(null);
  const [eventDockOpen, setEventDockOpen] = useState(false);
  const [eventDockHeight, setEventDockHeight] = useState(138);
  const [showMigration, setShowMigration] = useState(false);
  const [migration, setMigration] = useState<WorkspaceMigrationRecord | null>(null);
  const [migrationConfirmation, setMigrationConfirmation] = useState("");
  const [migrationVolumeId, setMigrationVolumeId] = useState("");
  const [migrationDatacenterId, setMigrationDatacenterId] = useState(datacenters[0]?.id ?? "");
  const [migrationDiskGb, setMigrationDiskGb] = useState(workspace.workspace_disk_gb);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [eventLog, setEventLog] = useState<StudioEvent[]>([]);
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [queuedJobs, setQueuedJobs] = useState<GenerationJob[]>([]);
  const [submissionRecovery, setSubmissionRecovery] = useState<PendingJobSubmission | null>(
    () => loadPendingJobSubmission(window.localStorage, workspace.id),
  );
  const [promptPreview, setPromptPreview] = useState<UnifiedPromptPreview | null>(null);
  const [controlPreview, setControlPreview] = useState<(VolumetricControlPreview & {
    url: string;
    subjectRegionId: string | null;
  }) | null>(null);
  const [poseControlCompatibility, setPoseControlCompatibility] = useState<KreaControlCheckpointInspection | null>(null);
  const [projectName, setProjectName] = useState("untitled.k2lab.json");
  const [faceDetections, setFaceDetections] = useState<DetectedFaceRecord[]>([]);
  const [selectedFaceIndices, setSelectedFaceIndices] = useState<number[]>([]);
  const [manualFacePaths, setManualFacePaths] = useState<number[][][]>([]);
  const [lassoMode, setLassoMode] = useState(false);
  const [faceDimensions, setFaceDimensions] = useState({ width: 1024, height: 1024 });
  const [latestOutputFileId, setLatestOutputFileId] = useState<string | null>(null);
  const [showFaceDetectorInstall, setShowFaceDetectorInstall] = useState(false);
  const [faceDetectorInstalling, setFaceDetectorInstalling] = useState(false);
  const [faceDetectorTransfer, setFaceDetectorTransfer] = useState<RemoteTransfer | null>(null);
  const [faceDetectorInstallError, setFaceDetectorInstallError] = useState("");
  const [workerMemory, setWorkerMemory] = useState<WorkerMemoryStatus | null>(null);
  const [memoryRefreshing, setMemoryRefreshing] = useState(false);
  const eventCursor = useRef<string | undefined>(undefined);
  const providerStale = useRef(Boolean(workspace.provider_freshness?.stale));
  const eventListRef = useRef<HTMLDivElement>(null);
  const eventResize = useRef<{ pointerId: number; startY: number; startHeight: number } | null>(null);
  const openProjectInput = useRef<HTMLInputElement>(null);
  const importPngInput = useRef<HTMLInputElement>(null);

  function appendEvent(messageText: string, kind: StudioEventKind = "info", createdAt = new Date().toISOString(), source?: StudioEventSource) {
    if (!messageText) return;
    setEventLog((current) => appendBoundedEvents(current, [{
      id: crypto.randomUUID(),
      createdAt,
      kind,
      source,
      message: messageText,
    }]));
  }

  function report(messageText: string, kind: StudioEventKind = "info", source?: StudioEventSource) {
    setMessage(messageText);
    appendEvent(messageText, kind, new Date().toISOString(), source);
  }

  function rememberSubmissionRecovery(next: PendingJobSubmission | null) {
    setSubmissionRecovery(next);
    if (next) savePendingJobSubmission(window.localStorage, next);
    else clearPendingJobSubmission(window.localStorage, workspace.id);
  }

  function trackSubmittedJobs(jobs: GenerationJob[]) {
    const distinct = uniqueJobs(jobs);
    if (distinct.length === 0) return;
    setJob(distinct[0]);
    setQueuedJobs(distinct.slice(1));
  }

  const uploadQueue = useUploadQueue(workspace.id, report);
  const pendingUploadCount = uploadQueue.items.filter((item) => (
    !["completed", "cancelled"].includes(item.state)
  )).length;

  useEffect(() => {
    setSubmissionRecovery(loadPendingJobSubmission(window.localStorage, workspace.id));
  }, [workspace.id]);

  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);

  useEffect(() => {
    if (!eventDockOpen || !eventListRef.current) return;
    eventListRef.current.scrollTop = eventListRef.current.scrollHeight;
  }, [eventDockOpen, eventLog]);

  useEffect(() => {
    if (developmentBackend || workspace.state !== "ready") return undefined;
    let cancelled = false;
    void controlPlane.files(workspace.id, "outputs").then((page) => {
      const latest = page.items.reduce<FileRecord | null>((newest, candidate) => (
        !newest || new Date(candidate.modified_at).getTime() > new Date(newest.modified_at).getTime()
          ? candidate
          : newest
      ), null);
      if (cancelled || !latest) return;
      setLatestOutputFileId((current) => current ?? latest.id);
      setResultUrl((current) => current ?? controlPlane.outputUrl(workspace.id, latest.id));
      setResultName((current) => current || latest.display_name);
      setComparePosition(1);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [developmentBackend, workspace.id, workspace.state]);

  useEffect(() => {
    if (developmentBackend || workspace.state === "deleted") return undefined;
    let cancelled = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const refreshed = await controlPlane.workspace(workspace.id);
        if (!cancelled) {
          const stale = Boolean(refreshed.provider_freshness?.stale);
          const transition = providerFreshnessEvent(providerStale.current, stale);
          if (transition) {
            appendEvent(
              transition.message,
              transition.kind,
              new Date().toISOString(),
              transition.source,
            );
          }
          providerStale.current = stale;
          onWorkspace(refreshed);
        }
      } catch (caught) {
        if (!cancelled) {
          if (isPassiveProviderTransient(caught)) {
            const transition = providerFreshnessEvent(providerStale.current, true);
            if (transition) {
              appendEvent(
                transition.message,
                transition.kind,
                new Date().toISOString(),
                transition.source,
              );
            }
            providerStale.current = true;
          } else {
            report(
              caught instanceof Error ? caught.message : "Could not refresh workspace status",
              "error",
              "workspace",
            );
          }
        }
      } finally {
        if (!cancelled) timer = scheduleProviderPoll(window.setTimeout, () => void refresh());
      }
    }
    timer = scheduleProviderPoll(window.setTimeout, () => void refresh());
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [developmentBackend, onWorkspace, workspace.id, workspace.state]);

  useEffect(() => {
    if (developmentBackend || workspace.state !== "ready") {
      setWorkerMemory(null);
      return undefined;
    }
    let cancelled = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const status = await controlPlane.workerMemory(workspace.id);
        if (!cancelled) setWorkerMemory(status);
      } catch {
        if (!cancelled) setWorkerMemory(null);
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void refresh(), 10_000);
      }
    }
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [developmentBackend, workspace.id, workspace.state]);

  const activeJobId = job && !["completed", "failed", "cancelled"].includes(job.state)
    ? job.id
    : null;

  useEffect(() => {
    if (!activeJobId) return undefined;
    const jobId = activeJobId;
    let cancelled = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const next = await controlPlane.job(workspace.id, jobId);
        const events = await controlPlane.jobEvents(
          workspace.id,
          jobId,
          eventCursor.current,
        );
        if (cancelled) return;
        eventCursor.current = events.next_cursor;
        if (events.items.length) {
          setMessage(events.items[events.items.length - 1].message);
          setEventLog((current) => appendBoundedEvents(current, events.items.map((event) => ({
            id: `${jobId}-${event.sequence}`,
            createdAt: event.created_at,
            kind: "worker" as const,
            source: "job" as const,
            message: event.message,
          }))));
        }
        setJob(next);
        if (next.state === "completed" && next.output_file_ids[0]) {
          const outputFileId = next.output_file_ids[0];
          setLatestOutputFileId(outputFileId);
          setResultUrl(controlPlane.outputUrl(workspace.id, outputFileId));
          setResultName(`k2lab-${outputFileId.slice(0, 12)}.png`);
          void allFiles("outputs").then((items) => {
            const output = items.find((file) => file.id === outputFileId);
            if (output) setResultName(output.display_name);
          }).catch(() => undefined);
          setComparePosition(next.kind === "generate" ? 1 : 0.5);
          report(queuedJobs.length ? `Batch image complete. ${queuedJobs.length} queued run(s) remain.` : "Remote job complete. The verified output is stored in cloud files.", "worker");
        } else if (next.error_message) {
          report(next.error_message, "error");
        }
        if (["completed", "failed", "cancelled"].includes(next.state) && queuedJobs.length) {
          const [following, ...remaining] = queuedJobs;
          eventCursor.current = undefined;
          setQueuedJobs(remaining);
          setJob(following);
        }
      } catch (caught) {
        if (!cancelled) {
          report(caught instanceof Error ? caught.message : "Could not refresh remote job", "error");
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void refresh(), 1_000);
      }
    }
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeJobId, queuedJobs, workspace.id]);

  useEffect(() => {
    if (!showCloud && !showMigration) return undefined;
    let cancelled = false;
    void controlPlane.migrations(workspace.id).then((items) => {
      if (!cancelled && items.length) setMigration(items[items.length - 1]);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [showCloud, showMigration, workspace.id]);

  const running = workspace.state === "ready";
  const activeCompute = ["provisioning", "starting", "ready", "stopping"].includes(workspace.state);
  const canExtend = !workspace.lease_unlimited && (workspace.state === "starting" || workspace.state === "ready");
  const canStart = workspace.state === "stopped" || workspace.state === "error";
  const showAssets = utilityPanel === "assets";
  const showTransfers = utilityPanel === "transfers";
  const showSetup = utilityPanel === "setup";
  const leaseMinutes = Math.max(0, Math.round((new Date(workspace.lease_expires_at).getTime() - Date.now()) / 60_000));
  const readiness = useMemo(() => Object.entries(workspace.readiness), [workspace.readiness]);

  function switchMode(next: StudioMode) {
    setMode(next);
    setDrawMode(null);
    setSelectedId(null);
    setActiveLayer(next === "edit" ? "targets" : "generation");
  }

  async function loadImage(file: File) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setSourceUrl(URL.createObjectURL(file));
    setSourceName(file.name);
    setCloudSource(null);
    setFaceDetections([]);
    setSelectedFaceIndices([]);
    setManualFacePaths([]);
    if (mode === "edit") {
      const bitmap = await createImageBitmap(file);
      setStudioSettings((current) => ({
        ...current,
        edit: { ...current.edit, width: bitmap.width, height: bitmap.height },
      }));
      bitmap.close();
      setRegions((items) => items.filter((item) => item.layer === "generation"));
      setActiveLayer("targets");
    }
    if (mode === "face") {
      const bitmap = await createImageBitmap(file);
      setFaceDimensions({ width: bitmap.width, height: bitmap.height });
      bitmap.close();
    }
    if (!developmentBackend && workspace.state === "ready") {
      report(`Uploading ${file.name} to persistent Inputs…`);
      try {
        const uploaded = await uploadWorkspaceFile(workspace.id, file, "inputs");
        setCloudSource(uploaded);
        report(`Loaded ${file.name}; remote input is ready.`);
      } catch (caught) {
        report(caught instanceof Error ? `Image loaded locally, but cloud upload failed: ${caught.message}` : "Image loaded locally, but cloud upload failed.", "error");
      }
    } else if (!developmentBackend) {
      report("Image loaded locally. Start the GPU workspace to upload it before a remote edit or face run.");
    }
  }

  function clearImage() {
    if (sourceUrl?.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
    setSourceUrl(null);
    setSourceName("");
    setCloudSource(null);
    setResultUrl(null);
    setResultName("");
    setFaceDetections([]);
    setSelectedFaceIndices([]);
    setManualFacePaths([]);
    report("Canvas image cleared.");
  }

  function resetProject() {
    if (!window.confirm("Start a new project? Unsaved browser changes will be cleared.")) return;
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setMode("generation");
    setActiveLayer("generation");
    setRegions([]);
    setSelectedId(null);
    setDrawMode(null);
    setSourceUrl(null);
    setSourceName("");
    setCloudSource(null);
    setResultUrl(null);
    setResultName("");
    setGlobalPrompts({ generation: "", reference: "", targets: "" });
    setStudioSettings(createStudioSettings());
    setPoseControlCompatibility(null);
    setLoras([]);
    setProjectName("untitled.k2lab.json");
    setFaceDetections([]);
    setSelectedFaceIndices([]);
    setManualFacePaths([]);
    setLassoMode(false);
    report("Started a new project with default settings.");
  }

  async function allFiles(kind: FileKind) {
    const items: FileRecord[] = [];
    let cursor: string | undefined;
    do {
      const page = await controlPlane.files(workspace.id, kind, cursor);
      items.push(...page.items);
      cursor = page.next_cursor ?? undefined;
    } while (cursor);
    return items;
  }

  async function restoreProject(document: unknown, name: string, source?: File) {
    const loaded = loadStudioProjectDocument(document);
    let loraFiles: FileRecord[] = [];
    let poseControlFiles: FileRecord[] = [];
    let upscalerFiles: FileRecord[] = [];
    let diffusionFiles: FileRecord[] = [];
    let textEncoderFiles: FileRecord[] = [];
    let vaeFiles: FileRecord[] = [];
    let faceDetectorFiles: FileRecord[] = [];
    let inputFiles: FileRecord[] = [];
    let outputFiles: FileRecord[] = [];
    try {
      // The Pod inventory is one persistent index. Load its sections sequentially so a
      // project restore cannot create an eight-connection burst through RunPod's proxy.
      loraFiles = await allFiles("loras");
      poseControlFiles = await allFiles("krea_control_loras");
      upscalerFiles = await allFiles("upscale_models");
      diffusionFiles = await allFiles("diffusion_models");
      textEncoderFiles = await allFiles("text_encoders");
      vaeFiles = await allFiles("vae");
      faceDetectorFiles = await allFiles("face_detection");
      inputFiles = await allFiles("inputs");
      outputFiles = await allFiles("outputs");
    } catch {
      // Project restoration remains usable while a stopped workspace inventory is unavailable.
    }
    const byName = (files: FileRecord[], target: string) => files.find(
      (file) => file.display_name.toLocaleLowerCase() === target.toLocaleLowerCase(),
    );
    loaded.loras = bindStudioLoraFiles(loaded.loras, loraFiles);
    const poseControl = byName(
      poseControlFiles,
      loaded.settings.generation.poseControlLoraModel,
    );
    if (poseControl) {
      loaded.settings.generation.poseControlLoraFileId = poseControl.id;
      void inspectPoseControlCheckpoint(poseControl.id, false);
    } else {
      setPoseControlCompatibility(null);
    }
    const depthCheckpoint = byName(
      poseControlFiles,
      loaded.settings.generation.depth.checkpointName,
    );
    if (depthCheckpoint) {
      loaded.settings.generation.depth.checkpointFileId = depthCheckpoint.id;
    }
    const depthImage = byName(
      inputFiles,
      loaded.settings.generation.depth.imageName,
    );
    if (depthImage) loaded.settings.generation.depth.imageFileId = depthImage.id;
    const upscaler = byName(upscalerFiles, loaded.settings.generation.upscaleModelName);
    if (upscaler) loaded.settings.generation.upscaleModelFileId = upscaler.id;
    const runtime = loaded.settings.runtime;
    const diffusion = byName(diffusionFiles, runtime.diffusionModelName);
    const textEncoder = byName(textEncoderFiles, runtime.textEncoderName);
    const vae = byName(vaeFiles, runtime.vaeName);
    const faceDetector = byName(faceDetectorFiles, runtime.faceDetectorName);
    if (diffusion) runtime.diffusionModelFileId = diffusion.id;
    if (textEncoder) runtime.textEncoderFileId = textEncoder.id;
    if (vae) runtime.vaeFileId = vae.id;
    if (faceDetector) runtime.faceDetectorFileId = faceDetector.id;
    setMode("generation");
    setActiveLayer("generation");
    setRegions(loaded.regions);
    setSelectedId(loaded.regions.find((region) => region.layer === "generation")?.id ?? null);
    setGlobalPrompts(loaded.prompts);
    setStudioSettings(loaded.settings);
    setLoras(loaded.loras);
    setResultUrl(null);
    setResultName("");
    const restoredSource = byName([...inputFiles, ...outputFiles], loaded.sourceName);
    setCloudSource(restoredSource ?? null);
    setFaceDetections([]);
    setSelectedFaceIndices([]);
    setManualFacePaths([]);
    setLassoMode(false);
    setFaceDimensions({ width: loaded.settings.generation.width, height: loaded.settings.generation.height });
    if (source) {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
      setSourceUrl(URL.createObjectURL(source));
      setSourceName(source.name);
    } else {
      setSourceUrl(restoredSource ? controlPlane.fileUrl(workspace.id, restoredSource.id) : null);
      setSourceName(restoredSource?.display_name ?? loaded.sourceName);
    }
    const safeName = name.toLocaleLowerCase().endsWith(".json") ? name : `${name}.k2lab.json`;
    setProjectName(safeName);
    const missing = loaded.loras.filter((lora) => !lora.fileId).map((lora) => lora.name);
    const openedMessage = missing.length
      ? `Opened ${name}. Upload or select missing cloud LoRA asset(s): ${missing.join(", ")}.`
      : `Opened ${name}.`;
    report(
      loaded.migrationNotices.length
        ? `${openedMessage} ${loaded.migrationNotices.join(" ")}`
        : openedMessage,
      loaded.migrationNotices.length ? "worker" : undefined,
    );
  }

  async function openProject(file: File) {
    setBusy(true);
    try {
      await restoreProject(JSON.parse(await file.text()), file.name);
    } catch (caught) {
      report(caught instanceof Error ? `Project open failed: ${caught.message}` : "Project open failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function importProjectPng(file: File) {
    setBusy(true);
    try {
      await restoreProject(await projectDocumentFromPng(file), file.name, file);
      setProjectName("untitled.k2lab.json");
      if (!developmentBackend && workspace.state === "ready") {
        const uploaded = await uploadWorkspaceFile(workspace.id, file, "inputs");
        setCloudSource(uploaded);
        report(`Imported project metadata and uploaded ${file.name} for remote use.`);
      } else {
        report(`Imported project metadata from ${file.name}. Start the workspace to upload it for remote use.`);
      }
    } catch (caught) {
      report(caught instanceof Error ? `PNG import failed: ${caught.message}` : "PNG import failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function saveProject(saveAs = false) {
    let name = projectName;
    if (saveAs) {
      const chosen = window.prompt("Project filename", projectName);
      if (!chosen) return;
      name = chosen.toLocaleLowerCase().endsWith(".json") ? chosen : `${chosen}.k2lab.json`;
      setProjectName(name);
    }
    if (!name || name.includes("/") || name.includes("\\") || name === "." || name === "..") {
      report("Project filename must be a filename, not a path.", "error");
      return;
    }
    if (workspace.state !== "ready") {
      report("Start the workspace before saving a project to persistent storage.", "error");
      return;
    }
    const projectDocument = buildProjectDocument(regions, globalPrompts, studioSettings, loras, cloudSource?.display_name ?? null);
    try {
      await controlPlane.saveProject(workspace.id, name, projectDocument);
      report(`Saved ${name} to workspace Assets › Projects. Download it there when needed.`);
    } catch (caught) {
      report(caught instanceof Error ? `Project save failed: ${caught.message}` : "Project save failed.", "error");
    }
  }

  async function openCloudProject(file: FileRecord) {
    setBusy(true);
    try {
      const response = await fetch(controlPlane.fileUrl(workspace.id, file.id));
      if (!response.ok) throw new Error(`Cloud project read failed (${response.status})`);
      await restoreProject(await response.json(), file.display_name);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not open cloud project", "error");
    } finally {
      setBusy(false);
    }
  }

  async function lifecycle(action: "start" | "stop" | "extend") {
    setBusy(true);
    setMessage("");
    try {
      const next = action === "start"
        ? await controlPlane.startWorkspace(workspace.id, startWithoutTimeLimit)
        : action === "stop"
          ? await controlPlane.stopWorkspace(workspace.id)
          : await controlPlane.extendLease(workspace.id);
      onWorkspace(next);
      if (action === "start") setStartWithoutTimeLimit(false);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Workspace action failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function connectMigratedPod() {
    setBusy(true);
    setMessage("");
    try {
      const next = await controlPlane.connectMigratedPod(
        workspace.id,
        connectPodId.trim(),
        connectWithoutTimeLimit,
      );
      onWorkspace(next);
      setShowConnectPod(false);
      setConnectPodId("");
      setConnectWithoutTimeLimit(false);
      report(`Connected migrated RunPod Pod ${next.provider_resource_id}.`);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not connect the migrated Pod", "error");
    } finally {
      setBusy(false);
    }
  }

  async function terminate() {
    setBusy(true);
    setMessage("");
    try {
      await controlPlane.terminateWorkspace(workspace.id, deleteConfirmation);
      onDelete();
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Workspace deletion failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function advanceMigration(initial: WorkspaceMigrationRecord) {
    let next = initial;
    while (["preparing", "copying", "verifying"].includes(next.state)) {
      next = await controlPlane.resumeMigration(workspace.id, next.id);
      setMigration(next);
    }
    if (next.state === "awaiting_confirmation") {
      onWorkspace(await controlPlane.workspace(workspace.id));
      report("Manifest verification succeeded. Test the portable workspace, then explicitly delete the retained original Pod.");
    } else if (next.error_message) {
      report(next.error_message, "error");
    }
  }

  async function beginMigration() {
    setBusy(true);
    setMessage("");
    try {
      const created = await controlPlane.createMigration(workspace.id, {
        network_volume_id: migrationVolumeId || null,
        workspace_disk_gb: migrationVolumeId
          ? networkVolumes.find((item) => item.id === migrationVolumeId)?.size_gb
          : migrationDiskGb,
        datacenter_priority_ids: migrationVolumeId || !migrationDatacenterId
          ? [] : [migrationDatacenterId],
      });
      setMigration(created);
      await advanceMigration(created);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Workspace migration failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function resumeMigration() {
    if (!migration) return;
    setBusy(true);
    try {
      await advanceMigration(migration);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not resume migration", "error");
    } finally {
      setBusy(false);
    }
  }

  async function confirmMigration() {
    if (!migration) return;
    setBusy(true);
    try {
      const completed = await controlPlane.confirmMigration(
        workspace.id, migration.id, migrationConfirmation,
      );
      setMigration(completed);
      setMigrationConfirmation("");
      setShowMigration(false);
      onWorkspace(await controlPlane.workspace(workspace.id));
      report("Migration complete. The original Pod and its regular volume were deleted.");
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not confirm migration", "error");
    } finally {
      setBusy(false);
    }
  }

  async function runRemoteJob() {
    const prefix = studioSettings.runtime.filenamePrefix.trim();
    if (!prefix || prefix.includes("/") || prefix.includes("\\") || prefix === "." || prefix === "..") {
      report("Choose a safe output filename prefix in Setup before running.", "error");
      setUtilityPanel("setup");
      return;
    }
    const missingLoras = loras.filter((lora) => !lora.fileId).map((lora) => lora.name);
    if (missingLoras.length) {
      report(`Bind missing cloud LoRA asset(s) before running: ${missingLoras.join(", ")}.`, "error");
      setAssetPurpose("lora");
      setUtilityPanel("assets");
      return;
    }
    const unresolvedModels = [
      [studioSettings.runtime.diffusionModelName, studioSettings.runtime.diffusionModelFileId],
      [studioSettings.runtime.textEncoderName, studioSettings.runtime.textEncoderFileId],
      [studioSettings.runtime.vaeName, studioSettings.runtime.vaeFileId],
      ...(mode === "face"
        ? [[studioSettings.runtime.faceDetectorName, studioSettings.runtime.faceDetectorFileId]]
        : []),
    ].filter(([name, id]) => name && !id).map(([name]) => name);
    if (unresolvedModels.length) {
      report(`Resolve missing model selection(s) in Setup: ${unresolvedModels.join(", ")}.`, "error");
      setUtilityPanel("setup");
      return;
    }
    if (mode !== "generation" && !cloudSource) {
      report("Choose an uploaded input or prior output from Cloud files first.", "error");
      setUtilityPanel("assets");
      return;
    }
    if (mode === "face" && !cloudSource?.display_name.toLocaleLowerCase().endsWith(".png")) {
      report("Face refinement requires a PNG source image.", "error");
      return;
    }
    if (mode === "face" && selectedFaceIndices.length === 0) {
      report("Detect faces or draw lassos, then select at least one face to refine.", "error");
      return;
    }
    if (mode === "face" && !loras.some((lora) => lora.active && lora.strength !== 0 && !lora.generation.global && lora.generation.regionIds.length > 0)) {
      report("Assign at least one enabled LoRA to a subject region before face refinement.", "error");
      return;
    }
    if (
      mode === "generation"
      && studioSettings.generation.poseGating
      && studioSettings.generation.poseHardGateSteps + studioSettings.generation.poseSoftGateSteps > 0
      && !regions.some((region) => (
        region.layer === "generation"
        && region.enabled
        && region.regionType === "subject"
        && region.pose?.enabled
      ))
    ) {
      report("Add and enable at least one subject mannequin before using volumetric pose gating.", "error");
      return;
    }
    if (mode === "generation" && studioSettings.generation.poseControlLoraEnabled) {
      if (!poseControlLoraAvailable) {
        report("This control plane or workspace protocol does not support Krea volumetric pose adapters. Update the workspace image; the adapter will not be silently disabled.", "error");
        return;
      }
      if (!studioSettings.generation.poseControlLoraFileId) {
        report("Select a verified Krea volumetric pose adapter checkpoint before generating.", "error");
        setAssetPurpose("pose-control");
        setUtilityPanel("assets");
        return;
      }
      if (!poseControlCompatibility?.compatible) {
        report("The selected Krea pose adapter is not verified as compatible. Review its checkpoint diagnostics before generating.", "error");
        return;
      }
      if (
        !poseControlCompatibility.verified
        && !studioSettings.generation.poseControlLegacyAcknowledged
      ) {
        report("Explicitly acknowledge the compatible unverified legacy checkpoint before generating.", "error");
        return;
      }
      if (!regions.some((region) => (
        region.layer === "generation"
        && region.enabled
        && region.regionType === "subject"
        && region.pose?.enabled
      ))) {
        report("Add and enable at least one subject mannequin before using the trained pose adapter.", "error");
        return;
      }
    }
    if (mode === "generation" && studioSettings.generation.depth.enabled) {
      if (!depthControlAvailable) {
        report("Depth control is disabled by this deployment's feature flags.", "error");
        return;
      }
      if (
        !depthRegionsAvailable
        && regions.some((region) => (
          region.layer === "generation"
          && (region.depthMode ?? "inherit") !== "inherit"
        ))
      ) {
        report("Regional depth weighting is disabled by this deployment's feature flags.", "error");
        return;
      }
      if (studioSettings.generation.poseControlLoraEnabled) {
        report("Depth control and volumetric pose control cannot be enabled in the same run.", "error");
        return;
      }
      if (!studioSettings.generation.depth.checkpointFileId) {
        report("Select the verified Krea depth adapter checkpoint before generating.", "error");
        setAssetPurpose("depth-checkpoint");
        setUtilityPanel("assets");
        return;
      }
      if (!studioSettings.generation.depth.imageFileId) {
        report("Select a grayscale PNG or TIFF depth image before generating.", "error");
        setAssetPurpose("depth-image");
        setUtilityPanel("assets");
        return;
      }
    }
    if (
      mode === "generation"
      && studioSettings.generation.poseGating
      && studioSettings.generation.poseSemanticMode === "prediction_composite"
    ) {
      if (!poseSemanticRoutingAvailable) {
        report("This control plane or workspace protocol does not support subject-semantic pose routing. Update the workspace image; the mode will not be changed automatically.", "error");
        return;
      }
      const posedSubjects = regions.filter((region) => (
        region.layer === "generation"
        && region.enabled
        && region.regionType === "subject"
        && region.pose?.enabled
      ));
      const missingDescription = posedSubjects.find((region) => (
        !region.prompt.trim() && !region.faceIdentityPrompt.trim()
      ));
      if (missingDescription) {
        report(`Add a subject or face-identity prompt for ${missingDescription.name} before using Prediction composite.`, "error");
        return;
      }
    }
    setBusy(true);
    setMessage("");
    eventCursor.current = undefined;
    try {
      await controlPlane.previewUnifiedPrompt(
        buildProjectDocument(regions, globalPrompts, studioSettings, loras, cloudSource?.display_name ?? null),
      );
      const kind: JobKind = mode === "generation" ? "generate" : mode === "edit" ? "edit_image" : "refine_faces";
      const runCount = mode === "generation" && studioSettings.generation.batchMode
        ? studioSettings.generation.batchCount : 1;
      const payloads: JobSubmitPayload[] = [];
      let lastSeed = studioSettings.generation.seed;
      for (let index = 0; index < runCount; index += 1) {
        let seed = studioSettings.generation.seed;
        if (mode === "generation" && studioSettings.generation.seedMode === "random") {
          seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
        } else if (mode === "generation" && studioSettings.generation.seedMode === "increment") {
          seed = (studioSettings.generation.seed + index) % 2147483648;
        }
        lastSeed = seed;
        const jobSettings = mode === "generation"
          ? { ...studioSettings, generation: { ...studioSettings.generation, seed } }
          : studioSettings;
        payloads.push({
          command_id: crypto.randomUUID(),
          kind,
          project_id: `studio-${workspace.id}`,
          project: buildProjectDocument(regions, globalPrompts, jobSettings, loras, cloudSource?.display_name ?? null),
          input_file_id: cloudSource?.id,
          diffusion_model_file_id: studioSettings.runtime.diffusionModelFileId || undefined,
          text_encoder_file_id: studioSettings.runtime.textEncoderFileId || undefined,
          vae_file_id: studioSettings.runtime.vaeFileId || undefined,
          face_detector_file_id: mode === "face"
            ? studioSettings.runtime.faceDetectorFileId || undefined
            : undefined,
          lora_file_ids: loras.map((lora) => lora.fileId),
          pose_control_lora_file_id: studioSettings.generation.poseControlLoraEnabled
            ? studioSettings.generation.poseControlLoraFileId
            : undefined,
          pose_control_allow_unverified_legacy: (
            studioSettings.generation.poseControlLoraEnabled
            && studioSettings.generation.poseControlLegacyAcknowledged
          ),
          depth_checkpoint_file_id: studioSettings.generation.depth.enabled
            ? studioSettings.generation.depth.checkpointFileId
            : undefined,
          depth_image_file_id: studioSettings.generation.depth.enabled
            ? studioSettings.generation.depth.imageFileId
            : undefined,
          upscale_model_file_id: studioSettings.generation.upscaleModelFileId || undefined,
          filename_prefix: studioSettings.runtime.filenamePrefix,
          selected_face_indices: mode === "face" ? selectedFaceIndices : undefined,
          manual_face_paths: mode === "face" ? manualFacePaths : undefined,
        });
      }
      const submitted: GenerationJob[] = [];
      for (let index = 0; index < payloads.length; index += 1) {
        try {
          submitted.push(await controlPlane.submitJob(workspace.id, payloads[index]));
        } catch (caught) {
          if (!isAmbiguousJobSubmissionError(caught)) throw caught;
          const lastError = caught instanceof Error
            ? caught.message
            : "The job submission did not receive an acknowledgment.";
          const recovery: PendingJobSubmission = {
            version: 1,
            workspaceId: workspace.id,
            payloads: payloads.slice(index),
            acknowledgedJobs: submitted,
            createdAt: new Date().toISOString(),
            lastError,
          };
          rememberSubmissionRecovery(recovery);
          trackSubmittedJobs(submitted);
          report(
            "Job submission acknowledgment timed out. The request may already be queued; use Recover submission instead of starting another job.",
            "error",
          );
          return;
        }
      }
      if (mode === "generation") {
        const nextSeed = studioSettings.generation.seedMode === "increment"
          ? (studioSettings.generation.seed + runCount) % 2147483648 : lastSeed;
        setStudioSettings({ ...studioSettings, generation: { ...studioSettings.generation, seed: nextSeed } });
      }
      trackSubmittedJobs(submitted);
      report(runCount > 1 ? `${runCount} remote batch runs queued.` : "Remote job queued.", "worker");
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not submit remote job", "error");
    } finally {
      setBusy(false);
    }
  }

  async function recoverTimedOutSubmission() {
    if (!submissionRecovery) return;
    setBusy(true);
    const recovered = [...submissionRecovery.acknowledgedJobs];
    try {
      for (let index = 0; index < submissionRecovery.payloads.length; index += 1) {
        const payload = submissionRecovery.payloads[index];
        try {
          recovered.push(await controlPlane.submitJob(workspace.id, payload));
        } catch (caught) {
          const lastError = caught instanceof Error
            ? caught.message
            : "The recovery request did not receive an acknowledgment.";
          const next: PendingJobSubmission = {
            ...submissionRecovery,
            payloads: submissionRecovery.payloads.slice(index),
            acknowledgedJobs: uniqueJobs(recovered),
            lastError,
          };
          rememberSubmissionRecovery(next);
          trackSubmittedJobs(recovered);
          report(
            isAmbiguousJobSubmissionError(caught)
              ? "Recovery still did not receive a job receipt. The same command ID remains saved; retry recovery when the agent responds."
              : `The agent rejected submission recovery: ${lastError}`,
            "error",
          );
          return;
        }
      }
      const jobs = uniqueJobs(recovered);
      rememberSubmissionRecovery(null);
      trackSubmittedJobs(jobs);
      eventCursor.current = undefined;
      report(
        jobs.length > 1
          ? `Recovered ${jobs.length} remote batch jobs without duplicating work.`
          : "Recovered the remote job without duplicating work.",
        "worker",
      );
    } finally {
      setBusy(false);
    }
  }

  async function cancelRemoteJob() {
    if (!job) return;
    setBusy(true);
    try {
      const cancelled = await controlPlane.cancelJob(workspace.id, job.id);
      for (const queued of queuedJobs) {
        await controlPlane.cancelJob(workspace.id, queued.id);
      }
      setJob(cancelled);
      setQueuedJobs([]);
      report("Remote job queue cancelled; worker memory was released.", "worker");
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not cancel remote job", "error");
    } finally {
      setBusy(false);
    }
  }

  async function releaseWorkerMemory() {
    setBusy(true);
    try {
      const released = await controlPlane.releaseWorkerMemory(workspace.id);
      rememberSubmissionRecovery(null);
      setJob(null);
      setQueuedJobs([]);
      eventCursor.current = undefined;
      report(released.cancelled_job_ids.length
        ? `Worker memory released; ${released.cancelled_job_ids.length} active job(s) cancelled.`
        : "Worker memory released. No active jobs were cancelled.", "worker");
      try {
        setWorkerMemory(await controlPlane.workerMemory(workspace.id));
      } catch {
        setWorkerMemory(null);
      }
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not release worker memory", "error");
    } finally {
      setBusy(false);
    }
  }

  async function refreshWorkerMemory() {
    setMemoryRefreshing(true);
    try {
      setWorkerMemory(await controlPlane.workerMemory(workspace.id));
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not refresh Pod RAM status", "error");
    } finally {
      setMemoryRefreshing(false);
    }
  }

  async function previewUnifiedPrompt() {
    setBusy(true);
    setMessage("");
    try {
      setPromptPreview(await controlPlane.previewUnifiedPrompt(
        buildProjectDocument(regions, globalPrompts, studioSettings, loras, cloudSource?.display_name ?? null),
      ));
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not compile the unified prompt", "error");
    } finally {
      setBusy(false);
    }
  }

  async function previewPoseControl(subjectRegionId: string | null = null) {
    setBusy(true);
    setMessage("");
    try {
      const preview = await controlPlane.previewVolumetricControl(
        buildProjectDocument(
          regions,
          globalPrompts,
          studioSettings,
          loras,
          cloudSource?.display_name ?? null,
        ),
        subjectRegionId,
      );
      if (controlPreview) URL.revokeObjectURL(controlPreview.url);
      setControlPreview({
        ...preview,
        url: URL.createObjectURL(preview.blob),
        subjectRegionId,
      });
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not render the control preview", "error");
    } finally {
      setBusy(false);
    }
  }

  async function inspectPoseControlCheckpoint(fileId: string, allowLegacy: boolean) {
    setBusy(true);
    try {
      setPoseControlCompatibility(
        await controlPlane.inspectKreaControlCheckpoint(
          workspace.id,
          fileId,
          allowLegacy,
        ),
      );
    } catch (caught) {
      setPoseControlCompatibility(null);
      report(caught instanceof Error ? caught.message : "Could not inspect the Krea pose adapter", "error");
    } finally {
      setBusy(false);
    }
  }

  function closeControlPreview() {
    if (controlPreview) URL.revokeObjectURL(controlPreview.url);
    setControlPreview(null);
  }

  async function runFaceDetection(faceDetectorFileId: string) {
    if (!cloudSource) return;
    setBusy(true);
    report("Detecting faces in the isolated worker…", "worker");
    try {
      const result = await controlPlane.detectFaces(workspace.id, {
        input_file_id: cloudSource.id,
        face_detector_file_id: faceDetectorFileId,
        threshold: studioSettings.face.detectorThreshold,
        provider: studioSettings.face.detectorProvider,
      });
      setFaceDimensions({ width: result.width, height: result.height });
      setFaceDetections(result.faces);
      setSelectedFaceIndices(result.faces.map((face) => face.index));
      setManualFacePaths([]);
      setLassoMode(false);
      report(`Detected ${result.faces.length} face(s) with ${result.execution_provider}.`, "worker");
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Face detection failed", "error");
    } finally {
      setBusy(false);
    }
  }

  async function detectFaces() {
    if (!cloudSource) {
      report("Choose an uploaded input or prior output before detecting faces.", "error");
      setAssetPurpose("source");
      setUtilityPanel("assets");
      return;
    }
    if (studioSettings.runtime.faceDetectorFileId) {
      await runFaceDetection(studioSettings.runtime.faceDetectorFileId);
      return;
    }
    setBusy(true);
    try {
      const installed = await allFiles("face_detection");
      const detector = installed[0];
      if (!detector) {
        setFaceDetectorInstallError("");
        setFaceDetectorTransfer(null);
        setShowFaceDetectorInstall(true);
        return;
      }
      setStudioSettings((current) => ({
        ...current,
        runtime: {
          ...current.runtime,
          faceDetectorFileId: detector.id,
          faceDetectorName: detector.display_name,
        },
      }));
      await runFaceDetection(detector.id);
    } catch (caught) {
      report(caught instanceof Error ? caught.message : "Could not inspect installed face detectors", "error");
    } finally {
      setBusy(false);
    }
  }

  async function installFaceDetector() {
    setFaceDetectorInstalling(true);
    setFaceDetectorInstallError("");
    try {
      report("Inspecting the pinned FantasyPortrait face detector…");
      await controlPlane.previewHuggingFace(workspace.id, FACE_DETECTOR_SOURCE, []);
      let transfer = await controlPlane.startHuggingFace(workspace.id, {
        source_url: FACE_DETECTOR_SOURCE,
        destination_kind: "face_detection",
        allow_patterns: [],
        allow_unsafe_format: false,
      });
      setFaceDetectorTransfer(transfer);
      report("Downloading face_det.onnx into persistent workspace storage…");
      for (let attempt = 0; attempt < 600 && !["completed", "failed", "cancelled"].includes(transfer.state); attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        transfer = await controlPlane.transfer(workspace.id, transfer.id);
        setFaceDetectorTransfer(transfer);
      }
      if (transfer.state !== "completed") {
        throw new Error(transfer.error_message ?? `Face detector installation ${transfer.state}.`);
      }
      const detector = transfer.files.find((file) => file.kind === "face_detection");
      if (!detector) throw new Error("The face detector download completed without an installed ONNX file.");
      setStudioSettings((current) => ({
        ...current,
        runtime: {
          ...current.runtime,
          faceDetectorFileId: detector.id,
          faceDetectorName: detector.display_name,
        },
      }));
      setShowFaceDetectorInstall(false);
      report(`Installed and selected ${detector.display_name}.`, "worker");
      await runFaceDetection(detector.id);
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : "Face detector installation failed";
      setFaceDetectorInstallError(detail);
      report(detail, "error");
    } finally {
      setFaceDetectorInstalling(false);
    }
  }

  function toggleFace(index: number) {
    setSelectedFaceIndices((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current, index].sort((left, right) => left - right));
  }

  function addManualFacePath(path: number[][]) {
    const paths = [...manualFacePaths, path];
    const faces = paths.map((points, index) => {
      const xs = points.map((point) => point[0]);
      const ys = points.map((point) => point[1]);
      return {
        index,
        box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] as [number, number, number, number],
        score: 1,
      };
    });
    setManualFacePaths(paths);
    setFaceDetections(faces);
    setSelectedFaceIndices(faces.map((face) => face.index));
  }

  function useLatestFaceSource() {
    if (!latestOutputFileId) {
      report("No completed first-pass output is available in this browser session.", "error");
      return;
    }
    const source: FileRecord = {
      id: latestOutputFileId,
      kind: "outputs",
      display_name: "Latest first pass",
      size_bytes: 0,
      sha256: "",
      modified_at: new Date().toISOString(),
    };
    setCloudSource(source);
    setSourceName(source.display_name);
    setSourceUrl(controlPlane.outputUrl(workspace.id, source.id));
    setFaceDimensions({ width: studioSettings.generation.width, height: studioSettings.generation.height });
    setFaceDetections([]);
    setSelectedFaceIndices([]);
    setManualFacePaths([]);
    setMode("face");
    report("Using the latest completed output for face refinement. Detect faces next.");
  }

  useEffect(() => {
    function projectShortcut(event: KeyboardEvent) {
      if (!(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLocaleLowerCase();
      if (key === "n" && !event.shiftKey) {
        event.preventDefault();
        resetProject();
      } else if (key === "o" && event.shiftKey) {
        event.preventDefault();
        importPngInput.current?.click();
      } else if (key === "o") {
        event.preventDefault();
        openProjectInput.current?.click();
      } else if (key === "s") {
        event.preventDefault();
        void saveProject(event.shiftKey);
      }
    }
    window.addEventListener("keydown", projectShortcut);
    return () => window.removeEventListener("keydown", projectShortcut);
  });

  function resizeEventDock(event: React.PointerEvent<HTMLDivElement>) {
    const resize = eventResize.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    const maximum = Math.max(120, Math.floor(window.innerHeight * 0.55));
    setEventDockHeight(Math.max(82, Math.min(maximum, resize.startHeight + resize.startY - event.clientY)));
  }

  function finishEventDockResize(event: React.PointerEvent<HTMLDivElement>) {
    if (eventResize.current?.pointerId !== event.pointerId) return;
    eventResize.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div
      className={`studio-shell ${eventDockOpen ? "with-event-dock" : ""}`}
      style={{ "--event-dock-height": `${eventDockHeight}px` } as CSSProperties}
    >
      <header className="studio-topbar">
        <div className="brand-lockup"><span className="brand-mark">K2</span><span><strong>Region Lab</strong><small>Cloud studio</small></span></div>
        <div className="project-actions">
          <button onClick={resetProject}>New</button>
          <button onClick={() => openProjectInput.current?.click()}>Open</button>
          <button onClick={() => importPngInput.current?.click()}>Import PNG</button>
          <button onClick={() => void saveProject(false)}>Save</button>
          <button onClick={() => void saveProject(true)}>Save as</button>
          <input ref={openProjectInput} type="file" hidden accept=".json,.k2lab.json,application/json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void openProject(file); event.target.value = ""; }} />
          <input ref={importPngInput} type="file" hidden accept="image/png" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importProjectPng(file); event.target.value = ""; }} />
        </div>
        <div className="workspace-status">
          {developmentBackend && <span className="preview-chip">Preview backend</span>}
          <button className="workspace-chip" onClick={() => setShowCloud(!showCloud)}>
            <span className={`status-dot ${activeCompute ? "online" : "stopped"}`} />
            <span><strong>{workspace.name}</strong><small>{workspace.gpu.display_name} · {workspace.state}</small></span>
            <Icon name="chevronDown" />
          </button>
          {activeCompute && <button className="stop-gpu" disabled={busy || workspace.state === "stopping"} onClick={() => lifecycle("stop")}><Icon name="stop" /> Stop GPU now</button>}
        </div>
      </header>

      {showCloud && (
        <div className="cloud-popover glass-card">
          <div className="cloud-popover-head"><div><p className="kicker">Cloud workspace</p><h3>{workspace.name}</h3></div><span className={`state-badge ${workspace.state}`}>{workspace.state}</span></div>
          <dl className="summary-list compact-summary">
            <div><dt>Compute now</dt><dd>{activeCompute ? `$${workspace.estimated_compute_per_hour.toFixed(2)}/hr` : "$0.00/hr"}</dd></div>
            <div><dt>Storage</dt><dd>${workspace.estimated_storage_per_month.toFixed(2)}/mo</dd></div>
            <div><dt>Lease</dt><dd>{activeCompute ? workspace.lease_unlimited ? "No time limit" : `${leaseMinutes} min remaining` : "No active lease"}</dd></div>
          </dl>
          <div className="readiness-grid">{readiness.map(([name, ready]) => <span key={name} className={ready ? "ready" : "pending"}><Icon name={ready ? "check" : "clock"} /> {name}</span>)}</div>
          {workspace.provider_resource_id && !developmentBackend && (
            <div className="runpod-progress">
              <div><strong>Docker image startup</strong><small>Pod {workspace.provider_resource_id}</small></div>
              <a className="quiet-button" href="https://console.runpod.io/pods" target="_blank" rel="noreferrer">
                <Icon name="cloud" /> {workspace.state === "starting" ? "View Docker progress" : "Open in RunPod"}
              </a>
            </div>
          )}
          {workspace.error_message && <div className="error-banner">{workspace.error_message}</div>}
          {canStart && (
            <label className="check-row warning-check lease-limit-option">
              <input type="checkbox" checked={startWithoutTimeLimit} onChange={(event) => setStartWithoutTimeLimit(event.target.checked)} />
              <span><strong>No time limit</strong><small>The Pod will continue running and billing until you manually stop it.</small></span>
            </label>
          )}
          <div className="popover-actions">
            <button
              className="quiet-button"
              onClick={() => {
                setShowCloud(false);
                onWorkspaceMenu();
              }}
            >
              <Icon name="layers" /> All workspaces
            </button>
            {canExtend
              ? <button className="quiet-button" onClick={() => lifecycle("extend")}>Extend session</button>
              : canStart
                ? <button className="primary-button" onClick={() => lifecycle("start")}>{startWithoutTimeLimit ? "Start GPU without time limit" : "Start GPU"}</button>
                : null}
            {workspace.mode === "persistent_pod" && (
              <button className="quiet-button" onClick={() => setShowMigration(true)}>Migrate to portable storage</button>
            )}
            {canStart && workspace.mode === "persistent_pod" && !developmentBackend && (
              <button className="quiet-button" onClick={() => { setMessage(""); setShowCloud(false); setShowConnectPod(true); }}>Connect migrated Pod</button>
            )}
            {workspace.retained_original_provider_resource_id && (
              <button className="quiet-button" onClick={() => setShowMigration(true)}>Confirm verified migration</button>
            )}
            <button className="danger-text-button" disabled={Boolean(workspace.retained_original_provider_resource_id)} title={workspace.retained_original_provider_resource_id ? "Confirm the verified migration first" : undefined} onClick={() => setShowDelete(true)}>Delete cloud workspace</button>
          </div>
          <button className="quiet-button full-button" disabled={busy || !running || developmentBackend} onClick={() => void releaseWorkerMemory()}><Icon name="stop" /> Release worker memory</button>
          <p className="field-help">{workspace.mode === "portable_workspace"
            ? "Stopping terminates the Pod and retains the network volume. Deleting this workspace also retains that volume for safety."
            : "Stopping retains the attached volume. Deleting permanently removes it."}</p>
        </div>
      )}

      <aside className="studio-rail">
        <div className="mode-rail">
          <RailButton icon="spark" label="Generate" active={mode === "generation"} onClick={() => switchMode("generation")} />
          <RailButton icon="edit" label="Edit" active={mode === "edit"} onClick={() => switchMode("edit")} />
          <RailButton icon="face" label="Faces" active={mode === "face"} onClick={() => switchMode("face")} />
        </div>
        <div className="utility-rail">
          <RailButton icon="folder" label={pendingUploadCount ? `Assets · ${pendingUploadCount}` : "Assets"} active={showAssets} onClick={() => { setAssetPurpose("source"); setUtilityPanel("assets"); }} />
          <RailButton icon="transfer" label="Transfers" active={showTransfers} onClick={() => setUtilityPanel("transfers")} />
          <RailButton icon="events" label="Events" active={eventDockOpen} onClick={() => {
            setUtilityPanel(null);
            setEventDockOpen((current) => !current);
          }} />
          <RailButton icon="settings" label="Setup" active={showSetup} onClick={() => setUtilityPanel("setup")} />
        </div>
      </aside>

      <main className="studio-main">
        <div className="mode-context-bar">
          <div><p className="kicker">Workspace</p><h1>{mode === "generation" ? "Image generation" : mode === "edit" ? "Image editing" : "Face refinement"}</h1></div>
          {mode === "edit" && (
            <div className="layer-switcher">
              <button className={activeLayer === "reference" ? "active" : ""} onClick={() => { setActiveLayer("reference"); setSelectedId(null); }}><Icon name="layers" /> Reference layer</button>
              <button className={activeLayer === "targets" ? "active" : ""} onClick={() => { setActiveLayer("targets"); setSelectedId(null); }}><Icon name="edit" /> Edit targets</button>
            </div>
          )}
        </div>
        <div className="workspace-grid">
          <RegionCanvas
            mode={mode}
            activeLayer={activeLayer}
            sourceUrl={sourceUrl}
            sourceName={sourceName}
            resultUrl={resultUrl}
            resultName={resultName}
            regions={regions}
            selectedId={selectedId}
            drawMode={drawMode}
            comparePosition={comparePosition}
            canvasWidth={mode === "edit" ? studioSettings.edit.width : mode === "face" ? faceDimensions.width : studioSettings.generation.width}
            canvasHeight={mode === "edit" ? studioSettings.edit.height : mode === "face" ? faceDimensions.height : studioSettings.generation.height}
            faces={faceDetections}
            selectedFaceIndices={selectedFaceIndices}
            manualFacePaths={manualFacePaths}
            lassoMode={lassoMode}
            onComparePosition={setComparePosition}
            onSelect={setSelectedId}
            onRegions={setRegions}
            onDrawMode={setDrawMode}
            onLoadImage={(file) => void loadImage(file)}
            onClearImage={clearImage}
            onToggleFace={toggleFace}
            onAddManualFacePath={addManualFacePath}
          />
          <Inspector
            mode={mode}
            activeLayer={activeLayer}
            regions={regions}
            selectedId={selectedId}
            globalPrompt={globalPrompts[activeLayer]}
            settings={studioSettings}
            loras={loras}
            onGlobalPrompt={(value) => setGlobalPrompts({ ...globalPrompts, [activeLayer]: value })}
            onSettings={setStudioSettings}
            onLoras={setLoras}
            onChooseLora={() => { setAssetPurpose("lora"); setUtilityPanel("assets"); }}
            onChoosePoseControlLora={() => { setAssetPurpose("pose-control"); setUtilityPanel("assets"); }}
            onChooseDepthCheckpoint={() => { setAssetPurpose("depth-checkpoint"); setUtilityPanel("assets"); }}
            onChooseDepthImage={() => { setAssetPurpose("depth-image"); setUtilityPanel("assets"); }}
            onPreviewPoseControl={() => void previewPoseControl()}
            poseControlLoraAvailable={poseControlLoraAvailable}
            depthControlAvailable={depthControlAvailable}
            depthRegionsAvailable={depthRegionsAvailable}
            poseControlCompatibility={poseControlCompatibility}
            onInspectPoseControlLegacy={(allow) => {
              const fileId = studioSettings.generation.poseControlLoraFileId;
              if (fileId) void inspectPoseControlCheckpoint(fileId, allow);
            }}
            onChooseUpscaleModel={() => { setAssetPurpose("upscale"); setUtilityPanel("assets"); }}
            onPreviewUnifiedPrompt={() => void previewUnifiedPrompt()}
            faces={faceDetections}
            selectedFaceIndices={selectedFaceIndices}
            manualFacePaths={manualFacePaths}
            lassoMode={lassoMode}
            onDetectFaces={() => void detectFaces()}
            onToggleFace={toggleFace}
            onSelectAllFaces={(selected) => setSelectedFaceIndices(selected ? faceDetections.map((face) => face.index) : [])}
            onLassoMode={setLassoMode}
            onUndoLasso={() => {
              const paths = manualFacePaths.slice(0, -1);
              setManualFacePaths(paths);
              setFaceDetections((current) => current.slice(0, paths.length));
              setSelectedFaceIndices(paths.map((_path, index) => index));
            }}
            onClearLassos={() => { setManualFacePaths([]); setFaceDetections([]); setSelectedFaceIndices([]); }}
            onUseLatestFaceSource={useLatestFaceSource}
            onRegions={setRegions}
            onSelect={setSelectedId}
            workerMemory={workerMemory}
            memoryRefreshing={memoryRefreshing}
            memoryActionsDisabled={busy || !running || developmentBackend}
            onRefreshMemory={() => void refreshWorkerMemory()}
            onReleaseMemory={() => void releaseWorkerMemory()}
          />
        </div>
      </main>

      {eventDockOpen && (
        <section className="event-dock" aria-label="Studio event log">
          <div
            className="event-dock-resize"
            role="separator"
            aria-label="Resize event log"
            aria-orientation="horizontal"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              eventResize.current = {
                pointerId: event.pointerId,
                startY: event.clientY,
                startHeight: eventDockHeight,
              };
            }}
            onPointerMove={resizeEventDock}
            onPointerUp={finishEventDockResize}
            onPointerCancel={finishEventDockResize}
          ><span /></div>
          <div className="event-dock-toolbar">
            <div><strong>Event history</strong><small>{eventLog.length} / {EVENT_LOG_LIMIT}</small></div>
            <span>Drag the handle to show more or fewer lines</span>
            <button className="quiet-button" disabled={eventLog.length === 0} onClick={() => setEventLog([])}>Clear</button>
            <button className="quiet-button" disabled={busy || !running || developmentBackend} onClick={() => void releaseWorkerMemory()}>Release memory</button>
            <button className="quiet-button" onClick={() => setEventDockOpen(false)}>Hide</button>
          </div>
          <div ref={eventListRef} className="event-list event-dock-list" role="log" aria-live="polite">
            {eventLog.length === 0
              ? <p className="field-help">Events from generation, transfers, saves, and face tools will appear here.</p>
              : eventLog.map((entry) => <article key={entry.id} className={`event-entry ${entry.kind}`}><time>{formatEventTime(entry.createdAt)}</time><span>{entry.source ?? entry.kind}</span><p>{entry.message}</p></article>)}
          </div>
        </section>
      )}

      <footer className="action-bar">
        <div className="action-status"><span className={`status-dot ${activeCompute ? "online" : "stopped"}`} /><span><strong>{job && !["completed", "failed", "cancelled"].includes(job.state) ? `Remote job ${job.state}` : running ? "Workspace ready" : activeCompute ? `Workspace ${workspace.state}` : "GPU stopped"}</strong><small>{workspace.provider_freshness?.stale ? "RunPod provider status stale · runtime and outputs use last known status" : message || workspace.error_message || (developmentBackend ? "Interface preview · remote jobs are disabled" : cloudSource ? `Cloud source: ${cloudSource.display_name}` : "Ready")}</small></span></div>
        <div className="memory-meter"><span>Job</span><div><i style={{ width: job?.progress_total ? `${Math.min(100, job.progress_current / job.progress_total * 100)}%` : "0%" }} /></div><small>{job?.progress_total ? `${job.progress_current}/${job.progress_total}` : running ? "Idle" : "Released"}</small></div>
        <button className="run-button" disabled={!running || developmentBackend || busy} title={developmentBackend ? "Remote generation jobs are disabled in preview mode" : undefined} onClick={() => void (job && !["completed", "failed", "cancelled"].includes(job.state) ? cancelRemoteJob() : runRemoteJob())}>
          <Icon name={job && !["completed", "failed", "cancelled"].includes(job.state) ? "stop" : mode === "face" ? "face" : mode === "edit" ? "wand" : "play"} />
          {job && !["completed", "failed", "cancelled"].includes(job.state) ? "Cancel remote job" : mode === "generation" ? "Generate image" : mode === "edit" ? "Run image edit" : "Refine faces"}
        </button>
      </footer>

      {submissionRecovery && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal submission-recovery-modal" role="alertdialog" aria-modal="true" aria-labelledby="submission-recovery-title" aria-describedby="submission-recovery-description">
            <div className="danger-icon submission-recovery-icon"><Icon name="cloud" /></div>
            <p className="kicker">Safe timeout recovery</p>
            <h2 id="submission-recovery-title">Job receipt was not received</h2>
            <p id="submission-recovery-description">The Pod may already have accepted this request. Recover submission resends the exact same command ID, so the existing job is returned without creating a duplicate. This recovery remains available after a page refresh.</p>
            <div className="submission-command-id"><span>Command ID</span><code>{submissionRecovery.payloads[0].command_id}</code></div>
            <p className="field-help">{submissionRecovery.lastError}</p>
            <p className="field-help">Cancel all remote work stops running and queued jobs and unloads the resident model. Workspace models, projects, inputs, and outputs are retained.</p>
            <div className="modal-actions submission-recovery-actions">
              <button className="danger-button" disabled={busy} onClick={() => void releaseWorkerMemory()}>Cancel all remote work</button>
              <button className="primary-button" disabled={busy} onClick={() => void recoverTimedOutSubmission()}>{busy ? "Recovering…" : "Recover submission"}</button>
            </div>
          </section>
        </div>
      )}

      {showDelete && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <div className="danger-icon"><Icon name="trash" /></div>
            <p className="kicker">Permanent action</p>
            <h2 id="delete-title">Delete cloud workspace?</h2>
            <p>{workspace.mode === "portable_workspace"
              ? "This removes the workspace and any active ephemeral Pod. The network volume is retained to prevent accidental data loss and continues to incur storage cost."
              : "This removes the Pod and its regular persistent volume. Models, projects, inputs, and outputs on that volume cannot be recovered."}</p>
            <label className="field-label" htmlFor="delete-confirmation">Type <strong>{workspace.name}</strong> to confirm</label>
            <input id="delete-confirmation" className="text-input" value={deleteConfirmation} onChange={(event) => setDeleteConfirmation(event.target.value)} />
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions"><button className="quiet-button" onClick={() => { setShowDelete(false); setDeleteConfirmation(""); }}>Cancel</button><button className="danger-button" disabled={busy || deleteConfirmation !== workspace.name} onClick={terminate}>{workspace.mode === "portable_workspace" ? "Delete workspace; retain volume" : "Delete workspace and files"}</button></div>
          </section>
        </div>
      )}
      {promptPreview && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal prompt-preview-modal" role="dialog" aria-modal="true" aria-labelledby="prompt-preview-title">
            <p className="kicker">Canonical compiler output</p>
            <h2 id="prompt-preview-title">Conditioning prompts</h2>
            <p>{promptPreview.regions.length} regional clause{promptPreview.regions.length === 1 ? "" : "s"} in front-to-back order. Pixel boxes are applied separately as a hidden soft attention grid.</p>
            <div className="conditioning-preview-list">
              {(promptPreview.conditioning_prompts.length
                ? promptPreview.conditioning_prompts
                : [{ kind: "full" as const, region_id: null, region_name: "Full scene", prompt: promptPreview.prompt }]
              ).map((item) => <section key={item.region_id ?? "full"}>
                <div className="section-inline-title">
                  <strong>{item.region_name}</strong>
                  {item.text_token_count != null && <span>{item.text_token_count} tokens</span>}
                </div>
                <textarea className="prompt-area prompt-preview-text" readOnly value={item.prompt} />
              </section>)}
            </div>
            <div className="preview-region-order">
              {promptPreview.regions.map((region, index) => <div key={region.id}><strong>{index + 1}. {region.name}</strong><span>{region.spatial_role}</span></div>)}
            </div>
            <div className="modal-actions"><button className="primary-button" onClick={() => setPromptPreview(null)}>Close</button></div>
          </section>
        </div>
      )}
      {controlPreview && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal prompt-preview-modal" role="dialog" aria-modal="true" aria-labelledby="control-preview-title">
            <p className="kicker">Canonical backend raster</p>
            <h2 id="control-preview-title">Volumetric pose control</h2>
            <div className="inline-actions">
              <button
                className={`tiny-button ${controlPreview.subjectRegionId === null ? "active" : ""}`}
                onClick={() => void previewPoseControl(null)}
              >
                All subjects
              </button>
              {regions.filter((region) => (
                region.layer === "generation"
                && region.enabled
                && region.regionType === "subject"
                && region.pose?.enabled
              )).map((region) => (
                <button
                  key={region.id}
                  className={`tiny-button ${controlPreview.subjectRegionId === region.id ? "active" : ""}`}
                  onClick={() => void previewPoseControl(region.id)}
                >
                  {region.name}
                </button>
              ))}
            </div>
            <img
              className="control-preview-image"
              src={controlPreview.url}
              alt="Canonical Krea volumetric pose control"
            />
            <div className="preview-region-order">
              <div><strong>Format</strong><span>{controlPreview.format}</span></div>
              <div><strong>Dimensions</strong><span>{controlPreview.width} × {controlPreview.height}</span></div>
              <div><strong>Coverage</strong><span>{(controlPreview.coverage * 100).toFixed(2)}%</span></div>
              <div><strong>SHA-256</strong><span><code>{controlPreview.sha256}</code></span></div>
            </div>
            <div className="modal-actions">
              <button className="primary-button" onClick={closeControlPreview}>Close</button>
            </div>
          </section>
        </div>
      )}
      {showFaceDetectorInstall && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal face-detector-modal" role="dialog" aria-modal="true" aria-labelledby="face-detector-install-title">
            <div className="danger-icon face-detector-icon"><Icon name="face" /></div>
            <p className="kicker">Optional model required</p>
            <h2 id="face-detector-install-title">Install face detection?</h2>
            <p>Face detection needs the 371 KB FantasyPortrait <code>face_det.onnx</code> model. K2 can download the pinned Apache-2.0 upstream file directly into this workspace and select it automatically.</p>
            {faceDetectorTransfer && (
              <div className="transfer-progress">
                <div><i style={{ width: `${faceDetectorTransfer.bytes_total ? Math.min(100, faceDetectorTransfer.bytes_complete / faceDetectorTransfer.bytes_total * 100) : 0}%` }} /></div>
                <span>{faceDetectorTransfer.state}{faceDetectorTransfer.bytes_total ? ` · ${Math.round(faceDetectorTransfer.bytes_complete / faceDetectorTransfer.bytes_total * 100)}%` : ""}</span>
              </div>
            )}
            {faceDetectorInstallError && <div className="error-banner">{faceDetectorInstallError}</div>}
            <p className="field-help">Source: <a href="https://huggingface.co/acvlab/FantasyPortrait/blob/14df15cac6721a1cabdb9ecbdc0fbd6d3e49154b/face_det.onnx" target="_blank" rel="noreferrer">acvlab/FantasyPortrait</a></p>
            <div className="modal-actions">
              <button className="quiet-button" disabled={faceDetectorInstalling} onClick={() => setShowFaceDetectorInstall(false)}>Not now</button>
              <button className="primary-button" disabled={faceDetectorInstalling} onClick={() => void installFaceDetector()}>{faceDetectorInstalling ? "Installing…" : faceDetectorInstallError ? "Retry install" : "Install and detect faces"}</button>
            </div>
          </section>
        </div>
      )}
      {showMigration && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="migration-title">
            <div className="danger-icon"><Icon name="transfer" /></div>
            <p className="kicker">Verified storage migration</p>
            <h2 id="migration-title">{migration?.state === "awaiting_confirmation" ? "Confirm the portable copy" : "Migrate to a network volume?"}</h2>
            {migration?.state === "awaiting_confirmation" ? (
              <>
                <p>The source and target SHA-256 manifests match. The original Pod is stopped and retained so you can test the portable workspace. Confirming permanently deletes its regular volume.</p>
                <label className="field-label" htmlFor="migration-confirmation">Type <strong>{workspace.name}</strong> to delete the original Pod</label>
                <input id="migration-confirmation" className="text-input" value={migrationConfirmation} onChange={(event) => setMigrationConfirmation(event.target.value)} />
              </>
            ) : (
              <>
                <p>Generation and transfers will stop while durable models, projects, inputs, outputs, and job state are copied. Switchover occurs only after file inventory and SHA-256 manifests match. The original Pod remains stopped until a separate confirmation.</p>
                {!migration && (
                  <div className="two-fields">
                    <label className="number-field">
                      <span>Target network volume</span>
                      <select className="text-input" value={migrationVolumeId} onChange={(event) => {
                        const volume = networkVolumes.find((item) => item.id === event.target.value);
                        setMigrationVolumeId(event.target.value);
                        if (volume) {
                          setMigrationDatacenterId(volume.datacenter_id);
                          setMigrationDiskGb(volume.size_gb);
                        }
                      }}>
                        <option value="">Create a new network volume</option>
                        {networkVolumes.map((volume) => <option value={volume.id} key={volume.id}>{volume.name} · {volume.size_gb} GB · {volume.datacenter_id}</option>)}
                      </select>
                    </label>
                    <label className="number-field">
                      <span>Datacenter</span>
                      <select className="text-input" disabled={Boolean(migrationVolumeId)} value={migrationDatacenterId} onChange={(event) => setMigrationDatacenterId(event.target.value)}>
                        {datacenters.map((datacenter) => <option value={datacenter.id} key={datacenter.id}>{datacenter.name} · {datacenter.location}</option>)}
                      </select>
                    </label>
                    {!migrationVolumeId && (
                      <label className="number-field">
                        <span>Target capacity</span>
                        <span className="number-input-wrap"><DraftNumberInput min={50} max={4000} value={migrationDiskGb} onCommit={setMigrationDiskGb} /><small>GB</small></span>
                      </label>
                    )}
                  </div>
                )}
              </>
            )}
            {migration && migration.bytes_total > 0 && (
              <p className="field-help">Copied {(migration.bytes_copied / 1_048_576).toFixed(1)} of {(migration.bytes_total / 1_048_576).toFixed(1)} MiB · {migration.state}</p>
            )}
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions">
              <button className="quiet-button" onClick={() => setShowMigration(false)}>Close</button>
              {migration?.state === "awaiting_confirmation" ? (
                <button className="danger-button" disabled={busy || migrationConfirmation !== workspace.name} onClick={confirmMigration}>Delete original Pod and volume</button>
              ) : migration && ["preparing", "copying", "verifying"].includes(migration.state) ? (
                <button className="primary-button" disabled={busy} onClick={resumeMigration}>{busy ? "Migrating…" : "Resume verified copy"}</button>
              ) : (
                <button className="primary-button" disabled={busy || workspace.mode !== "persistent_pod"} onClick={beginMigration}>{busy ? "Preparing migration…" : "Create volume and begin copy"}</button>
              )}
            </div>
          </section>
        </div>
      )}
      {showConnectPod && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="connect-pod-title">
            <div className="danger-icon"><Icon name="cloud" /></div>
            <p className="kicker">RunPod console migration</p>
            <h2 id="connect-pod-title">Connect the migrated Pod?</h2>
            <p>Enter the new Pod ID created by RunPod. K2 will verify that it retained this workspace's identity, agent credential, immutable image, GPU type, and persistent volume before replacing the old Pod ID.</p>
            <label className="field-label" htmlFor="migrated-pod-id">New RunPod Pod ID</label>
            <input id="migrated-pod-id" className="text-input" autoComplete="off" placeholder="e.g. a5fbpvr8eoykhk" value={connectPodId} onChange={(event) => setConnectPodId(event.target.value)} />
            <label className="check-row warning-check lease-limit-option">
              <input type="checkbox" checked={connectWithoutTimeLimit} onChange={(event) => setConnectWithoutTimeLimit(event.target.checked)} />
              <span><strong>No time limit if the migrated Pod is already running</strong><small>The Pod will continue running and billing until you manually stop it.</small></span>
            </label>
            <p className="field-help">This only changes the control-plane connection after verification. It does not start, stop, migrate, or delete either Pod.</p>
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions">
              <button className="quiet-button" onClick={() => { setShowConnectPod(false); setMessage(""); }}>Cancel</button>
              <button className="primary-button" disabled={busy || connectPodId.trim().length < 3} onClick={() => void connectMigratedPod()}>{busy ? "Verifying Pod…" : "Verify and connect"}</button>
            </div>
          </section>
        </div>
      )}
      {showAssets && <AssetPanel workspaceId={workspace.id} uploadQueue={uploadQueue} initialKind={assetPurpose === "lora" ? "loras" : assetPurpose === "pose-control" || assetPurpose === "depth-checkpoint" ? "krea_control_loras" : assetPurpose === "upscale" ? "upscale_models" : "inputs"} onEvent={(text, kind) => report(text, kind)} onClose={() => setUtilityPanel(null)} onSelect={(file) => {
        if (assetPurpose === "lora") {
          if (file.kind === "loras" && !loras.some((lora) => lora.fileId === file.id)) {
            const missingIndex = loras.findIndex((lora) => !lora.fileId && lora.name.toLocaleLowerCase() === file.display_name.toLocaleLowerCase());
            setLoras(missingIndex >= 0
              ? loras.map((lora, index) => index === missingIndex ? { ...lora, fileId: file.id, name: file.display_name } : lora)
              : [...loras, createStudioLora(file.id, file.display_name)]);
          }
          return;
        }
        if (assetPurpose === "upscale") {
          if (file.kind === "upscale_models") setStudioSettings({ ...studioSettings, generation: { ...studioSettings.generation, upscaleModelFileId: file.id, upscaleModelName: file.display_name } });
          return;
        }
        if (assetPurpose === "pose-control") {
          if (file.kind === "krea_control_loras") {
            setStudioSettings({
              ...studioSettings,
              generation: {
                ...studioSettings.generation,
                poseControlLoraFileId: file.id,
                poseControlLoraModel: file.display_name,
                poseControlLegacyAcknowledged: false,
              },
            });
            void inspectPoseControlCheckpoint(file.id, false);
          }
          return;
        }
        if (assetPurpose === "depth-checkpoint") {
          if (file.kind === "krea_control_loras") {
            setStudioSettings({
              ...studioSettings,
              generation: {
                ...studioSettings.generation,
                depth: {
                  ...studioSettings.generation.depth,
                  checkpointFileId: file.id,
                  checkpointName: file.display_name,
                },
              },
            });
          }
          return;
        }
        if (assetPurpose === "depth-image") {
          if (file.kind === "inputs") {
            setStudioSettings({
              ...studioSettings,
              generation: {
                ...studioSettings.generation,
                depth: {
                  ...studioSettings.generation.depth,
                  imageFileId: file.id,
                  imageName: file.display_name,
                },
              },
            });
          }
          return;
        }
        if (file.kind === "projects") { void openCloudProject(file); return; }
        if (file.kind !== "inputs" && file.kind !== "outputs") return;
        setCloudSource(file);
        setSourceName(file.display_name);
        setFaceDetections([]);
        setSelectedFaceIndices([]);
        setManualFacePaths([]);
        setSourceUrl(controlPlane.fileUrl(workspace.id, file.id));
      }} />}
      {showTransfers && <TransferPanel workspaceId={workspace.id} onEvent={(text, kind) => report(text, kind)} onClose={() => setUtilityPanel(null)} />}
      {showSetup && <SetupPanel workspaceId={workspace.id} settings={studioSettings} onSettings={setStudioSettings} onEvent={(text, kind) => report(text, kind)} onClose={() => setUtilityPanel(null)} onManageFiles={() => { setAssetPurpose("source"); setUtilityPanel("assets"); }} onTransfers={() => setUtilityPanel("transfers")} />}
    </div>
  );
}

function RailButton({ icon, label, active, onClick }: { icon: IconName; label: string; active: boolean; onClick: () => void }) {
  return <button className={`rail-button ${active ? "active" : ""}`} onClick={onClick}><Icon name={icon} /><span>{label}</span></button>;
}

function formatEventTime(value: string) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleTimeString();
}
