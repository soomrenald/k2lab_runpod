import type { GenerationJob, JobKind, JobSubmitPayload } from "./api";

const STORAGE_PREFIX = "k2lab.pending-job-submission.v1.";
const JOB_KINDS = new Set<JobKind>(["generate", "edit_image", "refine_faces"]);

export interface PendingJobSubmission {
  version: 1;
  workspaceId: string;
  payloads: JobSubmitPayload[];
  acknowledgedJobs: GenerationJob[];
  createdAt: string;
  lastError: string;
}

function storageKey(workspaceId: string): string {
  return `${STORAGE_PREFIX}${workspaceId}`;
}

function isJobSubmitPayload(value: unknown): value is JobSubmitPayload {
  if (!value || typeof value !== "object") return false;
  const payload = value as Partial<JobSubmitPayload>;
  return typeof payload.command_id === "string"
    && payload.command_id.length > 0
    && typeof payload.kind === "string"
    && JOB_KINDS.has(payload.kind as JobKind)
    && typeof payload.project_id === "string"
    && !!payload.project
    && typeof payload.project === "object"
    && typeof payload.filename_prefix === "string";
}

function isGenerationJob(value: unknown): value is GenerationJob {
  if (!value || typeof value !== "object") return false;
  const job = value as Partial<GenerationJob>;
  return typeof job.id === "string"
    && typeof job.command_id === "string"
    && typeof job.state === "string";
}

export function loadPendingJobSubmission(
  storage: Pick<Storage, "getItem">,
  workspaceId: string,
): PendingJobSubmission | null {
  try {
    const serialized = storage.getItem(storageKey(workspaceId));
    if (!serialized) return null;
    const value = JSON.parse(serialized) as Partial<PendingJobSubmission>;
    if (
      value.version !== 1
      || value.workspaceId !== workspaceId
      || !Array.isArray(value.payloads)
      || value.payloads.length === 0
      || !value.payloads.every(isJobSubmitPayload)
      || !Array.isArray(value.acknowledgedJobs)
      || !value.acknowledgedJobs.every(isGenerationJob)
      || typeof value.createdAt !== "string"
      || typeof value.lastError !== "string"
    ) {
      return null;
    }
    return value as PendingJobSubmission;
  } catch {
    return null;
  }
}

export function savePendingJobSubmission(
  storage: Pick<Storage, "setItem">,
  recovery: PendingJobSubmission,
): void {
  try {
    storage.setItem(storageKey(recovery.workspaceId), JSON.stringify(recovery));
  } catch {
    // The in-memory recovery remains usable if browser storage is unavailable.
  }
}

export function clearPendingJobSubmission(
  storage: Pick<Storage, "removeItem">,
  workspaceId: string,
): void {
  try {
    storage.removeItem(storageKey(workspaceId));
  } catch {
    // Storage may be disabled; clearing React state is still sufficient.
  }
}

export function isAmbiguousJobSubmissionError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (!error || typeof error !== "object") return false;
  const candidate = error as { code?: unknown; status?: unknown };
  if (typeof candidate.status === "number" && candidate.status >= 500) return true;
  if (typeof candidate.code !== "string") return false;
  return candidate.code === "agent_connection_failed"
    || candidate.code === "agent_proxy_error"
    || candidate.code.startsWith("agent_") && candidate.code.endsWith("_timeout");
}

export function uniqueJobs(jobs: GenerationJob[]): GenerationJob[] {
  const byId = new Map<string, GenerationJob>();
  for (const job of jobs) byId.set(job.id, job);
  return [...byId.values()];
}
