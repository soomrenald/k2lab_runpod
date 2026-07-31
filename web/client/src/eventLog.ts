export const EVENT_LOG_LIMIT = 1000;

export function appendBoundedEvents<T>(current: T[], additions: T[], limit = EVENT_LOG_LIMIT) {
  if (limit < 1) return [];
  return [...current, ...additions].slice(-limit);
}

export interface WorkerEventLike {
  message: string;
  payload: Record<string, unknown>;
}

export interface ParsedLoraCompatibility {
  instanceId: string;
  status: "compatible" | "incompatible";
  summary: string;
}

export function parseLoraCompatibility(event: WorkerEventLike): ParsedLoraCompatibility[] {
  if (!Array.isArray(event.payload.loras)) return [];
  return event.payload.loras.flatMap((value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [];
    const report = value as Record<string, unknown>;
    const runtimeId = String(report.id ?? "");
    const instanceId = runtimeId.replace(/^(?:reference|edit|face):/, "");
    if (!instanceId || typeof report.compatible !== "boolean") return [];
    const name = String(report.display_name ?? report.name ?? "LoRA");
    const matched = Number(report.matched_model_targets ?? 0);
    const total = Number(report.adapter_count ?? 0);
    const status = report.compatible ? "compatible" : "incompatible";
    const matchText = total > 0 ? ` · ${matched}/${total} Krea 2 targets` : "";
    return [{
      instanceId,
      status,
      summary: `${name}: ${status}${matchText}`,
    }];
  });
}

export function formatWorkerEvent(event: WorkerEventLike): string {
  const reports = parseLoraCompatibility(event);
  if (!reports.length) return event.message;
  return `${event.message}: ${reports.map((report) => report.summary).join("; ")}`;
}
