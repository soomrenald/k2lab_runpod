export const PROVIDER_POLL_INTERVAL_MS = 5_000;

export interface ProviderFreshnessEvent {
  kind: "warning" | "info";
  source: "provider";
  message: string;
}

export function providerFreshnessEvent(
  wasStale: boolean,
  isStale: boolean,
): ProviderFreshnessEvent | null {
  if (!wasStale && isStale) {
    return {
      kind: "warning",
      source: "provider",
      message: "RunPod status refresh timed out. Using the last known workspace status; completed jobs and cloud outputs are unaffected.",
    };
  }
  if (wasStale && !isStale) {
    return {
      kind: "info",
      source: "provider",
      message: "RunPod provider status is reachable again.",
    };
  }
  return null;
}

export function isPassiveProviderTransient(
  error: unknown,
): error is Error & { code: string } {
  return (
    error instanceof Error
    && "code" in error
    && typeof error.code === "string"
    && ["provider_timeout", "provider_unavailable"].includes(error.code)
  );
}

export function scheduleProviderPoll(
  schedule: (callback: () => void, delay: number) => number,
  callback: () => void,
): number {
  return schedule(callback, PROVIDER_POLL_INTERVAL_MS);
}
