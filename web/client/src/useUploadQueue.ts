import { createSHA256, sha256 } from "hash-wasm";
import { useCallback, useEffect, useRef, useState } from "react";
import type { FileKind, FileRecord, UploadSession } from "./api";
import { controlPlane } from "./api";
import { findExisting } from "./uploads";

export type LocalUploadState =
  | "queued"
  | "hashing"
  | "uploading"
  | "pausing"
  | "paused"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export interface LocalUploadItem {
  id: string;
  file: File;
  destinationKind: FileKind;
  state: LocalUploadState;
  session: UploadSession | null;
  result: FileRecord | null;
  progress: number;
  speed: number;
  eta: number;
  error: string;
  createdAt: string;
}

export interface UploadQueueController {
  items: LocalUploadItem[];
  activeId: string | null;
  enqueue: (files: File[], destinationKind: FileKind) => void;
  pause: (id: string) => void;
  resume: (id: string) => void;
  cancel: (id: string) => void;
  clearFinished: () => void;
}

type UploadIntent = "run" | "pause" | "cancel";
type UploadEvent = (message: string, kind: "info" | "error" | "worker") => void;

const CHUNK_SIZE = 8 * 1024 * 1024;

export function useUploadQueue(
  workspaceId: string,
  onEvent?: UploadEvent,
): UploadQueueController {
  const [items, setItems] = useState<LocalUploadItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const itemsRef = useRef<LocalUploadItem[]>([]);
  const activeRef = useRef<string | null>(null);
  const intents = useRef(new Map<string, UploadIntent>());
  const eventRef = useRef(onEvent);

  useEffect(() => {
    eventRef.current = onEvent;
  }, [onEvent]);

  const mutate = useCallback((operation: (current: LocalUploadItem[]) => LocalUploadItem[]) => {
    setItems((current) => {
      const next = operation(current);
      itemsRef.current = next;
      return next;
    });
  }, []);

  const updateItem = useCallback((
    id: string,
    updates: Partial<Omit<LocalUploadItem, "id" | "file" | "destinationKind" | "createdAt">>,
  ) => {
    mutate((current) => current.map((item) => (
      item.id === id ? { ...item, ...updates } : item
    )));
  }, [mutate]);

  const settleIntent = useCallback(async (
    item: LocalUploadItem,
    session: UploadSession | null,
  ) => {
    const intent = intents.current.get(item.id) ?? "run";
    if (intent === "run") return false;
    if (intent === "pause") {
      updateItem(item.id, { state: "paused", speed: 0, eta: 0 });
      eventRef.current?.(`Paused upload of ${item.file.name}.`, "info");
      return true;
    }
    try {
      if (session) await controlPlane.cancelUpload(workspaceId, session.id);
      updateItem(item.id, {
        state: "cancelled",
        session: session ? { ...session, state: "cancelled" } : null,
        speed: 0,
        eta: 0,
      });
      eventRef.current?.(`Cancelled upload of ${item.file.name}.`, "info");
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : "Could not cancel upload";
      updateItem(item.id, { state: "failed", error: detail, speed: 0, eta: 0 });
      eventRef.current?.(detail, "error");
    }
    return true;
  }, [updateItem, workspaceId]);

  const processItem = useCallback(async (item: LocalUploadItem) => {
    let session = item.session;
    try {
      intents.current.set(item.id, "run");
      updateItem(item.id, { state: "hashing", error: "", speed: 0, eta: 0 });
      const hasher = await createSHA256();
      hasher.init();
      for (let offset = 0; offset < item.file.size; offset += CHUNK_SIZE) {
        if (await settleIntent(item, session)) return;
        const buffer = await item.file.slice(offset, offset + CHUNK_SIZE).arrayBuffer();
        hasher.update(new Uint8Array(buffer));
        updateItem(item.id, {
          progress: Math.min(0.1, ((offset + buffer.byteLength) / item.file.size) * 0.1),
        });
      }
      const digest = hasher.digest("hex");
      if (await settleIntent(item, session)) return;

      const existing = await findExisting(
        workspaceId,
        item.destinationKind,
        item.file.name,
        item.file.size,
        digest,
      );
      if (existing) {
        updateItem(item.id, {
          state: "completed",
          result: existing,
          progress: 1,
          speed: 0,
          eta: 0,
        });
        eventRef.current?.(`Verified existing ${item.file.name}.`, "info");
        return;
      }

      if (session) {
        session = await controlPlane.uploadStatus(workspaceId, session.id);
      } else {
        const history = await controlPlane.uploads(workspaceId);
        session = history.find((candidate) => (
          candidate.state === "uploading"
          && candidate.display_name === item.file.name
          && candidate.destination_kind === item.destinationKind
          && candidate.size_bytes === item.file.size
          && candidate.sha256 === digest
        )) ?? await controlPlane.createUpload(workspaceId, {
          filename: item.file.name,
          destination_kind: item.destinationKind,
          size_bytes: item.file.size,
          sha256: digest,
          chunk_size_bytes: CHUNK_SIZE,
        });
      }
      updateItem(item.id, { state: "uploading", session });
      eventRef.current?.(
        `Uploading ${item.file.name} to ${item.destinationKind.replaceAll("_", " ")}.`,
        "info",
      );

      const completed = new Set(session.completed_chunks);
      let sent = completedBytes(session);
      const initialSent = sent;
      const started = performance.now();
      for (let index = 0; index < session.chunk_count; index += 1) {
        if (await settleIntent(item, session)) return;
        if (completed.has(index)) continue;
        const start = index * session.chunk_size_bytes;
        const buffer = await item.file.slice(
          start,
          start + session.chunk_size_bytes,
        ).arrayBuffer();
        if (await settleIntent(item, session)) return;
        await controlPlane.uploadChunk(
          workspaceId,
          session.id,
          index,
          buffer,
          await sha256(new Uint8Array(buffer)),
        );
        completed.add(index);
        sent += buffer.byteLength;
        session = {
          ...session,
          completed_chunks: [...completed].sort((left, right) => left - right),
          updated_at: new Date().toISOString(),
        };
        const elapsed = Math.max(0.001, (performance.now() - started) / 1000);
        const speed = (sent - initialSent) / elapsed;
        updateItem(item.id, {
          state: "uploading",
          session,
          progress: 0.1 + (sent / session.size_bytes) * 0.9,
          speed,
          eta: speed > 0 ? (session.size_bytes - sent) / speed : 0,
        });
      }
      if (await settleIntent(item, session)) return;
      const completedUpload = await controlPlane.completeUpload(workspaceId, session.id);
      session = {
        ...session,
        state: "completed",
        completed_chunks: Array.from(
          { length: session.chunk_count },
          (_value, index) => index,
        ),
        updated_at: new Date().toISOString(),
      };
      updateItem(item.id, {
        state: "completed",
        session,
        result: completedUpload.file,
        progress: 1,
        speed: 0,
        eta: 0,
      });
      eventRef.current?.(
        `${completedUpload.duplicate ? "Verified existing" : "Uploaded"} ${item.file.name}.`,
        "info",
      );
    } catch (caught) {
      if (await settleIntent(item, session)) return;
      const detail = caught instanceof Error ? caught.message : "Upload failed";
      updateItem(item.id, { state: "failed", session, error: detail, speed: 0, eta: 0 });
      eventRef.current?.(detail, "error");
    }
  }, [settleIntent, updateItem, workspaceId]);

  useEffect(() => {
    if (activeRef.current) return;
    const next = items.find((item) => item.state === "queued");
    if (!next) return;
    activeRef.current = next.id;
    setActiveId(next.id);
    void processItem(next).finally(() => {
      activeRef.current = null;
      setActiveId(null);
    });
  }, [items, processItem]);

  const enqueue = useCallback((files: File[], destinationKind: FileKind) => {
    if (!files.length) return;
    const createdAt = new Date().toISOString();
    const queued = files.map<LocalUploadItem>((file) => ({
      id: crypto.randomUUID(),
      file,
      destinationKind,
      state: "queued",
      session: null,
      result: null,
      progress: 0,
      speed: 0,
      eta: 0,
      error: "",
      createdAt,
    }));
    for (const item of queued) intents.current.set(item.id, "run");
    mutate((current) => [...current, ...queued]);
    eventRef.current?.(
      `Queued ${queued.length} local upload${queued.length === 1 ? "" : "s"}.`,
      "info",
    );
  }, [mutate]);

  const pause = useCallback((id: string) => {
    intents.current.set(id, "pause");
    if (activeRef.current === id) {
      updateItem(id, { state: "pausing" });
    } else {
      updateItem(id, { state: "paused", speed: 0, eta: 0 });
    }
  }, [updateItem]);

  const resume = useCallback((id: string) => {
    intents.current.set(id, "run");
    updateItem(id, { state: "queued", error: "", speed: 0, eta: 0 });
  }, [updateItem]);

  const cancel = useCallback((id: string) => {
    intents.current.set(id, "cancel");
    const item = itemsRef.current.find((candidate) => candidate.id === id);
    if (!item) return;
    if (activeRef.current === id) {
      updateItem(id, { state: "cancelling" });
      return;
    }
    updateItem(id, { state: "cancelling" });
    void settleIntent(item, item.session);
  }, [settleIntent, updateItem]);

  const clearFinished = useCallback(() => {
    mutate((current) => current.filter((item) => (
      !["completed", "cancelled"].includes(item.state)
    )));
  }, [mutate]);

  return { items, activeId, enqueue, pause, resume, cancel, clearFinished };
}

function completedBytes(session: UploadSession) {
  return session.completed_chunks.reduce((total, index) => {
    const start = index * session.chunk_size_bytes;
    return total + Math.min(session.chunk_size_bytes, session.size_bytes - start);
  }, 0);
}
