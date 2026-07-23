import { useEffect, useState } from "react";
import type {
  CapabilityManifest,
  CredentialStatus,
  DatacenterOption,
  GpuOption,
  NetworkVolumeOption,
  WorkspaceRecord,
} from "./api";
import { controlPlane } from "./api";
import { CloudOnboarding } from "./components/CloudOnboarding";
import { Icon } from "./components/Icon";
import { WorkspaceStudio } from "./components/WorkspaceStudio";

interface BootstrapState {
  capabilities: CapabilityManifest;
  credential: CredentialStatus;
  gpus: GpuOption[];
  datacenters: DatacenterOption[];
  networkVolumes: NetworkVolumeOption[];
  workspace: WorkspaceRecord | null;
}

export function App() {
  const [state, setState] = useState<BootstrapState | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        await controlPlane.openSession();
        const [capabilities, credential, workspaces] = await Promise.all([
          controlPlane.capabilities(),
          controlPlane.credentialStatus(),
          controlPlane.workspaces(),
        ]);
        const [gpus, datacenters, networkVolumes] = credential.configured
          ? await Promise.all([
              controlPlane.gpus(),
              controlPlane.datacenters(),
              controlPlane.networkVolumes(),
            ])
          : [[], [], []];
        if (!cancelled) {
          setState({
            capabilities,
            credential,
            gpus,
            datacenters,
            networkVolumes,
            workspace: workspaces.filter((item) => item.state !== "deleted").at(-1) ?? null,
          });
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not reach the control plane");
      }
    }
    bootstrap();
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return (
      <main className="fatal-shell">
        <div className="danger-icon"><Icon name="cloud" /></div>
        <p className="kicker">Control plane unavailable</p>
        <h1>Could not start K2 Region Lab</h1>
        <p>{error}</p>
        <code>uv run k2lab-web --reload</code>
        <button className="primary-button" onClick={() => window.location.reload()}>Try again</button>
      </main>
    );
  }

  if (!state) {
    return (
      <main className="loading-shell">
        <span className="loading-mark">K2</span>
        <span className="loading-bar"><i /></span>
        <p>Opening cloud studio…</p>
      </main>
    );
  }

  if (!state.workspace) {
    return (
      <div className="entry-shell">
        <header className="entry-header">
          <div className="brand-lockup"><span className="brand-mark">K2</span><span><strong>Region Lab</strong><small>Cloud studio</small></span></div>
          <div className="entry-meta"><span>Project schema v{state.capabilities.project_schema_version}</span><span className="divider" /> Persistent or portable cloud workspace</div>
        </header>
        <CloudOnboarding
          capabilities={state.capabilities}
          credential={state.credential}
          gpus={state.gpus}
          datacenters={state.datacenters}
          networkVolumes={state.networkVolumes}
          onCredential={(credential, gpus, datacenters, networkVolumes) =>
            setState({ ...state, credential, gpus, datacenters, networkVolumes })}
          onWorkspace={(workspace) => setState({ ...state, workspace })}
        />
      </div>
    );
  }

  if (
    state.workspace.state === "error"
    && ["provider_resource_not_found", "provider_resource_unavailable"].includes(
      state.workspace.error_code ?? "",
    )
  ) {
    return (
      <MissingWorkspace
        workspace={state.workspace}
        onWorkspace={(workspace) => setState({ ...state, workspace })}
        onForget={() => setState({ ...state, workspace: null })}
      />
    );
  }

  return (
    <WorkspaceStudio
      workspace={state.workspace}
      developmentBackend={state.capabilities.development_backend}
      datacenters={state.datacenters}
      networkVolumes={state.networkVolumes}
      onWorkspace={(workspace) => setState({ ...state, workspace })}
      onDelete={() => setState({ ...state, workspace: null })}
    />
  );
}

interface MissingWorkspaceProps {
  workspace: WorkspaceRecord;
  onWorkspace: (workspace: WorkspaceRecord) => void;
  onForget: () => void;
}

function MissingWorkspace({ workspace, onWorkspace, onForget }: MissingWorkspaceProps) {
  const [showConnect, setShowConnect] = useState(false);
  const [podId, setPodId] = useState("");
  const [leaseUnlimited, setLeaseUnlimited] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function connectPod() {
    setBusy(true);
    setError("");
    try {
      onWorkspace(await controlPlane.connectMigratedPod(workspace.id, podId.trim(), leaseUnlimited));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not connect the migrated Pod");
    } finally {
      setBusy(false);
    }
  }

  async function forgetWorkspace() {
    setBusy(true);
    setError("");
    try {
      await controlPlane.terminateWorkspace(workspace.id, workspace.name);
      onForget();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not clear the missing workspace");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="fatal-shell missing-workspace-shell">
      <div className="danger-icon"><Icon name="cloud" /></div>
      <p className="kicker">RunPod workspace unavailable</p>
      <h1>This Pod no longer exists</h1>
      <p>
        K2 still has the local record for <strong>{workspace.name}</strong>, but RunPod cannot find
        Pod <code>{workspace.provider_resource_id ?? "unknown"}</code>.
      </p>
      <p>
        If you migrated it in RunPod, connect the replacement Pod ID. Otherwise clear this stale
        record and create a workspace with the new image.
      </p>
      {showConnect && (
        <section className="missing-workspace-connect">
          <label className="field-label" htmlFor="replacement-pod-id">Replacement RunPod Pod ID</label>
          <input
            id="replacement-pod-id"
            className="text-input"
            autoComplete="off"
            placeholder="e.g. a5fbpvr8eoykhk"
            value={podId}
            onChange={(event) => setPodId(event.target.value)}
          />
          <label className="check-row warning-check">
            <input
              type="checkbox"
              checked={leaseUnlimited}
              onChange={(event) => setLeaseUnlimited(event.target.checked)}
            />
            <span>
              <strong>No time limit</strong>
              <small>The replacement Pod will keep running and billing until manually stopped.</small>
            </span>
          </label>
          <button
            className="primary-button full-button"
            disabled={busy || podId.trim().length < 3}
            onClick={() => void connectPod()}
          >
            {busy ? "Verifying Pod…" : "Verify and connect"}
          </button>
        </section>
      )}
      {error && <div className="error-banner">{error}</div>}
      <div className="missing-workspace-actions">
        <button className="quiet-button" disabled={busy} onClick={() => setShowConnect(!showConnect)}>
          {showConnect ? "Cancel migration connection" : "Connect migrated Pod"}
        </button>
        <button className="primary-button" disabled={busy} onClick={() => void forgetWorkspace()}>
          {busy ? "Clearing…" : "Create a new workspace"}
        </button>
      </div>
    </main>
  );
}
