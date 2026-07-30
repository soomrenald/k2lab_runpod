import { useRef, useState } from "react";
import type { DetectedFaceRecord, KreaControlCheckpointInspection, WorkerMemoryStatus } from "../api";
import type { RegionBox, RegionLayer, StudioMode } from "./RegionCanvas";
import { Icon } from "./Icon";
import { DraftNumberInput } from "./DraftNumberInput";
import { PoseGatingControls } from "./PoseGatingControls";
import {
  COMFYUI_SAMPLERS,
  COMFYUI_SCHEDULERS,
  defaultLoraTrigger,
  PROJECTOR_PRESETS,
  type GenerationSettings,
  type LoraLayerBinding,
  type ProjectorSettings,
  type PromptEmphasisState,
  type StudioLora,
  type StudioSettings,
  type VramMode,
} from "../studioProject";
import {
  promptEmphasisFromSelection,
  promptEmphasisMatches,
  reconcilePromptEmphases,
} from "../promptEmphasis.ts";
import { mirrorPose, squattingPose, standingPose } from "../pose.ts";

type InspectorTab = "prompt" | "regions" | "loras" | "advanced";

interface Props {
  mode: StudioMode;
  activeLayer: RegionLayer;
  regions: RegionBox[];
  selectedId: string | null;
  globalPrompt: string;
  settings: StudioSettings;
  loras: StudioLora[];
  onGlobalPrompt: (value: string) => void;
  onSettings: (settings: StudioSettings) => void;
  onLoras: (loras: StudioLora[]) => void;
  onChooseLora: () => void;
  onChoosePoseControlLora: () => void;
  onChooseDepthCheckpoint: () => void;
  onChooseDepthImage: () => void;
  onPreviewPoseControl: () => void;
  poseControlLoraAvailable: boolean;
  depthControlAvailable: boolean;
  depthRegionsAvailable: boolean;
  poseControlCompatibility: KreaControlCheckpointInspection | null;
  onInspectPoseControlLegacy: (allow: boolean) => void;
  onChooseUpscaleModel: () => void;
  onPreviewUnifiedPrompt: () => void;
  faces: DetectedFaceRecord[];
  selectedFaceIndices: number[];
  manualFacePaths: number[][][];
  lassoMode: boolean;
  onDetectFaces: () => void;
  onToggleFace: (index: number) => void;
  onSelectAllFaces: (selected: boolean) => void;
  onLassoMode: (enabled: boolean) => void;
  onUndoLasso: () => void;
  onClearLassos: () => void;
  onUseLatestFaceSource: () => void;
  onRegions: (regions: RegionBox[]) => void;
  onSelect: (id: string | null) => void;
  workerMemory: WorkerMemoryStatus | null;
  memoryRefreshing: boolean;
  memoryActionsDisabled: boolean;
  onRefreshMemory: () => void;
  onReleaseMemory: () => void;
}

export function Inspector(props: Props) {
  const {
    mode, activeLayer, regions, selectedId, globalPrompt, settings, loras,
    onGlobalPrompt, onSettings, onLoras, onChooseLora, onChoosePoseControlLora,
    onChooseDepthCheckpoint, onChooseDepthImage,
    onPreviewPoseControl,
    poseControlLoraAvailable,
    depthControlAvailable,
    depthRegionsAvailable,
    poseControlCompatibility, onInspectPoseControlLegacy,
    onChooseUpscaleModel,
    onPreviewUnifiedPrompt, faces, selectedFaceIndices, manualFacePaths, lassoMode,
    onDetectFaces, onToggleFace, onSelectAllFaces, onLassoMode, onUndoLasso,
    onClearLassos, onUseLatestFaceSource, onRegions, onSelect,
    workerMemory, memoryRefreshing, memoryActionsDisabled, onRefreshMemory, onReleaseMemory,
  } = props;
  const [tab, setTab] = useState<InspectorTab>("prompt");
  const [emphasisStrength, setEmphasisStrength] = useState(0.5);
  const globalPromptRef = useRef<HTMLTextAreaElement>(null);
  const regionPromptRef = useRef<HTMLTextAreaElement>(null);
  const visibleRegions = regions.filter((region) => region.layer === activeLayer);
  const selected = regions.find((region) => region.id === selectedId && region.layer === activeLayer) ?? null;
  const emphasisAvailable = mode === "generation" || (mode === "edit" && activeLayer === "reference");
  const emphases = mode === "generation"
    ? settings.generation.promptEmphases
    : settings.edit.referencePromptEmphases;

  function updateSelected(patch: Partial<RegionBox>) {
    if (!selected) return;
    onRegions(regions.map((region) => region.id === selected.id ? { ...region, ...patch } : region));
    if (emphasisAvailable && patch.prompt !== undefined) {
      reconcileEmphases(selected.id, patch.prompt);
    }
  }

  function updateGeneration(patch: Partial<GenerationSettings>) {
    if (patch.width !== undefined || patch.height !== undefined) {
      const width = patch.width ?? settings.generation.width;
      const height = patch.height ?? settings.generation.height;
      onRegions(regions.map((region) => {
        if (region.layer !== "generation") return region;
        const x = Math.min(region.x, width - 16);
        const y = Math.min(region.y, height - 16);
        return {
          ...region,
          x,
          y,
          width: Math.min(region.width, width - x),
          height: Math.min(region.height, height - y),
        };
      }));
    }
    onSettings({ ...settings, generation: { ...settings.generation, ...patch } });
  }

  function updateEdit(patch: Partial<StudioSettings["edit"]>) {
    onSettings({ ...settings, edit: { ...settings.edit, ...patch } });
  }

  function updateFace(patch: Partial<StudioSettings["face"]>) {
    onSettings({ ...settings, face: { ...settings.face, ...patch } });
  }

  function updateRuntime(patch: Partial<StudioSettings["runtime"]>) {
    onSettings({ ...settings, runtime: { ...settings.runtime, ...patch } });
  }

  function setEmphases(next: PromptEmphasisState[]) {
    if (mode === "generation") updateGeneration({ promptEmphases: next });
    else updateEdit({ referencePromptEmphases: next });
  }

  function sourceForEmphasis(scopeId: string, changedScopeId?: string, changedSource?: string) {
    if (scopeId === changedScopeId) return changedSource ?? "";
    if (scopeId === "__global__") return globalPrompt;
    const region = regions.find((item) => item.id === scopeId);
    return region?.enabled ? region.prompt : null;
  }

  function reconcileEmphases(changedScopeId: string, changedSource: string) {
    const next = reconcilePromptEmphases(
      emphases,
      (scopeId) => sourceForEmphasis(scopeId, changedScopeId, changedSource),
    );
    if (
      next.length !== emphases.length
      || next.some((item, index) => item !== emphases[index])
    ) {
      setEmphases(next);
    }
  }

  function addEmphasis(scopeId: "__global__" | string, editor: HTMLTextAreaElement | null) {
    if (!editor || editor.selectionStart === editor.selectionEnd) return;
    const selectedEmphasis = promptEmphasisFromSelection(
      editor.value,
      editor.selectionStart,
      editor.selectionEnd,
      scopeId,
    );
    if (!selectedEmphasis) return;
    const current = reconcilePromptEmphases(
      emphases,
      (itemScopeId) => sourceForEmphasis(itemScopeId),
    );
    setEmphases([...current, {
      id: crypto.randomUUID(),
      scopeId,
      phrase: selectedEmphasis.phrase,
      strength: emphasisStrength,
      occurrence: selectedEmphasis.occurrence,
    }]);
  }

  function moveSelected(offset: -1 | 1) {
    if (!selected) return;
    const peers = regions.filter((region) => region.layer === selected.layer);
    const peerIndex = peers.findIndex((region) => region.id === selected.id);
    const target = peers[peerIndex + offset];
    if (!target) return;
    const next = [...regions];
    const sourceIndex = next.findIndex((region) => region.id === selected.id);
    const targetIndex = next.findIndex((region) => region.id === target.id);
    [next[sourceIndex], next[targetIndex]] = [next[targetIndex], next[sourceIndex]];
    onRegions(next);
  }

  function removeSelectedRegion() {
    if (!selected) return;
    const regionId = selected.id;
    onRegions(regions.filter((region) => region.id !== regionId));
    onLoras(loras.map((lora) => {
      const generationIds = lora.generation.regionIds.filter((id) => id !== regionId);
      const referenceIds = lora.reference.regionIds.filter((id) => id !== regionId);
      const targetIds = lora.targets.regionIds.filter((id) => id !== regionId);
      return {
        ...lora,
        generation: generationIds.length || lora.generation.global
          ? { ...lora.generation, regionIds: generationIds }
          : { ...lora.generation, global: true, regionIds: [], routingMode: "standard" },
        reference: { ...lora.reference, enabled: lora.reference.global || referenceIds.length > 0, regionIds: referenceIds },
        targets: { ...lora.targets, enabled: lora.targets.global || targetIds.length > 0, regionIds: targetIds },
      };
    }));
    onSettings({
      ...settings,
      generation: { ...settings.generation, promptEmphases: settings.generation.promptEmphases.filter((item) => item.scopeId !== regionId) },
      edit: { ...settings.edit, referencePromptEmphases: settings.edit.referencePromptEmphases.filter((item) => item.scopeId !== regionId) },
    });
    onSelect(null);
  }

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <div><p className="kicker">Inspector</p><h2>{mode === "edit" ? "Image edit" : mode === "face" ? "Face refinement" : "Generation"}</h2></div>
        <span className="layer-count">{visibleRegions.length} region{visibleRegions.length === 1 ? "" : "s"}</span>
      </div>
      <nav className="inspector-tabs" aria-label="Inspector sections">
        {(["prompt", "regions", "loras", "advanced"] as InspectorTab[]).map((item) => (
          <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>
            {item === "prompt" ? "Prompt" : item === "regions" ? "Regions" : item === "loras" ? "LoRAs" : "Advanced"}
          </button>
        ))}
      </nav>

      <div className="inspector-content">
        {tab === "prompt" && <>
          <div className="inspector-section">
            <label className="field-label" htmlFor="global-prompt">
              {mode === "edit" && activeLayer === "targets" ? "Edit instruction" : mode === "edit" ? "Original global prompt · reference" : mode === "face" ? "Generation prompt · reference" : "Scene prompt"}
            </label>
            <textarea ref={globalPromptRef} id="global-prompt" className="prompt-area global-area"
              placeholder={mode === "edit" ? "Describe the overall edit intent…" : "Describe the complete scene…"}
              value={globalPrompt} onChange={(event) => {
                onGlobalPrompt(event.target.value);
                if (emphasisAvailable) {
                  reconcileEmphases("__global__", event.target.value);
                }
              }} />
            {mode === "edit" && activeLayer === "targets" && <p className="field-help">Combined with each target prompt. Leave blank for box-only instructions.</p>}
            {mode === "generation" && <>
              <label className="field-label" htmlFor="shared-visual-prompt">Shared visual context</label>
              <textarea
                id="shared-visual-prompt"
                className="prompt-area identity-area"
                placeholder="Lighting, lens, medium, palette, and visual style…"
                value={settings.generation.sharedVisualPrompt}
                onChange={(event) => updateGeneration({ sharedVisualPrompt: event.target.value })}
              />
              <p className="field-help">Lighting, lens, medium, palette, and visual style shared by the scene and every subject-only conditioning branch. Do not put scene objects or background content here.</p>
            </>}
          </div>
          {mode !== "face" && <div className="inspector-section">
            <div className="section-inline-title"><span>{selected ? selected.name : "Regional prompt"}</span>{selected && <span className="active-pill">Selected</span>}</div>
            {selected ? <>
              <input className="text-input compact-input" value={selected.name} onChange={(event) => updateSelected({ name: event.target.value })} />
              <label className="field-label">Box type</label>
              <div className="box-type-readout">
                <strong>{selected.regionType === "subject" ? "Subject box · mannequin" : "Region box · no mannequin"}</strong>
                <small>{selected.regionType === "subject" ? "Contributes volumetric pose gating when enabled." : "Use for objects, scenery, or background areas."}</small>
              </div>
              {selected.regionType === "region" && <>
                <label className="field-label">Spatial role</label>
                <select className="select-input compact-select" value={selected.spatialRole} onChange={(event) => updateSelected({ spatialRole: event.target.value as RegionBox["spatialRole"] })}>
                  <option value="auto">Auto (based on box width)</option><option value="subject">Subject target</option><option value="background">Background band</option>
                </select>
              </>}
              <label className="field-label">Region prompt</label>
              <textarea ref={regionPromptRef} className="prompt-area" value={selected.prompt}
                placeholder={mode === "edit" && activeLayer === "targets" ? "Describe the edit inside this box…" : "Describe this region…"}
                onChange={(event) => updateSelected({ prompt: event.target.value })} />
              {selected.regionType === "subject" && <>
                <label className="field-label">Face identity prompt</label>
                <textarea className="prompt-area identity-area" value={selected.faceIdentityPrompt}
                  placeholder="Stable facial identity, person class, face, and hair…"
                  onChange={(event) => updateSelected({ faceIdentityPrompt: event.target.value })} />
                <label className="check-row compact-check"><input type="checkbox" checked={selected.pose?.enabled ?? false} onChange={(event) => updateSelected({ pose: { ...(selected.pose ?? standingPose()), enabled: event.target.checked } })} /><span><strong>Enable this mannequin</strong></span></label>
                <div className="inline-actions pose-preset-actions">
                  <button className="tiny-button" onClick={() => updateSelected({ pose: standingPose() })}>Standing</button>
                  <button className="tiny-button" onClick={() => updateSelected({ pose: squattingPose() })}>Squatting</button>
                  <button className="tiny-button" onClick={() => updateSelected({ pose: mirrorPose(selected.pose ?? standingPose()) })}>Mirror</button>
                </div>
                <p className="field-help">Drag body joints and the three head controls directly on the canvas. The center head control moves it; the right and bottom controls resize it. Limbs may extend outside the subject box.</p>
              </>}
            </> : <div className="empty-inspector"><Icon name="layers" /><span>Select a region to edit its prompt.</span></div>}
          </div>}
          {emphasisAvailable && <div className="inspector-section emphasis-panel">
            <div className="section-inline-title"><span>Phrase emphasis</span></div>
            <LinkedValue label="Selected phrase boost" value={emphasisStrength} min={0} max={2} step={0.1} onChange={setEmphasisStrength} />
            <div className="inline-actions"><button className="tiny-button" onClick={() => addEmphasis("__global__", globalPromptRef.current)}>Emphasize global selection</button>{selected && <button className="tiny-button" onClick={() => addEmphasis(selected.id, regionPromptRef.current)}>Region selection</button>}</div>
            {emphases.map((item) => <div className={`emphasis-row ${promptEmphasisMatches(item, sourceForEmphasis(item.scopeId)) ? "" : "invalid"}`} key={item.id}>
              <span>{item.scopeId === "__global__" ? "Global" : regions.find((region) => region.id === item.scopeId)?.name ?? "Missing region"}: “{item.phrase}”</span>
              <DraftNumberInput min={0} max={2} step={0.1} value={item.strength} onCommit={(strength) => setEmphases(emphases.map((entry) => entry.id === item.id ? { ...entry, strength } : entry))} />
              <button className="icon-button danger" onClick={() => setEmphases(emphases.filter((entry) => entry.id !== item.id))}><Icon name="trash" /></button>
            </div>)}
          </div>}
        </>}

        {tab === "regions" && mode === "face" && <FaceSelectionPanel
          faces={faces}
          selectedFaceIndices={selectedFaceIndices}
          manualFacePaths={manualFacePaths}
          lassoMode={lassoMode}
          onDetect={onDetectFaces}
          onToggle={onToggleFace}
          onSelectAll={onSelectAllFaces}
          onLassoMode={onLassoMode}
          onUndoLasso={onUndoLasso}
          onClearLassos={onClearLassos}
          onUseLatest={onUseLatestFaceSource}
        />}

        {tab === "regions" && mode !== "face" && <div className="inspector-section region-panel">
          <div className="section-inline-title"><span>{activeLayer === "reference" ? "Reference regions · front to back" : activeLayer === "targets" ? "Edit targets · front to back" : "Scene regions · front to back"}</span></div>
          <div className="region-list">{visibleRegions.map((region, index) => <button className={`region-list-row ${selectedId === region.id ? "selected" : ""}`} key={region.id} onClick={() => onSelect(region.id)}>
            <span className="region-swatch" style={{ opacity: region.enabled ? 1 : 0.35 }}>{index + 1}</span>
            <span className="region-list-copy"><strong>{region.name}</strong><small>{region.regionType === "subject" ? "Subject · mannequin" : `Region · ${region.spatialRole}`} · {Math.round(region.width)} × {Math.round(region.height)}</small></span>
            <input type="checkbox" aria-label={`Enable ${region.name}`} checked={region.enabled} onClick={(event) => event.stopPropagation()} onChange={(event) => onRegions(regions.map((item) => item.id === region.id ? { ...item, enabled: event.target.checked } : item))} />
          </button>)}</div>
          {visibleRegions.length === 0 && <div className="empty-inspector"><Icon name="plus" /><span>Draw a box on the canvas to add a region.</span></div>}
          {selected && <>
            {mode === "generation" && depthRegionsAvailable && settings.generation.depth.enabled && <>
              <Choice
                label="Depth behavior"
                value={selected.depthMode ?? "inherit"}
                options={[["inherit", "Inherit"], ["emphasize", "Emphasize"], ["relax", "Relax"], ["ignore", "Ignore"]]}
                onChange={(depthMode) => onRegions(regions.map((item) => item.id === selected.id ? { ...item, depthMode: depthMode as NonNullable<RegionBox["depthMode"]> } : item))}
              />
              {(selected.depthMode === "emphasize" || selected.depthMode === "relax") && <LinkedValue
                label="Regional depth strength"
                value={selected.depthStrength ?? 1}
                min={selected.depthMode === "emphasize" ? 1 : 0}
                max={selected.depthMode === "emphasize" ? 3 : 1}
                step={0.05}
                onChange={(depthStrength) => onRegions(regions.map((item) => item.id === selected.id ? { ...item, depthStrength } : item))}
              />}
            </>}
            <div className="inline-actions region-depth-actions">
              <button className="tiny-button" disabled={visibleRegions[0]?.id === selected.id} onClick={() => moveSelected(-1)}>↑ Move forward</button>
              <button className="tiny-button" disabled={visibleRegions[visibleRegions.length - 1]?.id === selected.id} onClick={() => moveSelected(1)}>↓ Move backward</button>
            </div>
            <button className="danger-text-button" onClick={removeSelectedRegion}><Icon name="trash" /> Remove selected region</button>
          </>}
        </div>}

        {tab === "loras" && <LoraPanel activeLayer={activeLayer} regions={visibleRegions} loras={loras} onLoras={onLoras} onChoose={onChooseLora} />}

        {tab === "advanced" && <AdvancedPanel mode={mode} activeLayer={activeLayer} settings={settings} updateGeneration={updateGeneration} updateEdit={updateEdit} updateFace={updateFace} updateRuntime={updateRuntime} onChoosePoseControlLora={onChoosePoseControlLora} onChooseDepthCheckpoint={onChooseDepthCheckpoint} onChooseDepthImage={onChooseDepthImage} onPreviewPoseControl={onPreviewPoseControl} poseControlLoraAvailable={poseControlLoraAvailable} depthControlAvailable={depthControlAvailable} poseControlCompatibility={poseControlCompatibility} onInspectPoseControlLegacy={onInspectPoseControlLegacy} onChooseUpscaleModel={onChooseUpscaleModel} onPreviewUnifiedPrompt={onPreviewUnifiedPrompt} workerMemory={workerMemory} memoryRefreshing={memoryRefreshing} memoryActionsDisabled={memoryActionsDisabled} onRefreshMemory={onRefreshMemory} onReleaseMemory={onReleaseMemory} enabledMannequinCount={regions.filter((region) => region.layer === "generation" && region.enabled && region.regionType === "subject" && region.pose?.enabled).length} />}
      </div>
    </aside>
  );
}

function FaceSelectionPanel({ faces, selectedFaceIndices, manualFacePaths, lassoMode, onDetect, onToggle, onSelectAll, onLassoMode, onUndoLasso, onClearLassos, onUseLatest }: {
  faces: DetectedFaceRecord[];
  selectedFaceIndices: number[];
  manualFacePaths: number[][][];
  lassoMode: boolean;
  onDetect: () => void;
  onToggle: (index: number) => void;
  onSelectAll: (selected: boolean) => void;
  onLassoMode: (enabled: boolean) => void;
  onUndoLasso: () => void;
  onClearLassos: () => void;
  onUseLatest: () => void;
}) {
  return <div className="inspector-section face-selection-panel">
    <div className="section-inline-title"><span>Detected faces</span><span className="active-pill">{selectedFaceIndices.length} selected</span></div>
    <div className="inline-actions">
      <button className="tiny-button" onClick={onDetect}>Detect faces</button>
      <button className="tiny-button" onClick={onUseLatest}>Use latest first pass</button>
      <button className="tiny-button" onClick={() => onSelectAll(true)}>Select all</button>
      <button className="tiny-button" onClick={() => onSelectAll(false)}>Select none</button>
    </div>
    <div className="region-list face-list">{faces.map((face) => <button className={`region-list-row ${selectedFaceIndices.includes(face.index) ? "selected" : ""}`} key={face.index} onClick={() => onToggle(face.index)}>
      <span className="region-swatch">{face.index + 1}</span>
      <span className="region-list-copy"><strong>Face {face.index + 1}</strong><small>Confidence {face.score.toFixed(3)} · {Math.round(face.box[2] - face.box[0])} × {Math.round(face.box[3] - face.box[1])}</small></span>
      <input type="checkbox" readOnly checked={selectedFaceIndices.includes(face.index)} />
    </button>)}</div>
    {faces.length === 0 && <div className="empty-inspector"><Icon name="face" /><span>Choose a cloud source, then detect faces.</span></div>}
    <SectionTitle text="Manual face lassos" />
    <p className="field-help">Enable lasso drawing, then drag a closed path around each additional face on the image.</p>
    <div className="inline-actions">
      <button className={`tiny-button ${lassoMode ? "active" : ""}`} onClick={() => onLassoMode(!lassoMode)}>{lassoMode ? "Drawing lasso…" : "Draw lasso"}</button>
      <button className="tiny-button" disabled={manualFacePaths.length === 0} onClick={onUndoLasso}>Undo lasso</button>
      <button className="tiny-button" disabled={manualFacePaths.length === 0} onClick={onClearLassos}>Clear lassos</button>
    </div>
    <p className="field-help">{manualFacePaths.length} manual lasso{manualFacePaths.length === 1 ? "" : "s"} prepared.</p>
  </div>;
}

function LoraPanel({ activeLayer, regions, loras, onLoras, onChoose }: { activeLayer: RegionLayer; regions: RegionBox[]; loras: StudioLora[]; onLoras: (items: StudioLora[]) => void; onChoose: () => void }) {
  const bindingKey = activeLayer;
  function update(id: string, patch: Partial<StudioLora>) { onLoras(loras.map((lora) => lora.id === id ? { ...lora, ...patch } : lora)); }
  function updateBinding(lora: StudioLora, patch: Partial<LoraLayerBinding>) {
    update(lora.id, { [bindingKey]: { ...lora[bindingKey], ...patch } });
  }
  function toggleRegion(lora: StudioLora, regionId: string, checked: boolean) {
    const binding = lora[bindingKey];
    const regionIds = checked
      ? [...binding.regionIds, regionId]
      : binding.regionIds.filter((id) => id !== regionId);
    if (regionIds.length > 0) {
      updateBinding(lora, { enabled: true, global: false, regionIds });
    } else if (activeLayer === "generation") {
      updateBinding(lora, { enabled: true, global: true, regionIds: [], routingMode: "standard" });
    } else {
      updateBinding(lora, { enabled: false, global: false, regionIds: [] });
    }
  }
  return <div className="inspector-section lora-panel">
    <div className="section-inline-title"><span>LoRA library</span><button className="tiny-button" onClick={onChoose}><Icon name="plus" /> Add cloud LoRA</button></div>
    {loras.map((lora) => {
      const binding = lora[bindingKey];
      return <div className={`lora-card ${!lora.active ? "inactive" : ""}`} key={lora.id}>
        <div className="lora-title-row"><label className="toggle"><input type="checkbox" checked={lora.active} onChange={(event) => update(lora.id, { active: event.target.checked })} /><span /></label><div><strong>{lora.name}</strong><small>{binding.enabled ? binding.global ? "Global" : `${binding.regionIds.length} region(s)` : "Not used on this layer"}</small></div><button className="icon-button danger" onClick={() => onLoras(loras.filter((item) => item.id !== lora.id))}><Icon name="trash" /></button></div>
        <LinkedValue label="Strength" value={lora.strength} min={-4} max={4} step={0.05} onChange={(strength) => update(lora.id, { strength })} />
        {activeLayer !== "generation" && <label className="check-row compact-check"><input type="checkbox" checked={binding.enabled} onChange={(event) => updateBinding(lora, { enabled: event.target.checked, global: event.target.checked ? (binding.global || binding.regionIds.length === 0) : false, routingMode: event.target.checked && binding.regionIds.length === 0 ? "standard" : binding.routingMode })} /><span><strong>Use on this layer</strong></span></label>}
        {binding.enabled && <>
          <label className="check-row compact-check"><input type="checkbox" checked={binding.global} onChange={(event) => updateBinding(lora, { global: event.target.checked, regionIds: event.target.checked ? [] : binding.regionIds, routingMode: event.target.checked ? "standard" : binding.routingMode })} /><span><strong>Global</strong></span></label>
          {!binding.global && <div className="region-assignment-list">{regions.map((region) => <label className="check-row compact-check" key={region.id}><input type="checkbox" checked={binding.regionIds.includes(region.id)} onChange={(event) => toggleRegion(lora, region.id, event.target.checked)} /><span>{region.name}</span></label>)}</div>}
          <label className="field-label">Routing</label><select className="select-input compact-select" value={binding.routingMode} disabled={binding.global || binding.regionIds.length === 0} onChange={(event) => updateBinding(lora, { routingMode: event.target.value as LoraLayerBinding["routingMode"] })}><option value="standard">Standard regional</option><option value="character_identity">Character identity (face)</option></select>
          {binding.routingMode === "character_identity" && !binding.global && <><label className="field-label">Training trigger</label><input className="text-input compact-input" value={binding.triggerPhrase} placeholder="For example lface" onChange={(event) => updateBinding(lora, { triggerPhrase: event.target.value })} onBlur={() => { if (!binding.triggerPhrase.trim()) updateBinding(lora, { triggerPhrase: defaultLoraTrigger(lora.name) }); }} /><p className="field-help">Inserted automatically into the assigned region identity anchor; do not duplicate it in the visible prompt.</p></>}
        </>}
      </div>;
    })}
    {loras.length === 0 && <div className="drop-zone"><Icon name="upload" /><span>Add an uploaded `.safetensors` file from Cloud files.</span></div>}
  </div>;
}

function AdvancedPanel({ mode, activeLayer, settings, updateGeneration, updateEdit, updateFace, updateRuntime, onChoosePoseControlLora, onChooseDepthCheckpoint, onChooseDepthImage, onPreviewPoseControl, poseControlLoraAvailable, depthControlAvailable, poseControlCompatibility, onInspectPoseControlLegacy, onChooseUpscaleModel, onPreviewUnifiedPrompt, workerMemory, memoryRefreshing, memoryActionsDisabled, onRefreshMemory, onReleaseMemory, enabledMannequinCount }: {
  mode: StudioMode; activeLayer: RegionLayer; settings: StudioSettings;
  updateGeneration: (patch: Partial<GenerationSettings>) => void;
  updateEdit: (patch: Partial<StudioSettings["edit"]>) => void;
  updateFace: (patch: Partial<StudioSettings["face"]>) => void;
  updateRuntime: (patch: Partial<StudioSettings["runtime"]>) => void;
  onChoosePoseControlLora: () => void;
  onChooseDepthCheckpoint: () => void;
  onChooseDepthImage: () => void;
  onPreviewPoseControl: () => void;
  poseControlLoraAvailable: boolean;
  depthControlAvailable: boolean;
  poseControlCompatibility: KreaControlCheckpointInspection | null;
  onInspectPoseControlLegacy: (allow: boolean) => void;
  onChooseUpscaleModel: () => void;
  onPreviewUnifiedPrompt: () => void;
  workerMemory: WorkerMemoryStatus | null;
  memoryRefreshing: boolean;
  memoryActionsDisabled: boolean;
  onRefreshMemory: () => void;
  onReleaseMemory: () => void;
  enabledMannequinCount: number;
}) {
  const generation = settings.generation;
  const edit = settings.edit;
  const face = settings.face;
  if (mode === "face") return <div className="inspector-section advanced-panel">
    <NumberGrid items={[
      ["Steps", face.steps, 1, 100, 1, (steps) => updateFace({ steps })], ["Seed", face.seed, 0, 2147483647, 1, (seed) => updateFace({ seed })],
      ["Denoise", face.denoise, 0.05, 1, 0.05, (denoise) => updateFace({ denoise })], ["Padding", face.padding, 1, 4, 0.1, (padding) => updateFace({ padding })],
      ["Edge feather", face.feather, 0, 0.5, 0.02, (feather) => updateFace({ feather })], ["Blend", face.blend, 0, 1, 0.05, (blend) => updateFace({ blend })],
      ["Regional LoRA scale", face.loraScale, 0, 4, 0.05, (loraScale) => updateFace({ loraScale })], ["Detector threshold", face.detectorThreshold, 0.05, 0.95, 0.05, (detectorThreshold) => updateFace({ detectorThreshold })],
    ]} />
    <Choice label="Crop working resolution" value={face.cropSize} options={[[256, "256 px"], [512, "512 px"], [768, "768 px"], [1024, "1024 px"]]} onChange={(value) => updateFace({ cropSize: value as typeof face.cropSize })} />
    <Choice label="Detector device" value={face.detectorProvider} options={[["auto", "Auto (CUDA when available)"], ["cpu", "CPU"], ["cuda", "NVIDIA CUDA"]]} onChange={(value) => updateFace({ detectorProvider: value as typeof face.detectorProvider })} />
    <MemoryControls settings={settings} updateRuntime={updateRuntime} status={workerMemory} refreshing={memoryRefreshing} actionsDisabled={memoryActionsDisabled} onRefresh={onRefreshMemory} onRelease={onReleaseMemory} />
  </div>;
  const values = mode === "generation" ? generation : edit;
  const update = mode === "generation" ? updateGeneration : updateEdit;
  return <div className="inspector-section advanced-panel">
    <div className="settings-grid">
      <Choice label="Sampler" value={values.sampler} options={COMFYUI_SAMPLERS.map((value) => [value, value])} onChange={(sampler) => update({ sampler })} />
      <Choice label="Scheduler" value={values.scheduler} options={COMFYUI_SCHEDULERS.map((value) => [value, value])} onChange={(scheduler) => update({ scheduler })} />
    </div>
    {mode === "generation" ? <div className="settings-grid">
      <LinkedValue label="Steps" value={generation.steps} min={1} max={100} step={1} onChange={(steps) => updateGeneration({ steps })} />
      <SeedField
        value={generation.seed}
        mode={generation.seedMode}
        batchMode={generation.batchMode}
        onSeed={(seed) => updateGeneration({ seed })}
        onMode={(seedMode) => updateGeneration({ seedMode })}
      />
      <LinkedValue label="Width" value={generation.width} min={256} max={4096} step={16} onChange={(width) => updateGeneration({ width })} />
      <LinkedValue label="Height" value={generation.height} min={256} max={4096} step={16} onChange={(height) => updateGeneration({ height })} />
      <LinkedValue label="Inside boost" value={generation.insideBoost} min={0.1} max={10} step={0.1} onChange={(insideBoost) => updateGeneration({ insideBoost })} />
      <LinkedValue label="Outside penalty" value={generation.outsidePenalty} min={0} max={10} step={0.1} onChange={(outsidePenalty) => updateGeneration({ outsidePenalty })} />
      <LinkedValue label="Spatial falloff" value={generation.spatialFalloff} min={0} max={2048} step={16} onChange={(spatialFalloff) => updateGeneration({ spatialFalloff })} />
      <LinkedValue label="Late-step scale" value={generation.lateStepScale} min={0} max={1} step={0.05} onChange={(lateStepScale) => updateGeneration({ lateStepScale })} />
    </div> : <NumberGrid items={[
      ["Steps", edit.steps, 1, 100, 1, (steps) => updateEdit({ steps })], ["Seed · fixed", edit.seed, 0, 2147483647, 1, (seed) => updateEdit({ seed })],
      ["Denoise", edit.denoise, 0.05, 1, 0.05, (denoise) => updateEdit({ denoise })], ["Reference retention", edit.referenceRetention, 0, 1, 0.05, (referenceRetention) => updateEdit({ referenceRetention })],
      ["Latent feather", edit.latentFeather, 0, 256, 1, (latentFeather) => updateEdit({ latentFeather })], ["Composite feather", edit.compositeFeather, 0, 256, 1, (compositeFeather) => updateEdit({ compositeFeather })],
      ["Inside boost", edit.insideBoost, 0.1, 10, 0.1, (insideBoost) => updateEdit({ insideBoost })], ["Outside penalty", edit.outsidePenalty, 0, 10, 0.1, (outsidePenalty) => updateEdit({ outsidePenalty })],
      ["Spatial falloff", edit.spatialFalloff, 0, 2048, 1, (spatialFalloff) => updateEdit({ spatialFalloff })], ["Late-step scale", edit.lateStepScale, 0, 1, 0.01, (lateStepScale) => updateEdit({ lateStepScale })],
    ]} />}
    {mode === "generation" && <>
      <SectionTitle text="GPU memory" />
      <Choice
        label="Execution mode"
        value={settings.runtime.vramMode}
        options={[
          ["auto", "Auto (High VRAM at 40+ GiB)"],
          ["high_vram", "High VRAM · maximum performance"],
          ["dynamic", "Dynamic VRAM · balanced"],
          ["low_vram", "Low VRAM · maximum offload"],
        ]}
        onChange={(vramMode) => updateRuntime({ vramMode: vramMode as VramMode })}
      />
      <LinkedValue
        label="VRAM reserve · GiB"
        value={settings.runtime.reserveVramGb}
        min={0.5}
        max={16}
        step={0.5}
        onChange={(reserveVramGb) => updateRuntime({ reserveVramGb })}
      />
      <Check
        label="Keep baseline model loaded between runs"
        checked={settings.runtime.keepModelLoaded}
        onChange={(keepModelLoaded) => updateRuntime({ keepModelLoaded })}
      />
      <p className="field-help">
        With this option enabled, resolved High VRAM mode retains the worker and baseline
        transformer only while the configured reserve remains free. Other modes, memory pressure,
        model changes, OOM recovery, and Release memory safely discard the cache.
      </p>
    </>}
    <MemoryControls settings={settings} updateRuntime={updateRuntime} status={workerMemory} refreshing={memoryRefreshing} actionsDisabled={memoryActionsDisabled} onRefresh={onRefreshMemory} onRelease={onReleaseMemory} />
    {mode === "generation" && <Check label="Run generation in batch mode" checked={generation.batchMode} onChange={(batchMode) => updateGeneration({ batchMode, seedMode: batchMode && generation.seedMode === "fixed" ? "random" : generation.seedMode })} />}
    {mode === "generation" && generation.batchMode && <LinkedValue label="Batch runs" value={generation.batchCount} min={1} max={100} step={1} onChange={(batchCount) => updateGeneration({ batchCount })} />}
    {mode === "generation" && <Check label="Use unified spatial prompting" checked={generation.regionalPrompting} onChange={(regionalPrompting) => updateGeneration({ regionalPrompting })} />}
    {mode === "generation" && <button className="quiet-button full-button" onClick={onPreviewUnifiedPrompt}>Preview unified prompt…</button>}
    {mode === "generation" && depthControlAvailable && <>
      <SectionTitle text="Global depth control" />
      <Check
        label="Use a grayscale depth map"
        checked={generation.depth.enabled}
        onChange={(enabled) => updateGeneration({
          depth: { ...generation.depth, enabled },
          poseControlLoraEnabled: enabled ? false : generation.poseControlLoraEnabled,
        })}
      />
      {generation.depth.enabled && <>
        <button className="quiet-button full-button" onClick={onChooseDepthCheckpoint}>
          {generation.depth.checkpointName || "Choose Krea depth adapter checkpoint…"}
        </button>
        <button className="quiet-button full-button" onClick={onChooseDepthImage}>
          {generation.depth.imageName || "Choose grayscale PNG or TIFF depth map…"}
        </button>
        <Choice
          label="Normalization"
          value={generation.depth.normalization}
          options={[["minmax", "Min/max"], ["percentile", "Percentile"], ["none", "Preserve encoded range"]]}
          onChange={(normalization) => updateGeneration({ depth: { ...generation.depth, normalization: normalization as GenerationSettings["depth"]["normalization"] } })}
        />
        <Check
          label="Invert depth convention"
          checked={generation.depth.invert}
          onChange={(invert) => updateGeneration({ depth: { ...generation.depth, invert } })}
        />
        <div className="settings-grid">
          <LinkedValue label="Global strength" value={generation.depth.globalStrength} min={0} max={3} step={0.05} onChange={(globalStrength) => updateGeneration({ depth: { ...generation.depth, globalStrength } })} />
          <LinkedValue label="Mask feather" value={generation.depth.featherPixels} min={0} max={2048} step={8} onChange={(featherPixels) => updateGeneration({ depth: { ...generation.depth, featherPixels } })} />
          <LinkedValue label="Start" value={generation.depth.startPercent} min={0} max={generation.depth.endPercent} step={0.05} onChange={(startPercent) => updateGeneration({ depth: { ...generation.depth, startPercent } })} />
          <LinkedValue label="End" value={generation.depth.endPercent} min={generation.depth.startPercent} max={1} step={0.05} onChange={(endPercent) => updateGeneration({ depth: { ...generation.depth, endPercent } })} />
          <LinkedValue label="Gamma" value={generation.depth.gamma} min={0.1} max={4} step={0.05} onChange={(gamma) => updateGeneration({ depth: { ...generation.depth, gamma } })} />
        </div>
        <p className="field-help">The public adapter expects inverse depth: near objects white, far objects black. Use inversion only when the uploaded preview uses the opposite convention.</p>
      </>}
    </>}
    {mode === "generation" && <>
      <SectionTitle text="Volumetric pose gating" />
      <PoseGatingControls
        generation={generation}
        hasEnabledMannequin={enabledMannequinCount > 0}
        subjectCount={enabledMannequinCount}
        onChooseControlLora={onChoosePoseControlLora}
        onPreviewControl={onPreviewPoseControl}
        controlLoraAvailable={poseControlLoraAvailable}
        compatibility={poseControlCompatibility}
        onInspectLegacy={onInspectPoseControlLegacy}
        onChange={updateGeneration}
      />
      <p className="field-help">Filled mannequins define early denoising volumes and subject ownership. Ordinary region boxes remain available for objects and backgrounds and never add mannequin gating.</p>
    </>}
    <Check label="Separate overlapping subject targets" checked={values.subjectCompetition} onChange={(subjectCompetition) => update({ subjectCompetition })} />
    <Check label="Make subjects fill their boxes" checked={values.subjectFill} onChange={(subjectFill) => update({ subjectFill })} />
    {mode === "generation" && <Check label="Relax spatial guidance during late steps" checked={generation.relaxation} onChange={(relaxation) => updateGeneration({ relaxation })} />}
    <Check label="Adapt spatial guidance from regional LoRA delta" checked={values.loraAdaptation} onChange={(loraAdaptation) => update({ loraAdaptation })} />
    {values.loraAdaptation && <LinkedValue label="LoRA delta response" value={values.loraResponse} min={0} max={1} step={0.05} onChange={(loraResponse) => update({ loraResponse })} />}
    {mode === "edit" && <><Check label="Preserve reference identity" checked={edit.preserveIdentity} onChange={(preserveIdentity) => updateEdit({ preserveIdentity })} /><Check label="Edit entire image" checked={edit.editEntireImage} onChange={(editEntireImage) => updateEdit({ editEntireImage })} /></>}
    {mode === "generation" && <><SectionTitle text="Post-upscale" /><Check label="Post-upscale after releasing Krea VRAM" checked={generation.postUpscale} onChange={(postUpscale) => updateGeneration({ postUpscale })} />{generation.postUpscale && <><Choice label="Output scale" value={generation.upscaleScale} options={[[2, "2×"], [4, "4×"]]} onChange={(upscaleScale) => updateGeneration({ upscaleScale: upscaleScale as 2 | 4 })} /><Choice label="Upscaler" value={generation.upscaleMethod} options={[["lanczos", "CPU Lanczos"], ["model", "Neural model (tiled GPU)"]]} onChange={(upscaleMethod) => updateGeneration({ upscaleMethod: upscaleMethod as GenerationSettings["upscaleMethod"] })} />{generation.upscaleMethod === "model" && <button className="quiet-button full-button" onClick={onChooseUpscaleModel}>{generation.upscaleModelName || "Choose cloud upscaler model…"}</button>}</>}</>}
    {(mode === "generation" || activeLayer === "reference") && <ProjectorPanel projector={mode === "generation" ? generation.projector : edit.referenceProjector} onChange={(projector) => mode === "generation" ? updateGeneration({ projector }) : updateEdit({ referenceProjector: projector })} />}
  </div>;
}

function MemoryControls({ settings, updateRuntime, status, refreshing, actionsDisabled, onRefresh, onRelease }: {
  settings: StudioSettings;
  updateRuntime: (patch: Partial<StudioSettings["runtime"]>) => void;
  status: WorkerMemoryStatus | null;
  refreshing: boolean;
  actionsDisabled: boolean;
  onRefresh: () => void;
  onRelease: () => void;
}) {
  return <>
    <SectionTitle text="Pod RAM diagnostics" />
    <Check
      label="Enable system RAM safeguard"
      checked={settings.runtime.systemRamGuardEnabled}
      onChange={(systemRamGuardEnabled) => updateRuntime({ systemRamGuardEnabled })}
    />
    {!settings.runtime.systemRamGuardEnabled && <div className="memory-warning">
      K2 RAM preflight checks are disabled. Linux cgroup limits and the kernel OOM killer still apply,
      so the worker may be terminated if real resident memory is exhausted.
    </div>}
    {status ? <div className="memory-diagnostics">
      <div><span>Allocatable now</span><strong>{formatGib(status.allocatable_bytes)}</strong></div>
      <div><span>Actual non-cache use</span><strong>{formatGib(status.non_reclaimable_bytes)}</strong></div>
      <div><span>Clean reclaimable cache</span><strong>{formatGib(status.reclaimable_file_bytes)}</strong></div>
      <div><span>Raw cgroup charge</span><strong>{formatGib(status.current_bytes)} / {formatGib(status.total_bytes)}</strong></div>
      <div><span>Anonymous/process RAM</span><strong>{formatGib(status.anonymous_bytes)}</strong></div>
      <div><span>Shared + dirty files</span><strong>{formatGib((status.shared_memory_bytes ?? 0) + (status.dirty_file_bytes ?? 0) + (status.writeback_file_bytes ?? 0))}</strong></div>
      <div className="memory-worker-state"><span>Worker</span><strong>{status.worker_resident ? "Baseline resident" : status.worker_active ? "Active" : "Released"}</strong></div>
    </div> : <p className="field-help">Pod RAM telemetry is unavailable until the workspace agent is ready.</p>}
    <p className="field-help">
      Raw cgroup charge includes filesystem cache. K2 uses actual non-cache use for safeguards;
      clean cache is evictable and does not count against allocatable RAM.
    </p>
    <div className="inline-actions memory-actions">
      <button className="tiny-button" disabled={actionsDisabled || refreshing} onClick={onRefresh}>{refreshing ? "Refreshing…" : "Refresh RAM"}</button>
      <button className="tiny-button" disabled={actionsDisabled} onClick={onRelease}>Release worker memory</button>
    </div>
  </>;
}

function formatGib(value: number | null): string {
  if (value === null) return "Unavailable";
  return `${(value / 1024 ** 3).toFixed(2)} GiB`;
}

function ProjectorPanel({ projector, onChange }: { projector: ProjectorSettings; onChange: (value: ProjectorSettings) => void }) {
  return <><SectionTitle text="Projector" /><Check label="Apply global projector vector" checked={projector.enabled} onChange={(enabled) => onChange({ ...projector, enabled })} />{projector.enabled && <>
    <Choice label="Preset" value={projector.preset} options={[["filter_bypass2", "FilterBypass2"], ["filter_bypass3", "FilterBypass3"], ["skc3vo", "skc3vo"], ["z0jglf", "z0jglf"], ["custom", "Custom values"]]} onChange={(preset) => onChange({ ...projector, preset, values: PROJECTOR_PRESETS[preset] ? [...PROJECTOR_PRESETS[preset]] : projector.values })} />
    <div className="projector-grid">{projector.values.map((value, index) => <DraftNumberInput key={index} ariaLabel={`Projector vector ${index + 1}`} step={0.0001} min={-1000} max={1000} value={value} onCommit={(next) => { const values = [...projector.values]; values[index] = next; onChange({ ...projector, preset: "custom", values }); }} />)}</div>
    <LinkedValue label="Global multiplier" value={projector.multiplier} min={-20} max={20} step={0.1} onChange={(multiplier) => onChange({ ...projector, multiplier })} />
    <LinkedValue label="Face identity protection" value={projector.identityProtection} min={0} max={1} step={0.05} onChange={(identityProtection) => onChange({ ...projector, identityProtection })} />
  </>}</>;
}

type NumberItem = [string, number, number, number, number, (value: number) => void];
function NumberGrid({ items }: { items: NumberItem[] }) { return <div className="settings-grid">{items.map(([label, value, min, max, step, onChange]) => <LinkedValue key={label} label={label} value={value} min={min} max={max} step={step} onChange={onChange} />)}</div>; }
function SeedField({ value, mode, batchMode, onSeed, onMode }: {
  value: number;
  mode: GenerationSettings["seedMode"];
  batchMode: boolean;
  onSeed: (value: number) => void;
  onMode: (value: GenerationSettings["seedMode"]) => void;
}) {
  const options: readonly (readonly [GenerationSettings["seedMode"], string])[] = batchMode
    ? [["random", "Random"], ["increment", "Increment"]]
    : [["fixed", "Fixed"], ["random", "Random"], ["increment", "Increment"]];
  return <label className="seed-field">
    <span>Seed</span>
    <span className="seed-controls">
      <DraftNumberInput ariaLabel="Seed value" className="seed-input" value={value} min={0} max={2147483647} step={1} onCommit={onSeed} />
      <select aria-label="Seed behavior" className="select-input" value={mode} onChange={(event) => onMode(event.target.value as GenerationSettings["seedMode"])}>
        {options.map(([option, label]) => <option key={option} value={option}>{label}</option>)}
      </select>
    </span>
  </label>;
}
function Choice({ label, value, options, onChange }: { label: string; value: string | number; options: readonly (readonly [string | number, string])[]; onChange: (value: never) => void }) { return <label className="choice-field"><span>{label}</span><select className="select-input" value={value} onChange={(event) => { const match = options.find(([candidate]) => String(candidate) === event.target.value); if (match) onChange(match[0] as never); }}>{options.map(([option, text]) => <option key={option} value={option}>{text}</option>)}</select></label>; }
function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) { return <label className="check-row compact-check"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span><strong>{label}</strong></span></label>; }
function SectionTitle({ text }: { text: string }) { return <div className="advanced-section-title">{text}</div>; }
function LinkedValue({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  function change(next: number) { if (Number.isFinite(next)) onChange(Math.max(min, Math.min(max, next))); }
  return <div className="linked-value"><div className="linked-label"><span>{label}</span><DraftNumberInput value={value} min={min} max={max} step={step} onCommit={change} /></div><input className="range-input" type="range" value={value} min={min} max={max} step={step} onChange={(event) => change(Number(event.target.value))} /></div>;
}
