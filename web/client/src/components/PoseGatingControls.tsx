import { useEffect, useRef, useState } from "react";
import type {
  GenerationSettings,
  PoseSemanticMode,
  PoseSigmaMode,
  PoseSoftRelease,
} from "../studioProject";
import type { KreaControlCheckpointInspection } from "../api";
import {
  poseGateStrengths,
  poseKnotPhase,
  resamplePoseKnots,
} from "../poseGating";
import { DraftNumberInput } from "./DraftNumberInput";

interface Props {
  generation: GenerationSettings;
  hasEnabledMannequin: boolean;
  subjectCount: number;
  onChooseControlLora: () => void;
  onPreviewControl: () => void;
  controlLoraAvailable: boolean;
  compatibility: KreaControlCheckpointInspection | null;
  onInspectLegacy: (allow: boolean) => void;
  onChange: (patch: Partial<GenerationSettings>) => void;
}

export function PoseGatingControls({
  generation,
  hasEnabledMannequin,
  subjectCount,
  onChooseControlLora,
  onPreviewControl,
  controlLoraAvailable,
  compatibility,
  onInspectLegacy,
  onChange,
}: Props) {
  const hard = generation.poseHardGateSteps;
  const soft = generation.poseSoftGateSteps;
  const normal = generation.steps;
  const total = hard + soft + normal;
  const [dragKnot, setDragKnot] = useState<number | null>(null);
  const graphRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (
      generation.poseSigmaMode === "advanced"
      && generation.poseSigmaKnots.length !== total + 1
    ) {
      onChange({
        poseSigmaKnots: resamplePoseKnots(generation.poseSigmaKnots, total),
      });
    }
  }, [generation.poseSigmaKnots, generation.poseSigmaMode, onChange, total]);

  const knots = generation.poseSigmaMode === "advanced"
    ? resamplePoseKnots(generation.poseSigmaKnots, total)
    : [];
  const gate = poseGateStrengths(hard, soft, normal, generation.poseSoftRelease);
  const normalShare = 1 - generation.poseSigmaHardShare - generation.poseSigmaSoftShare;
  const weightedValid = (
    generation.poseSigmaHardShare >= 0
    && generation.poseSigmaSoftShare >= 0
    && normalShare > 0
    && (hard > 0 || generation.poseSigmaHardShare === 0)
    && (soft > 0 || generation.poseSigmaSoftShare === 0)
  );

  function updateSteps(key: "poseHardGateSteps" | "poseSoftGateSteps", value: number) {
    const next = Math.max(0, Math.min(100, Math.trunc(value)));
    onChange({
      [key]: next,
      ...(key === "poseHardGateSteps" && next === 0 ? { poseSigmaHardShare: 0 } : {}),
      ...(key === "poseSoftGateSteps" && next === 0 ? { poseSigmaSoftShare: 0 } : {}),
    } as Partial<GenerationSettings>);
  }

  function setPreset(hardShare: number, softShare: number) {
    onChange({
      poseSigmaMode: "phase_weighted",
      poseSigmaHardShare: hard ? hardShare : 0,
      poseSigmaSoftShare: soft ? softShare : 0,
    });
  }

  function moveGraphKnot(event: React.PointerEvent<SVGSVGElement>) {
    if (dragKnot === null || dragKnot <= 0 || dragKnot >= total) return;
    const bounds = graphRef.current!.getBoundingClientRect();
    const raw = (event.clientY - bounds.top - 12) / Math.max(1, bounds.height - 24);
    const progress = Math.max(
      knots[dragKnot - 1] + 0.001,
      Math.min(knots[dragKnot + 1] - 0.001, raw),
    );
    const next = [...knots];
    next[dragKnot] = progress;
    onChange({ poseSigmaKnots: next });
  }

  return <div className="pose-gating-panel">
    <div className="pose-control-adapter">
      <label className="check-row">
        <input
          type="checkbox"
          disabled={!controlLoraAvailable}
          checked={generation.poseControlLoraEnabled}
          onChange={(event) => onChange({ poseControlLoraEnabled: event.target.checked })}
        />
        <span><strong>Enable trained pose adapter</strong></span>
      </label>
      {!controlLoraAvailable && <div className="memory-warning">
        Update the workspace image to worker protocol 4 before enabling the trained adapter.
      </div>}
      {generation.poseControlLoraEnabled && <>
        {!hasEnabledMannequin && <div className="memory-warning">
          Add and enable at least one subject mannequin before generating.
        </div>}
        <button className="quiet-button full-button" onClick={onChooseControlLora}>
          {generation.poseControlLoraModel || "Choose Krea pose adapter checkpoint…"}
        </button>
        {!generation.poseControlLoraFileId && <div className="memory-warning">
          Select a verified Krea volumetric pose adapter checkpoint.
        </div>}
        <label className="field-label">Adapter strength</label>
        <DraftNumberInput
          min={0}
          max={2}
          step={0.05}
          value={generation.poseControlLoraStrength}
          onCommit={(poseControlLoraStrength) => onChange({ poseControlLoraStrength })}
        />
        <p className="field-help">
          The trained Krea Control-LoRA teaches the model that the hidden head, torso,
          and left/right limb colors represent a human pose. It complements Prediction
          composite and the hard/soft spatial gate.
        </p>
        <button className="quiet-button full-button" onClick={onPreviewControl}>
          Preview control image…
        </button>
        <div className="semantic-forward-estimate">
          <strong>{compatibility?.compatible ? "Compatible" : compatibility ? "Incompatible" : "Not inspected"}</strong>
          <small>{generation.poseControlFormat}</small>
          {compatibility?.checkpoint && <>
            <small>Rank {compatibility.checkpoint.rank} · {compatibility.checkpoint.metadata.k2lab_base_model || "unverified base"}</small>
            <small>{compatibility.checkpoint.compatible_block_pairs} block targets · SHA-256 {compatibility.checkpoint.sha256}</small>
            <small>Renderer {compatibility.checkpoint.metadata.k2lab_renderer_version || "unverified"} · palette {compatibility.checkpoint.metadata.k2lab_control_format_sha256 || "unverified"}</small>
          </>}
        </div>
        {compatibility?.errors.map((error) => <div className="memory-warning" key={error}>{error}</div>)}
        {compatibility?.warnings.map((warning) => <div className="memory-warning" key={warning}>{warning}</div>)}
        {compatibility && !compatibility.verified && <label className="check-row warning-check">
          <input
            type="checkbox"
            checked={generation.poseControlLegacyAcknowledged}
            onChange={(event) => {
              onChange({ poseControlLegacyAcknowledged: event.target.checked });
              onInspectLegacy(event.target.checked);
            }}
          />
          <span><strong>Acknowledge unverified legacy checkpoint</strong><small>Only compatible tensor structure is accepted; metadata cannot be verified.</small></span>
        </label>}
      </>}
    </div>
    <label className="check-row">
      <input
        type="checkbox"
        checked={generation.poseGating}
        onChange={(event) => onChange({
          poseGating: event.target.checked,
          poseHardGateSteps: event.target.checked && hard + soft === 0 ? 2 : hard,
          poseSoftGateSteps: event.target.checked && hard + soft === 0 ? 2 : soft,
        })}
      />
      <span><strong>Constrain generation to subject mannequins</strong></span>
    </label>
    {generation.poseGating && <>
      {!hasEnabledMannequin && <div className="memory-warning">
        Add and enable at least one subject mannequin before generating.
      </div>}
      <label className="field-label">Subject semantic routing</label>
      <select
        className="select-input"
        value={generation.poseSemanticMode}
        onChange={(event) => onChange({ poseSemanticMode: event.target.value as PoseSemanticMode })}
      >
        <option value="prediction_composite">Prediction composite — recommended</option>
        <option value="attention_isolation">Attention isolation</option>
        <option value="spatial_only">⚠ Spatial only — legacy diagnostic</option>
      </select>
      {generation.poseSemanticMode === "prediction_composite" && <p className="field-help">
        Computes a clean subject-only prediction for every enabled mannequin during hard and soft steps, then fuses it into that subject&apos;s exclusive pose volume. Slower but strongest.
      </p>}
      {generation.poseSemanticMode === "attention_isolation" && <p className="field-help">
        Uses one unified prediction but blocks scene and other-subject attention into each mannequin during gated steps. Faster, but unified text encoding can still carry leakage.
      </p>}
      {generation.poseSemanticMode === "spatial_only" && <div className="memory-warning">
        Spatial only restricts where denoising occurs without restricting what the unified scene prompt generates there.
      </div>}
      {generation.poseSemanticMode === "prediction_composite" && <div className="semantic-forward-estimate">
        <strong>Estimated Euler model forwards: {normal} + ({hard} + {soft}) × (1 + {subjectCount}) = {normal + (hard + soft) * (1 + subjectCount)}</strong>
        <small>One sampler trajectory; subject predictions are evaluated sequentially.</small>
      </div>}
      <div className="number-grid pose-step-grid">
        <label><span>Hard gate steps</span><DraftNumberInput min={0} max={100} step={1} value={hard} onCommit={(value) => updateSteps("poseHardGateSteps", value)} /></label>
        <label><span>Soft gate steps</span><DraftNumberInput min={0} max={100} step={1} value={soft} onCommit={(value) => updateSteps("poseSoftGateSteps", value)} /></label>
        <label><span>Normal steps</span><input className="text-input" value={normal} readOnly /></label>
        <label><span>Effective total</span><input className="text-input" value={total} readOnly /></label>
      </div>
      <p className="phase-equation">Hard {hard} + Soft {soft} + Normal {normal} = <strong>{total} total transitions</strong></p>
      <label className="field-label">Soft release</label>
      <select
        className="select-input"
        value={generation.poseSoftRelease}
        onChange={(event) => onChange({ poseSoftRelease: event.target.value as PoseSoftRelease })}
      >
        <option value="cosine">Cosine</option>
        <option value="linear">Linear</option>
        <option value="exponential">Exponential</option>
        <option value="stepped">Stepped</option>
      </select>
      <label className="field-label">Sigma schedule</label>
      <select
        className="select-input"
        value={generation.poseSigmaMode}
        onChange={(event) => onChange({ poseSigmaMode: event.target.value as PoseSigmaMode })}
      >
        <option value="automatic">Scheduler default</option>
        <option value="phase_weighted">Phase weighted</option>
        <option value="advanced">Advanced curve</option>
      </select>
      {generation.poseSigmaMode !== "automatic" && <p className="memory-warning">
        Experimental: nonstandard Turbo sigma allocation can materially change image quality.
      </p>}
      {generation.poseSigmaMode === "phase_weighted" && <div className="sigma-weighted">
        <div className="inline-actions">
          <button className="tiny-button" onClick={() => setPreset(0.20, 0.30)}>Balanced</button>
          <button className="tiny-button" onClick={() => setPreset(0.25, 0.40)}>Pose lock</button>
          <button className="tiny-button" onClick={() => setPreset(0.15, 0.25)}>Gentle</button>
          <button className="tiny-button" onClick={() => onChange({ poseSigmaMode: "automatic" })}>Scheduler default</button>
        </div>
        <div className="number-grid pose-share-grid">
          <label><span>Hard %</span><DraftNumberInput disabled={hard === 0} min={0} max={100} step={1} value={generation.poseSigmaHardShare * 100} onCommit={(value) => onChange({ poseSigmaHardShare: hard ? value / 100 : 0 })} /></label>
          <label><span>Soft %</span><DraftNumberInput disabled={soft === 0} min={0} max={100} step={1} value={generation.poseSigmaSoftShare * 100} onCommit={(value) => onChange({ poseSigmaSoftShare: soft ? value / 100 : 0 })} /></label>
          <label><span>Normal %</span><input className="text-input" value={`${Math.round(normalShare * 1000) / 10}%`} readOnly /></label>
        </div>
        {!weightedValid && <div className="memory-warning">Trajectory shares must leave a positive normal phase and zero-step phases must have a zero share.</div>}
      </div>}
      {generation.poseSigmaMode === "advanced" && <div className="sigma-editor">
        <svg
          ref={graphRef}
          className="sigma-graph"
          viewBox="0 0 640 220"
          role="img"
          aria-label="Normalized sigma trajectory and pose gate strength"
          onPointerMove={moveGraphKnot}
          onPointerUp={() => setDragKnot(null)}
          onPointerCancel={() => setDragKnot(null)}
        >
          <rect className="phase-hard" x="12" y="12" width={616 * hard / total} height="196" />
          <rect className="phase-soft" x={12 + 616 * hard / total} y="12" width={616 * soft / total} height="196" />
          <rect className="phase-normal" x={12 + 616 * (hard + soft) / total} y="12" width={616 * normal / total} height="196" />
          <polyline
            className="sigma-line"
            points={knots.map((value, index) => `${12 + 616 * index / total},${12 + 196 * value}`).join(" ")}
          />
          <polyline
            className="gate-line"
            points={gate.map((value, index) => `${12 + 616 * (index + 0.5) / total},${208 - 196 * value}`).join(" ")}
          />
          {knots.map((value, index) => <circle
            key={index}
            className={`sigma-knot ${index === 0 || index === total ? "locked" : ""}`}
            cx={12 + 616 * index / total}
            cy={12 + 196 * value}
            r="6"
            tabIndex={index === 0 || index === total ? -1 : 0}
            aria-label={`Boundary ${index}: ${Math.round(value * 1000) / 10}% progress`}
            onPointerDown={(event) => {
              if (index === 0 || index === total) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              setDragKnot(index);
            }}
          />)}
        </svg>
        <div className="sigma-knot-table">
          {knots.map((value, index) => <label key={index}>
            <span>{index} · {index === total ? "complete" : poseKnotPhase(index, hard, soft)}</span>
            <DraftNumberInput
              disabled={index === 0 || index === total}
              min={index ? (knots[index - 1] + 0.001) * 100 : 0}
              max={index < total ? (knots[index + 1] - 0.001) * 100 : 100}
              step={0.1}
              value={value * 100}
              onCommit={(nextValue) => {
                const next = [...knots];
                next[index] = nextValue / 100;
                onChange({ poseSigmaKnots: next });
              }}
            />
          </label>)}
        </div>
      </div>}
    </>}
  </div>;
}
