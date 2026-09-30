import { COMFYUI_SAMPLERS, COMFYUI_SCHEDULERS, type Image2ImageSettings } from "../studioProject";
import { DraftNumberInput } from "./DraftNumberInput";

export function Image2ImagePanel({ settings, onChange, onChooseLora }: {
  settings: Image2ImageSettings;
  onChange: (settings: Image2ImageSettings) => void;
  onChooseLora: () => void;
}) {
  const update = (patch: Partial<Image2ImageSettings>) => onChange({ ...settings, ...patch });
  function number(label: string, key: keyof Image2ImageSettings, min: number, max: number, step = 1) {
    return <label className="field-label" key={key}>{label}
      <DraftNumberInput value={Number(settings[key])} min={min} max={max} step={step}
        onCommit={(value) => update({ [key]: key === "width" || key === "height" ? Math.round(value / 32) * 32 : value })} />
    </label>;
  }
  return <aside className="inspector image2image-inspector">
    <div className="inspector-head"><div><p className="kicker">Instruction editing</p><h2>image2image</h2></div></div>
    <div className="inspector-content">
      <label className="field-label">Edit instruction
        <textarea className="prompt-input" rows={6} value={settings.prompt}
          placeholder="Describe what should change in the source image…"
          onChange={(event) => update({ prompt: event.target.value })} />
      </label>
      <button className="quiet-button" onClick={onChooseLora}>Choose Identity Edit LoRA</button>
      <p className="field-help">{settings.identity_lora_name || "Select krea2_identity_edit_v1_2.safetensors from Assets › LoRAs."}</p>
      <p className="field-help">Single source image. Uses image-grounded instructions and clean reference tokens. Regional Edit settings do not apply.</p>
      {number("Output width", "width", 256, Math.floor(Math.min(2048, 2097152 / settings.height) / 32) * 32, 32)}
      {number("Output height", "height", 256, Math.floor(Math.min(2048, 2097152 / settings.width) / 32) * 32, 32)}
      <p className="field-help">Dimensions must be multiples of 32, at most 2 megapixels total.</p>
      {number("Steps", "steps", 1, 100)}
      {number("Seed", "seed", 0, 2147483647)}
      {number("CFG", "cfg", 1, 10, 0.1)}
      <p className="field-help">Turbo: 8 steps / CFG 1. For removals, select Raw in Setup and try 20 steps / CFG 3. This tab uses the model selected in Setup.</p>
      {number("Identity Edit LoRA strength", "lora_strength", 0.01, 2, 0.05)}
      {number("Grounding resolution", "grounding_px", 384, 1536, 64)}
      {number("Reference fidelity", "ref_boost", 0, 10, 0.1)}
      <label className="field-label">Source fit
        <select value={settings.fit_mode} onChange={(event) => update({ fit_mode: event.target.value as "fit" | "crop" })}>
          <option value="fit">Fit · v1.2</option><option value="crop">Center crop · legacy</option>
        </select>
      </label>
      <label className="field-label">Sampler<select value={settings.sampler} onChange={(event) => update({ sampler: event.target.value })}>
        {COMFYUI_SAMPLERS.map((value) => <option key={value}>{value}</option>)}
      </select></label>
      <label className="field-label">Scheduler<select value={settings.scheduler} onChange={(event) => update({ scheduler: event.target.value })}>
        {COMFYUI_SCHEDULERS.map((value) => <option key={value}>{value}</option>)}
      </select></label>
    </div>
  </aside>;
}
