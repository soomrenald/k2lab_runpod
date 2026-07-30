import assert from "node:assert/strict";
import {
  buildProjectDocument,
  bindStudioLoraFiles,
  createStudioLora,
  createStudioSettings,
  defaultLoraTrigger,
  loadStudioProjectDocument,
  projectDocumentFromPng,
} from "../src/studioProject.ts";
import { appendBoundedEvents, EVENT_LOG_LIMIT } from "../src/eventLog.ts";
import { readFile } from "node:fs/promises";
import { poseGateStrengths } from "../src/poseGating.ts";
import {
  promptEmphasisFromSelection,
  promptEmphasisMatches,
  reconcilePromptEmphases,
} from "../src/promptEmphasis.ts";
import { standingPose } from "../src/pose.ts";

assert.deepEqual(
  promptEmphasisFromSelection("red coat, red coat", 10, 18, "__global__"),
  { phrase: "red coat", occurrence: 1 },
);
assert.deepEqual(
  promptEmphasisFromSelection("  portrait subject.  ", 2, 19, "person"),
  { phrase: "portrait subject", occurrence: 0 },
);
assert.equal(
  promptEmphasisMatches(
    { scopeId: "__global__", phrase: "portrait", occurrence: 1 },
    "portrait reference",
  ),
  false,
);
assert.deepEqual(
  reconcilePromptEmphases(
    [
      { id: "stale-copy", scopeId: "__global__", phrase: "portrait", occurrence: 4 },
      { id: "deleted", scopeId: "__global__", phrase: "missing", occurrence: 0 },
    ],
    () => "portrait reference",
  ),
  [{ id: "stale-copy", scopeId: "__global__", phrase: "portrait", occurrence: 0 }],
);

const settings = createStudioSettings();
const newLora = createStudioLora("opaque-new-lora", "folder/character.safetensors");
assert.equal(defaultLoraTrigger("folder/character.safetensors"), "character");
assert.equal(newLora.generation.triggerPhrase, "character");
assert.equal(newLora.reference.triggerPhrase, "character");
assert.equal(newLora.targets.triggerPhrase, "character");
const poseGateFixtures = JSON.parse(await readFile(
  new URL("../../../tests/fixtures/pose_gate_strengths.json", import.meta.url),
  "utf8",
));
for (const fixture of poseGateFixtures) {
  assert.deepEqual(
    poseGateStrengths(fixture.hard, fixture.soft, fixture.normal, fixture.schedule),
    fixture.values,
  );
}
settings.generation.seed = 8123;
settings.generation.seedMode = "increment";
settings.generation.batchMode = true;
settings.generation.batchCount = 4;
settings.generation.poseGating = true;
settings.generation.poseSemanticMode = "attention_isolation";
settings.generation.sharedVisualPrompt = "cinematic 35 mm photography";
settings.generation.poseHardGateSteps = 3;
settings.generation.poseSoftGateSteps = 2;
settings.generation.poseSigmaMode = "phase_weighted";
settings.generation.poseControlLoraEnabled = true;
settings.generation.poseControlLoraFileId = "opaque-pose-adapter";
settings.generation.poseControlLoraModel = "krea-volumetric-pose-r64.safetensors";
settings.generation.poseControlLoraStrength = 0.85;
settings.generation.depth.enabled = true;
settings.generation.depth.checkpointFileId = "opaque-depth-adapter";
settings.generation.depth.checkpointName = "depth-control-lora.safetensors";
settings.generation.depth.imageFileId = "opaque-depth-image";
settings.generation.depth.imageName = "depth_16bit.png";
settings.generation.depth.globalStrength = 1.25;
settings.generation.depth.normalization = "percentile";
settings.generation.depth.gamma = 0.9;
settings.generation.promptEmphases = [{
  id: "not-persisted", scopeId: "person", phrase: "red coat", strength: 1.2, occurrence: 7,
}, {
  id: "stale-not-persisted", scopeId: "person", phrase: "missing phrase", strength: 0.4, occurrence: 0,
}];
settings.edit.width = 768;
settings.edit.height = 1152;
settings.edit.referencePromptEmphases = [{
  id: "not-persisted", scopeId: "__global__", phrase: "portrait", strength: 0.7, occurrence: 1,
}];
settings.face.cropSize = 768;
settings.runtime.vramMode = "high_vram";
settings.runtime.reserveVramGb = 1.5;
settings.runtime.systemRamGuardEnabled = false;
settings.runtime.filenamePrefix = "portrait study";
settings.runtime.diffusionModelName = "chosen-transformer.safetensors";
settings.runtime.textEncoderName = "chosen-text.safetensors";
settings.runtime.vaeName = "chosen-vae.safetensors";
settings.runtime.faceDetectorName = "chosen-detector.onnx";
const regions = [
  { id: "person", name: "Person", layer: "generation", x: 80, y: 40, width: 400, height: 900, prompt: "red coat", faceIdentityPrompt: "green eyes", spatialRole: "subject", regionType: "subject", pose: standingPose(), depthMode: "relax", depthStrength: 0.25, depthStartPercent: 0.1, depthEndPercent: 0.8, enabled: true },
  { id: "wall", name: "Wall", layer: "generation", x: 0, y: 0, width: 1024, height: 1024, prompt: "brick wall", faceIdentityPrompt: "", spatialRole: "background", regionType: "region", pose: null, enabled: true },
  { id: "reference", name: "Reference", layer: "reference", x: 20, y: 30, width: 300, height: 700, prompt: "portrait", faceIdentityPrompt: "same person", spatialRole: "subject", regionType: "region", pose: null, enabled: true },
  { id: "target", name: "Target", layer: "targets", x: 350, y: 300, width: 200, height: 250, prompt: "blue jacket", faceIdentityPrompt: "", spatialRole: "subject", regionType: "region", pose: null, enabled: true },
];
const loras = [{
  id: "not-persisted",
  fileId: "opaque-cloud-id",
  name: "character.safetensors",
  active: true,
  strength: 0.85,
  generation: { enabled: true, global: false, regionIds: ["person"], routingMode: "character_identity", triggerPhrase: "lface" },
  reference: { enabled: true, global: false, regionIds: ["reference"], routingMode: "standard", triggerPhrase: "" },
  targets: { enabled: false, global: false, regionIds: [], routingMode: "standard", triggerPhrase: "" },
}];
const prompts = { generation: "studio portrait", reference: "portrait reference", targets: "change clothing" };
const first = buildProjectDocument(regions, prompts, settings, loras, "source.png");
assert.deepEqual(first.generation.prompt_emphases, [{
  scope_id: "person", phrase: "red coat", strength: 1.2, occurrence: 0,
}]);
assert.deepEqual(first.image_edit.reference_prompt_emphases, [{
  scope_id: "__global__", phrase: "portrait", strength: 0.7, occurrence: 0,
}]);
const loaded = loadStudioProjectDocument(first);
const second = buildProjectDocument(loaded.regions, loaded.prompts, loaded.settings, loaded.loras, loaded.sourceName);
assert.deepEqual(second, first);
assert.equal(second.image_edit.width, 768);
assert.equal(second.runtime.vram_mode, "high_vram");
assert.equal(second.runtime.reserve_vram_gb, 1.5);
assert.equal(second.runtime.system_ram_guard_enabled, false);
assert.deepEqual(second.regions.map((region) => [region.id, region.priority, region.spatial_role]), [
  ["person", 2, "subject"], ["wall", 1, "background"],
]);
assert.equal(second.regions[0].region_type, "subject");
assert.equal(second.version, 24);
assert.equal(second.generation.depth_control.enabled, true);
assert.equal(second.generation.depth_control.checkpoint, "depth-control-lora.safetensors");
assert.equal(second.generation.depth_control.depth_image, "depth_16bit.png");
assert.equal(second.generation.depth_control.global_strength, 1.25);
assert.equal(second.generation.depth_control.normalization.mode, "percentile");
assert.equal(second.generation.depth_control.normalization.gamma, 0.9);
assert.deepEqual(second.generation.depth_control.regions[0], {
  region_id: "person",
  mode: "relax",
  strength: 0.25,
  start_percent: 0.1,
  end_percent: 0.8,
});
assert.equal(second.generation.pose_gating_enabled, true);
assert.equal(second.generation.pose_semantic_mode, "attention_isolation");
assert.equal(second.generation.shared_visual_prompt, "cinematic 35 mm photography");
assert.equal(second.generation.pose_hard_gate_steps, 3);
assert.equal(second.generation.pose_control_lora_enabled, true);
assert.equal(second.generation.pose_control_lora_model, "krea-volumetric-pose-r64.safetensors");
assert.equal(second.generation.pose_control_lora_strength, 0.85);
assert.equal(second.generation.pose_control_format, "k2-volumetric-pose-control-v1");
assert.equal("pose_control_lora_file_id" in second.generation, false);
assert.equal(second.regions[0].pose.format, "k2-volumetric-pose-v1");
assert.equal(Object.keys(second.regions[0].pose.joints).length, 13);
assert.ok(second.regions[0].pose.head.rx > 0);
assert.equal(second.regions[1].region_type, "region");
assert.equal(second.regions[1].pose, null);

const version23 = structuredClone(first);
version23.version = 23;
delete version23.generation.depth_control;
for (const region of version23.regions) {
  delete region.depth_mode;
  delete region.depth_strength;
  delete region.depth_start_percent;
  delete region.depth_end_percent;
}
const migrated23 = loadStudioProjectDocument(version23);
assert.equal(migrated23.settings.generation.depth.enabled, false);
assert.equal(migrated23.settings.generation.depth.checkpointName, "");
assert.equal(migrated23.settings.generation.depth.imageName, "");
assert.equal(migrated23.regions[0].depthMode, "inherit");
assert.equal(migrated23.regions[0].depthStrength, 1);

const version22 = structuredClone(first);
version22.version = 22;
delete version22.generation.pose_control_lora_enabled;
delete version22.generation.pose_control_lora_model;
delete version22.generation.pose_control_lora_strength;
delete version22.generation.pose_control_format;
const migrated22 = loadStudioProjectDocument(version22);
assert.equal(migrated22.settings.generation.poseControlLoraEnabled, false);
assert.equal(migrated22.settings.generation.poseControlLoraFileId, "");
assert.equal(migrated22.settings.generation.poseControlLoraModel, "");
assert.equal(migrated22.settings.generation.poseControlLoraStrength, 1);
assert.equal(migrated22.settings.generation.poseSemanticMode, "attention_isolation");

const version21 = structuredClone(first);
version21.version = 21;
delete version21.generation.pose_semantic_mode;
delete version21.generation.shared_visual_prompt;
const migrated21 = loadStudioProjectDocument(version21);
assert.equal(migrated21.settings.generation.poseSemanticMode, "spatial_only");
assert.equal(migrated21.settings.generation.sharedVisualPrompt, "");
assert.ok(migrated21.migrationNotices.some((notice) => notice.includes("subject-semantic pose routing")));

const version20 = structuredClone(first);
version20.version = 20;
version20.generation.pose_conditioning_enabled = true;
version20.generation.pose_controlnet_model = "qwen-controlnet.safetensors";
delete version20.generation.pose_gating_enabled;
version20.regions[0].pose = {
  enabled: true,
  joints: [
    ...Object.entries(first.regions[0].pose.joints).map(([name, point]) => ({ name, ...point, enabled: true })),
    { name: "nose", x: 0.5, y: 0.1, enabled: true },
    { name: "left_eye", x: 0.54, y: 0.08, enabled: true },
    { name: "right_eye", x: 0.46, y: 0.08, enabled: true },
  ],
};
const migrated20 = loadStudioProjectDocument(version20);
assert.equal(migrated20.settings.generation.poseGating, false);
assert.equal(migrated20.regions[0].pose.joints.length, 13);
assert.ok(migrated20.regions[0].pose.head.rx > 0);
assert.ok(migrated20.migrationNotices[0].includes("Qwen pose-ControlNet settings were removed"));

const sanitizedProject = structuredClone(first);
sanitizedProject.loras[0].path = "opaque:opaque-cloud-id";
sanitizedProject.loras[0].display_name = "character.safetensors";
const loadedSanitized = loadStudioProjectDocument(sanitizedProject);
assert.equal(loadedSanitized.loras[0].fileId, "opaque-cloud-id");
assert.equal(loadedSanitized.loras[0].name, "character.safetensors");
assert.deepEqual(
  bindStudioLoraFiles(loadedSanitized.loras, [{
    id: "opaque-cloud-id",
    display_name: "people/character-v2.safetensors",
  }]).map((lora) => [lora.fileId, lora.name]),
  [["opaque-cloud-id", "people/character-v2.safetensors"]],
);
const legacyOpaqueProject = structuredClone(sanitizedProject);
delete legacyOpaqueProject.loras[0].display_name;
const loadedLegacyOpaque = loadStudioProjectDocument(legacyOpaqueProject);
assert.equal(loadedLegacyOpaque.loras[0].name, "Unresolved LoRA");
assert.equal(
  bindStudioLoraFiles(loadedLegacyOpaque.loras, [{
    id: "opaque-cloud-id",
    display_name: "character.safetensors",
  }])[0].name,
  "character.safetensors",
);

function textChunk(key, value) {
  const encoded = new TextEncoder().encode(`${key}\0${JSON.stringify(value)}`);
  const chunk = new Uint8Array(12 + encoded.length);
  new DataView(chunk.buffer).setUint32(0, encoded.length);
  chunk.set(new TextEncoder().encode("tEXt"), 4);
  chunk.set(encoded, 8);
  return chunk;
}

const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const png = new Blob([signature, textChunk("k2lab_project", first)]);
assert.deepEqual(await projectDocumentFromPng(png), first);
const legacyPng = new Blob([
  signature,
  textChunk("k2lab_project", legacyOpaqueProject),
  textChunk("loras", [{
    id: "opaque-cloud-id",
    display_name: "archive/character-from-report.safetensors",
  }]),
]);
const recoveredLegacyProject = await projectDocumentFromPng(legacyPng);
assert.equal(
  loadStudioProjectDocument(recoveredLegacyProject).loras[0].name,
  "character-from-report.safetensors",
);

const events = appendBoundedEvents([], Array.from({ length: EVENT_LOG_LIMIT + 25 }, (_value, index) => index));
assert.equal(events.length, EVENT_LOG_LIMIT);
assert.equal(events[0], 25);
assert.equal(events.at(-1), EVENT_LOG_LIMIT + 24);

console.log("studio project JSON and PNG round-trip contracts passed");
