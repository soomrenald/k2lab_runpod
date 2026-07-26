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
import {
  promptEmphasisFromSelection,
  promptEmphasisMatches,
  reconcilePromptEmphases,
} from "../src/promptEmphasis.ts";

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
settings.generation.seed = 8123;
settings.generation.seedMode = "increment";
settings.generation.batchMode = true;
settings.generation.batchCount = 4;
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
  { id: "person", name: "Person", layer: "generation", x: 80, y: 40, width: 400, height: 900, prompt: "red coat", faceIdentityPrompt: "green eyes", spatialRole: "subject", enabled: true },
  { id: "wall", name: "Wall", layer: "generation", x: 0, y: 0, width: 1024, height: 1024, prompt: "brick wall", faceIdentityPrompt: "", spatialRole: "background", enabled: true },
  { id: "reference", name: "Reference", layer: "reference", x: 20, y: 30, width: 300, height: 700, prompt: "portrait", faceIdentityPrompt: "same person", spatialRole: "subject", enabled: true },
  { id: "target", name: "Target", layer: "targets", x: 350, y: 300, width: 200, height: 250, prompt: "blue jacket", faceIdentityPrompt: "", spatialRole: "subject", enabled: true },
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
