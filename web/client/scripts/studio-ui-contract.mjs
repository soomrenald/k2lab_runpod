import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { committedNumber } from "../src/numericDraft.ts";

assert.equal(committedNumber("", 1, 100), null, "an empty editing draft must remain transient");
assert.equal(committedNumber("-", -10, 10), null, "an incomplete signed draft must remain transient");
assert.equal(committedNumber("27", 1, 100), 27);
assert.equal(committedNumber("200", 1, 100), 100);

const inspector = await readFile(new URL("../src/components/Inspector.tsx", import.meta.url), "utf8");
const promptSection = inspector.split('{tab === "prompt"', 2)[1].split('{tab === "regions"', 1)[0];
const regionsSection = inspector.split('{tab === "regions" && mode !== "face"', 2)[1].split('{tab === "loras"', 1)[0];

assert.ok(!promptSection.includes("moveSelected("), "Prompt tab must not own region depth controls");
assert.ok(regionsSection.includes("moveSelected(-1)"), "Regions tab must move the selected region forward");
assert.ok(regionsSection.includes("moveSelected(1)"), "Regions tab must move the selected region backward");

for (const relativePath of [
  "../src/components/Inspector.tsx",
  "../src/components/CloudOnboarding.tsx",
  "../src/components/WorkspaceStudio.tsx",
]) {
  const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  assert.ok(!/type="number"[^>]*onChange=/.test(source), `${relativePath} bypasses draft-safe numeric input`);
}

const workspaceStudio = await readFile(new URL("../src/components/WorkspaceStudio.tsx", import.meta.url), "utf8");
assert.ok(
  workspaceStudio.includes('href="https://console.runpod.io/pods"'),
  "Cloud workspace status must link to RunPod's Docker startup progress",
);
assert.ok(
  workspaceStudio.includes("workspace.provider_resource_id"),
  "RunPod startup progress must identify the provider Pod",
);
assert.ok(
  workspaceStudio.includes("startWithoutTimeLimit")
    && workspaceStudio.includes("continue running and billing until you manually stop it"),
  "Restarting a Pod must offer an explicitly warned unlimited lease",
);
assert.ok(
  workspaceStudio.includes("Connect migrated Pod")
    && workspaceStudio.includes("controlPlane.connectMigratedPod")
    && workspaceStudio.includes("Verify and connect"),
  "Stopped persistent workspaces must support verified RunPod console migration reassociation",
);

const onboarding = await readFile(new URL("../src/components/CloudOnboarding.tsx", import.meta.url), "utf8");
assert.ok(
  onboarding.includes("request.lease_unlimited")
    && onboarding.includes("keep running and billing until you manually stop it"),
  "Initial Pod creation must offer an explicitly warned unlimited lease",
);
assert.ok(
  workspaceStudio.includes('setUtilityPanel("assets")')
    && workspaceStudio.includes('setUtilityPanel("transfers")')
    && workspaceStudio.includes('setUtilityPanel("events")')
    && workspaceStudio.includes('setUtilityPanel("setup")'),
  "Utility rail panels must replace one another instead of stacking",
);

const transferPanel = await readFile(new URL("../src/components/TransferPanel.tsx", import.meta.url), "utf8");
assert.ok(transferPanel.includes("controlPlane.transfers(workspaceId)"), "Provider transfer history must restore when the panel opens");

const assetPanel = await readFile(new URL("../src/components/AssetPanel.tsx", import.meta.url), "utf8");
assert.ok(assetPanel.includes("controlPlane.uploads(workspaceId)"), "Local upload history must restore when the panel opens");

console.log("Studio UI contract passed");
