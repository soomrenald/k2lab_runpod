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

console.log("Studio UI contract passed");
