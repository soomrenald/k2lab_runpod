export interface PromptEmphasisRecord {
  scopeId: "__global__" | string;
  phrase: string;
  occurrence: number;
}

export interface SelectedPromptEmphasis {
  phrase: string;
  occurrence: number;
}

function occurrenceOffsets(source: string, phrase: string): number[] {
  if (!phrase) return [];
  const offsets: number[] = [];
  let start = 0;
  while (start <= source.length) {
    const offset = source.indexOf(phrase, start);
    if (offset < 0) break;
    offsets.push(offset);
    start = offset + phrase.length;
  }
  return offsets;
}

function compiledPromptSource(source: string): { text: string; leadingTrim: number } {
  const leadingTrim = source.length - source.trimStart().length;
  return { text: source.trim(), leadingTrim };
}

export function promptEmphasisFromSelection(
  source: string,
  selectionStart: number,
  selectionEnd: number,
  scopeId: "__global__" | string,
): SelectedPromptEmphasis | null {
  const start = Math.max(0, Math.min(selectionStart, source.length));
  const end = Math.max(start, Math.min(selectionEnd, source.length));
  const selection = source.slice(start, end);
  const leadingSelectionTrim = selection.length - selection.trimStart().length;
  let phrase = selection.trim();
  if (scopeId !== "__global__") {
    phrase = phrase.replace(/[.!?]+$/u, "").trimEnd();
  }
  if (!phrase) return null;

  const compiled = compiledPromptSource(source);
  const selectedOffset = start + leadingSelectionTrim - compiled.leadingTrim;
  const offsets = occurrenceOffsets(compiled.text, phrase);
  const occurrence = offsets.indexOf(selectedOffset);
  return occurrence < 0 ? null : { phrase, occurrence };
}

export function promptEmphasisMatches(
  item: PromptEmphasisRecord,
  source: string | null,
): boolean {
  if (source === null || !Number.isInteger(item.occurrence) || item.occurrence < 0) {
    return false;
  }
  return occurrenceOffsets(compiledPromptSource(source).text, item.phrase).length > item.occurrence;
}

export function reconcilePromptEmphases<T extends PromptEmphasisRecord>(
  items: readonly T[],
  sourceForScope: (scopeId: string) => string | null,
): T[] {
  const reconciled: T[] = [];
  for (const item of items) {
    const source = sourceForScope(item.scopeId);
    if (source === null) continue;
    const offsets = occurrenceOffsets(compiledPromptSource(source).text, item.phrase);
    if (!offsets.length) continue;
    const occurrence = Math.min(
      Math.max(0, Math.trunc(item.occurrence)),
      offsets.length - 1,
    );
    reconciled.push(
      occurrence === item.occurrence ? item : { ...item, occurrence },
    );
  }
  return reconciled;
}
