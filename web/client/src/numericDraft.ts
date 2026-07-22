export function committedNumber(draft: string, min: number, max: number): number | null {
  if (!draft.trim()) return null;
  const parsed = Number(draft);
  if (!Number.isFinite(parsed)) return null;
  return Math.max(min, Math.min(max, parsed));
}
