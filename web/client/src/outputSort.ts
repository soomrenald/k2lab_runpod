import type { FileRecord } from "./api";

export type OutputSort = "newest" | "oldest" | "name-asc" | "name-desc" | "size-desc" | "size-asc";

export function sortOutputFiles(files: FileRecord[], sort: OutputSort): FileRecord[] {
  const direction = sort === "oldest" || sort === "name-asc" || sort === "size-asc" ? 1 : -1;
  return [...files].sort((left, right) => {
    let comparison = 0;
    if (sort === "newest" || sort === "oldest") {
      comparison = timestamp(left.modified_at) - timestamp(right.modified_at);
    } else if (sort === "name-asc" || sort === "name-desc") {
      comparison = left.display_name.localeCompare(right.display_name, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    } else {
      comparison = left.size_bytes - right.size_bytes;
    }
    if (comparison !== 0) return comparison * direction;
    const name = left.display_name.localeCompare(right.display_name, undefined, {
      numeric: true,
      sensitivity: "base",
    });
    return name || left.id.localeCompare(right.id);
  });
}

function timestamp(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
}
