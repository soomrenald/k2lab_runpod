from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


GIB = 1024**3
VRAM_MODES = frozenset({"auto", "high_vram", "dynamic", "low_vram"})
CGROUP_UNLIMITED_THRESHOLD = 1 << 60


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    key: str
    label: str
    reserve_vram_gb: float
    warning_free_gb: float
    critical_free_gb: float
    minimum_system_ram_gb: float
    cpu_vae: bool = False
    oom_recovery: bool = True


@dataclass(frozen=True, slots=True)
class SystemMemorySnapshot:
    source: str
    total_bytes: int
    used_bytes: int
    available_bytes: int
    anonymous_bytes: int | None = None
    file_cache_bytes: int | None = None
    current_bytes: int | None = None
    reclaimable_file_bytes: int | None = None

    def to_payload(self) -> dict[str, int | str | None]:
        return {
            "source": self.source,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "available_bytes": self.available_bytes,
            "anonymous_bytes": self.anonymous_bytes,
            "file_cache_bytes": self.file_cache_bytes,
            "current_bytes": self.current_bytes,
            "reclaimable_file_bytes": self.reclaimable_file_bytes,
        }


MEMORY_POLICIES = (
    MemoryPolicy(
        "low_8gb",
        "Low VRAM (8 GB)",
        1.0,
        1.0,
        0.5,
        24.0,
        cpu_vae=True,
    ),
    MemoryPolicy(
        "safe_12gb",
        "Safe 12 GB",
        2.0,
        2.0,
        1.0,
        18.0,
        cpu_vae=True,
    ),
    MemoryPolicy("performance", "Performance", 2.0, 2.0, 1.0, 12.0),
    MemoryPolicy("balanced", "Balanced", 3.0, 3.0, 1.5, 12.0),
    MemoryPolicy("safe_16gb", "Safe 16 GB", 4.0, 4.0, 2.0, 14.0),
    MemoryPolicy("large_24gb", "Large VRAM (24+ GB)", 3.0, 3.0, 1.5, 12.0),
    MemoryPolicy("custom", "Custom / any GPU", 0.5, 0.5, 0.25, 4.0),
    MemoryPolicy(
        "emergency",
        "Emergency",
        5.5,
        5.0,
        2.5,
        16.0,
        cpu_vae=True,
        oom_recovery=False,
    ),
)


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_memory_stat(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(" ")
        if not separator:
            continue
        try:
            values[key] = int(raw)
        except ValueError:
            continue
    return values


def _read_proc_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        parts = raw.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return values


def system_memory_snapshot(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> SystemMemorySnapshot:
    """Report the effective container memory limit, with a host fallback."""

    candidates = (
        (
            "cgroup_v2",
            cgroup_root / "memory.current",
            cgroup_root / "memory.max",
            cgroup_root / "memory.stat",
        ),
        (
            "cgroup_v1",
            cgroup_root / "memory" / "memory.usage_in_bytes",
            cgroup_root / "memory" / "memory.limit_in_bytes",
            cgroup_root / "memory" / "memory.stat",
        ),
    )
    for source, usage_path, limit_path, stat_path in candidates:
        used = _read_integer(usage_path)
        limit = _read_integer(limit_path)
        if (
            used is None
            or limit is None
            or limit <= 0
            or limit >= CGROUP_UNLIMITED_THRESHOLD
        ):
            continue
        stat = _read_memory_stat(stat_path)
        anonymous = stat.get("anon", stat.get("total_rss"))
        file_cache = stat.get("file", stat.get("total_cache"))
        # memory.current includes filesystem pages populated by model downloads and
        # safetensors reads. The kernel can evict inactive file pages under pressure,
        # so use the cgroup working-set convention for allocation guards while keeping
        # the raw current value in telemetry.
        reclaimable_file = max(
            0,
            stat.get("inactive_file", stat.get("total_inactive_file", 0)),
        )
        working_set = max(0, used - reclaimable_file)
        available = min(limit, max(0, limit - working_set))
        return SystemMemorySnapshot(
            source=source,
            total_bytes=limit,
            used_bytes=working_set,
            available_bytes=available,
            anonymous_bytes=anonymous,
            file_cache_bytes=file_cache,
            current_bytes=max(0, used),
            reclaimable_file_bytes=reclaimable_file,
        )

    try:
        import psutil
    except ModuleNotFoundError:
        meminfo = _read_proc_meminfo()
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        if total <= 0 or available < 0:
            raise RuntimeError("could not determine system memory capacity")
        return SystemMemorySnapshot(
            source="proc_meminfo_host",
            total_bytes=total,
            used_bytes=max(0, total - available),
            available_bytes=available,
        )
    memory = psutil.virtual_memory()
    return SystemMemorySnapshot(
        source="psutil_host",
        total_bytes=int(memory.total),
        used_bytes=int(memory.total - memory.available),
        available_bytes=int(memory.available),
    )


def memory_policy(key: str) -> MemoryPolicy:
    for policy in MEMORY_POLICIES:
        if policy.key == key:
            return policy
    raise ValueError(f"unknown memory policy: {key}")


def effective_reserve_vram_gb(key: str, requested_gb: float) -> float:
    """Apply the selected policy's non-negotiable VRAM reserve floor."""
    return max(0.5, float(requested_gb), memory_policy(key).reserve_vram_gb)


def effective_minimum_system_ram_gb(key: str, requested_gb: float) -> float:
    """Apply the selected policy's non-negotiable system-RAM floor."""
    return max(4.0, float(requested_gb), memory_policy(key).minimum_system_ram_gb)


def oom_recovery_reserve_vram_gb(current_gb: float, total_gb: float) -> float:
    """Increase the reserve proportionally without consuming a small GPU.

    The original 16 GiB setup moves from a 4 GiB reserve to 5 GiB. Smaller and
    larger devices receive a bounded increase scaled to their actual capacity.
    """
    current = max(0.5, float(current_gb))
    total = max(1.0, float(total_gb))
    increase = max(0.5, min(1.5, total / 16.0))
    capacity_limit = max(0.5, total * 0.4)
    return current + min(increase, max(0.0, capacity_limit - current))


def resolve_vram_mode(requested: str, total_vram_gb: float) -> str:
    """Resolve the user-facing automatic mode before ComfyUI initializes."""
    if requested not in VRAM_MODES:
        raise ValueError(f"unknown VRAM mode: {requested}")
    if requested != "auto":
        return requested
    return "high_vram" if float(total_vram_gb) >= 40.0 else "dynamic"


def configure_comfy_vram_args(args: Any, mode: str) -> None:
    """Apply one validated mode before ComfyUI model management is imported."""
    if mode not in {"high_vram", "dynamic", "low_vram"}:
        raise ValueError(f"unresolved VRAM mode: {mode}")
    args.gpu_only = False
    args.novram = False
    args.highvram = mode == "high_vram"
    args.lowvram = mode == "low_vram"
    if hasattr(args, "enable_dynamic_vram"):
        args.enable_dynamic_vram = mode == "dynamic"
    if hasattr(args, "disable_dynamic_vram"):
        args.disable_dynamic_vram = mode != "dynamic"
