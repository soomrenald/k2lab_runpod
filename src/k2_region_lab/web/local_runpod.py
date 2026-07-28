from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


_IMAGE_DIGEST = re.compile(r"^\S+@sha256:[0-9a-fA-F]{64}$")
_CONFIG_FILENAME = "config.json"
_KEY_FILENAME = "credential.key"
_DATABASE_FILENAME = "state.sqlite3"
_LOG_FILENAME = "control-plane.log"


def default_state_directory(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    state_root = values.get("XDG_STATE_HOME")
    if state_root:
        return Path(state_root).expanduser() / "k2-region-lab"
    return Path.home() / ".local" / "state" / "k2-region-lab"


def validate_image_digest(value: str) -> str:
    digest = value.strip()
    if not _IMAGE_DIGEST.fullmatch(digest):
        raise ValueError(
            "Workspace image must use an immutable registry/name@sha256:<64-hex> digest."
        )
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read local configuration at {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Local configuration at {path} must contain a JSON object.")
    return value


def _write_private(path: Path, content: str, *, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, content.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o600)


def _credential_key(state_directory: Path) -> str:
    path = state_directory / _KEY_FILENAME
    if not path.exists():
        try:
            _write_private(path, Fernet.generate_key().decode("ascii") + "\n", exclusive=True)
        except FileExistsError:
            pass
    try:
        key = path.read_text(encoding="ascii").strip()
        Fernet(key.encode("ascii"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"The credential key at {path} is unreadable or invalid.") from error
    path.chmod(0o600)
    return key


def _choose_image_digest(
    *,
    explicit: str | None,
    environment: Mapping[str, str],
    stored: Mapping[str, Any],
    input_function: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> str:
    candidate = explicit or environment.get("K2LAB_RUNPOD_IMAGE_DIGEST")
    candidate = candidate or stored.get("workspace_image_digest")
    if not candidate:
        can_prompt = sys.stdin.isatty() if interactive is None else interactive
        if not can_prompt:
            raise RuntimeError(
                "No workspace image is configured. Pass --image registry/name@sha256:<digest>."
            )
        print(
            "A public, immutable K2 workspace image is required for RunPod Pods.\n"
            "You only need to enter this digest on the first run."
        )
        candidate = input_function("Workspace image digest: ")
    if not isinstance(candidate, str):
        raise ValueError("The stored workspace image digest is invalid.")
    return validate_image_digest(candidate)


def prepare_local_environment(
    *,
    state_directory: Path,
    image_digest: str | None,
    image_version: str | None,
    port: int,
    environment: Mapping[str, str] | None = None,
    input_function: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> tuple[dict[str, str], bool]:
    if not 1024 <= port <= 65535:
        raise ValueError("Port must be between 1024 and 65535.")
    state_directory = state_directory.expanduser().resolve()
    state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_directory.chmod(0o700)
    config_path = state_directory / _CONFIG_FILENAME
    stored = _read_json(config_path)
    values = os.environ if environment is None else environment
    digest = _choose_image_digest(
        explicit=image_digest,
        environment=values,
        stored=stored,
        input_function=input_function,
        interactive=interactive,
    )
    version = image_version or str(stored.get("workspace_image_version") or "0.3.0")
    next_config = {
        "workspace_image_digest": digest,
        "workspace_image_version": version,
    }
    changed = next_config != stored
    if changed:
        _write_private(config_path, json.dumps(next_config, indent=2, sort_keys=True) + "\n")
    key = _credential_key(state_directory)
    database_path = state_directory / _DATABASE_FILENAME
    runtime_environment = {
        "K2LAB_WEB_BACKEND": "runpod",
        "K2LAB_CREDENTIAL_FERNET_KEY": key,
        "K2LAB_DATABASE_URL": f"sqlite+aiosqlite:///{database_path.as_posix()}",
        "K2LAB_RUNPOD_IMAGE_DIGEST": digest,
        "K2LAB_RUNPOD_IMAGE_VERSION": version,
        "K2LAB_LOCAL_SINGLE_USER": "true",
        "K2LAB_LOCAL_PORT": str(port),
        "K2LAB_SERVE_WEB_UI": "true",
    }
    return runtime_environment, changed


def _open_browser_when_ready(url: str) -> None:
    health_url = f"{url}/api/v1/health"
    for _attempt in range(100):
        try:
            with urllib.request.urlopen(health_url, timeout=0.25) as response:  # noqa: S310
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(0.1)


def _local_control_plane_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/api/v1/health",
            timeout=0.5,
        ) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and isinstance(payload.get("backend"), str)
    )


def _port_available(port: int) -> bool:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        candidate.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        candidate.close()
    return True


def _configure_runtime_logging(log_path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    rotating = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "k2_region_lab"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(stream)
        logger.addHandler(rotating)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _follow_runtime_log(log_path: Path) -> int:
    if not log_path.exists():
        print(
            "K2 Region Lab is already running, but that process was started before "
            f"persistent logging was enabled.\nRestart it once to create {log_path}.",
            file=sys.stderr,
        )
        return 3
    print(f"K2 Region Lab is already running. Following {log_path}")
    print("Press Ctrl+C to stop following; the app will keep running.")
    source = log_path.open(encoding="utf-8", errors="replace")
    try:
        recent = source.readlines()[-50:]
        for line in recent:
            print(line, end="")
        while True:
            line = source.readline()
            if line:
                print(line, end="", flush=True)
                continue
            try:
                rotated = (
                    log_path.stat().st_ino != os.fstat(source.fileno()).st_ino
                    or log_path.stat().st_size < source.tell()
                )
            except OSError:
                rotated = False
            if rotated:
                source.close()
                source = log_path.open(encoding="utf-8", errors="replace")
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k2lab-runpod",
        description="Run the single-user K2 Region Lab RunPod control plane on this computer.",
    )
    parser.add_argument(
        "--image",
        help="Public immutable workspace image (registry/name@sha256:<digest>); saved after use.",
    )
    parser.add_argument("--image-version", help="Human-readable workspace image version.")
    parser.add_argument("--state-dir", type=Path, default=default_state_directory())
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser.")
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="If K2 is already running, report it and exit instead of following its log.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1024 <= args.port <= 65535:
        print("k2lab-runpod: Port must be between 1024 and 65535.", file=sys.stderr)
        return 2
    state_directory = args.state_dir.expanduser().resolve()
    log_path = state_directory / _LOG_FILENAME
    if _local_control_plane_running(args.port):
        if args.no_follow:
            print(f"K2 Region Lab is already running at http://127.0.0.1:{args.port}")
            print(f"Runtime log: {log_path}")
            return 0
        return _follow_runtime_log(log_path)
    if not _port_available(args.port):
        print(
            f"k2lab-runpod: Port {args.port} is already used by another application. "
            "Stop that application or select another port with --port.",
            file=sys.stderr,
        )
        return 2
    try:
        runtime_environment, config_changed = prepare_local_environment(
            state_directory=state_directory,
            image_digest=args.image,
            image_version=args.image_version,
            port=args.port,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"k2lab-runpod: {error}", file=sys.stderr)
        return 2

    os.umask(0o077)
    _configure_runtime_logging(log_path)
    os.environ.update(runtime_environment)
    url = f"http://127.0.0.1:{args.port}"
    print(f"K2 Region Lab state: {state_directory}")
    print(f"Runtime log: {log_path}")
    if config_changed:
        print("Saved the immutable workspace-image selection for future launches.")
    print(f"Open {url} and paste your restricted RunPod API key.")
    print("This mode can create billable RunPod resources. Press Ctrl+C to stop the control plane.")

    if not args.no_open:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    from k2_region_lab.web.app import app

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        proxy_headers=False,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
