# Async SQLite/ASGI hang diagnostic

Date: 2026-07-30

Feature branch: `feature/blender-depth-regions` at `1bef6ae`

Stable branch control: `main` at `9e2cfc64`

Pre-depth feature control: `4048ad6`

## Conclusion

**Deterministic infrastructure incompatibility.**

The two tests hang on this host on the feature branch, stable `main`, and the
feature branch's pre-depth base. They pass individually and in both orders in a
clean Ubuntu 24.04 container on the same machine. The complete clean-container
suite passes: `305 passed, 15 skipped, 16 subtests passed`.

The host cannot reliably wake an asyncio selector from another thread. A
minimal, repository-independent `loop.call_soon_threadsafe` reproduction hangs
until an unrelated loop timer fires. Both affected tests stop at exactly that
cross-thread handoff:

- the audit test waits for the idle `aiosqlite` worker to deliver its completed
  result to the event loop;
- the upload test waits for an idle executor worker to deliver an ASGI file
  inventory result to the event loop.

This is not a branch regression. No production flag was changed and the branch
was not merged.

## Exact tests and commands

1. `tests/test_runpod_backend.py::RunPodBackendTests::test_audit_store_redacts_nested_secrets_and_authenticated_urls`
2. `tests/test_workspace_agent.py::WorkspaceAgentTests::test_chunked_upload_resumes_verifies_and_updates_inventory`

The host diagnostic form used for each invocation was:

```bash
PYTHONPATH=/tmp \
K2_HANG_WATCHDOG_SECONDS=3 \
K2_HANG_DUMP=reports/test-hang-diagnostic/<case>.dump.txt \
timeout -s TERM -k 1s 9s \
  .venv/bin/python -m pytest -p k2_hang_probe -vv -s <ordered node IDs>
```

The watchdog plugin `/tmp/k2_hang_probe.py` captured Python threads, asyncio
tasks and primitive waiters, `aiosqlite` objects, file descriptors, child
processes, and FastAPI lifespan/application state. The outer GNU `timeout` was
independent of Python and produced exit status 124 for every host hang.

The depth-related subset used for the order checks was:

```bash
.venv/bin/python -m pytest -vv -s \
  tests/test_depth_evaluate.py \
  tests/test_depth_runtime.py \
  tests/test_depth_validate.py \
  tests/test_krea_control_lora.py \
  tests/test_project.py
```

It contains 32 cases in this environment: 29 pass and 3 Torch-dependent cases
skip.

## Test matrix

| Environment and order | Result |
|---|---|
| Feature: audit alone | timeout 124; stopped in `append_audit` |
| Feature: upload alone | timeout 124; stopped on inventory GET |
| Feature: audit, upload | timeout 124 in audit; upload not reached |
| Feature: upload, audit | timeout 124 in upload; audit not reached |
| Feature: depth subset, audit, upload | 29 passed, 3 skipped, then timeout in audit |
| Feature: depth subset, upload, audit | 29 passed, 3 skipped, then timeout in upload |
| Feature: audit, upload, depth subset | timeout in audit; later cases not reached |
| Feature: upload, audit, depth subset | timeout in upload; later cases not reached |
| Stable `main`: audit alone | identical timeout 124 |
| Stable `main`: upload alone | identical timeout 124 |
| Stable `main`: both orders | identical timeout 124 in first selected test |
| Pre-depth `4048ad6`: audit alone | identical timeout 124 |
| Pre-depth `4048ad6`: upload alone | identical timeout 124 |
| Ubuntu 24.04: each test and both orders | all pass, exit 0 |
| Ubuntu 24.04 with exact host package versions: each test | both pass, exit 0 |
| Ubuntu 24.04: complete suite | 305 passed, 15 skipped, exit 0 in 11.66 s |

The stable-branch control used an archive of `main` and the exact same host
interpreter and virtual environment as the feature run:

```bash
cd /tmp/k2lab-main-diagnostic-9e2cfc6
PYTHONPATH=/tmp/k2lab-main-diagnostic-9e2cfc6/src:/tmp \
  timeout -s TERM -k 1s 9s \
  /home/wolfhard/k2lab_runpod/.venv/bin/python \
  -m pytest -p k2_hang_probe -vv -s <ordered node IDs>
```

The pre-depth control used the same procedure against an archive of `4048ad6`.

The clean-container control used Ubuntu 24.04, Python 3.12.3, glibc 2.39, and
the repository's `[web,dev]` dependency set. A second control pinned the
host-sensitive packages exactly:

```text
pytest==9.1.1
aiosqlite==0.22.1
httpx==0.28.1
fastapi==0.139.2
sqlalchemy==2.0.51
anyio==4.14.2
starlette==1.3.1
uvicorn==0.51.0
```

The corrected full-suite container command ran from `/src` with the source
mounted read-only:

```bash
timeout -s TERM -k 2s 180s \
  /opt/diag-venv/bin/python -m pytest -q tests
```

## Timeout snapshots

### Audit/SQLite test

The feature snapshot records:

- test task pending at
  `tests/test_runpod_backend.py:656`, awaiting
  `self.state_store.append_audit(...)`;
- main event-loop thread blocked in `selectors.select`;
- `aiosqlite` worker blocked idle on its transaction queue (`tx.get`);
- one SQLite database file descriptor open;
- `aiosqlite.Connection`: running, no underlying operation active, queue size 0;
- no asyncio lock waiters;
- no subprocesses;
- no ASGI application involved.

The worker has finished the database work and has no queued operation, while
the loop still waits for the future completion notification.

### Upload/ASGI test

The feature snapshot records:

- test task pending at `tests/test_workspace_agent.py:1060`, awaiting
  `GET /v1/files?kind=inputs`;
- an ASGI middleware task waiting on an executor future;
- main event-loop thread blocked in `selectors.select`;
- `asyncio_0` executor thread idle in the futures worker loop;
- no SQLite connections;
- no subprocesses;
- the FastAPI app and its lifespan context present, with zero legacy
  startup/shutdown handlers;
- application state populated with the workspace layout and transfer,
  download, job, face, and migration managers;
- one unset asyncio event with one waiter, matching the ASGI response stream
  handoff.

The upload itself completed and verified before the inventory request stalled.

In both snapshots, the diagnostic thread's own
`loop.call_soon_threadsafe(snapshot_callback)` failed to wake the loop, so the
plugin fell back to a direct cross-thread task snapshot.

## Infrastructure isolation

A standalone host program with no project imports creates an asyncio future,
then asks a background thread to resolve it with
`loop.call_soon_threadsafe(future.set_result, 42)`. It times out with the main
thread in `selector.select()`. Adding an independent 500 ms event-loop timer
makes the same program complete in 0.501 seconds. This isolates the failure to
the host execution context's selector/self-pipe wakeup behavior rather than
SQLite, ASGI, pytest, or K2Lab.

See `host-call-soon-threadsafe-delayed.log` and
`host-call-soon-threadsafe-with-timer.log`.

## Changed-code involvement

The requested literal condition that no changed file is imported is not true:
the upload test module gained a separate depth test, and
`k2_region_lab.agent.jobs` is imported while the agent app is constructed.
That fact is not hidden or treated as sufficient isolation by itself.

The stronger behavioral isolation is:

- both hangs reproduce at the pre-depth commit `4048ad6`;
- both hangs reproduce on stable `main`;
- both tests pass from the current feature tree in the clean container;
- the audit test body and its SQLite state-store path are unchanged;
- the existing chunked-upload test body is unchanged;
- the executed-line trace for the upload test reaches only unchanged
  `JobManager` constructor/shutdown lines (185-214 and 1074-1076), not the
  depth payload block (509-609);
- neither pending stack contains a depth module;
- the minimal reproduction imports no repository code.

Therefore branch files can be imported as part of normal app setup, but no
depth-modified executable path causes or participates in the hang.

## Finite timeout

`pytest-timeout>=2.4,<3` is now a locked development dependency. Pytest has:

```toml
timeout = 30
timeout_method = "thread"
```

The thread method is intentional: it terminates even when the selector cannot
receive signals from a helper thread. Both target tests fail finitely with
thread dumps at about 3.25 seconds when the timeout is overridden to 3 seconds,
and the host full suite exits at the first affected test instead of hanging
indefinitely. The committed default is 30 seconds.

## Narrow waiver and residual risk

Proposed waiver scope: only the two host results named above, and only for this
branch validation. The clean-container results remain required; no other test
failure is waived.

Residual risk:

- the restricted host cannot directly validate successful cross-thread asyncio
  wakeups;
- a container shares the host kernel, although it provides a separate,
  production-like userspace and security context;
- imported branch modules still execute unchanged setup code before the upload
  test;
- the 30-second hard timeout stops the process at the first timeout, so a
  subsequent CI run must be used to discover additional timeouts.

This evidence supports the narrow waiver, but it remains pending explicit
approval. No merge should occur before that approval.
