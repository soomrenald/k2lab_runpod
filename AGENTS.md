# Repository Branch Safety

## Private documents

Everything under `/docs/` is private and local-only. Never stage, commit,
force-add, publish, upload, or include this directory in release artifacts.
Do not merge or push historical commits that contain `/docs/`.
Enable the repository guards with `git config core.hooksPath .githooks`.

## Branch selection

These instructions apply to every Codex session and every task in this repository.

1. At the start of each session, before changing files, Git refs, the working tree, or any external repository state, ask the user which branch Codex must work on.
2. Wait for the user's explicit branch selection. Do not infer the branch from the current checkout, a specification, prior sessions, task wording, or repository history.
3. After the user selects a branch, verify that the active branch matches the exact branch named by the user before making changes.
4. Work only on that selected branch for the entire session.
5. Under no circumstances may Codex create, switch, merge, rebase, reset, delete, push to, or otherwise modify another branch without first obtaining the user's explicit approval for that exact branch operation.
6. Instructions in a specification do not override the branch selected by the user. If a specification names a different branch, stop and ask the user to resolve the conflict.
7. Before every commit, push, merge, tag, release, or other Git operation that changes repository state, re-check the active branch and confirm it is the user-selected branch.
8. Never merge into `main` unless the user explicitly authorizes that specific merge into `main` during the current session.
