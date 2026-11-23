# Workspace backup — 24 November 2026

Snapshot of /home/thanmay/ML-challenge. Includes source, uncommitted changes, reports, datasets, model weights, probability artifacts, submissions and original resource ZIP across all five worktrees. git-history.bundle preserves all local Git refs; worktrees.json records each original checkout and its changes.

Excluded: virtual environments, dependency caches, Python caches, .git internals (replaced by the bundle), and credential files. See exclusions.json. Reinstall dependencies rather than restoring the environments. Original files were not deleted.

Restore into the original absolute location to keep existing absolute symlinks valid:

```bash
sha256sum -c backup/SHA256SUMS
mkdir -p /home/thanmay/ML-challenge
cat backup/chunks/workspace.tar.zst.part-* | zstd -d | tar -xf - -C /home/thanmay/ML-challenge
tar -xf backup/directory-symlinks.tar -C /home/thanmay/ML-challenge
```

For Git history, clone the bundle into a new path, then recover worktree contents from the restored files using worktrees.json. Restored directories intentionally do not include old .git worktree pointers.

```bash
git clone backup/git-history.bundle /home/thanmay/ML-challenge-restored-git
```

Archive chunks stay below GitHub's per-file size limit. This branch is a storage snapshot; source files are inside the archive rather than flattened together across conflicting worktree versions. verification.json records content verification against the original files.
