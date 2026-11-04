# Will-It-Jev — task/tier decoupling replacement bundle

This archive contains 15 complete replacement source/configuration files at repository-relative paths. The archive itself has no enclosing project directory, but desktop unzip tools may create a directory named after the archive. That is expected: `apply.sh` copies files from its own extracted directory into the Git working tree.

## Install

1. Extract the ZIP anywhere (for example, Finder may create `Will-It-Jev-task-tier-decoupling-root/` in Downloads).
2. In Terminal, `cd` to the root of your Will-It-Jev Git checkout.
3. Preview the installation:

   ```bash
   ./path/to/Will-It-Jev-task-tier-decoupling-root/apply.sh --dry-run
   ```

4. Apply it:

   ```bash
   ./path/to/Will-It-Jev-task-tier-decoupling-root/apply.sh
   ```

The script locates its own bundle directory and copies the 15 packaged files into the corresponding `src/` and `config/` paths in the current Git worktree. It does not assume the files have already been extracted into the repository. It backs up every overwritten file (and the deleted obsolete runner, if present) under `.git/task-tier-decoupling-backup/`.

After application, inspect the result:

```bash
git status --short
git diff --stat
```

## Files replaced

- `src/main.rs`
- `src/config/schema.rs`
- `src/pipeline/config.rs`
- `src/pipeline/mod.rs`
- `src/pipeline/runner.rs`
- `src/pipeline/tagger.rs`
- `src/pipeline/task.rs`
- `src/pipeline/tier.rs`
- `src/router/cascade.rs`
- `src/router/mod.rs`
- `src/router/service.rs`
- `src/router/tier2_runner.rs`
- `src/router/tier3_runner.rs`
- `config/router.toml`
- `config/tasks.toml`

The script also removes `src/router/tier4_runner.rs` if it exists; Tier 4 uses the shared gRPC `Tier3Runner` implementation.

## Verification caveat

These replacement files were prepared before PR #20 was merged. They have not been reconciled against the final merged `master` tree and have not been compiled or tested against that tree. Installation is now real and verifiable, but do not commit this upgrade until you review the diff and run:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
