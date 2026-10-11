# Validation evidence

Recorded 2026-10-12 on macOS arm64. Selected Python: **3.15.0**; uv0.13.0; native Cargo/Rust1.99.0. Commands below ran from the repository root. Exit124 means the documented time bound expired; exit139 means a native segmentation fault. An earlier failed attempt is retained even if a later rerun succeeds.

## Current scope and result

Release CPython3.15 wheel and source distribution build successfully. Native indicator crate has0unit tests; coordinator crate has3passing tests. EMA calculation and coordinator export were checked through the built extension. The sibling Jesse complete suite consumes this local source on Python3.14.8:1970passed,18existing skips (jesse-tests-4.log). No native numerical or binding mismatch was observed.

Current recorded status: **complete**. Interpreter fallback evidence and native prerequisites are described in [development](development.md). See [architecture](ayu-develop.md) for workflows, components, and existing source links.

## Executed commands

| Step | Exact command | Exit | Elapsed seconds | Evidence |
|---|---|---:|---:|---|
| python-sync | `uv sync --all-groups --all-extras --python 3.15` | 1 | 8.43 | [jesse-rust-python-sync.log](../../docs/maintenance-2026-10-12/jesse-rust-python-sync.log) |
| cargo-update | `cargo update` | 0 | 6.03 | [jesse-rust-cargo-update.log](../../docs/maintenance-2026-10-12/jesse-rust-cargo-update.log) |
| python315-modern-sync | `uv sync --all-groups --all-extras --python 3.15` | 1 | 44.2 | [jesse-rust-python315-modern-sync.log](../../docs/maintenance-2026-10-12/jesse-rust-python315-modern-sync.log) |
| python315-fixed-sync | `uv sync --all-groups --all-extras --python 3.15` | 0 | 64.92 | [jesse-rust-python315-fixed-sync.log](../../docs/maintenance-2026-10-12/jesse-rust-python315-fixed-sync.log) |
| cargo-coordinator-tests | `cargo test --manifest-path coordination/Cargo.toml` | 0 | 13.04 | [jesse-rust-cargo-coordinator-tests.log](../../docs/maintenance-2026-10-12/jesse-rust-cargo-coordinator-tests.log) |
| build | `uv build` | 0 | 47.24 | [jesse-rust-build.log](../../docs/maintenance-2026-10-12/jesse-rust-build.log) |
| native-import | `uv run --no-sync python -c 'import jesse_rust; import numpy as np; print(jesse_rust.ema(np.array([1., 2., 3., 4.]), 2)); print(callable(jesse_rust.run_coordinator))'` | 0 | 0.35 | [jesse-rust-native-import.log](../../docs/maintenance-2026-10-12/jesse-rust-native-import.log) |
| cargo-tests | `cargo test` | 0 | 19.78 | [jesse-rust-cargo-tests.log](../../docs/maintenance-2026-10-12/jesse-rust-cargo-tests.log) |
| build-current-metadata | `uv build` | 0 | 34.67 | [jesse-rust-build-current-metadata.log](../../docs/maintenance-2026-10-12/jesse-rust-build-current-metadata.log) |

## Test summaries and failures

### cargo-coordinator-tests

Exit 0; elapsed 13.04seconds.

```text
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.20s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### native-import

Exit 0; elapsed 0.35seconds.

```text
$ uv run --no-sync python -c import jesse_rust; import numpy as np; print(jesse_rust.ema(np.array([1., 2., 3., 4.]), 2)); print(callable(jesse_rust.run_coordinator))
[1.         1.66666667 2.55555556 3.51851852]
True
```

### cargo-tests

Exit 0; elapsed 19.78seconds.

```text
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

## Synchronization

All upstream branch and tag refs were fetched by the workspace synchronization procedure. Fork updates use non-forcing pushes; upstream tags also have a private local namespace to preserve divergent tags. The default branch was merged into feature/ayu_develop, preserving the modern package layout and upstream source changes. The workspace synchronization JSON and log retain per-ref outcomes and merge history. This document does not claim that any feature branch was pushed.
