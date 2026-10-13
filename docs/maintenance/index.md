# jesse-rust: portable verification records

<!-- current-documentation-provenance -->
Documentation reviewed: 2026-10-13 · Git source base: `edb3e71b7ef31ecad3a661e2aaa5e46ccd94a2b4` · [Current architecture and feature guide](../ayu-develop.md).
Historical execution commands/results below retain their original dates and scope; this metadata update is not a new application test run.
<!-- /current-documentation-provenance -->


Full raw logs are retained in the workspace maintenance evidence, not distributed here. Hashes identify those original logs. A successful build or selected test subset does not imply a full test-suite pass. Commands containing <workspace> describe the original execution environment.

[Machine-readable records](verification.json) · [Repository validation summary](../validation.md)

## Executed commands

| Check | Exit | Duration | Command | Outcomes |
| --- | ---: | ---: | --- | --- |
| build | 0 | unrecordeds | `uv build` | Successfully built dist/jesse_rust-1.3.1.tar.gz<br>Successfully built dist/jesse_rust-1.3.1-cp315-cp315-macosx_11_0_arm64.whl |
| build-current-metadata | 0 | unrecordeds | `uv build` | Successfully built dist/jesse_rust-1.3.1.tar.gz<br>Successfully built dist/jesse_rust-1.3.1-cp315-cp315-macosx_11_0_arm64.whl |
| cargo-coordinator-tests | 0 | unrecordeds | `cargo test --manifest-path coordination/Cargo.toml` | test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.20s<br>test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s<br>test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| cargo-tests | 0 | unrecordeds | `cargo test` | test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s |
| cargo-update | 0 | unrecordeds | `cargo update` | See exit status; no test-count assertion. |
| native-import | 0 | unrecordeds | `uv run --no-sync python -c import jesse_rust; import numpy as np; print(jesse_rust.ema(np.array([1., 2., 3., 4.]), 2)); print(callable(jesse_rust.run_coordinator))` | See exit status; no test-count assertion. |
| python-sync | 1 | unrecordeds | `uv sync --all-groups --all-extras --python 3.15` | See exit status; no test-count assertion. |
| python315-fixed-sync | 0 | unrecordeds | `uv sync --all-groups --all-extras --python 3.15` | See exit status; no test-count assertion. |
| python315-modern-sync | 1 | unrecordeds | `uv sync --all-groups --all-extras --python 3.15` | See exit status; no test-count assertion. |
