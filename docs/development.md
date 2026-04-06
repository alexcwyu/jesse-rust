# Jesse-Rust — Development Guide

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Rust toolchain | stable (1.75+) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Python | 3.10+ | system or pyenv |
| Maturin | 1.8+ | `pip install maturin` |
| NumPy | 1.26.4+ | `pip install numpy` |

---

## Project Structure

```
ext-systems/jesse-rust/
├── src/
│   ├── lib.rs                  # PyO3 module root — registers all ~40 functions
│   └── indicators.rs           # All indicator logic (~1 500 lines)
├── docs/                       # This documentation
│   ├── README.md
│   ├── architecture.md
│   ├── workflow.md
│   ├── state-management.md
│   └── development.md
├── Cargo.toml                  # Rust manifest + release profiles
├── pyproject.toml              # Python build metadata (Maturin backend)
├── __init__.py                 # Python shim for symbol re-export
├── MANIFEST.in                 # Source dist file inclusions
├── LICENSE                     # MIT
├── build-local.sh              # Dev install script
├── build-quick.sh              # Quick host-target wheel
├── build-all-wheels.sh         # Cross-compilation for all targets
├── build-comprehensive.sh      # Extended cross-compilation
└── build-local.ps1             # PowerShell equivalent of build-local.sh
```

**Source references**:
- `ext-systems/jesse-rust/src/lib.rs`
- `ext-systems/jesse-rust/src/indicators.rs`
- `ext-systems/jesse-rust/Cargo.toml`
- `ext-systems/jesse-rust/pyproject.toml`
- `ext-systems/jesse-rust/__init__.py`

---

## Setup: Development Install

```bash
cd ext-systems/jesse-rust

# Option A — use the helper script (recommended for first-time setup)
bash build-local.sh

# Option B — manual steps
pip install --upgrade maturin numpy
maturin develop --release

# Verify the build
python3 -c "
import jesse_rust, numpy as np
arr = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 102.5], dtype=np.float64)
print('RSI:', jesse_rust.rsi(arr, 5))
print('Functions:', [f for f in dir(jesse_rust) if not f.startswith('_')])
"
```

---

## Build Wheel for Distribution

```bash
# Host platform only (fast)
maturin build --release

# All supported targets (CI / release)
bash build-all-wheels.sh

# Publish to PyPI
maturin publish --skip-existing dist/*
```

Wheels land in `dist/` following the naming convention:
`jesse_rust-{version}-cp{python}-cp{python}-{platform}.whl`

---

## Configuration Reference

### `Cargo.toml` — Release Profiles

**Source**: `ext-systems/jesse-rust/Cargo.toml`

```toml
[profile.release]
opt-level = 3        # Maximum speed optimisation
lto = true           # Link-time optimisation (whole-program)
codegen-units = 1    # Single codegen unit for best inlining
panic = "abort"      # Smaller binary, faster panics
strip = "symbols"    # Remove debug symbols from binary

[profile.release-small]
inherits = "release"
opt-level = "z"      # Optimise for binary size instead of speed
```

To use the size-focused profile:
```bash
maturin build --profile release-small
```

### `pyproject.toml` — Maturin and cibuildwheel

**Source**: `ext-systems/jesse-rust/pyproject.toml`

| Key | Value | Meaning |
|---|---|---|
| `[tool.maturin] features` | `["pyo3/extension-module"]` | Required for cdylib Python extensions |
| `module-name` | `jesse_rust` | Name of the compiled `.so` / `.pyd` |
| `python-source` | `.` | Root of the Python package |
| `[tool.cibuildwheel] build` | `cp310-* cp311-* cp312-* cp313-*` | Target CPython versions |
| `skip` | `*-win32 *-manylinux_i686 *-musllinux_i686` | Excluded 32-bit targets |

### Maturin Feature Flags

`pyo3/extension-module` is mandatory when building a `cdylib` that will be loaded into an existing Python interpreter. Without it, PyO3 will link the Python library statically, causing symbol conflicts at runtime.

---

## Adding a New Indicator

1. Implement the function in `ext-systems/jesse-rust/src/indicators.rs`:

```rust
/// Calculate MyIndicator
#[pyfunction]
pub fn my_indicator(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let arr = source.as_array();
        let n = arr.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        // ... computation ...
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}
```

2. Register it in `ext-systems/jesse-rust/src/lib.rs`:

```rust
m.add_function(wrap_pyfunction!(my_indicator, m)?)?;
```

3. Rebuild:
```bash
maturin develop --release
```

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'jesse_rust'`

**Cause**: The Rust extension has not been compiled for the current Python environment.

**Fix**:
```bash
cd ext-systems/jesse-rust
pip install maturin
maturin develop --release
```

If the warning from `__init__.py` appears (`Warning: Rust native module 'jesse_rust' not compiled`), the Python shim loaded but found no `.so`/`.pyd` file.

---

### 2. `ImportError: dynamic module does not define module export function`

**Cause**: The wheel was built for a different Python version or architecture (ABI mismatch).

**Fix**: Rebuild for the exact Python interpreter in use:
```bash
maturin develop --release  # always targets the active interpreter
```
Or install the correct wheel from PyPI:
```bash
pip install --force-reinstall jesse-rust
```

---

### 3. Rust compilation fails: `error: toolchain 'stable-...' is not installed`

**Cause**: Rust is installed but the active toolchain or cross-compilation target is missing.

**Fix**:
```bash
rustup update stable
rustup target add x86_64-unknown-linux-gnu   # or the required target
```

---

### 4. `ValueError` / `IndexError` at runtime — "index out of bounds"

**Cause**: Input array is shorter than `period`. All indicator functions check `n <= period` and return early with an all-NaN array. If a panic still occurs, it points to a missing bounds check.

**Diagnosis**:
```python
import jesse_rust, numpy as np
arr = np.array([1.0, 2.0], dtype=np.float64)  # too short
result = jesse_rust.rsi(arr, 14)  # should return all-NaN, not panic
print(result)
```

**Fix**: Report on [GitHub Issues](https://github.com/jesse-ai/jesse-rust/issues) with the indicator name and input length.

---

### 5. Incorrect numerical results compared to a reference implementation

**Cause**: Differences in warm-up seeding (how the first EMA / smoothed value is initialised). `jesse-rust` typically seeds EMA-based indicators at `source[0]` rather than using a simple-average warm-up.

**Investigation**:
- Compare `ema(source, period)` at index `period - 1` against a pandas EWM result with `adjust=False`.
- Check the `alpha` formula: `jesse-rust` uses `alpha = 2 / (period + 1)` for standard EMA.

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `ema()` function.

---

### 6. `maturin publish` fails with `403 Forbidden`

**Cause**: Missing or expired PyPI API token.

**Fix**:
```bash
# Set token via environment variable
export MATURIN_PYPI_TOKEN=pypi-xxxxx
maturin publish --skip-existing dist/*
```

---

### 7. `aarch64` cross-compilation fails on Linux

**Cause**: `aarch64-linux-gnu-gcc` cross-linker is not installed.

**Fix** (Debian/Ubuntu):
```bash
sudo apt-get install gcc-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

**Source**: `ext-systems/jesse-rust/build-all-wheels.sh`, cross-compiler check block.

---

## Security Considerations

- **No network access**: The library performs no network calls. All input data is passed in by the caller.
- **Memory safety**: Rust's borrow checker eliminates use-after-free and buffer-overflow classes of bugs at compile time. The `PyReadonlyArray` wrappers enforce that input arrays are not mutated by Rust code.
- **Panic policy**: `panic = "abort"` in release builds. An unexpected panic (e.g. from an ndarray bounds check) will terminate the process rather than propagate an undefined state. Input validation (length checks, NaN propagation) prevents panics on well-formed inputs.
- **Integer overflow**: All loop counters are `usize`. Arithmetic on `period as f64` is used to avoid integer-truncation errors in smoothing formulas.
- **Dependency audit**: Run `cargo audit` periodically to check for known CVEs in crate dependencies.

```bash
cargo install cargo-audit
cargo audit
```

---

## See Also

- [README.md](README.md) — Overview and quick start
- [architecture.md](architecture.md) — Component breakdown and dependency table
- [workflow.md](workflow.md) — Build pipeline and data-flow diagrams
- [state-management.md](state-management.md) — Internal rolling-state patterns
