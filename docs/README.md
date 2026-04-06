# Jesse-Rust

> **Last Updated**: 2026-04-07T00:00:00Z
> **Git Hash**: `ef788a2`

High-performance Rust extension module providing technical indicators for the Jesse algorithmic trading framework. Compiled as a native Python extension (`.so`/`.pyd`) via PyO3 and Maturin, delivering 5–10x speedups over equivalent pure-Python implementations.

---

## Key Features

| Feature | Detail |
|---|---|
| Language | Rust (compiled to native Python extension) |
| Python Bridge | PyO3 0.20 + numpy crate |
| Array Interface | Zero-copy NumPy array I/O via `PyReadonlyArray1` / `PyReadonlyArray2` |
| Precision | `f64` throughout; `rust_decimal` for exact arithmetic utilities |
| Platforms | Linux x86_64/aarch64, macOS x86_64/arm64, Windows AMD64 |
| Python versions | 3.10, 3.11, 3.12, 3.13 |
| Build tool | Maturin 1.8+ |
| Optimization | `opt-level = 3`, LTO, `codegen-units = 1`, `panic = "abort"` |
| License | MIT |

---

## Indicators at a Glance

### Trend / Moving Averages
`ema`, `sma`, `smma`, `tema`, `dema`, `zlema`, `wma`, `vwma`, `kama`, `frama`, `t3`, `alligator`

### Momentum / Oscillators
`rsi`, `srsi`, `macd`, `stoch`, `stochf`, `adx`, `dx`, `di`, `dm`, `adosc`, `cvi`, `dti`, `fosc`

### Volatility / Bands
`bollinger_bands`, `bollinger_bands_width`, `atr`, `chop`, `chande`, `donchian`

### Structural / Composite
`ichimoku_cloud`, `vi`, `wt`, `vwap`

### Utilities
`shift`, `moving_std`, `sum_floats`, `subtract_floats`

---

## Quick Start

```bash
pip install jesse-rust
```

```python
import numpy as np
import jesse_rust

# Example: RSI on a close-price series
close_prices = np.array([100.0, 102.5, 101.3, 104.0, 103.8, 106.2, 105.0], dtype=np.float64)
rsi_values = jesse_rust.rsi(close_prices, period=14)

# Example: MACD
macd_line, signal_line, histogram = jesse_rust.macd(close_prices, fast_period=12, slow_period=26, signal_period=9)

# Example: Bollinger Bands
upper, middle, lower = jesse_rust.bollinger_bands(close_prices, period=20, devup=2.0, devdn=2.0)

# Example: Ichimoku Cloud (requires OHLCV candles — shape [n, 6])
candles = np.random.rand(100, 6)
conversion, base, span_a, span_b = jesse_rust.ichimoku_cloud(candles, 9, 26, 52, 26)
```

---

## Architecture Summary

```
ext-systems/jesse-rust/
├── src/
│   ├── lib.rs          # PyO3 module entry point — registers all functions
│   └── indicators.rs   # All indicator implementations (~1 500 lines)
├── Cargo.toml          # Rust package manifest + release profiles
├── pyproject.toml      # Python build metadata (Maturin backend)
├── __init__.py         # Python shim — re-exports Rust symbols
├── build-local.sh      # Dev build helper
└── build-all-wheels.sh # CI cross-compilation script
```

The library has a single compilation unit with no sub-crates. All 40+ indicator functions live in `src/indicators.rs` and are wired into the `jesse_rust` Python module in `src/lib.rs`.

---

## Documentation Index

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | System design, component breakdown, module diagram |
| [workflow.md](workflow.md) | Processing pipeline, Rust–Python data flow |
| [state-management.md](state-management.md) | Internal state machines, data structures |
| [development.md](development.md) | Setup, build, config reference, troubleshooting |

---

## Links

- **Jesse Framework**: <https://jesse.trade>
- **Jesse Docs**: <https://docs.jesse.trade>
- **GitHub**: <https://github.com/jesse-ai/jesse-rust>
- **PyPI**: <https://pypi.org/project/jesse-rust/>
- **PyO3**: <https://pyo3.rs/>
- **Maturin**: <https://github.com/PyO3/maturin>

---

## Tags

`rust` `python-extension` `pyo3` `maturin` `technical-indicators` `trading` `jesse` `numpy` `high-performance` `algorithmic-trading` `rsi` `macd` `bollinger-bands` `ichimoku`
