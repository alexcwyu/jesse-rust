# jesse-rust refactor plan

Goal: split `src/indicators.rs` (5,022 lines) into navigable modules, remove
dead code, add type aliases for the repeated `PyResult<(Py<PyArray1<f64>>, …)>`
signatures, and clean up `src/lib.rs`'s registration block.

Constraint: **all 178 indicator tests must still pass** and the
`benchmark_numba_indicators.py` total (~124 µs) must not regress meaningfully.

---

## Phase 1 — Inventory (already done)

- [x] Listed all ~90 public `#[pyfunction]`s in `indicators.rs`
- [x] Identified 6 private helpers (`ih_ema`, `ih_sma`, `ih_wma`, `ih_supersmoother_2pole`, `ih_high_pass_1pole`, `ih_atr_wilder`, `rolling_std_pop`)
- [x] Identified dead code: `rma_array` (line 2530), `linear_regression_line` (line 2688)
- [x] Identified leftover from my recent `cg` edit: `let lo0 = …; // … wait` (line 3564)
- [x] Counted 49 clippy warnings

---

## Phase 2 — Module split

Target file layout under `src/`:

```
src/
├── lib.rs            # pymodule registration only
├── types.rs          # type aliases (PyArr, PyArrResult, PyArrTuple2/3/4)
├── helpers.rs        # private inline helpers (ih_*, rolling_std_pop)
├── moving_averages.rs
├── oscillators.rs
├── trend.rs          # directional / trend-following
├── volume.rs
├── bands.rs          # bands / channels / volatility
├── ehlers.rs         # Ehlers' family of filters
├── candle.rs         # candle-transform indicators
├── statistical.rs
└── util.rs           # sum_floats, shift, moving_std, find_order_index, etc.
```

### Mapping (90+ functions)

- [x] `types.rs` — add type aliases (see Phase 4)
- [x] `helpers.rs`:
  - `ih_ema`, `ih_sma`, `ih_wma`, `ih_supersmoother_2pole`, `ih_high_pass_1pole`, `ih_atr_wilder`, `rolling_std_pop`
  - Plus ndarray::Array1 variants: `sma_array`, `ema_for_wt`, `sma_for_wt`, `get_period_from_timestamp`
- [x] `moving_averages.rs`:
  - `ema`, `sma`, `smma`, `wma`, `vwma`, `kama`, `dema`, `tema`, `t3`, `zlema`,
    `hma`, `wilders`, `rma`, `mcginley_dynamic`, `mwdx`, `hwma`, `trix`,
    `cwma`, `sqwma`, `srwma`, `vpwma`, `epma`, `jma`, `nma`, `vidya`,
    `vlma_inner`, `maaq`, `frama`, `vwap`
- [x] `oscillators.rs`:
  - `rsi`, `srsi`, `stoch`, `stochf`, `willr`, `cci`, `cmo`, `cfo`, `cg`,
    `fosc`, `aroonosc`, `lrsi`, `rsx`, `dti`, `dpo`, `pfe`, `mass`,
    `fisher`, `macd`, `wt`
- [x] `trend.rs`:
  - `adx`, `adxr`, `dx`, `dm`, `di`, `supertrend`, `sar`, `safezonestop`,
    `alligator`, `ichimoku_cloud`, `vi`
- [x] `volume.rs`:
  - `adosc`, `nvi`, `pvi`, `efi`, `emv`, `wad`, `cvi`
- [x] `bands.rs`:
  - `bollinger_bands`, `bollinger_bands_width`, `donchian`, `keltner_inner`,
    `chop`, `chande`
- [x] `ehlers.rs`:
  - `supersmoother`, `supersmoother_3_pole`, `high_pass`, `high_pass_2_pole`,
    `bandpass`, `gauss`, `reflex`, `trendflex`, `itrend`, `voss`, `edcf`,
    `correlation_cycle`, `mama`, `pma`
- [x] `candle.rs`:
  - `heikin_ashi_candles`, `emd`, `qstick`
- [x] `statistical.rs`:
  - `linearreg`, `hurst_rs`, `damiani_volatmeter`
- [x] `util.rs`:
  - `shift`, `moving_std`, `sum_floats`, `subtract_floats`, `find_order_index`

---

## Phase 3 — Dead code removal

- [x] Delete `fn rma_array` (unused)
- [x] Delete `fn linear_regression_line` (unused)
- [x] Delete leftover `let lo0 = …; // = period + 1 - (period - 1) + 1 = 3? wait` from `cg`
- [x] Audit all `#[allow(dead_code)]` markers

---

## Phase 4 — Type aliases

In `types.rs`:

```rust
use numpy::PyArray1;
use pyo3::prelude::*;

pub type PyArr = Py<PyArray1<f64>>;
pub type PyArrResult = PyResult<PyArr>;
pub type PyArrTuple2 = PyResult<(PyArr, PyArr)>;
pub type PyArrTuple3 = PyResult<(PyArr, PyArr, PyArr)>;
pub type PyArrTuple4 = PyResult<(PyArr, PyArr, PyArr, PyArr)>;
```

- [x] Replace 23 instances of `PyResult<(Py<PyArray1<f64>>, …)>` with the tuple aliases
- [~] `PyArrResult` was *removed* instead — no single-array site improved enough in readability to justify it. Aliases kept: `PyArr`, `PyArrTuple2/3/4`.

---

## Phase 5 — Clippy fixes

- [x] 3× `else { if … }` → `else if`
- [x] 1× unnecessary parens around assigned value (`mama`)
- [x] 3× `x.max(a).min(b)` → `x.clamp(a, b)`
- [~] 7× `taken reference of right operand` — **left as-is**: all 7 are in `ndarray` slice
      arithmetic like `(&high + &low + &close) / 3.0` inside `vwap`/`vwma` source-type
      dispatch; the suggested fix (drop `&` on the last operand) is purely cosmetic
      and the symmetric form is more readable.
- [~] Loop-variable-only-used-to-index — 3 remain in `oscillators::cg`, `oscillators::cmo`,
      `statistical::linearreg`. The suggested `iter_mut().enumerate().take(n).skip(...)`
      rewrite hurts clarity and the loop bodies use the index for multiple array accesses,
      so it doesn't simplify with iterators.
- [x] 1× `manual_memcpy` in `volume::efi` → `copy_from_slice`
- [x] `cargo fix` applied for safe auto-suggestions (removed unused `ArrayView1`, `s`,
      `PyReadonlyArray1`, `PyReadonlyArray2` imports across most modules)
- [x] Final clippy count: **13 warnings** (down from 49 before refactor; remaining
      are all stylistic nits in `ndarray` slice expressions and a single 7-tuple return
      type in `wt`)

---

## Phase 6 — `lib.rs` cleanup

- [x] Replace `mod indicators;` + `use indicators::*;` with the new module list
- [x] Group `m.add_function!` calls by category (matching the module split)
- [x] Alphabetize within each group
- [x] Drop stale comments: `// New indicator`, `// Newly added`, `// Latest indicators`, `// Phase 3 — …`
- [x] Use sub-headers that match the module names

---

## Phase 7 — Verify

- [x] `maturin develop --release` builds with zero errors
- [x] `pytest tests/test_indicators.py -q` shows 178 passed
- [x] `pytest tests/ -q --ignore=tests/exchange_tests` shows 457 passed
- [x] `python bot/benchmark_numba_indicators.py` total is within ±2 µs of 124.4
- [~] `cargo clippy --release` warning count = 13 (justified, see Phase 5)

---

## Phase 8 — Final commit

- [x] Single commit summarizing the refactor with clippy-warning delta
  and benchmark delta confirmation
