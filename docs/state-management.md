# Jesse-Rust — State Management

## Overview

`jesse-rust` is a **stateless** library: every exported function is a pure, deterministic transformation from input arrays to output arrays. There are no persistent objects, no global mutable state, and no side effects between calls. All rolling state lives on the Rust stack or in heap-allocated locals that are created, used, and dropped within a single function invocation.

---

## Internal State Structures

Despite the stateless API surface, several important in-function state patterns are used to achieve O(n) complexity.

### 1. Wilder Smoothing Scalars

**Used by**: `rsi`, `srsi`, `adx`, `atr`, `di`, `dm`

**Source**: `ext-systems/jesse-rust/src/indicators.rs`

```rust
// Example from rsi()
let mut avg_gain: f64 = sum_gain / period as f64;
let mut avg_loss: f64 = sum_loss / period as f64;
// ...
for i in (period + 1)..n {
    avg_gain = (avg_gain * (period as f64 - 1.0) + current_gain) / period as f64;
    avg_loss = (avg_loss * (period as f64 - 1.0) + current_loss) / period as f64;
}
```

State is a pair of `f64` scalars updated in a single forward pass. This avoids keeping any price history beyond the current element.

### 2. Monotonic Deque (Sliding-Window Min/Max)

**Used by**: `donchian`, `chop` (ATR-sum sliding max/min), `stoch`, `stochf`, `willr`

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `donchian()` and `chop()`

```rust
let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
```

Each deque stores `(array_index, value)` pairs and maintains a monotonic invariant:
- **max_deque**: decreasing by value (front = current maximum)
- **min_deque**: increasing by value (front = current minimum)

On each step:
1. Pop expired entries from the **front** (index <= i - period) in O(1).
2. Pop dominated entries from the **back** before inserting the new value in amortised O(1).
3. Read O(1) window max/min from the front.

Overall complexity: O(n) for the entire array.

### 3. Rolling Sum / Sum-of-Squares

**Used by**: `bollinger_bands`, `bollinger_bands_width`, `sma`, `moving_std`

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `bollinger_bands_width()` lines 500–558

```rust
let mut sum: f64 = 0.0;
let mut sum_sq: f64 = 0.0;
// Sliding window update:
sum = sum - old_val + new_val;
sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
let sma = sum / period as f64;
let variance = (sum_sq / period as f64) - (sma * sma);
```

Two scalar accumulators replace an O(period) re-scan per step, giving O(n) total.

### 4. Triple EMA Cascade

**Used by**: `tema`, `dema`, `t3`, `kama` (with Kaufman SC), `zlema`

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `tema()` lines 411–439

```rust
let mut ema1 = source_array[0];
let mut ema2 = ema1;
let mut ema3 = ema2;
for i in 1..n {
    ema1 = alpha * source_array[i] + (1.0 - alpha) * ema1;
    ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
    ema3 = alpha * ema2 + (1.0 - alpha) * ema3;
    result[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
}
```

Three scalar `f64` values carry forward the full EMA chain with zero heap allocation.

### 5. Circular Buffer (VecDeque as Fixed Queue)

**Used by**: `srsi` (RSI buffer for stochastic window), `srsi` (K-buffer for smoothing)

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `srsi()` lines 189–315

```rust
let mut rsi_buffer = std::collections::VecDeque::with_capacity(period_stoch);
// ...
rsi_buffer.push_back(rsi_val);
if rsi_buffer.len() > period_stoch {
    rsi_buffer.pop_front();
}
```

Pre-allocated `VecDeque` acts as a fixed-capacity circular buffer. Capacity is set at initialisation, so no re-allocation occurs during the main loop.

### 6. KAMA Efficiency Ratio State

**Used by**: `kama`

**Source**: `ext-systems/jesse-rust/src/indicators.rs`, `kama()` lines 67–131

```rust
let mut volatility_sum: f64 = 0.0;
// Pre-compute price_diffs Vec<f64> once
// Then rolling update:
volatility_sum += price_diffs[i - 1];
volatility_sum -= price_diffs[i - period - 1];
let er = if volatility_sum != 0.0 { change / volatility_sum } else { 0.0 };
let sc = (er * alpha_diff + slow_alpha).powi(2);
result[i] = result[i - 1] + sc * (source_array[i] - result[i - 1]);
```

A pre-computed `Vec<f64>` of absolute price differences enables O(1) rolling volatility updates.

---

## Output Array Initialization

```mermaid
stateDiagram-v2
    [*] --> Allocate: Array1::from_elem(n, f64::NAN)
    Allocate --> WarmUp: i < period
    WarmUp --> WarmUp: accumulate initial sums / smoothed values
    WarmUp --> Compute: i >= period
    Compute --> Compute: rolling update, write result[i]
    Compute --> Return: i == n-1
    Return --> [*]: PyArray1::from_array(py, &result).to_owned()
```

All output arrays are pre-filled with `f64::NAN`. The warm-up phase accumulates enough history to seed the rolling state; only then do valid values appear. Callers should treat any leading `NaN` values as "not enough data yet."

---

## Data Structures Summary

| Structure | Type | Indicator Examples | Purpose |
|---|---|---|---|
| Wilder scalars | `f64, f64` | `rsi`, `adx`, `atr` | O(1) exponential smoothing |
| Monotonic deque | `VecDeque<(usize, f64)>` | `donchian`, `chop`, `stoch` | O(1) sliding window min/max |
| Rolling sum pair | `f64, f64` | `bollinger_bands`, `sma` | O(1) mean and variance |
| Triple EMA scalars | `f64, f64, f64` | `tema`, `dema`, `t3` | cascaded EMA without allocation |
| Fixed circular buffer | `VecDeque<f64>` | `srsi` | fixed-window history |
| Pre-computed diff Vec | `Vec<f64>` | `kama` | O(1) rolling volatility sum |
| Output buffer | `Array1<f64>` | all functions | mutable result accumulator |

---

## GIL and Concurrency

Every function acquires the Python GIL for its entire execution:

```rust
Python::with_gil(|py| {
    // all computation happens here
})
```

This means:
- Calls from Python are serialised at the GIL level (standard CPython behaviour).
- No Rust-level concurrency is used inside individual functions.
- The GIL is released as soon as `with_gil` returns, allowing other Python threads to run between indicator calls.

For parallel computation across multiple indicators, callers can use Python `concurrent.futures.ProcessPoolExecutor` (separate processes bypass the GIL entirely).

---

## See Also

- [architecture.md](architecture.md) — Component overview and candle column convention
- [workflow.md](workflow.md) — Rust–Python data flow and computation patterns
- [development.md](development.md) — Setup and build instructions
