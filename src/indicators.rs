use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, s, ArrayView1};
use pyo3::types::PyDict;
use rust_decimal::Decimal;
use std::str::FromStr;

/// Calculate RSI (Relative Strength Index)
#[pyfunction]
pub fn rsi(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate initial sum of gains and losses
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        for i in 1..=period {
            let change = source_array[i] - source_array[i - 1];
            if change > 0.0 {
                sum_gain += change;
            } else {
                sum_loss += change.abs();
            }
        }

        let mut avg_gain = sum_gain / period as f64;
        let mut avg_loss = sum_loss / period as f64;

        // Calculate first RSI value
        if avg_loss == 0.0 {
            result[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Calculate subsequent RSI values using Wilder's smoothing
        for i in (period + 1)..n {
            let change = source_array[i] - source_array[i - 1];
            let (current_gain, current_loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, change.abs())
            };

            avg_gain = (avg_gain * (period as f64 - 1.0) + current_gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + current_loss) / period as f64;

            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate KAMA (Kaufman Adaptive Moving Average) - Optimized version
#[pyfunction]
pub fn kama(source: PyReadonlyArray1<f64>, period: usize, fast_length: usize, slow_length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::zeros(n);
        
        if n <= period {
            // Fill with source values when we don't have enough data
            for i in 0..n {
                result[i] = source_array[i];
            }
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate the efficiency ratio multiplier
        let fast_alpha = 2.0 / (fast_length as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_length as f64 + 1.0);
        let alpha_diff = fast_alpha - slow_alpha;
        
        // First 'period' values are same as source
        for i in 0..period {
            result[i] = source_array[i];
        }
        
        // Pre-calculate price differences for rolling volatility
        let mut price_diffs = Vec::with_capacity(n - 1);
        for i in 1..n {
            price_diffs.push((source_array[i] - source_array[i - 1]).abs());
        }
        
        // Initialize rolling volatility sum for the first window
        let mut volatility_sum = 0.0;
        for i in 0..(period - 1) {
            volatility_sum += price_diffs[i];
        }
        
        // Start the calculation after the initial period
        for i in period..n {
            // Calculate Efficiency Ratio using rolling volatility
            let change = (source_array[i] - source_array[i - period]).abs();
            
            // Update rolling volatility sum
            if i >= period {
                // Add new difference, remove old difference
                volatility_sum += price_diffs[i - 1]; // Current period's difference
                if i > period {
                    volatility_sum -= price_diffs[i - period - 1]; // Remove oldest difference
                }
            }
            
            let er = if volatility_sum != 0.0 { change / volatility_sum } else { 0.0 };
            
            // Calculate smoothing constant
            let sc = (er * alpha_diff + slow_alpha).powi(2);
            
            // Calculate KAMA
            result[i] = result[i - 1] + sc * (source_array[i] - result[i - 1]);
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Ichimoku Cloud
#[pyfunction]
pub fn ichimoku_cloud(
    candles: PyReadonlyArray2<f64>,
    conversion_line_period: usize,
    base_line_period: usize,
    lagging_line_period: usize,
    displacement: usize
) -> PyResult<(f64, f64, f64, f64)> {
    Python::with_gil(|_py| {
        let candles_array = candles.as_array();
        
        // Get the high and low price columns
        let high_prices = candles_array.slice(s![.., 3]);
        let low_prices = candles_array.slice(s![.., 4]);
        
        // Calculate for earlier period (displaced)
        let earlier_high = high_prices.slice(s![..-((displacement as isize) - 1)]);
        let earlier_low = low_prices.slice(s![..-((displacement as isize) - 1)]);
        
        // Helper function to get period high and low
        let get_period_hl = |highs: ndarray::ArrayView1<f64>, lows: ndarray::ArrayView1<f64>, period: usize| -> (f64, f64) {
            let n = highs.len();
            if n < period {
                return (f64::NAN, f64::NAN);
            }
            
            // Instead of negative slicing, use the last 'period' elements
            let start_idx = n.saturating_sub(period);
            let period_high = highs.slice(s![start_idx..]).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low = lows.slice(s![start_idx..]).fold(f64::INFINITY, |a, &b| a.min(b));
            
            (period_high, period_low)
        };
        
        // Earlier periods calculations
        let (small_ph, small_pl) = get_period_hl(earlier_high, earlier_low, conversion_line_period);
        let (mid_ph, mid_pl) = get_period_hl(earlier_high, earlier_low, base_line_period);
        let (long_ph, long_pl) = get_period_hl(earlier_high, earlier_low, lagging_line_period);
        
        let early_conversion_line = (small_ph + small_pl) / 2.0;
        let early_base_line = (mid_ph + mid_pl) / 2.0;
        let span_a = (early_conversion_line + early_base_line) / 2.0;
        let span_b = (long_ph + long_pl) / 2.0;
        
        // Current period calculations
        let (current_small_ph, current_small_pl) = get_period_hl(high_prices, low_prices, conversion_line_period);
        let (current_mid_ph, current_mid_pl) = get_period_hl(high_prices, low_prices, base_line_period);
        
        let current_conversion_line = (current_small_ph + current_small_pl) / 2.0;
        let current_base_line = (current_mid_ph + current_mid_pl) / 2.0;
        
        Ok((current_conversion_line, current_base_line, span_a, span_b))
    })
}

/// Calculate SRSI (Stochastic RSI) - Optimized version
#[pyfunction]
pub fn srsi(source: PyReadonlyArray1<f64>, period: usize, period_stoch: usize, k_period: usize, d_period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut k_values = Array1::<f64>::from_elem(n, f64::NAN);
        let mut d_values = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }

        // Inline RSI calculation with rolling stochastic
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        
        // Calculate initial RSI values
        for i in 1..=period {
            let change = source_array[i] - source_array[i - 1];
            if change > 0.0 {
                sum_gain += change;
            } else {
                sum_loss += change.abs();
            }
        }
        
        let mut avg_gain = sum_gain / period as f64;
        let mut avg_loss = sum_loss / period as f64;
        
        // Rolling buffers for stochastic calculation
        let mut rsi_buffer = std::collections::VecDeque::with_capacity(period_stoch);
        let mut k_buffer = std::collections::VecDeque::with_capacity(k_period);
        
        // Process each data point
        for i in period..n {
            // Calculate RSI for current point
            let rsi_val = if i == period {
                // First RSI value
                if avg_loss == 0.0 {
                    100.0
                } else {
                    let rs = avg_gain / avg_loss;
                    100.0 - (100.0 / (1.0 + rs))
                }
            } else {
                // Subsequent RSI values using Wilder's smoothing
                let change = source_array[i] - source_array[i - 1];
                let (current_gain, current_loss) = if change > 0.0 {
                    (change, 0.0)
                } else {
                    (0.0, change.abs())
                };
                
                avg_gain = (avg_gain * (period as f64 - 1.0) + current_gain) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + current_loss) / period as f64;
                
                if avg_loss == 0.0 {
                    100.0
                } else {
                    let rs = avg_gain / avg_loss;
                    100.0 - (100.0 / (1.0 + rs))
                }
            };
            
            // Add to RSI buffer
            rsi_buffer.push_back(rsi_val);
            if rsi_buffer.len() > period_stoch {
                rsi_buffer.pop_front();
            }
            
            // Calculate %K when we have enough RSI values
            if rsi_buffer.len() == period_stoch {
                let rsi_min = rsi_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let rsi_max = rsi_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                let k_val = if rsi_max != rsi_min {
                    100.0 * (rsi_val - rsi_min) / (rsi_max - rsi_min)
                } else {
                    f64::NAN
                };
                
                if k_period > 1 {
                    // Smooth %K
                    k_buffer.push_back(k_val);
                    if k_buffer.len() > k_period {
                        k_buffer.pop_front();
                    }
                    
                    if k_buffer.len() == k_period && k_buffer.iter().all(|&x| !x.is_nan()) {
                        let k_smoothed = k_buffer.iter().sum::<f64>() / k_period as f64;
                        k_values[i] = k_smoothed;
                    }
                } else {
                    // No smoothing needed
                    k_values[i] = k_val;
                }
            }
        }
        
        // Calculate %D (SMA of %K) using rolling sum
        if d_period > 0 {
            let mut d_buffer = std::collections::VecDeque::with_capacity(d_period);
            
            for i in 0..n {
                if !k_values[i].is_nan() {
                    d_buffer.push_back(k_values[i]);
                    if d_buffer.len() > d_period {
                        d_buffer.pop_front();
                    }
                    
                    if d_buffer.len() == d_period {
                        d_values[i] = d_buffer.iter().sum::<f64>() / d_period as f64;
                    }
                }
            }
        }
        
        Ok((
            PyArray1::from_array(py, &k_values).to_owned(),
            PyArray1::from_array(py, &d_values).to_owned()
        ))
    })
}

/// Calculate ADX (Average Directional Index)
#[pyfunction]
pub fn adx(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut adx_result = Array1::<f64>::from_elem(n, f64::NAN);

        let required_len = 2 * period;
        if n <= required_len {
            return Ok(PyArray1::from_array(py, &adx_result).to_owned());
        }

        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);

        // State for Wilder smoothing
        let mut tr_smooth: f64 = 0.0;
        let mut plus_dm_smooth: f64 = 0.0;
        let mut minus_dm_smooth: f64 = 0.0;
        
        // Buffer for DX values to calculate the first ADX
        let mut dx_buffer: Vec<f64> = Vec::with_capacity(period);

        // Main calculation loop
        for i in 1..n {
            // 1. Calculate raw TR, +DM, -DM for current step `i`
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);

            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];

            let mut current_plus_dm = 0.0;
            if h_diff > l_diff && h_diff > 0.0 {
                current_plus_dm = h_diff;
            }

            let mut current_minus_dm = 0.0;
            if l_diff > h_diff && l_diff > 0.0 {
                current_minus_dm = l_diff;
            }

            // 2. Update smoothed values
            if i <= period {
                // Accumulate for the first smoothed value
                tr_smooth += current_tr;
                plus_dm_smooth += current_plus_dm;
                minus_dm_smooth += current_minus_dm;
            } else {
                // Apply Wilder's smoothing formula
                tr_smooth = tr_smooth - (tr_smooth / period as f64) + current_tr;
                plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period as f64) + current_plus_dm;
                minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period as f64) + current_minus_dm;
            }
            
            // From index `period` onwards, we can calculate DI and DX
            if i >= period {
                let mut current_dx = 0.0;
                if tr_smooth != 0.0 {
                    let di_plus = 100.0 * plus_dm_smooth / tr_smooth;
                    let di_minus = 100.0 * minus_dm_smooth / tr_smooth;
                    let di_sum = di_plus + di_minus;
                    if di_sum != 0.0 {
                        current_dx = 100.0 * (di_plus - di_minus).abs() / di_sum;
                    }
                }
                
                // Store DX value for initial ADX calculation, or calculate ADX
                if i < required_len {
                    dx_buffer.push(current_dx);
                } else {
                    if i == required_len {
                        // First ADX value is the average of the buffer
                        let dx_sum: f64 = dx_buffer.iter().sum();
                        adx_result[i] = dx_sum / period as f64;
                    } else {
                        // Subsequent ADX values are smoothed
                        if !adx_result[i - 1].is_nan() {
                           adx_result[i] = (adx_result[i - 1] * (period - 1) as f64 + current_dx) / period as f64;
                        }
                    }
                }
            }
        }

        Ok(PyArray1::from_array(py, &adx_result).to_owned())
    })
}

/// Calculate TEMA (Triple Exponential Moving Average)
#[pyfunction]
pub fn tema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::zeros(n);

        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema1 = source_array[0];
        let mut ema2 = ema1;
        let mut ema3 = ema2;

        result[0] = 3.0 * ema1 - 3.0 * ema2 + ema3;

        for i in 1..n {
            ema1 = alpha * source_array[i] + (1.0 - alpha) * ema1;
            ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
            ema3 = alpha * ema2 + (1.0 - alpha) * ema3;
            result[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate MACD (Moving Average Convergence/Divergence)
#[pyfunction]
pub fn macd(source: PyReadonlyArray1<f64>, fast_period: usize, slow_period: usize, signal_period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut macd_line_result = Array1::<f64>::zeros(n);
        let mut signal_line_result = Array1::<f64>::zeros(n);
        let mut hist_result = Array1::<f64>::zeros(n);

        if n == 0 {
            return Ok((
                PyArray1::from_array(py, &macd_line_result).to_owned(),
                PyArray1::from_array(py, &signal_line_result).to_owned(),
                PyArray1::from_array(py, &hist_result).to_owned(),
            ));
        }

        let alpha_fast = 2.0 / (fast_period as f64 + 1.0);
        let alpha_slow = 2.0 / (slow_period as f64 + 1.0);
        let alpha_signal = 2.0 / (signal_period as f64 + 1.0);

        let mut ema_fast = source_array[0];
        let mut ema_slow = source_array[0];
        
        let macd_val = ema_fast - ema_slow;
        let macd_val_cleaned = if macd_val.is_nan() { 0.0 } else { macd_val };
        
        let mut signal_ema = macd_val_cleaned;

        macd_line_result[0] = macd_val_cleaned;
        signal_line_result[0] = signal_ema;
        hist_result[0] = macd_val - signal_ema;

        for i in 1..n {
            ema_fast = alpha_fast * source_array[i] + (1.0 - alpha_fast) * ema_fast;
            ema_slow = alpha_slow * source_array[i] + (1.0 - alpha_slow) * ema_slow;

            let macd_val = ema_fast - ema_slow;
            let macd_val_cleaned = if macd_val.is_nan() { 0.0 } else { macd_val };
            
            signal_ema = alpha_signal * macd_val_cleaned + (1.0 - alpha_signal) * signal_ema;
            
            let hist_val = macd_val - signal_ema;

            macd_line_result[i] = macd_val_cleaned;
            signal_line_result[i] = signal_ema;
            hist_result[i] = hist_val;
        }

        Ok((
            PyArray1::from_array(py, &macd_line_result).to_owned(),
            PyArray1::from_array(py, &signal_line_result).to_owned(),
            PyArray1::from_array(py, &hist_result).to_owned(),
        ))
    })
}

/// Calculate Bollinger Bands Width - Optimized version
#[pyfunction]
pub fn bollinger_bands_width(source: PyReadonlyArray1<f64>, period: usize, mult: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Use rolling window for efficient calculation
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        // Initialize first window
        for i in 0..period {
            let val = source_array[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // Calculate first BBW value
        let sma = sum / period as f64;
        let variance = (sum_sq / period as f64) - (sma * sma);
        let std_dev = variance.sqrt();
        
        // Calculate Bollinger Bands Width
        if sma != 0.0 {
            let upper_band = sma + mult * std_dev;
            let lower_band = sma - mult * std_dev;
            result[period - 1] = (upper_band - lower_band) / sma;
        }
        
        // Calculate subsequent values using rolling window
        for i in period..n {
            let old_val = source_array[i - period];
            let new_val = source_array[i];
            
            // Update rolling sums
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
            
            // Calculate SMA and standard deviation
            let sma = sum / period as f64;
            let variance = (sum_sq / period as f64) - (sma * sma);
            let std_dev = variance.sqrt();
            
            // Calculate Bollinger Bands Width
            if sma != 0.0 {
                let upper_band = sma + mult * std_dev;
                let lower_band = sma - mult * std_dev;
                result[i] = (upper_band - lower_band) / sma;
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Bollinger Bands - Optimized version
#[pyfunction]
pub fn bollinger_bands(source: PyReadonlyArray1<f64>, period: usize, devup: f64, devdn: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut upper_band = Array1::<f64>::from_elem(n, f64::NAN);
        let mut middle_band = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lower_band = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok((
                PyArray1::from_array(py, &upper_band).to_owned(),
                PyArray1::from_array(py, &middle_band).to_owned(),
                PyArray1::from_array(py, &lower_band).to_owned()
            ));
        }

        // Use rolling window for efficient calculation
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        // Initialize first window
        for i in 0..period {
            let val = source_array[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // Calculate first Bollinger Bands values
        let sma = sum / period as f64;
        let variance = (sum_sq / period as f64) - (sma * sma);
        let std_dev = variance.sqrt();
        
        middle_band[period - 1] = sma;
        upper_band[period - 1] = sma + devup * std_dev;
        lower_band[period - 1] = sma - devdn * std_dev;
        
        // Calculate subsequent values using rolling window
        for i in period..n {
            let old_val = source_array[i - period];
            let new_val = source_array[i];
            
            // Update rolling sums
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
            
            // Calculate SMA and standard deviation
            let sma = sum / period as f64;
            let variance = (sum_sq / period as f64) - (sma * sma);
            let std_dev = variance.sqrt();
            
            // Calculate bands
            middle_band[i] = sma;
            upper_band[i] = sma + devup * std_dev;
            lower_band[i] = sma - devdn * std_dev;
        }

        Ok((
            PyArray1::from_array(py, &upper_band).to_owned(),
            PyArray1::from_array(py, &middle_band).to_owned(),
            PyArray1::from_array(py, &lower_band).to_owned()
        ))
    })
}

/// Shift array by a given number of periods
#[pyfunction]
pub fn shift(source: PyReadonlyArray1<f64>, periods: isize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if periods == 0 {
            // No shift, return copy of source
            for i in 0..n {
                result[i] = source_array[i];
            }
        } else if periods > 0 {
            // Shift right (positive periods)
            let shift_amount = periods as usize;
            if shift_amount < n {
                for i in shift_amount..n {
                    result[i] = source_array[i - shift_amount];
                }
            }
        } else {
            // Shift left (negative periods)
            let shift_amount = (-periods) as usize;
            if shift_amount < n {
                for i in 0..(n - shift_amount) {
                    result[i] = source_array[i + shift_amount];
                }
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate moving standard deviation
#[pyfunction]
pub fn moving_std(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate moving standard deviation
        for i in (period - 1)..n {
            let start_idx = i + 1 - period;
            let window = source_array.slice(s![start_idx..=i]);
            
            // Calculate mean
            let mean = window.sum() / period as f64;
            
            // Calculate variance
            let variance = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / period as f64;
            
            // Calculate standard deviation
            result[i] = variance.sqrt();
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Simple Moving Average (SMA)
#[pyfunction]
pub fn sma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate first SMA value
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..period {
            if !source_array[i].is_nan() {
                sum += source_array[i];
                count += 1;
            }
        }
        if count > 0 {
            result[period - 1] = sum / count as f64;
        }

        // Calculate subsequent SMA values using sliding window
        for i in period..n {
            if !source_array[i - period].is_nan() {
                sum -= source_array[i - period];
                count -= 1;
            }
            if !source_array[i].is_nan() {
                sum += source_array[i];
                count += 1;
            }
            if count > 0 {
                result[i] = sum / count as f64;
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate SMMA (Smoothed Moving Average)
#[pyfunction]
pub fn smma(source: PyReadonlyArray1<f64>, length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        // Convert PyArray to rust ndarray
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Create output array
        let mut result = Array1::<f64>::zeros(n);
        
        if n < length {
            // Return array of NaNs if we don't have enough data
            for i in 0..n {
                result[i] = f64::NAN;
            }
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate first SMMA value (SMA)
        let alpha = 1.0 / (length as f64);
        let mut total = 0.0;
        for i in 0..length {
            total += source_array[i];
        }
        let init_val = total / (length as f64);
        
        // Set first value
        result[length - 1] = init_val;
        
        // Calculate subsequent SMMA values
        for i in length..n {
            result[i] = alpha * source_array[i] + (1.0 - alpha) * result[i - 1];
        }
        
        // Fill initial values with NaN
        for i in 0..(length - 1) {
            result[i] = f64::NAN;
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Alligator indicator (Jaw, Teeth, Lips) - Optimized version
#[pyfunction]
pub fn alligator(source: PyReadonlyArray1<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Initialize result arrays
        let mut jaw = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 13 {  // Need at least 13 periods for jaw (longest period)
            return Ok((
                PyArray1::from_array(py, &jaw).to_owned(),
                PyArray1::from_array(py, &teeth).to_owned(),
                PyArray1::from_array(py, &lips).to_owned()
            ));
        }
        
        // SMMA parameters
        let jaw_period = 13;
        let teeth_period = 8;
        let lips_period = 5;
        
        let jaw_alpha = 1.0 / jaw_period as f64;
        let teeth_alpha = 1.0 / teeth_period as f64;
        let lips_alpha = 1.0 / lips_period as f64;
        
        // Calculate initial SMA values for each line
        let mut jaw_sum = 0.0;
        let mut teeth_sum = 0.0;
        let mut lips_sum = 0.0;
        
        // Accumulate sums for initial values
        for i in 0..jaw_period {
            jaw_sum += source_array[i];
            if i < teeth_period {
                teeth_sum += source_array[i];
            }
            if i < lips_period {
                lips_sum += source_array[i];
            }
        }
        
        // Initialize SMMA values (without shifts)
        let mut jaw_smma = jaw_sum / jaw_period as f64;
        let mut teeth_smma = teeth_sum / teeth_period as f64;
        let mut lips_smma = lips_sum / lips_period as f64;
        
        // Calculate all SMMA values first, then apply shifts
        let mut jaw_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Set initial SMMA values
        jaw_unshifted[jaw_period - 1] = jaw_smma;
        teeth_unshifted[teeth_period - 1] = teeth_smma;
        lips_unshifted[lips_period - 1] = lips_smma;
        
        // Calculate subsequent SMMA values using single pass
        for i in jaw_period..n {
            // Update jaw (13-period SMMA)
            jaw_smma = jaw_alpha * source_array[i] + (1.0 - jaw_alpha) * jaw_smma;
            jaw_unshifted[i] = jaw_smma;
            
            // Update teeth (8-period SMMA) if we have enough data
            if i >= teeth_period {
                teeth_smma = teeth_alpha * source_array[i] + (1.0 - teeth_alpha) * teeth_smma;
                teeth_unshifted[i] = teeth_smma;
            }
            
            // Update lips (5-period SMMA) if we have enough data
            if i >= lips_period {
                lips_smma = lips_alpha * source_array[i] + (1.0 - lips_alpha) * lips_smma;
                lips_unshifted[i] = lips_smma;
            }
        }
        
        // Apply shifts inline (forward shifts)
        // Jaw: shift by 8 periods forward
        for i in 0..(n - 8) {
            if i + 8 < n && !jaw_unshifted[i].is_nan() {
                jaw[i + 8] = jaw_unshifted[i];
            }
        }
        
        // Teeth: shift by 5 periods forward
        for i in 0..(n - 5) {
            if i + 5 < n && !teeth_unshifted[i].is_nan() {
                teeth[i + 5] = teeth_unshifted[i];
            }
        }
        
        // Lips: shift by 3 periods forward
        for i in 0..(n - 3) {
            if i + 3 < n && !lips_unshifted[i].is_nan() {
                lips[i + 3] = lips_unshifted[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &jaw).to_owned(),
            PyArray1::from_array(py, &teeth).to_owned(),
            PyArray1::from_array(py, &lips).to_owned()
        ))
    })
} 

/// Calculate DI (Directional Indicator) - Optimized version
#[pyfunction]
pub fn di(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result arrays
        let mut plus_di = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_di = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 2 || n < period + 1 {
            return Ok((
                PyArray1::from_array(py, &plus_di).to_owned(),
                PyArray1::from_array(py, &minus_di).to_owned()
            ));
        }
        
        // Extract OHLCV data (assuming standard format: open, high, low, close, volume)
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        
        // Initialize smoothed values for Wilder's smoothing
        let mut tr_smooth = 0.0;
        let mut plus_dm_smooth = 0.0;
        let mut minus_dm_smooth = 0.0;
        
        // Calculate initial sums for the first 'period' values
        for i in 1..=period {
            // Calculate True Range (TR)
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            // Calculate Directional Movements (+DM and -DM)
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Accumulate for initial smoothed values
            tr_smooth += current_tr;
            plus_dm_smooth += current_plus_dm;
            minus_dm_smooth += current_minus_dm;
        }
        
        // Calculate first DI values at index 'period'
        if tr_smooth > 0.0 {
            plus_di[period] = 100.0 * plus_dm_smooth / tr_smooth;
            minus_di[period] = 100.0 * minus_dm_smooth / tr_smooth;
        } else {
            plus_di[period] = 0.0;
            minus_di[period] = 0.0;
        }
        
        // Calculate subsequent DI values using Wilder's smoothing
        for i in (period + 1)..n {
            // Calculate current TR, +DM, -DM
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Apply Wilder's smoothing: smoothed[i] = (smoothed[i-1] * (period - 1) + current) / period
            tr_smooth = (tr_smooth * (period - 1) as f64 + current_tr) / period as f64;
            plus_dm_smooth = (plus_dm_smooth * (period - 1) as f64 + current_plus_dm) / period as f64;
            minus_dm_smooth = (minus_dm_smooth * (period - 1) as f64 + current_minus_dm) / period as f64;
            
            // Calculate DI values
            if tr_smooth > 0.0 {
                plus_di[i] = 100.0 * plus_dm_smooth / tr_smooth;
                minus_di[i] = 100.0 * minus_dm_smooth / tr_smooth;
            } else {
                plus_di[i] = 0.0;
                minus_di[i] = 0.0;
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_di).to_owned(),
            PyArray1::from_array(py, &minus_di).to_owned()
        ))
    })
}

/// Calculate CHOP (Choppiness Index) - Ultra-optimized version
#[pyfunction]
pub fn chop(candles: PyReadonlyArray2<f64>, period: usize, scalar: f64, drift: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate True Range (TR) array
        let mut tr = Array1::<f64>::zeros(n);
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
                          let hc = (high[i] - close[i - 1]).abs();
              let lc = (low[i] - close[i - 1]).abs();
              tr[i] = hl.max(hc).max(lc);
        }
        
        // Calculate ATR using Wilder's smoothing or simple average based on drift
        let mut atr = Array1::<f64>::from_elem(n, f64::NAN);
        
        if drift == 1 {
            // Simple case: ATR = TR
            atr.assign(&tr);
        } else {
            // Wilder's smoothing for ATR
            // Calculate initial ATR as simple average of first 'drift' values
            let mut sum = 0.0;
            for i in 0..drift {
                sum += tr[i];
            }
            atr[drift - 1] = sum / drift as f64;
            
            // Apply Wilder's smoothing for subsequent values
            let alpha = 1.0 / drift as f64;
            for i in drift..n {
                atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1];
            }
        }
        
        // Pre-calculate log10(period) for efficiency
        let log_period = (period as f64).log10();
        
        // Use rolling sum algorithm for ATR sum (O(n) instead of O(n*p))
        let mut atr_sum = 0.0;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        
        // Initialize the first window
        for i in 0..period {
            if !atr[i].is_nan() {
                atr_sum += atr[i];
            }
            if high[i] > highest {
                highest = high[i];
            }
            if low[i] < lowest {
                lowest = low[i];
            }
        }
        
        // Use deque-like structures for efficient rolling max/min
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::new();
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::new();
        
        // Initialize deques for the first window
        for i in 0..period {
            // Maintain max deque (decreasing order)
            while !max_deque.is_empty() && max_deque.back().unwrap().1 <= high[i] {
                max_deque.pop_back();
            }
            max_deque.push_back((i, high[i]));
            
            // Maintain min deque (increasing order)
            while !min_deque.is_empty() && min_deque.back().unwrap().1 >= low[i] {
                min_deque.pop_back();
            }
            min_deque.push_back((i, low[i]));
        }
        
        // Calculate first CHOP value
        if atr_sum > 0.0 {
            let range = highest - lowest;
            if range > 0.0 {
                result[period - 1] = (scalar * (atr_sum.log10() - range.log10())) / log_period;
            }
        }
        
        // Rolling window calculation for subsequent values
        for i in period..n {
            // Update rolling ATR sum
            if !atr[i].is_nan() {
                atr_sum += atr[i];
            }
            if !atr[i - period].is_nan() {
                atr_sum -= atr[i - period];
            }
            
            // Remove elements outside the window from deques
            while !max_deque.is_empty() && max_deque.front().unwrap().0 <= i - period {
                max_deque.pop_front();
            }
            while !min_deque.is_empty() && min_deque.front().unwrap().0 <= i - period {
                min_deque.pop_front();
            }
            
            // Add new element to deques
            while !max_deque.is_empty() && max_deque.back().unwrap().1 <= high[i] {
                max_deque.pop_back();
            }
            max_deque.push_back((i, high[i]));
            
            while !min_deque.is_empty() && min_deque.back().unwrap().1 >= low[i] {
                min_deque.pop_back();
            }
            min_deque.push_back((i, low[i]));
            
            // Get current max and min
            let current_highest = max_deque.front().unwrap().1;
            let current_lowest = min_deque.front().unwrap().1;
            
            // Calculate CHOP
            if atr_sum > 0.0 {
                let range = current_highest - current_lowest;
                if range > 0.0 {
                    result[i] = (scalar * (atr_sum.log10() - range.log10())) / log_period;
                }
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate ATR (Average True Range) - Ultra-optimized version
#[pyfunction]
pub fn atr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate True Range (TR) inline and ATR simultaneously
        let mut tr_sum = 0.0;
        
        // Calculate first TR value
        let first_tr = high[0] - low[0];
        tr_sum += first_tr;
        
        // Calculate subsequent TR values and accumulate for first period
        for i in 1..period {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        
        // First ATR value is the simple average of the first 'period' true ranges
        result[period - 1] = tr_sum / period as f64;
        
        // Calculate subsequent ATR values using Wilder's smoothing
        // Using the optimized formula: ATR[i] = (ATR[i-1] * (period - 1) + TR[i]) / period
        // Which can be rewritten as: ATR[i] = ATR[i-1] + (TR[i] - ATR[i-1]) / period
        let alpha = 1.0 / period as f64;
        
        for i in period..n {
            // Calculate current True Range
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            
            // Apply Wilder's smoothing: ATR[i] = ATR[i-1] + alpha * (TR[i] - ATR[i-1])
            result[i] = result[i - 1] + alpha * (tr - result[i - 1]);
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Chande (Chandelier Exit) - Ultra-optimized version
#[pyfunction]
pub fn chande(
    candles: PyReadonlyArray2<f64>, 
    period: usize, 
    mult: f64, 
    direction: &str
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Pre-allocate ATR array
        let mut atr = vec![f64::NAN; n];
        
        // Calculate ATR using rolling window - first pass
        let mut tr_sum = 0.0;
        
        // Calculate first TR value
        let first_tr = high[0] - low[0];
        tr_sum += first_tr;
        
        // Calculate subsequent TR values and accumulate for first period
        for i in 1..period {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        
        // First ATR value
        atr[period - 1] = tr_sum / period as f64;
        
        // Calculate subsequent ATR values using Wilder's smoothing
        let alpha = 1.0 / period as f64;
        for i in period..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            atr[i] = atr[i - 1] + alpha * (tr - atr[i - 1]);
        }
        
        // Calculate Chandelier Exit using rolling max/min with deques
        use std::collections::VecDeque;
        
        if direction == "long" {
            // For long: use rolling maximum of high prices
            let mut max_deque: VecDeque<(usize, f64)> = VecDeque::new();
            
            // Initialize the first window
            for i in 0..period.min(n) {
                // Remove elements that are smaller than current (maintain decreasing order)
                while let Some(&(_, val)) = max_deque.back() {
                    if val <= high[i] {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((i, high[i]));
            }
            
            // Calculate chandelier exit for the first valid position
            if period <= n {
                let max_high = max_deque.front().unwrap().1;
                result[period - 1] = max_high - atr[period - 1] * mult;
            }
            
            // Slide the window for remaining positions
            for i in period..n {
                // Remove elements outside the window
                while let Some(&(idx, _)) = max_deque.front() {
                    if idx <= i - period {
                        max_deque.pop_front();
                    } else {
                        break;
                    }
                }
                
                // Add new element maintaining decreasing order
                while let Some(&(_, val)) = max_deque.back() {
                    if val <= high[i] {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((i, high[i]));
                
                // Calculate chandelier exit
                let max_high = max_deque.front().unwrap().1;
                result[i] = max_high - atr[i] * mult;
            }
        } else if direction == "short" {
            // For short: use rolling minimum of low prices
            let mut min_deque: VecDeque<(usize, f64)> = VecDeque::new();
            
            // Initialize the first window
            for i in 0..period.min(n) {
                // Remove elements that are larger than current (maintain increasing order)
                while let Some(&(_, val)) = min_deque.back() {
                    if val >= low[i] {
                        min_deque.pop_back();
                    } else {
                        break;
                    }
                }
                min_deque.push_back((i, low[i]));
            }
            
            // Calculate chandelier exit for the first valid position
            if period <= n {
                let min_low = min_deque.front().unwrap().1;
                result[period - 1] = min_low + atr[period - 1] * mult;
            }
            
            // Slide the window for remaining positions
            for i in period..n {
                // Remove elements outside the window
                while let Some(&(idx, _)) = min_deque.front() {
                    if idx <= i - period {
                        min_deque.pop_front();
                    } else {
                        break;
                    }
                }
                
                // Add new element maintaining increasing order
                while let Some(&(_, val)) = min_deque.back() {
                    if val >= low[i] {
                        min_deque.pop_back();
                    } else {
                        break;
                    }
                }
                min_deque.push_back((i, low[i]));
                
                // Calculate chandelier exit
                let min_low = min_deque.front().unwrap().1;
                result[i] = min_low + atr[i] * mult;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Donchian Channels - Ultra-optimized version
#[pyfunction]
pub fn donchian(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result arrays
        let mut upperband = Array1::<f64>::from_elem(n, f64::NAN);
        let mut middleband = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lowerband = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            let result = PyDict::new(py);
            result.set_item("upperband", PyArray1::from_array(py, &upperband).to_owned())?;
            result.set_item("middleband", PyArray1::from_array(py, &middleband).to_owned())?;
            result.set_item("lowerband", PyArray1::from_array(py, &lowerband).to_owned())?;
            return Ok(result.into());
        }
        
        // Extract high and low data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for O(1) front/back operations
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        
        // Initialize first window
        for i in 0..period.min(n) {
            let high_val = high[i];
            let low_val = low[i];
            
            // Maintain decreasing order for max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Maintain increasing order for min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
        }
        
        // Set first valid result
        if period <= n {
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            upperband[period - 1] = max_high;
            lowerband[period - 1] = min_low;
            middleband[period - 1] = (max_high + min_low) * 0.5;
        }
        
        // Process remaining elements with optimized sliding window
        for i in period..n {
            let high_val = high[i];
            let low_val = low[i];
            
            // Remove expired elements from front of max deque (O(1) amortized)
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Remove expired elements from front of min deque (O(1) amortized)
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element to max deque (maintain decreasing order)
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Add new element to min deque (maintain increasing order)
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate results
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            upperband[i] = max_high;
            lowerband[i] = min_low;
            middleband[i] = (max_high + min_low) * 0.5;
        }
        
        // Return as dictionary
        let result = PyDict::new(py);
        result.set_item("upperband", PyArray1::from_array(py, &upperband).to_owned())?;
        result.set_item("middleband", PyArray1::from_array(py, &middleband).to_owned())?;
        result.set_item("lowerband", PyArray1::from_array(py, &lowerband).to_owned())?;
        Ok(result.into())
    })
}

/// Calculate Williams' %R - Ultra-optimized version
#[pyfunction]
pub fn willr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract required data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for efficient sliding window
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        
        // Initialize first window
        for i in 0..period.min(n) {
            let high_val = high[i];
            let low_val = low[i];
            
            // Maintain decreasing order for max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Maintain increasing order for min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
        }
        
        // Set first valid result
        if period <= n {
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            let denom = max_high - min_low;
            if denom == 0.0 {
                result[period - 1] = 0.0;
            } else {
                result[period - 1] = ((max_high - close[period - 1]) / denom) * -100.0;
            }
        }
        
        // Process remaining elements
        for i in period..n {
            let high_val = high[i];
            let low_val = low[i];
            
            // Remove expired elements from max deque
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Remove expired elements from min deque
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element to max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Add new element to min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate Williams' %R
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            let denom = max_high - min_low;
            if denom == 0.0 {
                result[i] = 0.0;
            } else {
                result[i] = ((max_high - close[i]) / denom) * -100.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Weighted Moving Average - Ultra-optimized version
#[pyfunction]
pub fn wma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period || period == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Pre-calculate weight sum for efficiency
        let weight_sum = (period * (period + 1)) / 2;
        let weight_sum_f64 = weight_sum as f64;
        
        // Calculate WMA for each position
        for i in (period - 1)..n {
            let mut weighted_sum = 0.0;
            
            // Calculate weighted sum with safe indexing
            for j in 0..period {
                let weight = (j + 1) as f64;
                let idx = i.saturating_sub(period - 1) + j;
                if idx < n {
                    let value = source_array[idx];
                    weighted_sum += weight * value;
                }
            }
            
            result[i] = weighted_sum / weight_sum_f64;
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Volume Weighted Moving Average - Ultra-optimized version
#[pyfunction]
pub fn vwma(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract required data
        let close = candles_array.column(2);
        let volume = candles_array.column(5);
        
        // Pre-calculate price * volume for efficiency
        let mut weighted_prices = Array1::<f64>::zeros(n);
        for i in 0..n {
            weighted_prices[i] = close[i] * volume[i];
        }
        
        // Use cumulative sums for ultra-fast rolling calculation
        let mut cumsum_weighted = Array1::<f64>::zeros(n);
        let mut cumsum_volume = Array1::<f64>::zeros(n);
        
        cumsum_weighted[0] = weighted_prices[0];
        cumsum_volume[0] = volume[0];
        
        for i in 1..n {
            cumsum_weighted[i] = cumsum_weighted[i - 1] + weighted_prices[i];
            cumsum_volume[i] = cumsum_volume[i - 1] + volume[i];
        }
        
        // Calculate VWMA using cumulative sums
        for i in 0..n {
            let start_idx = if i >= period { i - period + 1 } else { 0 };
            let end_idx = i;
            
            let sum_weighted = if start_idx == 0 {
                cumsum_weighted[end_idx]
            } else {
                cumsum_weighted[end_idx] - cumsum_weighted[start_idx - 1]
            };
            
            let sum_volume = if start_idx == 0 {
                cumsum_volume[end_idx]
            } else {
                cumsum_volume[end_idx] - cumsum_volume[start_idx - 1]
            };
            
            if sum_volume == 0.0 {
                result[i] = f64::NAN;
            } else {
                result[i] = sum_weighted / sum_volume;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Stochastic Oscillator - Ultra-optimized version
#[pyfunction]
pub fn stoch(candles: PyReadonlyArray2<f64>, fastk_period: usize, slowk_period: usize, _slowk_matype: usize, slowd_period: usize, _slowd_matype: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        if n < fastk_period {
            let k_values = Array1::<f64>::from_elem(n, f64::NAN);
            let d_values = Array1::<f64>::from_elem(n, f64::NAN);
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }
        
        // Extract price data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate rolling highs and lows using simple approach to match Python exactly
        let mut hh = Array1::<f64>::from_elem(n, f64::NAN);
        let mut ll = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in (fastk_period - 1)..n {
            let start_idx = i + 1 - fastk_period;
            let end_idx = i + 1;
            
            let window_high = high.slice(s![start_idx..end_idx]);
            let window_low = low.slice(s![start_idx..end_idx]);
            
            hh[i] = window_high.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ll[i] = window_low.fold(f64::INFINITY, |a, &b| a.min(b));
        }
        
        // Calculate raw %K values
        let mut raw_k = Array1::<f64>::from_elem(n, f64::NAN);
        for i in (fastk_period - 1)..n {
            let hh_val = hh[i];
            let ll_val = ll[i];
            let close_val = close[i];
            
            if hh_val > ll_val {
                raw_k[i] = 100.0 * (close_val - ll_val) / (hh_val - ll_val);
            }
        }
        
        // Apply smoothing for %K (slow K)
        let smoothed_k = sma_array(&raw_k, slowk_period);
        
        // Apply smoothing for %D
        let smoothed_d = sma_array(&smoothed_k, slowd_period);
        
        Ok((
            PyArray1::from_array(py, &smoothed_k).to_owned(),
            PyArray1::from_array(py, &smoothed_d).to_owned()
        ))
    })
}

/// Calculate Stochastic Fast - Ultra-optimized version
#[pyfunction]
pub fn stochf(candles: PyReadonlyArray2<f64>, fastk_period: usize, fastd_period: usize, fastd_matype: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut k_values = Array1::<f64>::from_elem(n, f64::NAN);
        let mut d_values = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < fastk_period {
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }
        
        // Extract price data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for efficient sliding window max/min
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        
        // Initialize first window
        for i in 0..fastk_period.min(n) {
            let high_val = high[i];
            let low_val = low[i];
            
            // Maintain decreasing order for max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Maintain increasing order for min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
        }
        
        // Calculate fast stochastic values
        // First valid value
        if n >= fastk_period {
            let hh = max_deque.front().unwrap().1;
            let ll = min_deque.front().unwrap().1;
            if hh > ll {
                k_values[fastk_period - 1] = 100.0 * (close[fastk_period - 1] - ll) / (hh - ll);
            } else {
                k_values[fastk_period - 1] = 50.0; // Default when no range
            }
        }
        
        // Sliding window for remaining values
        for i in fastk_period..n {
            // Remove elements outside window
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - fastk_period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - fastk_period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element
            let high_val = high[i];
            let low_val = low[i];
            
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate %K
            let hh = max_deque.front().unwrap().1;
            let ll = min_deque.front().unwrap().1;
            if hh > ll {
                k_values[i] = 100.0 * (close[i] - ll) / (hh - ll);
            } else {
                k_values[i] = 50.0;
            }
        }
        
        // Apply smoothing to get %D
        let smoothed_d = if fastd_matype == 0 {
            // SMA
            sma_array(&k_values, fastd_period)
        } else {
            // Other MA types - simplified, using SMA for now
            sma_array(&k_values, fastd_period)
        };
        
        d_values = smoothed_d;
        
        Ok((
            PyArray1::from_array(py, &k_values).to_owned(),
            PyArray1::from_array(py, &d_values).to_owned()
        ))
    })
}

/// Calculate Directional Movement (DM) - Ultra-optimized version
#[pyfunction]
pub fn dm(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut plus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &plus_dm).to_owned(),
                PyArray1::from_array(py, &minus_dm).to_owned()
            ));
        }
        
        // Extract price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate raw directional movements
        let mut raw_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut raw_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];
            
            if up_move > down_move && up_move > 0.0 {
                raw_plus[i] = up_move;
            } else {
                raw_plus[i] = 0.0;
            }
            
            if down_move > up_move && down_move > 0.0 {
                raw_minus[i] = down_move;
            } else {
                raw_minus[i] = 0.0;
            }
        }
        
        // Apply Wilder's smoothing
        if n > period {
            // Calculate initial sum
            let mut sum_plus = 0.0;
            let mut sum_minus = 0.0;
            
            for i in 1..=period {
                sum_plus += raw_plus[i];
                sum_minus += raw_minus[i];
            }
            
            plus_dm[period] = sum_plus;
            minus_dm[period] = sum_minus;
            
            // Apply Wilder's smoothing formula
            for i in (period + 1)..n {
                plus_dm[i] = plus_dm[i - 1] - (plus_dm[i - 1] / period as f64) + raw_plus[i];
                minus_dm[i] = minus_dm[i - 1] - (minus_dm[i - 1] / period as f64) + raw_minus[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_dm).to_owned(),
            PyArray1::from_array(py, &minus_dm).to_owned()
        ))
    })
}

/// Calculate DEMA (Double Exponential Moving Average) - Ultra-optimized version
#[pyfunction]
pub fn dema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = source_array[0];
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        // Optimized: Conservative optimizations for better performance
        // Pre-calculate constants for better performance
        let two = 2.0;
        
        // Pre-allocate arrays with the exact pattern as Python
        let mut ema1 = Array1::<f64>::zeros(n);
        ema1[0] = source_array[0];
        
        // Step 1: Calculate EMA1 efficiently  
        for i in 1..n {
            ema1[i] = alpha * source_array[i] + one_minus_alpha * ema1[i - 1];
        }
        
        // Step 2: Calculate EMA2 (EMA of EMA1) efficiently
        let mut ema2 = Array1::<f64>::zeros(n);
        ema2[0] = ema1[0];
        
        for i in 1..n {
            ema2[i] = alpha * ema1[i] + one_minus_alpha * ema2[i - 1];
        }
        
        // Step 3: Calculate DEMA = 2*EMA1 - EMA2 efficiently
        for i in 0..n {
            result[i] = two * ema1[i] - ema2[i];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Helper function to calculate SMA on an array
fn sma_array(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::from_elem(n, f64::NAN);
    
    if n < period {
        return result;
    }
    
    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        let end_idx = i + 1;
        
        // Calculate SMA for window, handling NaN values
        let window = source.slice(s![start_idx..end_idx]);
        let mut sum = 0.0;
        let mut count = 0;
        
        for &val in window.iter() {
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }
    
    result
}

/// Calculate ADOSC (Chaikin A/D Oscillator) - Ultra-optimized version
#[pyfunction]
pub fn adosc(candles: PyReadonlyArray2<f64>, fast_period: usize, slow_period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = 0.0;
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract price and volume data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        let volume = candles_array.column(5);
        
        // Step 1: Calculate Money Flow Multiplier and Money Flow Volume
        let mut mf_volume = Array1::<f64>::zeros(n);
        for i in 0..n {
            let price_range = high[i] - low[i];
            let multiplier = if price_range != 0.0 {
                ((close[i] - low[i]) - (high[i] - close[i])) / price_range
            } else {
                0.0
            };
            mf_volume[i] = multiplier * volume[i];
        }
        
        // Step 2: Calculate A/D Line (cumulative sum of money flow volume)
        let mut ad_line = Array1::<f64>::zeros(n);
        ad_line[0] = mf_volume[0];
        for i in 1..n {
            ad_line[i] = ad_line[i - 1] + mf_volume[i];
        }
        
        // Step 3: Calculate EMAs of A/D Line
        let fast_alpha = 2.0 / (fast_period as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_period as f64 + 1.0);
        let fast_one_minus_alpha = 1.0 - fast_alpha;
        let slow_one_minus_alpha = 1.0 - slow_alpha;
        
        // Calculate fast EMA
        let mut fast_ema = Array1::<f64>::zeros(n);
        fast_ema[0] = ad_line[0];
        for i in 1..n {
            fast_ema[i] = fast_alpha * ad_line[i] + fast_one_minus_alpha * fast_ema[i - 1];
        }
        
        // Calculate slow EMA
        let mut slow_ema = Array1::<f64>::zeros(n);
        slow_ema[0] = ad_line[0];
        for i in 1..n {
            slow_ema[i] = slow_alpha * ad_line[i] + slow_one_minus_alpha * slow_ema[i - 1];
        }
        
        // Step 4: Calculate ADOSC = Fast EMA - Slow EMA
        for i in 0..n {
            result[i] = fast_ema[i] - slow_ema[i];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate EMA (Exponential Moving Average) - Optimized version
#[pyfunction]
pub fn ema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Return NaN if period is greater than the data length
        if period > n {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = source_array[0];
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        // Initialize first value
        result[0] = source_array[0];
        
        // Calculate EMA efficiently  
        for i in 1..n {
            result[i] = alpha * source_array[i] + one_minus_alpha * result[i - 1];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate CVI (Chaikins Volatility Indicator) - Ultra-optimized version
#[pyfunction]
pub fn cvi(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract high and low price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate high-low difference
        let mut hl_diff = Array1::<f64>::zeros(n);
        for i in 0..n {
            hl_diff[i] = high[i] - low[i];
        }
        
        // Calculate EMA of the high-low difference
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        let mut ema_diff = Array1::<f64>::zeros(n);
        ema_diff[0] = hl_diff[0];
        
        for i in 1..n {
            ema_diff[i] = alpha * hl_diff[i] + one_minus_alpha * ema_diff[i - 1];
        }
        
        // Calculate rate of change
        for i in period..n {
            if ema_diff[i - period] != 0.0 {
                result[i] = ((ema_diff[i] - ema_diff[i - period]) / ema_diff[i - period]) * 100.0;
            } else {
                result[i] = 0.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate DTI (Dynamic Trend Index) by William Blau - Optimized version
#[pyfunction]
pub fn dti(candles: PyReadonlyArray2<f64>, r: usize, s: usize, u: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= r || n <= s || n <= u {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract high and low price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Shift high and low by 1 period
        let mut high_1 = Array1::<f64>::from_elem(n, f64::NAN);
        let mut low_1 = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in 1..n {
            high_1[i] = high[i - 1];
            low_1[i] = low[i - 1];
        }
        
        // Compute upward and downward movements
        let mut xhmu = Array1::<f64>::zeros(n);
        let mut xlmd = Array1::<f64>::zeros(n);
        
        for i in 1..n {
            let high_diff = high[i] - high_1[i];
            if high_diff > 0.0 {
                xhmu[i] = high_diff;
            }
            
            let low_diff = low[i] - low_1[i];
            if low_diff < 0.0 {
                xlmd[i] = -low_diff;
            }
        }
        
        // Calculate xPrice and xPriceAbs
        let mut xprice = Array1::<f64>::zeros(n);
        let mut xprice_abs = Array1::<f64>::zeros(n);
        
        for i in 0..n {
            xprice[i] = xhmu[i] - xlmd[i];
            xprice_abs[i] = xprice[i].abs();
        }
        
        // Apply triple EMA to xPrice
        let r_alpha = 2.0 / (r as f64 + 1.0);
        let s_alpha = 2.0 / (s as f64 + 1.0);
        let u_alpha = 2.0 / (u as f64 + 1.0);
        
        let r_one_minus_alpha = 1.0 - r_alpha;
        let s_one_minus_alpha = 1.0 - s_alpha;
        let u_one_minus_alpha = 1.0 - u_alpha;
        
        // First stage EMA on xPrice
        let mut temp = Array1::<f64>::zeros(n);
        temp[0] = xprice[0];
        for i in 1..n {
            temp[i] = r_alpha * xprice[i] + r_one_minus_alpha * temp[i - 1];
        }
        
        // Second stage EMA
        let mut temp2 = Array1::<f64>::zeros(n);
        temp2[0] = temp[0];
        for i in 1..n {
            temp2[i] = s_alpha * temp[i] + s_one_minus_alpha * temp2[i - 1];
        }
        
        // Third stage EMA (xuXA)
        let mut xu_xa = Array1::<f64>::zeros(n);
        xu_xa[0] = temp2[0];
        for i in 1..n {
            xu_xa[i] = u_alpha * temp2[i] + u_one_minus_alpha * xu_xa[i - 1];
        }
        
        // Apply triple EMA to xPriceAbs
        // First stage EMA on xPriceAbs
        let mut temp_abs = Array1::<f64>::zeros(n);
        temp_abs[0] = xprice_abs[0];
        for i in 1..n {
            temp_abs[i] = r_alpha * xprice_abs[i] + r_one_minus_alpha * temp_abs[i - 1];
        }
        
        // Second stage EMA
        let mut temp2_abs = Array1::<f64>::zeros(n);
        temp2_abs[0] = temp_abs[0];
        for i in 1..n {
            temp2_abs[i] = s_alpha * temp_abs[i] + s_one_minus_alpha * temp2_abs[i - 1];
        }
        
        // Third stage EMA (xuXAAbs)
        let mut xu_xa_abs = Array1::<f64>::zeros(n);
        xu_xa_abs[0] = temp2_abs[0];
        for i in 1..n {
            xu_xa_abs[i] = u_alpha * temp2_abs[i] + u_one_minus_alpha * xu_xa_abs[i - 1];
        }
        
        // Calculate Val1 and Val2
        let val1 = xu_xa.mapv(|x| x * 100.0);
        let val2 = xu_xa_abs;
        
        // Calculate DTI value
        for i in 0..n {
            if val2[i] != 0.0 {
                result[i] = val1[i] / val2[i];
            } else {
                result[i] = 0.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate ZLEMA (Zero-Lag Exponential Moving Average)
#[pyfunction]
pub fn zlema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Return early if we don't have enough data
        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate lag
        let lag = (period - 1) / 2;
        
        // Calculate the smoothing factor
        let alpha = 2.0 / (period as f64 + 1.0);
        
        // Pre-calculate the EMA data = price + (price - price_lag)
        let mut ema_data = Array1::<f64>::from_elem(n, 0.0);
        for i in lag..n {
            ema_data[i] = source_array[i] + (source_array[i] - source_array[i - lag]);
        }
        
        // First value with enough data is just the ema_data at that point
        result[lag] = ema_data[lag];
        
        // Calculate ZLEMA for the rest of the points
        for i in (lag + 1)..n {
            result[i] = alpha * ema_data[i] + (1.0 - alpha) * result[i - 1];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate EMA for internal use in Wavetrend indicator
fn ema_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    // Not enough data
    if n == 0 {
        return result;
    }
    
    // Calculate alpha
    let alpha = 2.0 / (period as f64 + 1.0);
    
    // First value is the source
    result[0] = source[0];
    
    // Calculate EMA for the rest
    for i in 1..n {
        result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1];
    }
    
    result
}

/// Calculate SMA for internal use in Wavetrend indicator
fn sma_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    if n == 0 || period == 0 {
        return result;
    }
    
    // Calculate initial values with partial windows
    let mut cumsum = 0.0;
    for i in 0..period.min(n) {
        cumsum += source[i];
        result[i] = cumsum / (i as f64 + 1.0);
    }
    
    if n <= period {
        return result;
    }
    
    // For the remaining windows, use a rolling approach
    for i in period..n {
        result[i] = result[i-1] + (source[i] - source[i-period]) / period as f64;
    }
    
    result
}

/// Calculate Wavetrend indicator
#[pyfunction]
pub fn wt(
    candles: PyReadonlyArray2<f64>,
    wtchannellen: usize,
    wtaveragelen: usize,
    wtmalen: usize,
    oblevel: f64,
    oslevel: f64,
    source_type: &str
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        // Extract data from candles based on source_type
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        
        // Get source data based on source_type
        let src = match source_type {
            "open" => {
                let opens = candles_array.slice(s![.., 1]);
                opens.to_owned()
            },
            "high" => {
                let highs = candles_array.slice(s![.., 3]);
                highs.to_owned()
            },
            "low" => {
                let lows = candles_array.slice(s![.., 4]);
                lows.to_owned()
            },
            "close" => {
                let closes = candles_array.slice(s![.., 2]);
                closes.to_owned()
            },
            "hlc3" => {
                let highs = candles_array.slice(s![.., 3]);
                let lows = candles_array.slice(s![.., 4]);
                let closes = candles_array.slice(s![.., 2]);
                
                let mut result = Array1::<f64>::zeros(n);
                for i in 0..n {
                    result[i] = (highs[i] + lows[i] + closes[i]) / 3.0;
                }
                result
            },
            "ohlc4" => {
                let opens = candles_array.slice(s![.., 1]);
                let highs = candles_array.slice(s![.., 3]);
                let lows = candles_array.slice(s![.., 4]);
                let closes = candles_array.slice(s![.., 2]);
                
                let mut result = Array1::<f64>::zeros(n);
                for i in 0..n {
                    result[i] = (opens[i] + highs[i] + lows[i] + closes[i]) / 4.0;
                }
                result
            },
            _ => {
                // Default to close
                let closes = candles_array.slice(s![.., 2]);
                closes.to_owned()
            }
        };
        
        // Calculate Wavetrend components
        let esa = ema_for_wt(&src, wtchannellen);
        
        // Calculate absolute difference
        let mut abs_diff = Array1::<f64>::zeros(n);
        for i in 0..n {
            abs_diff[i] = (src[i] - esa[i]).abs();
        }
        
        let de = ema_for_wt(&abs_diff, wtchannellen);
        
        // Calculate CI (avoid division by zero)
        let mut ci = Array1::<f64>::zeros(n);
        for i in 0..n {
            ci[i] = if de[i] == 0.0 {
                0.0
            } else {
                (src[i] - esa[i]) / (0.015 * de[i])
            };
        }
        
        // Calculate wt1 and wt2
        let wt1 = ema_for_wt(&ci, wtaveragelen);
        let wt2 = sma_for_wt(&wt1, wtmalen);
        
        // Calculate additional components
        let mut wtvwap = Array1::<f64>::zeros(n);
        let mut wtcrossup = Array1::<bool>::from_elem(n, false);
        let mut wtcrossdown = Array1::<bool>::from_elem(n, false);
        let mut wtoversold = Array1::<bool>::from_elem(n, false);
        let mut wtoverbought = Array1::<bool>::from_elem(n, false);
        
        for i in 0..n {
            wtvwap[i] = wt1[i] - wt2[i];
            wtcrossup[i] = wt2[i] - wt1[i] <= 0.0;
            wtcrossdown[i] = wt2[i] - wt1[i] >= 0.0;
            wtoversold[i] = wt2[i] <= oslevel;
            wtoverbought[i] = wt2[i] >= oblevel;
        }
        
        Ok((
            PyArray1::from_array(py, &wt1).to_owned(),
            PyArray1::from_array(py, &wt2).to_owned(),
            PyArray1::from_array(py, &wtcrossup).to_owned(),
            PyArray1::from_array(py, &wtcrossdown).to_owned(),
            PyArray1::from_array(py, &wtoversold).to_owned(),
            PyArray1::from_array(py, &wtoverbought).to_owned(),
            PyArray1::from_array(py, &wtvwap).to_owned()
        ))
    })
}

/// Calculate RMA (Relative Moving Average) for internal use
fn rma_array(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    if n == 0 || period == 0 {
        return result;
    }
    
    let alpha = 1.0 / period as f64;
    result[0] = source[0];
    
    for i in 1..n {
        result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1];
    }
    
    result
}

/// Calculate DX (Directional Movement Index) - matches Jesse's Python implementation exactly
/// Uses Jesse's rma seeding behavior (newseries[-1] = source[-1])
#[pyfunction]
pub fn dx(
    candles: PyReadonlyArray2<f64>,
    di_length: usize,
    adx_smoothing: usize,
    _sequential: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];

        if n == 0 {
            let empty = Array1::<f64>::from_elem(0, f64::NAN);
            return Ok((
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
            ));
        }

        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);

        // Build plusDM, minusDM, true_range arrays (matches _fast_dm_tr)
        let mut plus_dm = vec![0.0f64; n];
        let mut minus_dm = vec![0.0f64; n];
        let mut tr = vec![0.0f64; n];
        tr[0] = high[0usize] - low[0usize];
        for i in 1..n {
            let up = high[i] - high[i - 1];
            let down = low[i - 1] - low[i];
            if up > down && up > 0.0 { plus_dm[i] = up; }
            if down > up && down > 0.0 { minus_dm[i] = down; }
            let a = high[i] - low[i];
            let b = (high[i] - close[i - 1]).abs();
            let c = (low[i] - close[i - 1]).abs();
            tr[i] = a.max(b).max(c);
        }

        // Jesse-style rma: r[0] = alpha * src[0] + (1-alpha) * src[n-1], then EMA
        let alpha_di = 1.0 / di_length as f64;
        let one_minus_di = 1.0 - alpha_di;
        let mut tr_rma = vec![0.0f64; n];
        let mut plus_rma = vec![0.0f64; n];
        let mut minus_rma = vec![0.0f64; n];
        tr_rma[0] = alpha_di * tr[0] + one_minus_di * tr[n - 1];
        plus_rma[0] = alpha_di * plus_dm[0] + one_minus_di * plus_dm[n - 1];
        minus_rma[0] = alpha_di * minus_dm[0] + one_minus_di * minus_dm[n - 1];
        for i in 1..n {
            tr_rma[i] = alpha_di * tr[i] + one_minus_di * tr_rma[i - 1];
            plus_rma[i] = alpha_di * plus_dm[i] + one_minus_di * plus_rma[i - 1];
            minus_rma[i] = alpha_di * minus_dm[i] + one_minus_di * minus_rma[i - 1];
        }

        // plusDI, minusDI, directional_index
        let mut plus_di = vec![0.0f64; n];
        let mut minus_di = vec![0.0f64; n];
        let mut di_idx = vec![0.0f64; n];
        for i in 0..n {
            if tr_rma[i] != 0.0 {
                plus_di[i] = 100.0 * plus_rma[i] / tr_rma[i];
                minus_di[i] = 100.0 * minus_rma[i] / tr_rma[i];
            }
            let di_sum = plus_di[i] + minus_di[i];
            let di_diff = (plus_di[i] - minus_di[i]).abs();
            di_idx[i] = if di_sum == 0.0 { di_diff } else { di_diff / di_sum };
        }

        // adx = 100 * rma(directional_index, adx_smoothing) — same Jesse-style seeding
        let alpha_adx = 1.0 / adx_smoothing as f64;
        let one_minus_adx = 1.0 - alpha_adx;
        let mut adx = vec![0.0f64; n];
        adx[0] = 100.0 * (alpha_adx * di_idx[0] + one_minus_adx * di_idx[n - 1]);
        for i in 1..n {
            adx[i] = one_minus_adx * adx[i - 1] + 100.0 * alpha_adx * di_idx[i];
        }

        Ok((
            PyArray1::from_vec(py, adx).to_owned(),
            PyArray1::from_vec(py, plus_di).to_owned(),
            PyArray1::from_vec(py, minus_di).to_owned(),
        ))
    })
}


/// Calculate FOSC (Forecast Oscillator)
#[pyfunction]
pub fn fosc(
    source: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source = source.as_array();
        let n = source.len();
        let mut result = Array1::<f64>::from_elem(n, 0.0);

        if n < period || period == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        let period_f64 = period as f64;
        let sum_x: f64 = (0..period).map(|i| i as f64).sum();
        let mean_x = sum_x / period_f64;
        let sum_xx: f64 = (0..period).map(|i| (i as f64)*(i as f64)).sum();
        let denom = sum_xx - sum_x*mean_x;
        
        // Prefix sums of y and k*y
        let mut prefix_y = Array1::<f64>::zeros(n);
        let mut prefix_xy = Array1::<f64>::zeros(n);
        let mut cum_y = 0.0;
        let mut cum_xy = 0.0;
        for i in 0..n {
            cum_y += source[i];
            cum_xy += (i as f64) * source[i];
            prefix_y[i] = cum_y;
            prefix_xy[i] = cum_xy;
        }
        
        for end in (period-1)..n {
            let start = end + 1 - period;
            let sum_y = if start == 0 { prefix_y[end] } else { prefix_y[end] - prefix_y[start-1] };
            let raw_sum_xy = if start == 0 { prefix_xy[end] } else { prefix_xy[end] - prefix_xy[start-1] };
            // Shift x values so that start becomes 0
            let sum_xy = raw_sum_xy - (start as f64)*sum_y;
            let mean_y = sum_y / period_f64;
            let slope = (sum_xy - sum_x*mean_y)/denom;
            let intercept = mean_y - slope*mean_x;
            let predicted = slope*(period_f64-1.0)+intercept;
            let actual = source[end];
            result[end] = if actual != 0.0 { 100.0*(actual-predicted)/actual } else { 0.0 };
        }
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Helper function for linear regression calculation for FOSC
fn linear_regression_line(x: &Array1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len() as f64;
    let sum_x: f64 = x.sum();
    let sum_y: f64 = y.sum();
    
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    
    for i in 0..x.len() {
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    let mut result = Array1::<f64>::zeros(x.len());
    for i in 0..x.len() {
        result[i] = slope * x[i] + intercept;
    }
    
    result
}

/// Calculate FRAMA (Fractal Adaptive Moving Average)
#[pyfunction]
pub fn frama(
    candles: PyReadonlyArray2<f64>,
    window: usize,
    fc: usize,
    sc: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = vec![f64::NAN; n];

        // Adjust window to be even if it's not
        let mut n_local = window;
        if n_local % 2 == 1 { n_local += 1; }
        if n < n_local { return Ok(PyArray1::from_vec(py, result).to_owned()); }

        // Extract high/low/close to plain Vec<f64> for fast indexing
        let high: Vec<f64> = candles_array.slice(s![.., 3]).to_vec();
        let low: Vec<f64> = candles_array.slice(s![.., 4]).to_vec();
        let close: Vec<f64> = candles_array.slice(s![.., 2]).to_vec();

        let w = (2.0 / (sc as f64 + 1.0)).ln();
        let half = n_local / 2;
        let inv_half = 1.0 / half as f64;
        let inv_full = 1.0 / n_local as f64;
        let ln2 = 2.0f64.ln();
        let sc_f = sc as f64;
        let fc_f = fc as f64;
        let min_alpha = 2.0 / (sc_f + 1.0);

        let mut d = vec![f64::NAN; n];
        let mut alphas = vec![f64::NAN; n];

        for i in n_local..n {
            // Window is candles[i-n_local..i]. Indices in our extracted arrays: [i-n_local, i).
            let start = i - n_local;
            let mid = start + half;
            // Second half (slice [half..]): indices [mid, i)
            let mut v1_hi = f64::NEG_INFINITY;
            let mut v1_lo = f64::INFINITY;
            for k in mid..i {
                if high[k] > v1_hi { v1_hi = high[k]; }
                if low[k] < v1_lo { v1_lo = low[k]; }
            }
            // First half (slice [..half]): indices [start, mid)
            let mut v2_hi = f64::NEG_INFINITY;
            let mut v2_lo = f64::INFINITY;
            for k in start..mid {
                if high[k] > v2_hi { v2_hi = high[k]; }
                if low[k] < v2_lo { v2_lo = low[k]; }
            }
            // Full window max/min by combining halves
            let hi = v1_hi.max(v2_hi);
            let lo = v1_lo.min(v2_lo);
            let n1 = (v1_hi - v1_lo) * inv_half;
            let n2 = (v2_hi - v2_lo) * inv_half;
            let n3 = (hi - lo) * inv_full;

            if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                d[i] = ((n1 + n2).ln() - n3.ln()) / ln2;
            } else if i > n_local {
                d[i] = d[i - 1];
            } else {
                d[i] = 0.0;
            }

            let old_alpha = (w * (d[i] - 1.0)).exp().clamp(0.1, 1.0);
            let old_n = (2.0 - old_alpha) / old_alpha;
            let n_val = (sc_f - fc_f) * ((old_n - 1.0) / (sc_f - 1.0)) + fc_f;
            let alpha = (2.0 / (n_val + 1.0)).clamp(min_alpha, 1.0);
            alphas[i] = alpha;
        }

        // Calculate FRAMA EMA
        let seed: f64 = close[..n_local].iter().sum::<f64>() / n_local as f64;
        result[n_local - 1] = seed;
        for i in n_local..n {
            result[i] = alphas[i] * close[i] + (1.0 - alphas[i]) * result[i - 1];
        }

        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}

/// Get period identifier from timestamp based on anchor
fn get_period_from_timestamp(timestamp: f64, anchor: &str) -> i64 {
    // Convert timestamp from milliseconds to seconds
    let seconds = (timestamp / 1000.0) as i64;
    
    match anchor.to_uppercase().as_str() {
        "D" => seconds / 86400,        // Daily: 24 * 60 * 60 seconds
        "H" => seconds / 3600,         // Hourly: 60 * 60 seconds  
        "M" => seconds / 60,           // Minute: 60 seconds
        "4H" => seconds / 14400,       // 4 Hour: 4 * 60 * 60 seconds
        "12H" => seconds / 43200,      // 12 Hour: 12 * 60 * 60 seconds
        "W" => seconds / 604800,       // Weekly: 7 * 24 * 60 * 60 seconds
        "MN" => {
            // Monthly - approximate as 30 days for simplicity
            seconds / 2592000  // 30 * 24 * 60 * 60 seconds
        },
        _ => seconds / 86400,          // Default to daily
    }
}

/// Calculate VWAP (Volume Weighted Average Price)
#[pyfunction]
pub fn vwap(
    candles: PyReadonlyArray2<f64>,
    source_type: &str,
    anchor: &str,
    sequential: bool
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract source prices and volumes based on source_type
        let source = match source_type.to_lowercase().as_str() {
            "open" => candles_array.slice(s![.., 1]).to_owned(),
            "high" => candles_array.slice(s![.., 3]).to_owned(),
            "low" => candles_array.slice(s![.., 4]).to_owned(),
            "close" => candles_array.slice(s![.., 2]).to_owned(),
            "hl2" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                (&high + &low) / 2.0
            },
            "hlc3" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            },
            "ohlc4" => {
                let open = candles_array.slice(s![.., 1]);
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&open + &high + &low + &close) / 4.0
            },
            _ => {
                // Default to hlc3
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            }
        };
        
        let volume = candles_array.slice(s![.., 5]).to_owned();
        let timestamps = candles_array.slice(s![.., 0]).to_owned();
        
        // Calculate VWAP with anchoring logic
        let mut cum_vol = 0.0;
        let mut cum_vol_price = 0.0;
        let mut current_period = if n > 0 { get_period_from_timestamp(timestamps[0usize], anchor) } else { 0 };
        
        for i in 0..n {
            let period = get_period_from_timestamp(timestamps[i], anchor);
            
            // Reset if we've moved to a new period
            if period != current_period {
                cum_vol = 0.0;
                cum_vol_price = 0.0;
                current_period = period;
            }
            
            let vol_price = volume[i] * source[i];
            cum_vol_price += vol_price;
            cum_vol += volume[i];
            
            if cum_vol != 0.0 {
                result[i] = cum_vol_price / cum_vol;
            }
        }
        
        if sequential {
            Ok(PyArray1::from_array(py, &result).to_owned())
        } else {
            let last_result = Array1::<f64>::from_elem(1, if n > 0 { result[n-1] } else { f64::NAN });
            Ok(PyArray1::from_array(py, &last_result).to_owned())
        }
    })
}

/// Calculate VI (Vortex Indicator)
#[pyfunction] 
pub fn vi(
    candles: PyReadonlyArray2<f64>,
    period: usize,
    sequential: bool
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut vi_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut vi_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ));
        }
        
        let close = candles_array.slice(s![.., 2]).to_owned();
        let high = candles_array.slice(s![.., 3]).to_owned();
        let low = candles_array.slice(s![.., 4]).to_owned();
        
        // Calculate True Range, VP and VM for each period
        let mut tr = Array1::<f64>::zeros(n);
        let mut vp = Array1::<f64>::zeros(n);
        let mut vm = Array1::<f64>::zeros(n);
        
        // First candle
        if n > 0 {
            tr[0usize] = high[0usize] - low[0usize];
        }
        
        // Calculate TR, VP, VM for each candle
        for i in 1..n {
            let hl = high[i] - low[i];
            let hpc = (high[i] - close[i - 1]).abs();
            let lpc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hpc).max(lpc);
            
            vp[i] = (high[i] - low[i - 1]).abs();
            vm[i] = (low[i] - high[i - 1]).abs();
        }
        
        // Calculate rolling sums and VI values
        for i in period..n {
            let start_idx = i + 1 - period;
            
            let sum_tr: f64 = tr.slice(s![start_idx..=i]).sum();
            let sum_vp: f64 = vp.slice(s![start_idx..=i]).sum();
            let sum_vm: f64 = vm.slice(s![start_idx..=i]).sum();
            
            if sum_tr != 0.0 {
                vi_plus[i] = sum_vp / sum_tr;
                vi_minus[i] = sum_vm / sum_tr;
            }
        }
        
        if sequential {
            Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ))
        } else {
            let plus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_plus[n-1] } else { f64::NAN });
            let minus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_minus[n-1] } else { f64::NAN });
            Ok((
                PyArray1::from_array(py, &plus_result).to_owned(),
                PyArray1::from_array(py, &minus_result).to_owned()
            ))
        }
    })
}

/// Calculate T3 (Triple Exponential Moving Average)
#[pyfunction]
pub fn t3(
    candles: PyReadonlyArray2<f64>,
    period: usize,
    vfactor: f64,
    source_type: &str,
    sequential: bool
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        
        if n == 0 {
            let result = Array1::<f64>::from_elem(0, f64::NAN);
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract source based on source_type
        let source = match source_type.to_lowercase().as_str() {
            "open" => candles_array.slice(s![.., 1]).to_owned(),
            "high" => candles_array.slice(s![.., 3]).to_owned(),
            "low" => candles_array.slice(s![.., 4]).to_owned(),
            "close" => candles_array.slice(s![.., 2]).to_owned(),
            "hl2" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                (&high + &low) / 2.0
            },
            "hlc3" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            },
            "ohlc4" => {
                let open = candles_array.slice(s![.., 1]);
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&open + &high + &low + &close) / 4.0
            },
            _ => candles_array.slice(s![.., 2]).to_owned(), // Default to close
        };
        
        let k = 2.0 / (period as f64 + 1.0);
        let k_rev = 1.0 - k;
        
        // Calculate weights based on volume factor
        let w1 = -vfactor.powi(3);
        let w2 = 3.0 * vfactor.powi(2) + 3.0 * vfactor.powi(3);
        let w3 = -6.0 * vfactor.powi(2) - 3.0 * vfactor - 3.0 * vfactor.powi(3);
        let w4 = 1.0 + 3.0 * vfactor + vfactor.powi(3) + 3.0 * vfactor.powi(2);
        
        // Initialize EMAs
        let mut e1 = Array1::<f64>::zeros(n);
        let mut e2 = Array1::<f64>::zeros(n);
        let mut e3 = Array1::<f64>::zeros(n);
        let mut e4 = Array1::<f64>::zeros(n);
        let mut e5 = Array1::<f64>::zeros(n);
        let mut e6 = Array1::<f64>::zeros(n);
        let mut t3_result = Array1::<f64>::zeros(n);
        
        // Initialize first values
        if n > 0 {
            e1[0usize] = source[0usize];
            e2[0usize] = e1[0usize];
            e3[0usize] = e2[0usize];
            e4[0usize] = e3[0usize];
            e5[0usize] = e4[0usize];
            e6[0usize] = e5[0usize];
            t3_result[0usize] = w1 * e6[0usize] + w2 * e5[0usize] + w3 * e4[0usize] + w4 * e3[0usize];
        }
        
        // Calculate all EMAs
        for i in 1..n {
            e1[i] = k * source[i] + k_rev * e1[i-1];
            e2[i] = k * e1[i] + k_rev * e2[i-1];
            e3[i] = k * e2[i] + k_rev * e3[i-1];
            e4[i] = k * e3[i] + k_rev * e4[i-1];
            e5[i] = k * e4[i] + k_rev * e5[i-1];
            e6[i] = k * e5[i] + k_rev * e6[i-1];
            t3_result[i] = w1 * e6[i] + w2 * e5[i] + w3 * e4[i] + w4 * e3[i];
        }
        
        if sequential {
            Ok(PyArray1::from_array(py, &t3_result).to_owned())
        } else {
            let last_result = Array1::<f64>::from_elem(1, if n > 0 { t3_result[n-1] } else { f64::NAN });
            Ok(PyArray1::from_array(py, &last_result).to_owned())
        }
    })
}

/// Sums two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn sum_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise addition
    let result_decimal = decimal1 + decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}

/// Subtracts two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn subtract_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise subtraction
    let result_decimal = decimal1 - decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}
// ============================================================
// Private helper functions for shared computation
// ============================================================

fn ih_ema(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![0.0f64; n];
    if n == 0 { return r; }
    let alpha = 2.0 / (period as f64 + 1.0);
    r[0] = source[0];
    for i in 1..n { r[i] = alpha * source[i] + (1.0 - alpha) * r[i-1]; }
    r
}

fn ih_sma(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![f64::NAN; n];
    if period == 0 || n < period { return r; }
    let pf = period as f64;
    let mut sum: f64 = source[..period].iter().sum();
    r[period-1] = sum / pf;
    for i in period..n { sum += source[i] - source[i-period]; r[i] = sum / pf; }
    r
}

fn ih_wma(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![0.0f64; n];
    if period == 0 { return r; }
    let ws: f64 = (1..=period).map(|x| x as f64).sum();
    for i in period.saturating_sub(1)..n {
        if i + 1 < period { continue; }
        let mut w = 0.0;
        for j in 0..period { w += source[i - j] * (period - j) as f64; }
        r[i] = w / ws;
    }
    r
}

fn ih_supersmoother_2pole(source: &[f64], period: f64) -> Vec<f64> {
    let n = source.len();
    let mut r = source.to_vec();
    let pi = std::f64::consts::PI;
    let a = (-1.414 * pi / period).exp();
    let b = 2.0 * a * (1.414 * pi / period).cos();
    let c1 = (1.0 + a * a - b) / 2.0;
    for i in 2..n {
        r[i] = c1 * (source[i] + source[i-1]) + b * r[i-1] - a * a * r[i-2];
    }
    r
}

fn ih_high_pass_1pole(source: &[f64], period: f64) -> Vec<f64> {
    let n = source.len();
    let mut r = source.to_vec();
    let pi = std::f64::consts::PI;
    let sv = (2.0 * pi / period).sin();
    let cv = (2.0 * pi / period).cos();
    let alpha = 1.0 + (sv - 1.0) / cv;
    let c = 1.0 - alpha / 2.0;
    for i in 1..n {
        r[i] = c * source[i] - c * source[i-1] + (1.0 - alpha) * r[i-1];
    }
    r
}

fn ih_atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        tr[i] = (high[i] - low[i]).max((high[i] - close[i-1]).abs()).max((low[i] - close[i-1]).abs());
    }
    let mut r = vec![f64::NAN; n];
    if n < period { return r; }
    r[period-1] = tr[..period].iter().sum::<f64>() / period as f64;
    for i in period..n {
        r[i] = (r[i-1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }
    r
}

// ============================================================
// Phase 3 — New Indicator Implementations
// ============================================================

/// Cubed Weighted Moving Average
#[pyfunction]
pub fn cwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powi(3)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Squared Weighted Moving Average
#[pyfunction]
pub fn sqwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powi(2)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Square Root Weighted Moving Average
#[pyfunction]
pub fn srwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).sqrt()).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Variable Power Weighted Moving Average
#[pyfunction]
pub fn vpwma(source: PyReadonlyArray1<f64>, period: usize, power: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Pre-compute weights once
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powf(power)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// End Point Moving Average
#[pyfunction]
pub fn epma(source: PyReadonlyArray1<f64>, period: usize, offset: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let start = period + offset + 1;
        if n <= start { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1))
            .map(|i| (period as isize - i as isize - offset as isize) as f64)
            .collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in start..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// QStick — SMA of (close - open)
#[pyfunction]
pub fn qstick(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut r = vec![0.0f64; n];
        if period == 0 || n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_p = 1.0 / period as f64;
        let mut sum: f64 = (0..period).map(|k| c[[k, 2]] - c[[k, 1]]).sum();
        r[period - 1] = sum * inv_p;
        for i in period..n {
            sum += (c[[i, 2]] - c[[i, 1]]) - (c[[i - period, 2]] - c[[i - period, 1]]);
            r[i] = sum * inv_p;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// RMA — Wilder's EMA (alpha = 1/length)
#[pyfunction]
pub fn rma(source: PyReadonlyArray1<f64>, length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let alpha = 1.0 / length as f64;
        let mut r: Vec<f64> = src.to_vec();
        // Seed with last element to match Jesse's original rma_fast behavior:
        // rma_fast uses newseries[i-1] which wraps to newseries[-1] = source[-1] when i=0
        r[0] = alpha * src[0] + (1.0 - alpha) * src[n - 1];
        for i in 1..n {
            r[i] = alpha * src[i] + (1.0 - alpha) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Wilders Smoothing (same as rma with period param)
#[pyfunction]
pub fn wilders(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let pf = period as f64;
        let alpha = 1.0 / pf;
        let one_minus_alpha = 1.0 - alpha;
        // r[i] = (r[i-1] * (period - 1) + src[i]) / period = (1-1/period)*r[i-1] + (1/period)*src[i]
        for i in 1..n {
            r[i] = alpha * src[i] + one_minus_alpha * r[i - 1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// McGinley Dynamic
#[pyfunction]
pub fn mcginley_dynamic(source: PyReadonlyArray1<f64>, period: usize, k: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        r[0] = src[0];
        for i in 1..n {
            let prev = r[i-1];
            let ratio = src[i] / prev;
            let denom = (k * period as f64 * ratio.powi(4)).max(1.0);
            r[i] = prev + (src[i] - prev) / denom;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MWDX Average
#[pyfunction]
pub fn mwdx(source: PyReadonlyArray1<f64>, factor: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let val2 = 2.0 / factor - 1.0;
        let fac = 2.0 / (val2 + 1.0);
        let mut r: Vec<f64> = src.to_vec();
        for i in 1..n {
            r[i] = fac * src[i] + (1.0 - fac) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Holt-Winter Moving Average
#[pyfunction]
pub fn hwma(source: PyReadonlyArray1<f64>, na: f64, nb: f64, nc: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![0.0f64; n];
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let (mut last_a, mut last_v, mut last_f) = (0.0f64, 0.0f64, src[0]);
        for i in 0..n {
            let f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * src[i];
            let v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f);
            let a = (1.0 - nc) * last_a + nc * (v - last_v);
            r[i] = f + v + 0.5 * a;
            last_a = a; last_f = f; last_v = v;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// TRIX — triple EMA of log(price), percent change * 10000
#[pyfunction]
pub fn trix(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let log_src: Vec<f64> = src.iter().map(|&x| x.ln()).collect();
        let e1 = ih_ema(&log_src, period);
        let e2 = ih_ema(&e1, period);
        let e3 = ih_ema(&e2, period);
        let mut r = vec![f64::NAN; n];
        for i in 1..n { r[i] = (e3[i] - e3[i-1]) * 10000.0; }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// DPO — Detrended Price Oscillator
#[pyfunction]
pub fn dpo(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let shift = period / 2 + 1;
        let sma_vals = ih_sma(src.as_slice().unwrap(), period);
        let invalid = period - 1 + shift;
        let mut r = vec![f64::NAN; n];
        for i in invalid..n {
            if i >= shift {
                r[i] = src[i - shift] - sma_vals[i];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CCI — Commodity Channel Index
#[pyfunction]
pub fn cci(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let tp: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]] + c[[i,2]]) / 3.0).collect();
        let mut r = vec![f64::NAN; n];
        for i in (period-1)..n {
            let start = i + 1 - period;
            let sma: f64 = tp[start..=i].iter().sum::<f64>() / period as f64;
            let md: f64 = tp[start..=i].iter().map(|&x| (x - sma).abs()).sum::<f64>() / period as f64;
            r[i] = if md == 0.0 { 0.0 } else { (tp[i] - sma) / (0.015 * md) };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CMO — Chande Momentum Oscillator
#[pyfunction]
pub fn cmo(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n <= period || period == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Pre-split diffs into positive and negative magnitudes
        let mut pos_arr = vec![0.0f64; n - 1];
        let mut neg_arr = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            let d = src[i + 1] - src[i];
            if d > 0.0 { pos_arr[i] = d; } else if d < 0.0 { neg_arr[i] = -d; }
        }
        // Rolling window sum over [i-period, i) which is pos_arr/neg_arr indices [i-period, i-1]
        // First window for i=period: indices [0..period]
        let mut pos: f64 = pos_arr[..period].iter().sum();
        let mut neg: f64 = neg_arr[..period].iter().sum();
        let denom = pos + neg;
        r[period] = if denom == 0.0 { 0.0 } else { 100.0 * (pos - neg) / denom };
        for i in (period + 1)..n {
            // Slide: remove pos_arr[i-period-1], add pos_arr[i-1]
            pos += pos_arr[i - 1] - pos_arr[i - period - 1];
            neg += neg_arr[i - 1] - neg_arr[i - period - 1];
            let denom = pos + neg;
            r[i] = if denom == 0.0 { 0.0 } else { 100.0 * (pos - neg) / denom };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CFO — Chande Forecast Oscillator
#[pyfunction]
pub fn cfo(source: PyReadonlyArray1<f64>, period: usize, scalar: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if period == 0 || n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let pf = period as f64;
        let sx: f64 = (0..period).map(|j| j as f64).sum();
        let sxx: f64 = (0..period).map(|j| (j as f64).powi(2)).sum();
        let denom = pf * sxx - sx * sx;
        // sxy_i = B_i - (i-period+1)*A_i where A_i = sum src[k] and B_i = sum k*src[k] over window
        let mut a: f64 = src.slice(s![0..period]).iter().sum();
        let mut b: f64 = (0..period).map(|k| (k as f64) * src[k]).sum();
        // First valid index: i = period - 1
        let i0 = period - 1;
        {
            let sxy = b - (i0 as f64 - pf + 1.0) * a;
            let slope = (pf * sxy - sx * a) / denom;
            let intercept = (a - slope * sx) / pf;
            let reg_val = intercept + slope * (pf - 1.0);
            r[i0] = if src[i0] != 0.0 { scalar * (src[i0] - reg_val) / src[i0] } else { f64::NAN };
        }
        for i in period..n {
            let kout = i - period;
            a += src[i] - src[kout];
            b += (i as f64) * src[i] - (kout as f64) * src[kout];
            let sxy = b - (i as f64 - pf + 1.0) * a;
            let slope = (pf * sxy - sx * a) / denom;
            let intercept = (a - slope * sx) / pf;
            let reg_val = intercept + slope * (pf - 1.0);
            r[i] = if src[i] != 0.0 { scalar * (src[i] - reg_val) / src[i] } else { f64::NAN };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CG — Center of Gravity
#[pyfunction]
pub fn cg(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if period < 2 || n <= period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // For window k in [i-period+2, i] (length period-1):
        //   denom = sum src[k]
        //   num   = sum (1 + i - k) * src[k] = (1+i)*denom - sum k*src[k]
        // Maintain rolling sums.
        let w = period - 1;
        let first_i = period + 1;
        let lo0 = first_i - w + 1; // = period + 1 - (period - 1) + 1 = 3? wait
        // window for i=first_i: k in [first_i - w + 1, first_i] = [first_i - period + 2, first_i]
        // For period=10, first_i=11, window = [3, 11] inclusive (length 9).
        let start_k = first_i + 2 - period;
        let mut denom: f64 = src.slice(s![start_k..=first_i]).iter().sum();
        let mut sum_kx: f64 = (start_k..=first_i).map(|k| (k as f64) * src[k]).sum();
        let num0 = (1.0 + first_i as f64) * denom - sum_kx;
        r[first_i] = if denom != 0.0 { -num0 / denom } else { 0.0 };
        for i in (first_i + 1)..n {
            // Add k=i, remove k=i-(period-1)=i-period+1... wait window length w=period-1
            // For i, window is [i - period + 2, i]
            let kin = i;
            let kout = i - period + 1; // (i-1) - (period-1) + 1 = i - period + 1
            denom += src[kin] - src[kout];
            sum_kx += (kin as f64) * src[kin] - (kout as f64) * src[kout];
            let num = (1.0 + i as f64) * denom - sum_kx;
            r[i] = if denom != 0.0 { -num / denom } else { 0.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Aroon Oscillator
#[pyfunction]
pub fn aroonosc(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut r = vec![f64::NAN; n];
        for i in (period-1)..n {
            let start = i + 1 - period;
            let mut best_val = c[[start,3]];
            let mut best_idx = 0usize;
            let mut worst_val = c[[start,4]];
            let mut worst_idx = 0usize;
            for j in 0..period {
                if c[[start+j,3]] > best_val { best_val = c[[start+j,3]]; best_idx = j; }
                if c[[start+j,4]] < worst_val { worst_val = c[[start+j,4]]; worst_idx = j; }
            }
            r[i] = 100.0 * (best_idx as f64 - worst_idx as f64) / period as f64;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// ADXR — Average Directional Movement Index Rating
#[pyfunction]
pub fn adxr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut tr = vec![0.0f64; n];
        let mut dmp = vec![0.0f64; n];
        let mut dmm = vec![0.0f64; n];
        tr[0] = c[[0,3]] - c[[0,4]];
        for i in 1..n {
            tr[i] = (c[[i,3]] - c[[i,4]]).max((c[[i,3]] - c[[i-1,2]]).abs()).max((c[[i,4]] - c[[i-1,2]]).abs());
            let up = c[[i,3]] - c[[i-1,3]];
            let dn = c[[i-1,4]] - c[[i,4]];
            dmp[i] = if up > dn && up > 0.0 { up } else { 0.0 };
            dmm[i] = if dn > up && dn > 0.0 { dn } else { 0.0 };
        }
        let mut str_ = vec![0.0f64; n];
        let mut sdmp = vec![0.0f64; n];
        let mut sdmm = vec![0.0f64; n];
        str_[0] = tr[0]; sdmp[0] = dmp[0]; sdmm[0] = dmm[0];
        for i in 1..n {
            str_[i] = str_[i-1] - str_[i-1] / period as f64 + tr[i];
            sdmp[i] = sdmp[i-1] - sdmp[i-1] / period as f64 + dmp[i];
            sdmm[i] = sdmm[i-1] - sdmm[i-1] / period as f64 + dmm[i];
        }
        let mut dx = vec![0.0f64; n];
        for i in 0..n {
            if str_[i] != 0.0 {
                let di_p = sdmp[i] / str_[i] * 100.0;
                let di_m = sdmm[i] / str_[i] * 100.0;
                let s = di_p + di_m;
                dx[i] = if s != 0.0 { (di_p - di_m).abs() / s * 100.0 } else { 0.0 };
            }
        }
        let mut adx_val = vec![f64::NAN; n];
        for i in (period-1)..n {
            adx_val[i] = dx[i+1-period..=i].iter().sum::<f64>() / period as f64;
        }
        let mut result = vec![f64::NAN; n];
        for i in period..n {
            if !adx_val[i].is_nan() && !adx_val[i-period].is_nan() {
                result[i] = (adx_val[i] + adx_val[i-period]) / 2.0;
            }
        }
        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}

/// EFI — Elder's Force Index
#[pyfunction]
pub fn efi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        if n < 2 { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]).to_owned()); }
        let raw: Vec<f64> = (1..n).map(|i| (src[i] - src[i-1]) * c[[i,5]]).collect();
        let ema_vals = ih_ema(&raw, period);
        let mut r = vec![f64::NAN; n];
        for i in period..n { r[i] = ema_vals[i - 1]; }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// EMV — Ease of Movement
#[pyfunction]
pub fn emv(candles: PyReadonlyArray2<f64>, length: usize, div: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut emv_raw = vec![0.0f64; n];
        for i in 1..n {
            let hl2_prev = (c[[i-1,3]] + c[[i-1,4]]) / 2.0;
            let hl2_curr = (c[[i,3]] + c[[i,4]]) / 2.0;
            let hl2_chg = hl2_curr - hl2_prev;
            let vol = c[[i,5]];
            emv_raw[i] = if vol != 0.0 { div * hl2_chg * (c[[i,3]] - c[[i,4]]) / vol } else { 0.0 };
        }
        let mut r = vec![0.0f64; n];
        for i in (length-1)..n {
            r[i] = emv_raw[i+1-length..=i].iter().sum::<f64>() / length as f64;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// WAD — Williams Accumulation/Distribution
/// Note: uses candles[:,1] (open) as "close" per original code
#[pyfunction]
pub fn wad(candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut ad = vec![0.0f64; n];
        for i in 1..n {
            let close = c[[i,1]]; // WAD uses open column as "close"
            let prev_close = c[[i-1,1]];
            ad[i] = if close > prev_close {
                close - c[[i,4]].min(prev_close)
            } else if close < prev_close {
                close - c[[i,3]].max(prev_close)
            } else { 0.0 };
        }
        let mut r = vec![0.0f64; n];
        let mut cum = 0.0f64;
        for i in 0..n { cum += ad[i]; r[i] = cum; }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// NVI — Negative Volume Index
#[pyfunction]
pub fn nvi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        let mut r = vec![1000.0f64; n];
        for i in 1..n {
            if c[[i,5]] < c[[i-1,5]] {
                r[i] = r[i-1] * (1.0 + (src[i] - src[i-1]) / src[i-1]);
            } else {
                r[i] = r[i-1];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// PVI — Positive Volume Index
#[pyfunction]
pub fn pvi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        let mut r = vec![1000.0f64; n];
        for i in 1..n {
            if c[[i,5]] > c[[i-1,5]] {
                r[i] = r[i-1] * (1.0 + (src[i] - src[i-1]) / src[i-1]);
            } else {
                r[i] = r[i-1];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MASS — Mass Index
#[pyfunction]
pub fn mass(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let hl: Vec<f64> = (0..n).map(|i| c[[i,3]] - c[[i,4]]).collect();
        let e1 = ih_ema(&hl, 9);
        let e2 = ih_ema(&e1, 9);
        let ratio: Vec<f64> = (0..n).map(|i| if e2[i] != 0.0 { e1[i] / e2[i] } else { 0.0 }).collect();
        let mut r = vec![0.0f64; n];
        for i in (period-1)..n {
            r[i] = ratio[i+1-period..=i].iter().sum();
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// PFE — Polarized Fractal Efficiency
#[pyfunction]
pub fn pfe(source: PyReadonlyArray1<f64>, period: usize, smoothing: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let ln = period - 1; // lookback = period - 1

        // Compute ln-th order differences (matches Python's np.diff(source, ln))
        // Each iteration reduces length by 1; after ln iterations: length = n - ln
        let mut d: Vec<f64> = src.clone();
        for _ in 0..ln {
            d = d.windows(2).map(|w| w[1] - w[0]).collect();
        }
        // d has length n - ln

        // a[i] = sqrt(d[i]^2 + period^2)
        let a: Vec<f64> = d.iter().map(|&x| (x * x + (period as f64).powi(2)).sqrt()).collect();

        // First differences and their sqrt(1 + dx^2) terms; length = n - 1
        let sqrt_term: Vec<f64> = src.windows(2).map(|w| {
            let dx = w[1] - w[0];
            (1.0 + dx * dx).sqrt()
        }).collect();

        // Rolling sum of sqrt_term with window = ln; matches rolling_sum(sqrt_term, ln)
        // Result: first (ln-1) elements are NaN, then (n-ln) valid sums
        // Total length: n - 1 (same as sqrt_term)
        let mut b: Vec<f64> = vec![f64::NAN; ln - 1];
        if sqrt_term.len() >= ln {
            let init: f64 = sqrt_term[..ln].iter().sum();
            b.push(init);
            for i in ln..sqrt_term.len() {
                let prev = *b.last().unwrap();
                b.push(prev + sqrt_term[i] - sqrt_term[i - ln]);
            }
        }
        // b has length n - 1

        // Align to source length (same_length behavior: prepend NaN to shorter array)
        // a_sl: prepend (n - (n-ln)) = ln NaN → a_sl[i] = a[i-ln] for i >= ln
        // b_sl: prepend (n - (n-1)) = 1 NaN → b_sl[i] = b[i-1] for i >= 1
        // diff_sl: same as a_sl, d[i-ln] for i >= ln
        let mut pfetmp = vec![0.0f64; n];
        let mut sign = vec![-1.0f64; n]; // default: -1 (NaN > 0 is False)
        for i in ln..n {
            let a_val = a[i - ln];
            let b_val = if i >= 1 { b[i - 1] } else { f64::NAN };
            pfetmp[i] = if b_val.is_nan() || b_val == 0.0 {
                0.0
            } else {
                100.0 * a_val / b_val
            };
            if d[i - ln] > 0.0 {
                sign[i] = 1.0;
            }
        }

        let signed: Vec<f64> = sign.iter().zip(pfetmp.iter()).map(|(s, p)| s * p).collect();
        let r = ih_ema(&signed, smoothing);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// HMA — Hull Moving Average
#[pyfunction]
pub fn hma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let half = period / 2;
        let sq = (period as f64).sqrt() as usize;
        let wma_half = ih_wma(&src, half);
        let wma_full = ih_wma(&src, period);
        let raw: Vec<f64> = (0..n).map(|i| 2.0 * wma_half[i] - wma_full[i]).collect();
        let r = ih_wma(&raw, sq);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Linear Regression Value
#[pyfunction]
pub fn linearreg(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let mean_x = (period as f64 - 1.0) / 2.0;
        let sxx: f64 = (0..period).map(|j| (j as f64 - mean_x).powi(2)).sum();
        for i in (period-1)..n {
            let win = &src.as_slice().unwrap()[i+1-period..=i];
            let mean_y = win.iter().sum::<f64>() / period as f64;
            let sxy: f64 = win.iter().enumerate().map(|(j, &y)| (y - mean_y) * (j as f64 - mean_x)).sum();
            r[i] = mean_y + mean_x * (sxy / sxx);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Laguerre RSI
#[pyfunction]
pub fn lrsi(candles: PyReadonlyArray2<f64>, alpha: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let price: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let gamma = 1.0 - alpha;
        let (mut l0, mut l1, mut l2, mut l3) = (price[0], price[0], price[0], price[0]);
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let new_l0 = alpha * price[i] + gamma * l0;
            let new_l1 = -gamma * new_l0 + l0 + gamma * l1;
            let new_l2 = -gamma * new_l1 + l1 + gamma * l2;
            let new_l3 = -gamma * new_l2 + l2 + gamma * l3;
            l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
            let mut cu = 0.0f64;
            let mut cd = 0.0f64;
            if l0 >= l1 { cu += l0 - l1; } else { cd += l1 - l0; }
            if l1 >= l2 { cu += l1 - l2; } else { cd += l2 - l1; }
            if l2 >= l3 { cu += l2 - l3; } else { cd += l3 - l2; }
            r[i] = if cu + cd == 0.0 { 0.0 } else { cu / (cu + cd) };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MAAQ — Moving Average Adaptive Q
#[pyfunction]
pub fn maaq(source: PyReadonlyArray1<f64>, period: usize, fast_period: usize, slow_period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let fast_sc = 2.0 / (fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (slow_period as f64 + 1.0);
        let mut diff = vec![0.0f64; n];
        for i in 1..n { diff[i] = (src[i] - src[i - 1]).abs(); }
        let mut r = src.to_vec();
        if n <= period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Initial rolling-sum window [period+1-period..=period] = [1..=period]
        let mut noise: f64 = diff[1..=period].iter().sum();
        let i0 = period;
        let signal0 = (src[i0] - src[i0 - period]).abs();
        let ratio0 = if noise != 0.0 { signal0 / noise } else { 0.0 };
        let temp0 = (ratio0 * fast_sc + slow_sc).powi(2);
        r[i0] = r[i0 - 1] + temp0 * (src[i0] - r[i0 - 1]);
        for i in (period + 1)..n {
            // Slide: remove diff[i-period], add diff[i]
            noise += diff[i] - diff[i - period];
            let signal = (src[i] - src[i - period]).abs();
            let ratio = if noise != 0.0 { signal / noise } else { 0.0 };
            let temp = (ratio * fast_sc + slow_sc).powi(2);
            r[i] = r[i - 1] + temp * (src[i] - r[i - 1]);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// VIDYA — Variable Index Dynamic Average
#[pyfunction]
pub fn vidya(source: PyReadonlyArray1<f64>, length: usize, fix_cmo: bool, select: bool) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let alpha = 2.0 / (length as f64 + 1.0);
        let cmo_length = if fix_cmo { 9usize } else { length };
        let mut momm = vec![0.0f64; n];
        for i in 1..n { momm[i] = src[i] - src[i-1]; }
        let mut vidya_r = src.to_vec();
        for i in 1..n {
            let start = if i + 1 > cmo_length { i + 1 - cmo_length } else { 0 };
            let (mut sm1, mut sm2) = (0.0f64, 0.0f64);
            for k in start..=i {
                if momm[k] >= 0.0 { sm1 += momm[k]; } else { sm2 -= momm[k]; }
            }
            let k_val = if select {
                let tot = sm1 + sm2;
                if tot != 0.0 { ((sm1 - sm2) / tot * 100.0).abs() / 100.0 } else { 0.0 }
            } else {
                let start2 = if i + 1 > length { i + 1 - length } else { 0 };
                let slice = &src.as_slice().unwrap()[start2..=i];
                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let var: f64 = slice.iter().map(|&x| (x-mean).powi(2)).sum::<f64>() / slice.len() as f64;
                var.sqrt()
            };
            let eff_alpha = alpha * k_val;
            vidya_r[i] = eff_alpha * src[i] + (1.0 - eff_alpha) * vidya_r[i-1];
        }
        Ok(PyArray1::from_vec(py, vidya_r).to_owned())
    })
}

/// VLMA inner loop (called from Python after computing deviation bands)
#[pyfunction]
pub fn vlma_inner(
    source: PyReadonlyArray1<f64>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>,
    d: PyReadonlyArray1<f64>,
    min_period: usize,
    max_period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let av = a.as_array();
        let bv = b.as_array();
        let cv = c.as_array();
        let dv = d.as_array();
        let n = src.len();
        let mut r = src.to_vec();
        let mut period = max_period as f64;
        for i in 1..n {
            let nz_period = period;
            let next_period = if bv[i] <= src[i] && src[i] <= cv[i] {
                nz_period + 1.0
            } else if src[i] < av[i] || src[i] > dv[i] {
                nz_period - 1.0
            } else { nz_period };
            period = next_period.max(min_period as f64).min(max_period as f64);
            let sc = 2.0 / (period + 1.0);
            r[i] = src[i] * sc + (1.0 - sc) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// NMA — Natural Moving Average
#[pyfunction]
pub fn nma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let clipped: Vec<f64> = src.iter().map(|&x| x.max(1e-10)).collect();
        let ln: Vec<f64> = clipped.iter().map(|&x| x.ln() * 1000.0).collect();
        let mut r = vec![f64::NAN; n];
        if period < 1 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Precompute |ln[k] - ln[k-1]| for k >= 1
        let mut abs_diff = vec![0.0f64; n];
        for k in 1..n {
            abs_diff[k] = (ln[k] - ln[k - 1]).abs();
        }
        // Precompute weights: w[i] = sqrt(i+1) - sqrt(i) for i in 0..period
        let weights: Vec<f64> = (0..period).map(|i| (i as f64 + 1.0).sqrt() - (i as f64).sqrt()).collect();
        let last_i = period - 1;
        for j in (period + 1)..n {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..period {
                let oi = abs_diff[j - i];
                num += oi * weights[i];
                den += oi;
            }
            let ratio = if den != 0.0 { num / den } else { 0.0 };
            r[j] = clipped[j - last_i] * ratio + clipped[j - last_i - 1] * (1.0 - ratio);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// JMA — Jurik Moving Average
#[pyfunction]
pub fn jma(source: PyReadonlyArray1<f64>, period: usize, phase: f64, power: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let phase_ratio = if phase < -100.0 { 0.5 } else if phase > 100.0 { 2.5 } else { phase / 100.0 + 1.5 };
        let beta = 0.45 * (period as f64 - 1.0) / (0.45 * (period as f64 - 1.0) + 2.0);
        let alpha = beta.powi(power as i32);
        let mut e0 = vec![0.0f64; n];
        let mut e1 = vec![0.0f64; n];
        let mut e2 = vec![0.0f64; n];
        let mut jma_v = src.to_vec();
        for i in 1..n {
            e0[i] = (1.0 - alpha) * src[i] + alpha * e0[i-1];
            e1[i] = (src[i] - e0[i]) * (1.0 - beta) + beta * e1[i-1];
            e2[i] = (e0[i] + phase_ratio * e1[i] - jma_v[i-1]) * (1.0 - alpha).powi(2) + alpha.powi(2) * e2[i-1];
            jma_v[i] = e2[i] + jma_v[i-1];
        }
        Ok(PyArray1::from_vec(py, jma_v).to_owned())
    })
}

/// RSX — Relative Strength Xtra
#[pyfunction]
pub fn rsx(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        let (mut f0, mut f8) = (0.0f64, 0.0f64);
        let (mut f18, mut f20) = (0.0f64, 0.0f64);
        let (mut f28, mut f30) = (0.0f64, 0.0f64);
        let (mut f38, mut f40) = (0.0f64, 0.0f64);
        let (mut f48, mut f50) = (0.0f64, 0.0f64);
        let (mut f58, mut f60) = (0.0f64, 0.0f64);
        let (mut f68, mut f70) = (0.0f64, 0.0f64);
        let (mut f78, mut f80) = (0.0f64, 0.0f64);
        let (mut f88, mut f90) = (0.0f64, 0.0f64);
        let mut v14 = 0.0f64;
        let mut v20 = 0.0f64;
        for i in period..n {
            if f90 == 0.0 {
                f90 = 1.0; f0 = 0.0;
                f88 = if period >= 6 { period as f64 - 1.0 } else { 5.0 };
                f8 = 100.0 * src[i];
                f18 = 3.0 / (period as f64 + 2.0);
                f20 = 1.0 - f18;
            } else {
                f90 = if f88 <= f90 { f88 + 1.0 } else { f90 + 1.0 };
                let f10 = f8;
                f8 = 100.0 * src[i];
                let v8 = f8 - f10;
                f28 = f20 * f28 + f18 * v8;
                f30 = f18 * f28 + f20 * f30;
                let vc = f28 * 1.5 - f30 * 0.5;
                f38 = f20 * f38 + f18 * vc;
                f40 = f18 * f38 + f20 * f40;
                let v10 = f38 * 1.5 - f40 * 0.5;
                f48 = f20 * f48 + f18 * v10;
                f50 = f18 * f48 + f20 * f50;
                v14 = f48 * 1.5 - f50 * 0.5;
                f58 = f20 * f58 + f18 * v8.abs();
                f60 = f18 * f58 + f20 * f60;
                let v18 = f58 * 1.5 - f60 * 0.5;
                f68 = f20 * f68 + f18 * v18;
                f70 = f18 * f68 + f20 * f70;
                let v1c = f68 * 1.5 - f70 * 0.5;
                f78 = f20 * f78 + f18 * v1c;
                f80 = f18 * f78 + f20 * f80;
                v20 = f78 * 1.5 - f80 * 0.5;
                if f88 >= f90 && f8 != f10 { f0 = 1.0; }
                if f88 == f90 && f0 == 0.0 { f90 = 0.0; }
            }
            r[i] = if f88 < f90 && v20 > 1e-10 {
                ((v14 / v20 + 1.0) * 50.0).min(100.0).max(0.0)
            } else { 50.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// SuperSmoother 2-pole
#[pyfunction]
pub fn supersmoother(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let r = ih_supersmoother_2pole(&src, period as f64);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// SuperSmoother 3-pole
#[pyfunction]
pub fn supersmoother_3_pole(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let a = (-pi / period as f64).exp();
        let b = 2.0 * a * (1.738 * pi / period as f64).cos();
        let c = a * a;
        let mut r = src.to_vec();
        for i in 3..n {
            r[i] = (1.0 - c * c - b + b * c) * src[i]
                 + (b + c) * r[i-1]
                 + (-c - b * c) * r[i-2]
                 + c * c * r[i-3];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// High Pass 1-pole
#[pyfunction]
pub fn high_pass(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let r = ih_high_pass_1pole(&src, period as f64);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// High Pass 2-pole
#[pyfunction]
pub fn high_pass_2_pole(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let k = 0.707f64;
        let sv = (2.0 * pi * k / period as f64).sin();
        let cv = (2.0 * pi * k / period as f64).cos();
        let alpha = 1.0 + (sv - 1.0) / cv;
        let c = (1.0 - alpha / 2.0).powi(2);
        let mut r = src.to_vec();
        for i in 2..n {
            r[i] = c * src[i]
                 - 2.0 * c * src[i-1]
                 + c * src[i-2]
                 + 2.0 * (1.0 - alpha) * r[i-1]
                 - (1.0 - alpha).powi(2) * r[i-2];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Bandpass Filter → (bp, bp_normalized, signal, trigger)
#[pyfunction]
pub fn bandpass(source: PyReadonlyArray1<f64>, period: usize, bandwidth: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let hp = ih_high_pass_1pole(&src, 4.0 * period as f64 / bandwidth);
        let beta = (2.0 * pi / period as f64).cos();
        let gamma = (2.0 * pi * bandwidth / period as f64).cos();
        let alpha = 1.0 / gamma - (1.0 / (gamma * gamma) - 1.0).sqrt();
        let mut bp = hp.clone();
        for i in 2..n {
            bp[i] = 0.5 * (1.0 - alpha) * (hp[i] - hp[i-2])
                  + beta * (1.0 + alpha) * bp[i-1]
                  - alpha * bp[i-2];
        }
        // AGC
        let k = 0.991f64;
        let mut peak = bp.clone();
        for i in 1..n {
            peak[i] = peak[i-1] * k;
            if bp[i].abs() > peak[i] { peak[i] = bp[i].abs(); }
        }
        let bp_norm: Vec<f64> = (0..n).map(|i| if peak[i] != 0.0 { bp[i] / peak[i] } else { 0.0 }).collect();
        let trigger = ih_high_pass_1pole(&bp_norm, period as f64 / bandwidth / 1.5);
        let signal: Vec<f64> = (0..n).map(|i| {
            if bp_norm[i] < trigger[i] { 1.0 } else if trigger[i] < bp_norm[i] { -1.0 } else { 0.0 }
        }).collect();
        Ok((
            PyArray1::from_vec(py, bp).to_owned(),
            PyArray1::from_vec(py, bp_norm).to_owned(),
            PyArray1::from_vec(py, signal).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}

/// Gaussian Filter
#[pyfunction]
pub fn gauss(source: PyReadonlyArray1<f64>, period: usize, poles: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src_arr = source.as_array();
        let n_total = src_arr.len();
        let src: Vec<f64> = src_arr.iter().filter(|x| !x.is_nan()).cloned().collect();
        let to_fill = n_total - src.len();
        let pi = std::f64::consts::PI;
        let beta = (1.0 - (2.0 * pi / period as f64).cos()) / (2.0f64.powf(1.0 / poles as f64) - 1.0);
        let alpha = -beta + (beta * beta + 2.0 * beta).sqrt();
        let m = src.len();
        let mut fil = vec![0.0f64; poles + m];
        let coeff: Vec<f64> = match poles {
            1 => vec![alpha, 1.0 - alpha],
            2 => vec![alpha.powi(2), 2.0 * (1.0 - alpha), -(1.0 - alpha).powi(2)],
            3 => vec![alpha.powi(3), 3.0*(1.0-alpha), -3.0*(1.0-alpha).powi(2), (1.0-alpha).powi(3)],
            _ => vec![alpha.powi(4), 4.0*(1.0-alpha), -6.0*(1.0-alpha).powi(2), 4.0*(1.0-alpha).powi(3), -(1.0-alpha).powi(4)],
        };
        for i in 0..m {
            let val: f64 = coeff[0] * src[i] + coeff[1..].iter().enumerate().map(|(k, &c)| {
                let idx = poles + i - 1 - k;
                c * fil[idx]
            }).sum::<f64>();
            fil[poles + i] = val;
        }
        let fil_slice = &fil[poles..];
        let mut r = vec![f64::NAN; n_total];
        let start = if to_fill > 0 { to_fill } else { 0 };
        for (k, &v) in fil_slice.iter().enumerate() {
            if start + k < n_total { r[start + k] = v; }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Reflex indicator
#[pyfunction]
pub fn reflex(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let ssf = ih_supersmoother_2pole(&src, period as f64 / 2.0);
        let n = ssf.len();
        let mut rf = vec![0.0f64; n];
        let mut ms = vec![0.0f64; n];
        let mut sums = vec![0.0f64; n];
        for i in period..n {
            let slope = (ssf[i-period] - ssf[i]) / period as f64;
            let mut my_sum = 0.0f64;
            for t in 1..=period {
                my_sum += (ssf[i] + t as f64 * slope) - ssf[i-t];
            }
            my_sum /= period as f64;
            sums[i] = my_sum;
            ms[i] = 0.04 * sums[i] * sums[i] + 0.96 * ms[i-1];
            if ms[i] > 0.0 { rf[i] = sums[i] / ms[i].sqrt(); }
        }
        Ok(PyArray1::from_vec(py, rf).to_owned())
    })
}

/// TrendFlex indicator
#[pyfunction]
pub fn trendflex(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let ssf = ih_supersmoother_2pole(&src, period as f64 / 2.0);
        let n = ssf.len();
        let mut tf = vec![0.0f64; n];
        let mut ms = vec![0.0f64; n];
        let mut sums = vec![0.0f64; n];
        for i in period..n {
            let mut my_sum = 0.0f64;
            for t in 1..=period { my_sum += ssf[i] - ssf[i-t]; }
            my_sum /= period as f64;
            sums[i] = my_sum;
            ms[i] = 0.04 * sums[i] * sums[i] + 0.96 * ms[i-1];
            if ms[i] != 0.0 { tf[i] = sums[i] / ms[i].sqrt(); }
        }
        Ok(PyArray1::from_vec(py, tf).to_owned())
    })
}

/// Instantaneous Trendline → (signal, it, trigger)
#[pyfunction]
pub fn itrend(source: PyReadonlyArray1<f64>, alpha: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut it = src.to_vec();
        for i in 2..7.min(n) {
            it[i] = (src[i] + 2.0 * src[i-1] + src[i-2]) / 4.0;
        }
        for i in 7..n {
            it[i] = (alpha - alpha * alpha / 4.0) * src[i]
                  + alpha * alpha / 2.0 * src[i-1]
                  - (alpha - 3.0 * alpha * alpha / 4.0) * src[i-2]
                  + 2.0 * (1.0 - alpha) * it[i-1]
                  - (1.0 - alpha).powi(2) * it[i-2];
        }
        let mut trigger = vec![0.0f64; n];
        let mut signal = vec![0.0f64; n];
        for i in 0..n {
            let lag2 = if i >= 20 { it[i-20] } else { it[i] };
            trigger[i] = 2.0 * it[i] - lag2;
            signal[i] = if trigger[i] > it[i] { 1.0 } else if trigger[i] < it[i] { -1.0 } else { 0.0 };
        }
        Ok((
            PyArray1::from_vec(py, signal).to_owned(),
            PyArray1::from_vec(py, it).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}

/// Voss Predictive Filter → (voss, filt)
#[pyfunction]
pub fn voss(source: PyReadonlyArray1<f64>, period: usize, predict: usize, bandwidth: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let order = 3 * predict;
        let f1 = (2.0 * pi / period as f64).cos();
        let g1 = (bandwidth * 2.0 * pi / period as f64).cos();
        let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();
        let mut filt = vec![0.0f64; n];
        let mut voss_v = vec![0.0f64; n];
        for i in 0..n {
            if i > period && i > 5 && i > order {
                filt[i] = 0.5 * (1.0 - s1) * (src[i] - src[i-2])
                        + f1 * (1.0 + s1) * filt[i-1]
                        - s1 * filt[i-2];
            }
        }
        for i in 0..n {
            if i > period && i > 5 && i > order {
                let sumc: f64 = (0..order).map(|count| {
                    (count + 1) as f64 / order as f64 * voss_v[i - (order - count)]
                }).sum();
                voss_v[i] = (3.0 + order as f64) / 2.0 * filt[i] - sumc;
            }
        }
        Ok((
            PyArray1::from_vec(py, voss_v).to_owned(),
            PyArray1::from_vec(py, filt).to_owned(),
        ))
    })
}

/// EDCF — Ehlers Distance Coefficient Filter
#[pyfunction]
pub fn edcf(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n < 2 * period || period < 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Precompute dist[k] = sum_{lb=1}^{period-1} (src[k] - src[k-lb])^2
        let mut dist = vec![0.0f64; n];
        for k in (period - 1)..n {
            let mut d_sum = 0.0f64;
            for lb in 1..period {
                let d = src[k] - src[k - lb];
                d_sum += d * d;
            }
            dist[k] = d_sum;
        }
        // r[j] uses k = j-i in [j-period+1, j], so rolling window of size `period` ending at j.
        let first_j = 2 * period;
        let mut num: f64 = 0.0;
        let mut coef: f64 = 0.0;
        for k in (first_j - period + 1)..=first_j {
            num += dist[k] * src[k];
            coef += dist[k];
        }
        r[first_j] = if coef != 0.0 { num / coef } else { 0.0 };
        for j in (first_j + 1)..n {
            // Slide window: remove k=j-period, add k=j
            num -= dist[j - period] * src[j - period];
            coef -= dist[j - period];
            num += dist[j] * src[j];
            coef += dist[j];
            r[j] = if coef != 0.0 { num / coef } else { 0.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Correlation Cycle → (real, imag, angle, state)
#[pyfunction]
pub fn correlation_cycle(source: PyReadonlyArray1<f64>, period: usize, threshold: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pix2 = 4.0 * (1.0f64).asin();
        let period = period.max(2);
        let pf = period as f64;
        let mut real_part = vec![f64::NAN; n];
        let mut imag_part = vec![f64::NAN; n];

        // Precompute trig terms (depend only on j and period)
        let yc: Vec<f64> = (0..period).map(|j| (pix2 * (j as f64 + 1.0) / pf).cos()).collect();
        let ys: Vec<f64> = (0..period).map(|j| -((pix2 * (j as f64 + 1.0) / pf).sin())).collect();
        let ry: f64 = yc.iter().sum();
        let iy: f64 = ys.iter().sum();
        let ryy: f64 = yc.iter().map(|v| v * v).sum();
        let iyy: f64 = ys.iter().map(|v| v * v).sum();
        let t2 = pf * ryy - ry * ry;
        let u2 = pf * iyy - iy * iy;
        if !(t2 > 0.0 && u2 > 0.0) {
            // degenerate — fall through with NaN results
        }
        let t2_sqrt = if t2 > 0.0 { t2.sqrt() } else { 0.0 };
        let u2_sqrt = if u2 > 0.0 { u2.sqrt() } else { 0.0 };

        for i in period..n {
            let (mut rx, mut rxx, mut rxy, mut ixy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..period {
                let v = src[i - j - 1];
                let x = if v.is_nan() { 0.0 } else { v };
                rx += x;
                rxx += x * x;
                rxy += x * yc[j];
                ixy += x * ys[j];
            }
            let t1 = pf * rxx - rx * rx;
            if t1 > 0.0 && t2_sqrt > 0.0 {
                real_part[i] = (pf * rxy - rx * ry) / (t1.sqrt() * t2_sqrt);
            }
            if t1 > 0.0 && u2_sqrt > 0.0 {
                imag_part[i] = (pf * ixy - rx * iy) / (t1.sqrt() * u2_sqrt);
            }
        }
        let half_pi = 1.0f64.asin();
        // Initialize to NaN to match Python behavior (imagPart==0 gives 0.0, NaN gives NaN)
        let mut angle = vec![f64::NAN; n];
        for i in 0..n {
            let im = imag_part[i];
            let re = real_part[i];
            if im.is_nan() {
                // stays NaN
            } else if im == 0.0 {
                angle[i] = 0.0;
                // no subtraction since imag > 0 is false
            } else {
                let mut a = ((re / im).atan() + half_pi).to_degrees();
                if im > 0.0 { a -= 180.0; }
                angle[i] = a;
            }
        }
        // Vectorized update: use original angles (clone) to match Python's non-sequential np.where
        let orig_angle = angle.clone();
        for i in 1..n {
            let prior = orig_angle[i-1];
            // NaN > x is false in Rust, so NaN prior means no update
            if prior > angle[i] && prior - angle[i] < 270.0 { angle[i] = prior; }
        }
        let mut state = vec![0.0f64; n];
        for i in 1..n {
            let prior = angle[i-1];
            if (angle[i] - prior).abs() < threshold {
                state[i] = if angle[i] >= 0.0 { 1.0 } else if angle[i] < 0.0 { -1.0 } else { 0.0 };
            }
        }
        Ok((
            PyArray1::from_vec(py, real_part).to_owned(),
            PyArray1::from_vec(py, imag_part).to_owned(),
            PyArray1::from_vec(py, angle).to_owned(),
            PyArray1::from_vec(py, state).to_owned(),
        ))
    })
}

/// Heikin Ashi Candles → (open, close, high, low)
#[pyfunction]
pub fn heikin_ashi_candles(candles: PyReadonlyArray2<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut ha_open = vec![f64::NAN; n];
        let mut ha_close = vec![f64::NAN; n];
        let mut ha_high = vec![f64::NAN; n];
        let mut ha_low = vec![f64::NAN; n];
        for i in 1..n {
            ha_open[i] = (c[[i-1,1]] + c[[i-1,2]]) / 2.0;
            ha_close[i] = (c[[i,1]] + c[[i,2]] + c[[i,3]] + c[[i,4]]) / 4.0;
            ha_high[i] = c[[i,3]].max(ha_open[i]).max(ha_close[i]);
            ha_low[i] = c[[i,4]].min(ha_open[i]).min(ha_close[i]);
        }
        Ok((
            PyArray1::from_vec(py, ha_open).to_owned(),
            PyArray1::from_vec(py, ha_close).to_owned(),
            PyArray1::from_vec(py, ha_high).to_owned(),
            PyArray1::from_vec(py, ha_low).to_owned(),
        ))
    })
}

/// Keltner Channel inner (takes pre-computed MA) → (upper, middle, lower)
#[pyfunction]
pub fn keltner_inner(
    ma_values: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
    multiplier: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let ma = ma_values.as_array();
        let h = high.as_array();
        let l = low.as_array();
        let cl = close.as_array();
        let n = ma.len();
        let h_sl = h.as_slice().unwrap();
        let l_sl = l.as_slice().unwrap();
        let cl_sl = cl.as_slice().unwrap();
        let atr_vals = ih_atr_wilder(h_sl, l_sl, cl_sl, period);
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        for i in 0..n {
            if !atr_vals[i].is_nan() {
                upper[i] = ma[i] + atr_vals[i] * multiplier;
                lower[i] = ma[i] - atr_vals[i] * multiplier;
            }
        }
        Ok((
            PyArray1::from_vec(py, upper).to_owned(),
            PyArray1::from_vec(py, ma.to_vec()).to_owned(),
            PyArray1::from_vec(py, lower).to_owned(),
        ))
    })
}

/// SuperTrend → (trend, changed)
#[pyfunction]
pub fn supertrend(candles: PyReadonlyArray2<f64>, period: usize, factor: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let h: Vec<f64> = (0..n).map(|i| c[[i,3]]).collect();
        let l: Vec<f64> = (0..n).map(|i| c[[i,4]]).collect();
        let cl: Vec<f64> = (0..n).map(|i| c[[i,2]]).collect();
        let atr_vals = ih_atr_wilder(&h, &l, &cl, period);
        let mut upper_basic = vec![0.0f64; n];
        let mut lower_basic = vec![0.0f64; n];
        let mut upper_band = vec![0.0f64; n];
        let mut lower_band = vec![0.0f64; n];
        for i in 0..n {
            let mid = (c[[i,3]] + c[[i,4]]) / 2.0;
            let atr = if atr_vals[i].is_nan() { 0.0 } else { atr_vals[i] };
            upper_basic[i] = mid + factor * atr;
            lower_basic[i] = mid - factor * atr;
            upper_band[i] = upper_basic[i];
            lower_band[i] = lower_basic[i];
        }
        let mut trend = vec![0.0f64; n];
        let mut changed = vec![0.0f64; n];
        let idx = period.saturating_sub(1);
        if idx < n {
            trend[idx] = if c[[idx,2]] <= upper_band[idx] { upper_band[idx] } else { lower_band[idx] };
        }
        for i in period..n {
            let p = i - 1;
            let prev_cl = c[[p,2]];
            upper_band[i] = if prev_cl <= upper_band[p] {
                upper_basic[i].min(upper_band[p])
            } else { upper_basic[i] };
            lower_band[i] = if prev_cl >= lower_band[p] {
                lower_basic[i].max(lower_band[p])
            } else { lower_basic[i] };
            if trend[p] == upper_band[p] {
                if c[[i,2]] <= upper_band[i] {
                    trend[i] = upper_band[i]; changed[i] = 0.0;
                } else {
                    trend[i] = lower_band[i]; changed[i] = 1.0;
                }
            } else {
                if c[[i,2]] >= lower_band[i] {
                    trend[i] = lower_band[i]; changed[i] = 0.0;
                } else {
                    trend[i] = upper_band[i]; changed[i] = 1.0;
                }
            }
        }
        Ok((
            PyArray1::from_vec(py, trend).to_owned(),
            PyArray1::from_vec(py, changed).to_owned(),
        ))
    })
}

/// EMD — Empirical Mode Decomposition → (upperband, middleband, lowerband)
#[pyfunction]
pub fn emd(candles: PyReadonlyArray2<f64>, period: usize, delta: f64, fraction: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let price: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let pi = std::f64::consts::PI;
        let beta = (2.0 * pi / period as f64).cos();
        let gamma_v = 1.0 / (4.0 * pi * delta / period as f64).cos();
        let alpha = gamma_v - (gamma_v * gamma_v - 1.0).sqrt();
        let mut bp = vec![0.0f64; n];
        for i in 0..n {
            if i > 2 {
                bp[i] = 0.5 * (1.0 - alpha) * (price[i] - price[i-2])
                      + beta * (1.0 + alpha) * bp[i-1]
                      - alpha * bp[i-2];
            } else if i == 2 {
                bp[i] = 0.5 * (1.0 - alpha) * (price[i] - price[i-2]);
            }
        }
        // SMA of bp over 2*period
        let mean = ih_sma(&bp, 2*period);
        let mut peak = bp.clone();
        let mut valley = bp.clone();
        for i in 0..n {
            peak[i] = peak[i-1.min(i)];
            valley[i] = valley[i-1.min(i)];
            if i > 2 {
                if bp[i-1] > bp[i] && bp[i-1] > bp[i-2] { peak[i] = bp[i-1]; }
                if bp[i-1] < bp[i] && bp[i-1] < bp[i-2] { valley[i] = bp[i-1]; }
            }
        }
        let avg_peak: Vec<f64> = {
            let sp = ih_sma(&peak, 50);
            sp.iter().map(|&x| x * fraction).collect()
        };
        let avg_valley: Vec<f64> = {
            let sv = ih_sma(&valley, 50);
            sv.iter().map(|&x| x * fraction).collect()
        };
        Ok((
            PyArray1::from_vec(py, avg_peak).to_owned(),
            PyArray1::from_vec(py, mean).to_owned(),
            PyArray1::from_vec(py, avg_valley).to_owned(),
        ))
    })
}

/// Fisher Transform → (fisher, signal)
#[pyfunction]
pub fn fisher(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mid: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let mut fisher_v = vec![0.0f64; n];
        let mut fisher_sig = vec![0.0f64; n];
        let mut value1 = 0.0f64;
        for i in period..n {
            let max_h = mid[i+1-period..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_l = mid[i+1-period..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let mut value0 = if max_h - min_l == 0.0 { 0.0 } else {
                0.33 * 2.0 * ((mid[i] - min_l) / (max_h - min_l) - 0.5) + 0.67 * value1
            };
            value0 = value0.min(0.999).max(-0.999);
            fisher_v[i] = 0.5 * ((1.0 + value0) / (1.0 - value0)).ln() + 0.5 * fisher_v[i-1];
            fisher_sig[i] = fisher_v[i-1];
            value1 = value0;
        }
        Ok((
            PyArray1::from_vec(py, fisher_v).to_owned(),
            PyArray1::from_vec(py, fisher_sig).to_owned(),
        ))
    })
}

/// MAMA — MESA Adaptive Moving Average → (mama, fama)
#[pyfunction]
pub fn mama(source: PyReadonlyArray1<f64>, fastlimit: f64, slowlimit: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let mut sp = vec![0.0f64; n];
        let mut dt = vec![0.0f64; n];
        let mut q1 = vec![0.0f64; n];
        let mut i1 = vec![0.0f64; n];
        let mut ji = vec![0.0f64; n];
        let mut jq = vec![0.0f64; n];
        let mut i2 = vec![0.0f64; n];
        let mut q2 = vec![0.0f64; n];
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        let mut p1 = vec![0.0f64; n];
        let mut p2: f64;
        let mut p3 = vec![0.0f64; n];
        let mut p_arr = vec![0.0f64; n];
        let mut phase = vec![0.0f64; n];
        let mut mama_v = vec![0.0f64; n];
        let mut fama_v = vec![0.0f64; n];
        let g = |i: usize, arr: &Vec<f64>| -> f64 { if i < arr.len() { arr[i] } else { 0.0 } };
        let gs = |i: usize| -> f64 { if i < n { src[i] } else { 0.0 } };
        for i in 1..n {
            sp[i] = (4.0*gs(i) + 3.0*gs(i.saturating_sub(1)) + 2.0*gs(i.saturating_sub(2)) + gs(i.saturating_sub(3))) / 10.0;
            let pv = g(i.saturating_sub(1), &p_arr);
            let c075p = 0.075 * pv + 0.54;
            dt[i] = (0.0962*sp[i] + 0.5769*g(i.saturating_sub(2),&sp) - 0.5769*g(i.saturating_sub(4),&sp) - 0.0962*g(i.saturating_sub(6),&sp)) * c075p;
            q1[i] = (0.0962*dt[i] + 0.5769*g(i.saturating_sub(2),&dt) - 0.5769*g(i.saturating_sub(4),&dt) - 0.0962*g(i.saturating_sub(6),&dt)) * c075p;
            i1[i] = g(i.saturating_sub(3), &dt);
            ji[i] = (0.0962*i1[i] + 0.5769*g(i.saturating_sub(2),&i1) - 0.5769*g(i.saturating_sub(4),&i1) - 0.0962*g(i.saturating_sub(6),&i1)) * c075p;
            jq[i] = (0.0962*q1[i] + 0.5769*g(i.saturating_sub(2),&q1) - 0.5769*g(i.saturating_sub(4),&q1) - 0.0962*g(i.saturating_sub(6),&q1)) * c075p;
            let i2t = i1[i] - jq[i];
            let q2t = q1[i] + ji[i];
            i2[i] = 0.2*i2t + 0.8*g(i.saturating_sub(1),&i2);
            q2[i] = 0.2*q2t + 0.8*g(i.saturating_sub(1),&q2);
            let ret = i2[i]*g(i.saturating_sub(1),&i2) + q2[i]*g(i.saturating_sub(1),&q2);
            let imt = i2[i]*g(i.saturating_sub(1),&q2) - q2[i]*g(i.saturating_sub(1),&i2);
            re[i] = 0.2*ret + 0.8*g(i.saturating_sub(1),&re);
            im[i] = 0.2*imt + 0.8*g(i.saturating_sub(1),&im);
            p1[i] = if im[i] != 0.0 && re[i] != 0.0 { 2.0*pi/(im[i]/re[i]).atan() } else { pv };
            let p1v = p1[i];
            p2 = if p1v > 1.5*pv { 1.5*pv } else if p1v < 0.67*pv { 0.67*pv } else { p1v };
            p3[i] = p2.max(6.0).min(50.0);
            p_arr[i] = 0.2*p3[i] + 0.8*pv;
            phase[i] = if i1[i] != 0.0 { (q1[i]/i1[i]).atan() * 180.0/pi } else { 0.0 };
            let dphase = ((g(i.saturating_sub(1),&phase) - phase[i]).max(1.0));
            let alpha_t = (fastlimit / dphase).max(slowlimit).min(fastlimit);
            mama_v[i] = alpha_t*src[i] + (1.0-alpha_t)*mama_v[i.saturating_sub(1)];
            fama_v[i] = 0.5*alpha_t*mama_v[i] + (1.0-0.5*alpha_t)*fama_v[i.saturating_sub(1)];
        }
        Ok((
            PyArray1::from_vec(py, mama_v).to_owned(),
            PyArray1::from_vec(py, fama_v).to_owned(),
        ))
    })
}

/// PMA — Predictive Moving Average → (predict, trigger)
#[pyfunction]
pub fn pma(source: PyReadonlyArray1<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut predict = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];
        let mut wma1 = vec![0.0f64; n];
        for j in 6..n {
            wma1[j] = (7.0*src[j] + 6.0*src[j-1] + 5.0*src[j-2] + 4.0*src[j-3] + 3.0*src[j-4] + 2.0*src[j-5] + src[j-6]) / 28.0;
            let wma2 = (7.0*wma1[j] + 6.0*wma1[j-1] + 5.0*wma1[j-2] + 4.0*wma1[j-3] + 3.0*wma1[j-4] + 2.0*wma1[j-5] + wma1[j-6]) / 28.0;
            predict[j] = 2.0 * wma1[j] - wma2;
        }
        for j in 6..n {
            if !predict[j].is_nan() && j >= 3 {
                let p3 = if predict[j-3].is_nan() { 0.0 } else { predict[j-3] };
                let p2 = if predict[j-2].is_nan() { 0.0 } else { predict[j-2] };
                let p1 = if predict[j-1].is_nan() { 0.0 } else { predict[j-1] };
                trigger[j] = (4.0*predict[j] + 3.0*p1 + 2.0*p2 + p3) / 10.0;
            }
        }
        Ok((
            PyArray1::from_vec(py, predict).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}

/// Parabolic SAR
#[pyfunction]
pub fn sar(candles: PyReadonlyArray2<f64>, acceleration: f64, maximum: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        if n == 0 { return Ok(PyArray1::from_vec(py, vec![]).to_owned()); }
        let high: Vec<f64> = (0..n).map(|i| c[[i,3]]).collect();
        let low: Vec<f64> = (0..n).map(|i| c[[i,4]]).collect();
        let mut sar_v = vec![0.0f64; n];
        let mut uptrend = high[1] > high[0];
        sar_v[0] = if uptrend { low[0] } else { high[0] };
        let mut ep = if uptrend { high[0] } else { low[0] };
        let mut af = acceleration;
        for i in 1..n {
            let prev_sar = sar_v[i-1];
            let mut sar_temp = if uptrend {
                let mut s = prev_sar + af * (ep - prev_sar);
                if i >= 2 { s = s.min(low[i-1]).min(low[i-2]); } else { s = s.min(low[i-1]); }
                s
            } else {
                let mut s = prev_sar - af * (prev_sar - ep);
                if i >= 2 { s = s.max(high[i-1]).max(high[i-2]); } else { s = s.max(high[i-1]); }
                s
            };
            if uptrend {
                if low[i] < sar_temp {
                    sar_temp = ep; uptrend = false; af = acceleration; ep = low[i];
                } else if high[i] > ep {
                    ep = high[i]; af = (af + acceleration).min(maximum);
                }
            } else {
                if high[i] > sar_temp {
                    sar_temp = ep; uptrend = true; af = acceleration; ep = high[i];
                } else if low[i] < ep {
                    ep = low[i]; af = (af + acceleration).min(maximum);
                }
            }
            sar_v[i] = sar_temp;
        }
        Ok(PyArray1::from_vec(py, sar_v).to_owned())
    })
}

/// SafeZone Stops
#[pyfunction]
pub fn safezonestop(candles: PyReadonlyArray2<f64>, period: usize, mult: f64, max_lookback: usize, direction: bool) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut diff_high = vec![0.0f64; n];
        let mut diff_low = vec![0.0f64; n];
        for i in 1..n {
            let dh = c[[i,3]] - c[[i-1,3]];
            let dl = c[[i-1,4]] - c[[i,4]];
            diff_high[i] = if dh.is_nan() { 0.0 } else { dh };
            diff_low[i] = if dl.is_nan() { 0.0 } else { dl };
        }
        let mut raw_plus = vec![0.0f64; n];
        let mut raw_minus = vec![0.0f64; n];
        for i in 0..n {
            raw_plus[i] = if diff_high[i] > diff_low[i] && diff_high[i] > 0.0 { diff_high[i] } else { 0.0 };
            raw_minus[i] = if diff_low[i] > diff_high[i] && diff_low[i] > 0.0 { diff_low[i] } else { 0.0 };
        }
        // Wilder smoothing: smoothed[i] = alpha * smoothed[i-1] + raw[i], alpha = (period-1)/period
        let alpha = 1.0 - 1.0 / period as f64;
        let mut plus_dm = vec![0.0f64; n];
        let mut minus_dm = vec![0.0f64; n];
        plus_dm[0] = raw_plus[0]; minus_dm[0] = raw_minus[0];
        for i in 1..n {
            plus_dm[i] = alpha * plus_dm[i-1] + raw_plus[i];
            minus_dm[i] = alpha * minus_dm[i-1] + raw_minus[i];
        }
        let mut intermediate = vec![0.0f64; n];
        for i in 1..n {
            intermediate[i] = if direction {
                c[[i-1,4]] - mult * minus_dm[i]  // long: last_low - mult * minus_dm
            } else {
                c[[i-1,3]] + mult * plus_dm[i]   // short: last_high + mult * plus_dm
            };
        }
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let start = i.saturating_sub(max_lookback - 1);
            r[i] = if direction {
                intermediate[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                intermediate[start..=i].iter().cloned().fold(f64::INFINITY, f64::min)
            };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Hurst RS (R/S method for Hurst exponent)
#[pyfunction]
pub fn hurst_rs(x: PyReadonlyArray1<f64>, min_chunksize: usize, max_chunksize: usize, num_chunksize: usize) -> PyResult<f64> {
    Python::with_gil(|_py| {
        let x_arr = x.as_array();
        let xv: Vec<f64> = x_arr.to_vec();
        let n = xv.len();
        let max_cs = max_chunksize + 1;
        let step = if num_chunksize > 1 { (max_cs - min_chunksize) as f64 / (num_chunksize - 1) as f64 } else { 1.0 };
        let chunk_sizes: Vec<usize> = (0..num_chunksize).map(|i| (min_chunksize as f64 + i as f64 * step) as usize).collect();
        let mut rs_values = vec![0.0f64; num_chunksize];
        for (ci, &cs) in chunk_sizes.iter().enumerate() {
            let nchunks = n / cs;
            let mut rs_sum = 0.0f64;
            let mut valid = 0usize;
            for idx in 0..nchunks {
                let chunk = &xv[idx*cs..(idx+1)*cs];
                let mean = chunk.iter().sum::<f64>() / cs as f64;
                let mut cum = 0.0f64;
                let mut z = Vec::with_capacity(cs);
                for &v in chunk { cum += v - mean; z.push(cum); }
                let r_val = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                          - z.iter().cloned().fold(f64::INFINITY, f64::min);
                let var: f64 = chunk.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cs as f64;
                let s = var.sqrt();
                if s > 0.0 { rs_sum += r_val / s; valid += 1; }
            }
            rs_values[ci] = if valid > 0 { rs_sum / valid as f64 } else { 0.0 };
        }
        // Linear regression: log(rs) = H*log(chunk_size) + c
        let log_cs: Vec<f64> = chunk_sizes.iter().map(|&x| (x as f64).ln()).collect();
        let log_rs: Vec<f64> = rs_values.iter().map(|&x| if x > 0.0 { x.ln() } else { 0.0 }).collect();
        let m = num_chunksize as f64;
        let sx: f64 = log_cs.iter().sum();
        let sy: f64 = log_rs.iter().sum();
        let sxx: f64 = log_cs.iter().map(|&x| x*x).sum();
        let sxy: f64 = log_cs.iter().zip(log_rs.iter()).map(|(&x,&y)| x*y).sum();
        let h = (m*sxy - sx*sy) / (m*sxx - sx*sx);
        Ok(h)
    })
}

/// Population rolling std with window
fn rolling_std_pop(source: &[f64], window: usize) -> Vec<f64> {
    let n = source.len();
    let mut result = vec![0.0f64; n];
    if window == 0 || window > n { return result; }
    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for i in 0..window {
        sum += source[i];
        sum_sq += source[i] * source[i];
    }
    let w = window as f64;
    let mean = sum / w;
    let var = (sum_sq / w - mean * mean).max(0.0);
    result[window - 1] = var.sqrt();
    for i in window..n {
        sum += source[i] - source[i - window];
        sum_sq += source[i] * source[i] - source[i - window] * source[i - window];
        let mean = sum / w;
        let var = (sum_sq / w - mean * mean).max(0.0);
        result[i] = var.sqrt();
    }
    result
}

/// Damiani Volatmeter — returns (vol, anti)
#[pyfunction]
pub fn damiani_volatmeter(
    candles: PyReadonlyArray2<f64>,
    source: PyReadonlyArray1<f64>,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let src: Vec<f64> = source.as_array().to_vec();
        let n = candles_array.shape()[0];
        let lag_s = 0.5f64;

        let high: Vec<f64> = candles_array.slice(s![.., 3]).to_vec();
        let low: Vec<f64> = candles_array.slice(s![.., 4]).to_vec();
        let close: Vec<f64> = candles_array.slice(s![.., 2]).to_vec();

        let atrvis = ih_atr_wilder(&high, &low, &close, vis_atr);
        let atrsed = ih_atr_wilder(&high, &low, &close, sed_atr);

        // u[i] = atrvis[i] / atrsed[i] for i >= sed_std
        let mut u = vec![0.0f64; n];
        for i in sed_std..n {
            if atrsed[i] != 0.0 && !atrsed[i].is_nan() && !atrvis[i].is_nan() {
                u[i] = atrvis[i] / atrsed[i];
            }
        }

        // lfilter with b=[1.0], a=[1.0, -0.5, 0.0, 0.5]:
        //   y[i] = u[i] + 0.5*y[i-1] - 0.5*y[i-3]
        let mut vol = vec![0.0f64; n];
        for i in 0..n {
            let mut y = u[i];
            if i >= 1 { y += lag_s * vol[i - 1]; }
            if i >= 3 { y -= lag_s * vol[i - 3]; }
            vol[i] = y;
        }

        // Rolling std and threshold calculation
        let mut t = vec![0.0f64; n];
        if n >= sed_std {
            let std_vis = rolling_std_pop(&src, vis_std);
            let std_sed = rolling_std_pop(&src, sed_std);
            for idx in sed_std..n {
                let v = std_vis[idx - 1];
                let s = std_sed[idx - 1];
                t[idx] = threshold - v / s;
            }
        }

        Ok((
            PyArray1::from_vec(py, vol).to_owned(),
            PyArray1::from_vec(py, t).to_owned(),
        ))
    })
}

/// Find index of a row in a 2D array that matches the given 1D array exactly.
/// Returns -1 if no match. Mirrors np.all(orders[i] == order_array) used by FuturesExchange.
#[pyfunction]
pub fn find_order_index(
    orders: PyReadonlyArray2<f64>,
    order_array: PyReadonlyArray1<f64>,
) -> PyResult<i64> {
    let orders = orders.as_array();
    let target = order_array.as_array();
    let n_rows = orders.shape()[0];
    let n_cols = orders.shape()[1];
    if target.len() != n_cols { return Ok(-1); }
    'outer: for i in 0..n_rows {
        for j in 0..n_cols {
            if orders[[i, j]] != target[j] { continue 'outer; }
        }
        return Ok(i as i64);
    }
    Ok(-1)
}
