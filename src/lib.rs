use pyo3::prelude::*;

mod indicators;

use indicators::*;

/// A Python module implemented in Rust.
#[pymodule]
fn jesse_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Indicators
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(kama, m)?)?;
    m.add_function(wrap_pyfunction!(ichimoku_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(srsi, m)?)?;
    m.add_function(wrap_pyfunction!(adx, m)?)?;
    m.add_function(wrap_pyfunction!(tema, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands_width, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(adosc, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(cvi, m)?)?;
    m.add_function(wrap_pyfunction!(dti, m)?)?;
    m.add_function(wrap_pyfunction!(dx, m)?)?; // New indicator
    m.add_function(wrap_pyfunction!(fosc, m)?)?; // New indicator
    m.add_function(wrap_pyfunction!(frama, m)?)?; // New indicator
    
    // Utility functions (now in indicators.rs)
    m.add_function(wrap_pyfunction!(shift, m)?)?;
    m.add_function(wrap_pyfunction!(moving_std, m)?)?;
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(smma, m)?)?;
    m.add_function(wrap_pyfunction!(alligator, m)?)?;
    m.add_function(wrap_pyfunction!(di, m)?)?;
    m.add_function(wrap_pyfunction!(chop, m)?)?;
    m.add_function(wrap_pyfunction!(atr, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::chande, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::donchian, m)?)?;
    
    // New optimized indicators
    m.add_function(wrap_pyfunction!(willr, m)?)?;
    m.add_function(wrap_pyfunction!(wma, m)?)?;
    m.add_function(wrap_pyfunction!(vwma, m)?)?;
    
    // Performance optimized indicators
    m.add_function(wrap_pyfunction!(stoch, m)?)?;
    m.add_function(wrap_pyfunction!(stochf, m)?)?;
    m.add_function(wrap_pyfunction!(dm, m)?)?;
    m.add_function(wrap_pyfunction!(dema, m)?)?;
    
    // Newly added indicators
    m.add_function(wrap_pyfunction!(zlema, m)?)?;
    m.add_function(wrap_pyfunction!(wt, m)?)?;
    
    // Latest indicators
    m.add_function(wrap_pyfunction!(vwap, m)?)?;
    m.add_function(wrap_pyfunction!(vi, m)?)?;
    m.add_function(wrap_pyfunction!(t3, m)?)?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(sum_floats, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_floats, m)?)?;

    // Phase 3 — Power-weighted MAs
    m.add_function(wrap_pyfunction!(cwma, m)?)?;
    m.add_function(wrap_pyfunction!(sqwma, m)?)?;
    m.add_function(wrap_pyfunction!(srwma, m)?)?;
    m.add_function(wrap_pyfunction!(vpwma, m)?)?;
    m.add_function(wrap_pyfunction!(epma, m)?)?;
    m.add_function(wrap_pyfunction!(qstick, m)?)?;

    // Phase 3 — EMA variants
    m.add_function(wrap_pyfunction!(rma, m)?)?;
    m.add_function(wrap_pyfunction!(wilders, m)?)?;
    m.add_function(wrap_pyfunction!(mcginley_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(mwdx, m)?)?;
    m.add_function(wrap_pyfunction!(hwma, m)?)?;
    m.add_function(wrap_pyfunction!(trix, m)?)?;
    m.add_function(wrap_pyfunction!(dpo, m)?)?;

    // Phase 3 — Oscillators
    m.add_function(wrap_pyfunction!(cci, m)?)?;
    m.add_function(wrap_pyfunction!(cmo, m)?)?;
    m.add_function(wrap_pyfunction!(cfo, m)?)?;
    m.add_function(wrap_pyfunction!(cg, m)?)?;
    m.add_function(wrap_pyfunction!(aroonosc, m)?)?;
    m.add_function(wrap_pyfunction!(adxr, m)?)?;
    m.add_function(wrap_pyfunction!(efi, m)?)?;
    m.add_function(wrap_pyfunction!(emv, m)?)?;
    m.add_function(wrap_pyfunction!(wad, m)?)?;
    m.add_function(wrap_pyfunction!(nvi, m)?)?;
    m.add_function(wrap_pyfunction!(pvi, m)?)?;
    m.add_function(wrap_pyfunction!(mass, m)?)?;
    m.add_function(wrap_pyfunction!(pfe, m)?)?;

    // Phase 3 — Complex MAs
    m.add_function(wrap_pyfunction!(hma, m)?)?;
    m.add_function(wrap_pyfunction!(linearreg, m)?)?;
    m.add_function(wrap_pyfunction!(lrsi, m)?)?;
    m.add_function(wrap_pyfunction!(maaq, m)?)?;
    m.add_function(wrap_pyfunction!(vidya, m)?)?;
    m.add_function(wrap_pyfunction!(vlma_inner, m)?)?;
    m.add_function(wrap_pyfunction!(nma, m)?)?;
    m.add_function(wrap_pyfunction!(jma, m)?)?;
    m.add_function(wrap_pyfunction!(rsx, m)?)?;

    // Phase 3 — Ehlers filters
    m.add_function(wrap_pyfunction!(supersmoother, m)?)?;
    m.add_function(wrap_pyfunction!(supersmoother_3_pole, m)?)?;
    m.add_function(wrap_pyfunction!(high_pass, m)?)?;
    m.add_function(wrap_pyfunction!(high_pass_2_pole, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass, m)?)?;
    m.add_function(wrap_pyfunction!(gauss, m)?)?;
    m.add_function(wrap_pyfunction!(reflex, m)?)?;
    m.add_function(wrap_pyfunction!(trendflex, m)?)?;
    m.add_function(wrap_pyfunction!(itrend, m)?)?;
    m.add_function(wrap_pyfunction!(voss, m)?)?;
    m.add_function(wrap_pyfunction!(edcf, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_cycle, m)?)?;

    // Phase 3 — Candle-based
    m.add_function(wrap_pyfunction!(heikin_ashi_candles, m)?)?;
    m.add_function(wrap_pyfunction!(keltner_inner, m)?)?;
    m.add_function(wrap_pyfunction!(supertrend, m)?)?;
    m.add_function(wrap_pyfunction!(emd, m)?)?;
    m.add_function(wrap_pyfunction!(fisher, m)?)?;
    m.add_function(wrap_pyfunction!(mama, m)?)?;
    m.add_function(wrap_pyfunction!(pma, m)?)?;
    m.add_function(wrap_pyfunction!(sar, m)?)?;

    // Phase 3 — Statistical
    m.add_function(wrap_pyfunction!(safezonestop, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_rs, m)?)?;
    m.add_function(wrap_pyfunction!(damiani_volatmeter, m)?)?;

    // Non-indicator utilities
    m.add_function(wrap_pyfunction!(find_order_index, m)?)?;

    Ok(())
}
