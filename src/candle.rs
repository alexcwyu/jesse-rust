//! Candle transforms / decompositions (Heikin Ashi, EMD, QStick).

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::types::{PyArrTuple3, PyArrTuple4};

use crate::helpers::{ih_sma};

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

/// Heikin Ashi Candles → (open, close, high, low)
#[pyfunction]
pub fn heikin_ashi_candles(candles: PyReadonlyArray2<f64>) -> PyArrTuple4 {
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

/// EMD — Empirical Mode Decomposition → (upperband, middleband, lowerband)
#[pyfunction]
pub fn emd(candles: PyReadonlyArray2<f64>, period: usize, delta: f64, fraction: f64) -> PyArrTuple3 {
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
