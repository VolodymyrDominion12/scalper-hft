use pyo3::prelude::*;

/// A basic Risk Gate implementation in Rust for HFT speeds.
/// In Python, this was checking daily loss limits, cooldowns, etc.
/// Moving it to Rust allows sub-microsecond pre-trade checks.
#[pyclass]
pub struct RiskGate {
    daily_loss_limit_pct: f64,
    day_start_equity: f64,
    current_equity: f64,
}

#[pymethods]
impl RiskGate {
    #[new]
    pub fn new(daily_loss_limit_pct: f64, initial_equity: f64) -> Self {
        RiskGate {
            daily_loss_limit_pct,
            day_start_equity: initial_equity,
            current_equity: initial_equity,
        }
    }

    /// Check if a new entry is allowed based on the daily loss limit.
    pub fn is_entry_allowed(&self) -> bool {
        let max_loss_equity = self.day_start_equity * (1.0 - self.daily_loss_limit_pct);
        self.current_equity > max_loss_equity
    }

    /// Mark-to-market update
    pub fn update_equity(&mut self, new_equity: f64) {
        self.current_equity = new_equity;
    }
}

/// The main Python module exposing Rust components.
#[pymodule]
fn scalper_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RiskGate>()?;
    Ok(())
}
