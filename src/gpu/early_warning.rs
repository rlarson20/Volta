//! Early warning system for predictive GPU failure detection
//!
//! Analyzes resource usage trends to predict and warn about potential
//! GPU failures before they occur. Uses linear regression to detect
//! problematic patterns in memory usage and pending operation counts.

use std::collections::VecDeque;
use std::time::Instant;

/// Health status with predictive warnings based on trend analysis
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Resources are healthy with stable or decreasing trends
    Healthy,
    /// Resources show concerning trends that may lead to problems
    Warning(String),
    /// Resources show critical trends that will likely cause failures
    Critical(String),
}

/// Early warning system using trend analysis
///
/// Monitors resource usage over time and uses linear regression to detect
/// problematic trends before they cause failures.
///
/// # Example
///
/// ```ignore
/// use volta::gpu::early_warning::EarlyWarningSystem;
///
/// let mut ews = EarlyWarningSystem::new();
///
/// // Periodically check health during operations
/// for _ in 0..100 {
///     // ... GPU operations ...
///
///     match ews.check_health() {
///         HealthStatus::Critical(msg) => {
///             eprintln!("CRITICAL: {}", msg);
///             break;
///         }
///         HealthStatus::Warning(msg) => {
///             eprintln!("Warning: {}", msg);
///         }
///         _ => {}
///     }
/// }
/// ```
pub struct EarlyWarningSystem {
    memory_trend: VecDeque<usize>,
    pending_trend: VecDeque<u32>,
    last_check: Instant,
    max_history: usize,
}

impl EarlyWarningSystem {
    /// Create a new early warning system
    ///
    /// # Arguments
    /// * `max_history` - Maximum number of samples to keep (default: 10)
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(10)
    }

    /// Create a new early warning system with custom history size
    ///
    /// # Arguments
    /// * `max_history` - Maximum number of samples to keep
    #[must_use]
    pub fn with_capacity(max_history: usize) -> Self {
        Self {
            memory_trend: VecDeque::with_capacity(max_history),
            pending_trend: VecDeque::with_capacity(max_history),
            last_check: Instant::now(),
            max_history,
        }
    }

    /// Check health and update trends
    ///
    /// Samples current resource usage and analyzes trends to predict
    /// potential problems. Returns health status with warnings based
    /// on trend analysis.
    pub fn check_health(&mut self) -> HealthStatus {
        let current_memory = crate::gpu::monitor::get_process_memory_mb();
        let current_pending = crate::gpu::gpu_pending_count();

        // Update trends
        self.memory_trend.push_back(current_memory);
        self.pending_trend.push_back(current_pending);

        // Keep only recent history
        while self.memory_trend.len() > self.max_history {
            self.memory_trend.pop_front();
        }
        while self.pending_trend.len() > self.max_history {
            self.pending_trend.pop_front();
        }

        self.last_check = Instant::now();

        // Analyze trends
        let memory_status = self.analyze_memory_trend();
        let pending_status = self.analyze_pending_trend();

        // Return worst status
        match (memory_status, pending_status) {
            (HealthStatus::Critical(m), _) | (_, HealthStatus::Critical(m)) => {
                HealthStatus::Critical(m)
            }
            (HealthStatus::Warning(m), _) | (_, HealthStatus::Warning(m)) => {
                HealthStatus::Warning(m)
            }
            _ => HealthStatus::Healthy,
        }
    }

    fn analyze_memory_trend(&self) -> HealthStatus {
        if self.memory_trend.len() < 3 {
            return HealthStatus::Healthy;
        }

        let values: Vec<f64> = self.memory_trend.iter().map(|&m| m as f64).collect();

        match calculate_trend(&values) {
            TrendDirection::Increasing(rate) if rate > 0.15 => HealthStatus::Critical(format!(
                "Memory increasing rapidly: {:.1}%/sample",
                rate * 100.0
            )),
            TrendDirection::Increasing(rate) if rate > 0.08 => {
                HealthStatus::Warning(format!("Memory increasing: {:.1}%/sample", rate * 100.0))
            }
            TrendDirection::Increasing(_)
            | TrendDirection::Decreasing(_)
            | TrendDirection::Stable => HealthStatus::Healthy,
        }
    }

    fn analyze_pending_trend(&self) -> HealthStatus {
        if self.pending_trend.len() < 3 {
            return HealthStatus::Healthy;
        }

        let values: Vec<f64> = self.pending_trend.iter().map(|&p| f64::from(p)).collect();

        match calculate_trend(&values) {
            TrendDirection::Increasing(rate) if rate > 0.20 => HealthStatus::Critical(format!(
                "Pending ops increasing rapidly: {:.1}%/sample",
                rate * 100.0
            )),
            TrendDirection::Increasing(rate) if rate > 0.10 => HealthStatus::Warning(format!(
                "Pending ops increasing: {:.1}%/sample",
                rate * 100.0
            )),
            TrendDirection::Increasing(_)
            | TrendDirection::Decreasing(_)
            | TrendDirection::Stable => HealthStatus::Healthy,
        }
    }

    /// Reset all trend data
    ///
    /// Clears historical samples and starts fresh. Useful when starting
    /// a new phase of work with different characteristics.
    pub fn reset(&mut self) {
        self.memory_trend.clear();
        self.pending_trend.clear();
        self.last_check = Instant::now();
    }

    /// Get current trend data (diagnostic)
    ///
    /// Returns the current memory and pending operation trends for
    /// debugging and analysis.
    #[must_use]
    pub fn trends(&self) -> TrendData {
        TrendData {
            memory_samples: self.memory_trend.iter().copied().collect(),
            pending_samples: self.pending_trend.iter().copied().collect(),
        }
    }
}

impl Default for EarlyWarningSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Current trend data snapshot
#[derive(Debug, Clone)]
pub struct TrendData {
    /// Recent memory usage samples (MB)
    pub memory_samples: Vec<usize>,
    /// Recent pending operation count samples
    pub pending_samples: Vec<u32>,
}

/// Direction and magnitude of a trend
#[derive(Debug, Clone, PartialEq)]
enum TrendDirection {
    /// Increasing trend with relative rate per sample
    Increasing(f64),
    /// Decreasing trend with relative rate per sample
    Decreasing(f64),
    /// Stable trend (no significant change)
    Stable,
}

/// Calculate trend using linear regression
///
/// Uses simple least-squares linear regression to find the slope of the
/// data. Returns the trend direction and relative rate of change.
///
/// # Arguments
/// * `values` - Time series data to analyze
fn calculate_trend(values: &[f64]) -> TrendDirection {
    const THRESHOLD: f64 = 0.01; // 1% per sample

    if values.len() < 2 {
        return TrendDirection::Stable;
    }

    // Simple linear regression: y = mx + b
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    // Calculate slope (rate of change)
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

    // Calculate average value for normalization
    let avg_y = sum_y / n;

    // Normalize slope by average value to get relative rate
    let relative_slope = if avg_y.abs() > 1e-6 {
        slope / avg_y
    } else {
        0.0
    };

    if relative_slope > THRESHOLD {
        TrendDirection::Increasing(relative_slope)
    } else if relative_slope < -THRESHOLD {
        TrendDirection::Decreasing(-relative_slope)
    } else {
        TrendDirection::Stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ews_creation() {
        let ews = EarlyWarningSystem::new();
        assert_eq!(ews.memory_trend.len(), 0);
        assert_eq!(ews.pending_trend.len(), 0);
    }

    #[test]
    fn test_ews_with_capacity() {
        let ews = EarlyWarningSystem::with_capacity(5);
        assert_eq!(ews.max_history, 5);
    }

    #[test]
    fn test_trend_calculation_stable() {
        let values = vec![10.0, 10.0, 10.0, 10.0];
        let trend = calculate_trend(&values);
        assert_eq!(trend, TrendDirection::Stable);
    }

    #[test]
    fn test_trend_calculation_increasing() {
        let values = vec![10.0, 15.0, 20.0, 25.0];
        let trend = calculate_trend(&values);
        match trend {
            TrendDirection::Increasing(rate) => {
                assert!(rate > 0.0, "Expected positive rate");
            }
            TrendDirection::Decreasing(_) | TrendDirection::Stable => {
                panic!("Expected increasing trend")
            }
        }
    }

    #[test]
    fn test_trend_calculation_decreasing() {
        let values = vec![25.0, 20.0, 15.0, 10.0];
        let trend = calculate_trend(&values);
        match trend {
            TrendDirection::Decreasing(rate) => {
                assert!(rate > 0.0, "Expected positive rate");
            }
            TrendDirection::Increasing(_) | TrendDirection::Stable => {
                panic!("Expected decreasing trend")
            }
        }
    }

    #[test]
    fn test_trend_insufficient_data() {
        let values = vec![10.0];
        let trend = calculate_trend(&values);
        assert_eq!(trend, TrendDirection::Stable);
    }

    #[test]
    fn test_reset() {
        let mut ews = EarlyWarningSystem::new();
        ews.memory_trend.push_back(100);
        ews.pending_trend.push_back(10);

        ews.reset();

        assert_eq!(ews.memory_trend.len(), 0);
        assert_eq!(ews.pending_trend.len(), 0);
    }

    #[test]
    fn test_trends_snapshot() {
        let mut ews = EarlyWarningSystem::new();
        ews.memory_trend.push_back(100);
        ews.memory_trend.push_back(200);

        let trends = ews.trends();
        assert_eq!(trends.memory_samples.len(), 2);
        assert_eq!(*trends.memory_samples.first().unwrap_or(&0), 100);
        assert_eq!(*trends.memory_samples.get(1).unwrap_or(&0), 200);
    }

    #[test]
    fn test_health_status_equality() {
        let healthy = HealthStatus::Healthy;
        let warning = HealthStatus::Warning("test".into());
        let critical = HealthStatus::Critical("test".into());

        assert_eq!(healthy, HealthStatus::Healthy);
        assert_ne!(warning, healthy);
        assert_ne!(critical, healthy);
    }
}
