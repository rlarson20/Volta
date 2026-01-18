//! Emergency resource monitoring for GPU operations
//!
//! Provides system-level resource monitoring to detect critical memory
//! pressure before system freezes occur. macOS-specific implementation
//! using the `ps` command for memory tracking.

use std::process::Command;

/// System resource health status
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceStatus {
    /// Resources are healthy
    Healthy,
    /// Resources are elevated but not critical (>70% memory or >50 pending ops)
    Warning(String),
    /// Resources are critical and operation should abort (>90% memory or >100 pending ops)
    Critical(String),
}

/// Check system resource health
///
/// Monitors two key indicators:
/// 1. **Process memory usage** - Via macOS `ps` command (fails safe on other platforms)
/// 2. **GPU pending operation count** - Proxy for GPU memory pressure
///
/// # Thresholds
///
/// **Critical** (operation should abort):
/// - Process memory >90% of system RAM
/// - GPU pending ops >100
///
/// **Warning** (elevated but operational):
/// - Process memory >70% of system RAM
/// - GPU pending ops >50
///
/// # Platform Support
///
/// - **macOS**: Full memory monitoring via `ps` command
/// - **Other platforms**: Only monitors GPU pending count (memory always returns healthy)
///
/// # Example
///
/// ```ignore
/// use volta::gpu::monitor::{check_system_resources, ResourceStatus};
///
/// match check_system_resources() {
///     ResourceStatus::Critical(msg) => {
///         panic!("Test aborted - critical resources: {}", msg);
///     }
///     ResourceStatus::Warning(msg) => {
///         eprintln!("Warning: {}", msg);
///     }
///     ResourceStatus::Healthy => {}
/// }
/// ```
#[must_use]
pub fn check_system_resources() -> ResourceStatus {
    let memory_ratio = get_process_memory_ratio();
    let pending_count = crate::gpu::gpu_pending_count();

    // Critical thresholds - operation should abort
    if memory_ratio > 0.9 {
        return ResourceStatus::Critical(format!(
            "Process memory >90% ({:.1}% of system RAM)",
            memory_ratio * 100.0
        ));
    }

    if pending_count > 100 {
        return ResourceStatus::Critical(format!("GPU pending count very high: {}", pending_count));
    }

    // Warning thresholds - elevated but operational
    if memory_ratio > 0.7 {
        return ResourceStatus::Warning(format!(
            "Process memory >70% ({:.1}% of system RAM)",
            memory_ratio * 100.0
        ));
    }

    if pending_count > 50 {
        return ResourceStatus::Warning(format!("GPU pending count elevated: {}", pending_count));
    }

    ResourceStatus::Healthy
}

/// Get process memory usage as ratio of system memory (macOS only)
///
/// Uses the `ps` command to query process resident set size (RSS).
/// Returns 0.0 (healthy) on any errors to avoid false positives.
///
/// # Implementation Details
///
/// - Uses `ps -o rss= -p <pid>` to get RSS in kilobytes
/// - Assumes 24GB system memory (M2 Mac target platform)
/// - ~1ms execution time, negligible overhead
///
/// # Fail-Safe Behavior
///
/// Returns 0.0 (healthy) if:
/// - `ps` command fails to execute
/// - Output parsing fails
/// - Any other errors occur
#[cfg(target_os = "macos")]
fn get_process_memory_ratio() -> f64 {
    let pid = std::process::id();

    // Execute: ps -o rss= -p <pid>
    let output = match Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        Ok(out) => out,
        Err(_) => return 0.0, // Fail-safe: assume healthy
    };

    // Parse RSS value (in KB)
    let rss_str = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };

    let rss_kb: usize = match rss_str.trim().parse() {
        Ok(kb) => kb,
        Err(_) => return 0.0,
    };

    // Convert to ratio of 24GB (M2 Mac system memory)
    // TODO: Could use sysinfo crate for cross-platform memory detection
    const SYSTEM_MEMORY_KB: usize = 24 * 1024 * 1024;
    rss_kb as f64 / SYSTEM_MEMORY_KB as f64
}

#[cfg(not(target_os = "macos"))]
fn get_process_memory_ratio() -> f64 {
    0.0 // Not implemented for non-macOS, return healthy
}

/// Get current process memory in MB (diagnostic)
///
/// Useful for logging and debugging memory usage patterns.
/// Returns 0 if memory monitoring is unavailable.
///
/// # Example
///
/// ```ignore
/// use volta::gpu::monitor::get_process_memory_mb;
///
/// println!("Current memory usage: {}MB", get_process_memory_mb());
/// ```
#[must_use]
pub fn get_process_memory_mb() -> usize {
    let ratio = get_process_memory_ratio();
    const SYSTEM_MEMORY_MB: usize = 24 * 1024;
    (ratio * SYSTEM_MEMORY_MB as f64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_check_does_not_panic() {
        // Should never panic, even if monitoring fails
        let _status = check_system_resources();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_memory_ratio_is_reasonable() {
        let ratio = get_process_memory_ratio();
        // Memory ratio should be between 0 and 1
        assert!(ratio >= 0.0);
        assert!(ratio <= 1.0);
    }

    #[test]
    fn test_memory_mb_non_negative() {
        let mb = get_process_memory_mb();
        // Should be non-negative
        assert!(mb < 100_000); // Sanity check: < 100GB
    }

    #[test]
    fn test_resource_status_display() {
        // Ensure ResourceStatus variants can be created and compared
        let healthy = ResourceStatus::Healthy;
        let warning = ResourceStatus::Warning("test".into());
        let critical = ResourceStatus::Critical("test".into());

        assert_eq!(healthy, ResourceStatus::Healthy);
        assert_ne!(warning, healthy);
        assert_ne!(critical, healthy);
    }
}
