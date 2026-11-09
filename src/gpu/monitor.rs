//! Emergency resource monitoring for GPU operations
//!
//! Provides system-level resource monitoring to detect critical memory
//! pressure before system freezes occur. Process memory is read from `ps` and
//! `sysctl` on macOS and from `/proc` on Linux.

#[cfg(target_os = "macos")]
use std::process::Command;

/// System resource health status
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// 1. **Process memory usage** - macOS and Linux (fails safe elsewhere)
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
/// - **macOS** and **Linux**: Full memory monitoring
/// - **Other platforms**: Only monitors GPU pending count (memory always returns healthy)
///
/// # Example
///
/// ```no_run
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
        return ResourceStatus::Critical(format!("GPU pending count very high: {pending_count}"));
    }

    // Warning thresholds - elevated but operational
    if memory_ratio > 0.7 {
        return ResourceStatus::Warning(format!(
            "Process memory >70% ({:.1}% of system RAM)",
            memory_ratio * 100.0
        ));
    }

    if pending_count > 50 {
        return ResourceStatus::Warning(format!("GPU pending count elevated: {pending_count}"));
    }

    ResourceStatus::Healthy
}

/// Process resident set size in KiB, or `None` if it cannot be read.
#[cfg(target_os = "macos")]
fn process_rss_kb() -> Option<u64> {
    // Execute: ps -o rss= -p <pid>
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

/// Total physical memory in KiB, or `None` if it cannot be read.
#[cfg(target_os = "macos")]
fn total_memory_kb() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let bytes: u64 = String::from_utf8(output.stdout).ok()?.trim().parse().ok()?;
    Some(bytes / 1024)
}

/// Value in KiB of a `Key:   <n> kB` line in a `/proc` file.
#[cfg(target_os = "linux")]
fn proc_kb(path: &str, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let rest = text.lines().find_map(|line| line.strip_prefix(key))?;
    rest.strip_prefix(':')?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Process resident set size in KiB, or `None` if it cannot be read.
#[cfg(target_os = "linux")]
fn process_rss_kb() -> Option<u64> {
    proc_kb("/proc/self/status", "VmRSS")
}

/// Total physical memory in KiB, or `None` if it cannot be read.
#[cfg(target_os = "linux")]
fn total_memory_kb() -> Option<u64> {
    proc_kb("/proc/meminfo", "MemTotal")
}

/// Not measured on this platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_rss_kb() -> Option<u64> {
    None
}

/// Not measured on this platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn total_memory_kb() -> Option<u64> {
    None
}

/// Log once that memory pressure cannot be measured, so a blind monitor is not silent.
fn warn_unmeasured() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!("process memory is not measurable here; memory pressure checks are disabled");
    });
}

/// Get process memory usage as a ratio of total system memory
///
/// Reads the process resident set size (RSS) and the machine's physical
/// memory, so the thresholds in [`check_system_resources`] scale to the host.
/// Returns 0.0 (healthy) when either cannot be read, to avoid false positives.
///
/// # Implementation Details
///
/// - **macOS**: `ps -o rss= -p <pid>` for RSS, `sysctl -n hw.memsize` for total RAM
/// - **Linux**: `VmRSS` from `/proc/self/status`, `MemTotal` from `/proc/meminfo`
/// - **Other platforms**: not measured
///
/// Total RAM is read once and cached.
///
/// # Fail-Safe Behavior
///
/// Returns 0.0 (healthy), and logs a warning once, if a command or file
/// cannot be read or parsed, or if total memory is reported as zero.
fn get_process_memory_ratio() -> f64 {
    static TOTAL_KB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let total = *TOTAL_KB.get_or_init(total_memory_kb);
    match (process_rss_kb(), total) {
        (Some(rss), Some(total)) if total > 0 => rss as f64 / total as f64,
        _ => {
            warn_unmeasured();
            0.0
        }
    }
}

/// Get current process memory in MB (diagnostic)
///
/// Useful for logging and debugging memory usage patterns.
/// Returns 0 if memory monitoring is unavailable.
///
/// # Example
///
/// ```no_run
/// use volta::gpu::monitor::get_process_memory_mb;
///
/// println!("Current memory usage: {}MB", get_process_memory_mb());
/// ```
#[must_use]
pub fn get_process_memory_mb() -> usize {
    process_rss_kb().map_or_else(
        || {
            warn_unmeasured();
            0
        },
        |kb| usize::try_from(kb / 1024).unwrap_or(usize::MAX),
    )
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
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn test_memory_ratio_is_reasonable() {
        let ratio = get_process_memory_ratio();
        // A running test process has a nonzero RSS, so zero means the read failed.
        assert!(ratio > 0.0, "RSS or total memory was not read");
        assert!(ratio <= 1.0);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn test_memory_mb_matches_ratio() {
        let total_mb = total_memory_kb().expect("total memory") / 1024;
        let from_ratio = get_process_memory_ratio() * total_mb as f64;
        let mb = get_process_memory_mb() as f64;
        // Two reads a moment apart; allow for allocation between them.
        assert!((mb - from_ratio).abs() < 64.0, "{mb} vs {from_ratio}");
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
