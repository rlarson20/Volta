//! Interactive terminal progress bar for training loops.
//!
//! This module deliberately writes to stderr (carriage-return redraws of a
//! single line), which is not a log record and would be meaningless through the
//! `log` facade. Callers opt in by constructing a [`ProgressBar`].
#![allow(
    clippy::print_stderr,
    reason = "a progress bar redraws one stderr line by design; it is not a log record"
)]

use std::io::{self, Write};

/// Simple progress bar for training loops
pub struct ProgressBar {
    /// Number of steps that make a full bar.
    total: usize,
    /// Steps completed so far.
    current: usize,
    /// Label printed before the bar.
    prefix: String,
    /// Bar width in characters.
    width: usize,
}

impl ProgressBar {
    /// Create a new progress bar
    #[must_use]
    pub fn new(total: usize, prefix: &str) -> Self {
        Self {
            total,
            current: 0,
            prefix: prefix.to_string(),
            width: 40,
        }
    }

    /// Update progress and display
    pub fn update(&mut self, current: usize) {
        self.current = current;
        self.render();
    }

    /// Increment by 1 and display
    pub fn inc(&mut self) {
        self.current += 1;
        self.render();
    }

    /// Finish the progress bar
    pub fn finish(&self) {
        eprint!("\r");
        let _ = io::stderr().flush();
    }

    /// Redraws the bar on the current stderr line.
    fn render(&self) {
        // An empty bar (total 0) shows 0%.
        let percent = self
            .current
            .saturating_mul(100)
            .checked_div(self.total)
            .unwrap_or(0);

        let filled = (self.current * self.width)
            .checked_div(self.total)
            .map_or(0, |v| v.min(self.width));

        let bar: String = "█".repeat(filled) + &"░".repeat(self.width - filled);

        eprint!(
            "\r{} [{}] {:3}% ({}/{})",
            self.prefix, bar, percent, self.current, self.total
        );
        let _ = io::stderr().flush();
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        self.finish();
    }
}
