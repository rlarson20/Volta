use std::io::{self, Write};

/// Simple progress bar for training loops
pub struct ProgressBar {
    total: usize,
    current: usize,
    prefix: String,
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

    fn render(&self) {
        let percent = if self.total > 0 {
            (self.current as f32 / self.total as f32 * 100.0) as usize
        } else {
            0
        };

        let filled = if self.total > 0 {
            (self.current * self.width / self.total).min(self.width)
        } else {
            0
        };

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
