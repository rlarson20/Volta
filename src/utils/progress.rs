//! Interactive terminal progress bar for training loops.
//!
//! This module deliberately writes to stderr (carriage-return redraws of a
//! single line), which is not a log record and would be meaningless through the
//! `log` facade. Callers opt in by constructing a [`ProgressBar`].
#![allow(
    clippy::print_stderr,
    reason = "a progress bar redraws one stderr line by design; it is not a log record"
)]

use std::cell::Cell;
use std::io::{self, IsTerminal, Write};

/// Simple progress bar for training loops.
///
/// On a terminal the bar redraws one stderr line, and only when the shown percentage or
/// the filled width changes. When stderr is not a terminal (redirected to a file or a
/// pipe) nothing is drawn until [`ProgressBar::finish`], which writes the final line once.
pub struct ProgressBar {
    /// Number of steps that make a full bar.
    total: usize,
    /// Steps completed so far.
    current: usize,
    /// Label printed before the bar.
    prefix: String,
    /// Bar width in characters.
    width: usize,
    /// The `(percent, filled)` last drawn, so an update that changes neither is not redrawn.
    last_drawn: Cell<Option<(usize, usize)>>,
    /// Whether [`ProgressBar::finish`] has run, so `Drop` does not finish a second time.
    finished: Cell<bool>,
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
            last_drawn: Cell::new(None),
            finished: Cell::new(false),
        }
    }

    /// Update progress and display
    pub fn update(&mut self, current: usize) {
        self.current = current;
        self.render();
    }

    /// Increment by 1 and display
    pub fn inc(&mut self) {
        self.current = self.current.saturating_add(1);
        self.render();
    }

    /// Finish the progress bar: draw its final state and end the line with `\n`, so the
    /// bar stays visible and the next output starts on a new line. Runs once; a second
    /// call, or the `Drop` after an explicit call, does nothing.
    pub fn finish(&self) {
        if self.finished.replace(true) {
            return;
        }
        eprintln!("\r{}", self.line());
        let _ = io::stderr().flush();
    }

    /// `(percent, filled)`: the shown percentage and the number of filled cells. Both are
    /// clamped to a full bar, so a `current` past `total` shows 100%, not 250%. An empty
    /// bar (`total` 0) shows 0%.
    fn progress(&self) -> (usize, usize) {
        let shown = self.current.min(self.total);
        // `shown * scale / total` in u128, so the product cannot overflow; the quotient is at
        // most `scale` because `shown <= total`.
        let part = |scale: usize| -> usize {
            if self.total == 0 {
                return 0;
            }
            let q = (shown as u128) * (scale as u128) / (self.total as u128);
            usize::try_from(q).expect("shown <= total, so the quotient is at most scale")
        };
        (part(100), part(self.width))
    }

    /// The bar's text without the leading `\r`: prefix, bar, percentage and the raw
    /// `current/total` counter.
    fn line(&self) -> String {
        let (percent, filled) = self.progress();
        let bar: String = "█".repeat(filled) + &"░".repeat(self.width - filled);
        format!(
            "{} [{}] {:3}% ({}/{})",
            self.prefix, bar, percent, self.current, self.total
        )
    }

    /// Whether an update should redraw: on a terminal, when the shown progress changed
    /// since the last draw (recording it as drawn).
    fn should_redraw(&self, is_terminal: bool) -> bool {
        let now = self.progress();
        if !is_terminal || self.last_drawn.get() == Some(now) {
            return false;
        }
        self.last_drawn.set(Some(now));
        true
    }

    /// Redraws the bar on the current stderr line, if [`Self::should_redraw`] says so.
    fn render(&self) {
        if self.should_redraw(io::stderr().is_terminal()) {
            eprint!("\r{}", self.line());
            let _ = io::stderr().flush();
        }
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bar that must not print when dropped by a test.
    fn quiet(total: usize) -> ProgressBar {
        let p = ProgressBar::new(total, "x");
        p.finished.set(true);
        p
    }

    /// #127: going past `total` shows 100% and a full bar; the counter keeps the raw value.
    #[test]
    fn percentage_is_clamped_past_total() {
        let mut p = quiet(2);
        p.current = 5;
        assert_eq!(p.progress(), (100, 40));
        assert!(p.line().ends_with("100% (5/2)"), "{}", p.line());
    }

    #[test]
    fn line_shows_partial_progress() {
        let mut p = quiet(4);
        p.current = 1;
        assert_eq!(p.progress(), (25, 10));
        let line = p.line();
        assert!(line.starts_with("x ["), "{line}");
        assert_eq!(line.matches('█').count(), 10);
        assert_eq!(line.matches('░').count(), 30);
        assert!(line.ends_with(" 25% (1/4)"), "{line}");
    }

    #[test]
    fn empty_bar_shows_zero() {
        let mut p = quiet(0);
        p.current = 3;
        assert_eq!(p.progress(), (0, 0));
    }

    /// A huge `current` does not overflow the filled-width product.
    #[test]
    fn huge_current_does_not_overflow() {
        let mut p = quiet(usize::MAX);
        p.current = usize::MAX;
        assert_eq!(p.progress().1, 40);
    }

    /// #127: on a terminal, an update redraws only when the percentage or the filled
    /// width changed; off a terminal, updates never redraw.
    #[test]
    fn redraws_only_on_visible_change() {
        let mut p = quiet(1000);
        assert!(p.should_redraw(true), "first draw");
        p.current = 1; // 0% and no filled cell: unchanged
        assert!(!p.should_redraw(true));
        p.current = 10; // 1%
        assert!(p.should_redraw(true));
        assert!(!p.should_redraw(true), "same state twice");
        p.current = 500;
        assert!(!p.should_redraw(false), "not a terminal");
    }

    /// `finish` runs once, so `Drop` after an explicit `finish` does not print again.
    #[test]
    fn finish_runs_once() {
        let p = ProgressBar::new(1, "x");
        assert!(!p.finished.get());
        p.finish();
        assert!(p.finished.get());
        p.finish();
        assert!(p.finished.get());
    }
}
