//! Embedding cache management and memory pressure handling
//!
//! Implements LRU caching for embeddings with automatic eviction under memory pressure.

use super::BaseStrategy;
use super::types::{EXPECTED_EMBEDDING_DIM, MEMORY_PRESSURE_THRESHOLD};
use lru::LruCache;
use std::sync::Arc;
use sysinfo::System;
use tokio::sync::RwLock;

impl BaseStrategy {
    /// Check if system is under memory pressure
    /// Returns true if available memory is below MEMORY_PRESSURE_THRESHOLD
    pub(super) fn check_memory_pressure() -> bool {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let available = sys.available_memory() as f64;
        let total = sys.total_memory() as f64;

        if total == 0.0 {
            return false; // Safety check
        }

        let available_ratio = available / total;
        available_ratio < MEMORY_PRESSURE_THRESHOLD
    }

    /// Aggressively evict cache entries when under memory pressure
    /// Reduces cache to target_ratio of max size (e.g., 25% = 250 entries from 1000)
    pub(super) async fn aggressive_cache_evict(
        cache: &Arc<RwLock<LruCache<String, Vec<f32>>>>,
        stats: &Arc<crate::types::EmbeddingCacheStats>,
        target_ratio: f64,
    ) {
        let mut cache_lock = cache.write().await;
        let current_len = cache_lock.len();
        let max_size = cache_lock.cap().get();
        let target_size = (max_size as f64 * target_ratio).ceil() as usize;

        if current_len <= target_size {
            return; // Already below target
        }

        let to_evict = current_len - target_size;

        tracing::warn!(
            "Memory pressure detected! Evicting {} cache entries ({}% reduction)",
            to_evict,
            ((to_evict as f64 / current_len as f64) * 100.0) as usize
        );

        // LRU cache pop_lru() removes least recently used entries
        for _ in 0..to_evict {
            if let Some((_, embedding)) = cache_lock.pop_lru() {
                let entry_size = estimate_entry_size(&embedding);
                stats.record_eviction(entry_size);
            }
        }

        tracing::info!(
            "Cache reduced from {} to {} entries",
            current_len,
            cache_lock.len()
        );
    }

    /// Validate embedding has expected dimensions
    /// Returns true if embedding is valid for caching
    pub(super) fn validate_embedding_size(embedding: &[f32]) -> bool {
        let dim = embedding.len();

        if dim != EXPECTED_EMBEDDING_DIM {
            tracing::warn!(
                "Embedding dimension mismatch: expected {}, got {}",
                EXPECTED_EMBEDDING_DIM,
                dim
            );
            return false;
        }

        true
    }

    /// Get current cache statistics (for logging/monitoring)
    pub fn get_cache_stats(&self) -> crate::types::EmbeddingCacheSnapshot {
        self.embedding_stats.snapshot()
    }
}

/// Estimate memory size of a cache entry (key + value)
/// Embedding: dimension * 4 bytes (f32)
/// String key: length * 1 byte (approximate)
pub(super) fn estimate_entry_size(embedding: &[f32]) -> usize {
    std::mem::size_of_val(embedding)
}

/// Estimate memory size including string key
pub(super) fn estimate_entry_size_with_key(key: &str, embedding: &[f32]) -> usize {
    key.len() + estimate_entry_size(embedding)
}
