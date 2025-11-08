//! Base reasoning strategy implementation
//!
//! Provides common functionality for all reasoning strategies including:
//! - Thought evaluation and scoring
//! - Semantic coherence calculation with cached embeddings
//! - Node operations with retry logic
//! - Cache management and memory pressure handling
//! - Metrics collection

use crate::state::StateManager;
use crate::types::ThoughtNode;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

// Module declarations
mod cache;
mod evaluation;
mod metrics;
mod node_ops;
mod semantic;
pub mod types;

// Public re-exports
pub use types::{
    AsyncPath, AsyncTask, ClearedSignal, Metric, MetricStream, Reasoning, ReasoningError,
    Strategy, TaskStream, NOVELTY_MARKERS,
};

/// Base strategy implementation that provides common functionality
pub struct BaseStrategy {
    pub state_manager: Arc<StateManager>,

    // LRU cache for embedding vectors (text -> Vec<f32>)
    // Size configurable via CONFIG.cache_size (~1.6MB for 1000x400-dim embeddings)
    embedding_cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,

    // Telemetry for cache performance and memory monitoring
    embedding_stats: Arc<crate::types::EmbeddingCacheStats>,
}

impl Clone for BaseStrategy {
    fn clone(&self) -> Self {
        Self {
            state_manager: Arc::clone(&self.state_manager),
            embedding_cache: Arc::clone(&self.embedding_cache),
            embedding_stats: Arc::clone(&self.embedding_stats),
        }
    }
}

/// Default cache size for embedding vectors.
/// APPROVED PANIC: Compile-time constant with non-zero value
/// - Evaluated at compile time, will fail during build if invalid
/// - Value 1000 is clearly non-zero in source code
/// - No runtime failure possible
const DEFAULT_EMBEDDING_CACHE_SIZE: NonZeroUsize = NonZeroUsize::new(1000).unwrap();

impl BaseStrategy {
    pub fn new(state_manager: Arc<StateManager>) -> Self {
        Self {
            state_manager,
            embedding_cache: Arc::new(RwLock::new(LruCache::new(
                DEFAULT_EMBEDDING_CACHE_SIZE,
            ))),
            embedding_stats: Arc::new(crate::types::EmbeddingCacheStats::new()),
        }
    }

    /// Safe division that returns fallback on NaN/infinite/divide-by-zero
    #[inline]
    pub(crate) fn safe_divide(numerator: f64, denominator: f64, fallback: f64) -> f64 {
        if denominator == 0.0 || denominator.is_nan() || numerator.is_nan() {
            return fallback;
        }
        let result = numerator / denominator;
        if result.is_nan() || result.is_infinite() {
            fallback
        } else {
            result
        }
    }

    /// Safe weighted sum with NaN checking for each component.
    /// Skips NaN components rather than propagating them.
    #[inline]
    pub(crate) fn safe_weighted_sum(components: &[(f64, f64)]) -> f64 {
        let mut sum = 0.0;
        let mut valid_weight_sum = 0.0;

        for (value, weight) in components {
            if value.is_nan() || weight.is_nan() {
                tracing::warn!(
                    "NaN detected in weighted sum component: value={}, weight={}",
                    value,
                    weight
                );
                continue;
            }
            sum += value * weight;
            valid_weight_sum += weight;
        }

        if valid_weight_sum == 0.0 {
            return 0.5;
        }

        let result = sum / valid_weight_sum;
        if result.is_nan() {
            0.5
        } else {
            result
        }
    }
}

/// Default implementation of Strategy for BaseStrategy
impl Strategy for BaseStrategy {
    fn process_thought(&self, _request: crate::types::ReasoningRequest) -> Reasoning {
        TaskStream::from_error(ReasoningError::Other(
            "Base strategy does not implement process_thought".into(),
        ))
    }

    fn get_best_path(&self) -> AsyncPath {
        let state_manager = Arc::clone(&self.state_manager);

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let nodes = state_manager.get_all_nodes().await;
            if nodes.is_empty() {
                let _ = tx.send(Ok(vec![]));
                return;
            }

            // Find highest scoring complete path
            let mut completed_nodes: Vec<ThoughtNode> =
                nodes.into_iter().filter(|n| n.is_complete).collect();

            if completed_nodes.is_empty() {
                let _ = tx.send(Ok(vec![]));
                return;
            }

            // Sort with deterministic tie-breaking
            completed_nodes.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or_else(|| {
                        // Should never happen with preventive validation
                        tracing::error!(
                            "Unexpected NaN in completed nodes sort - validation failure at assignment"
                        );
                        std::cmp::Ordering::Equal
                    })
                    .then_with(|| a.id.cmp(&b.id)) // Deterministic tie-breaking
            });

            let path = state_manager.get_path(&completed_nodes[0].id).await;
            let _ = tx.send(Ok(path));
        });

        AsyncTask::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        // Convert AsyncTask to TaskStream
        let async_metrics = self.get_base_metrics();
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            match async_metrics.await {
                Ok(metrics) => {
                    let _ = tx.send(Ok(metrics)).await;
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                }
            }
        });

        TaskStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        AsyncTask::from_value(())
    }
}
