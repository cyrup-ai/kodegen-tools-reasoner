//! Metrics collection for base strategy
//!
//! Provides methods for collecting and reporting strategy performance metrics.

use super::BaseStrategy;
use super::types::AsyncTask;
use crate::types::StrategyMetrics;
use std::sync::Arc;
use tokio::sync::oneshot;

impl BaseStrategy {
    /// Get base metrics
    pub fn get_base_metrics(&self) -> AsyncTask<StrategyMetrics> {
        let state_manager = Arc::clone(&self.state_manager);
        let cache_snapshot = self.get_cache_stats();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let nodes = state_manager.get_all_nodes().await;

            let avg_score = if nodes.is_empty() {
                0.0
            } else {
                nodes.iter().map(|n| n.score).sum::<f64>() / nodes.len() as f64
            };

            let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);

            let mut extra = std::collections::HashMap::new();
            extra.insert("cache_hits".to_string(), cache_snapshot.hits.into());
            extra.insert("cache_misses".to_string(), cache_snapshot.misses.into());
            extra.insert(
                "cache_evictions".to_string(),
                cache_snapshot.evictions.into(),
            );
            extra.insert(
                "cache_size_bytes".to_string(),
                cache_snapshot.size_bytes.into(),
            );

            let metrics = StrategyMetrics {
                name: String::from("BaseStrategy"),
                nodes_explored: nodes.len(),
                average_score: avg_score,
                max_depth,
                active: None,
                extra,
            };

            let _ = tx.send(Ok(metrics));
        });

        AsyncTask::new(rx)
    }
}
