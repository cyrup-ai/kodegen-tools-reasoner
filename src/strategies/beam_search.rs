use crate::state::StateManager;
use crate::strategies::base::{
    AsyncPath, AsyncTask, BaseStrategy, ClearedSignal, MetricStream, Reasoning, Strategy,
    TaskStream,
};
use crate::types::{CONFIG, ReasoningRequest, ReasoningResponse, StrategyMetrics, ThoughtNode};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

pub struct BeamSearchStrategy {
    base: BaseStrategy,
    beam_width: usize,
    beams: Arc<Mutex<HashMap<usize, Vec<ThoughtNode>>>>,
}

impl BeamSearchStrategy {
    pub fn new(state_manager: Arc<StateManager>, beam_width: Option<usize>) -> Self {
        Self {
            base: BaseStrategy::new(state_manager),
            beam_width: beam_width.unwrap_or(CONFIG.beam_width),
            beams: Arc::new(Mutex::new(HashMap::new())),
        }
    }

}

impl Strategy for BeamSearchStrategy {
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning {
        let self_clone = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(8);

        tokio::spawn(async move {
            let node_id = Uuid::new_v4().to_string();
            let parent_node = match &request.parent_id {
                Some(parent_id) => match self_clone.base.get_node_with_retry(parent_id, None).await {
                    Ok(node) => node,
                    Err(e) => {
                        BaseStrategy::log_channel_send_error(
                            tx.send(Err(e)).await,
                            "parent node error"
                        );
                        return;
                    }
                },
                None => None,
            };

            // CHECK 1: Before expensive evaluation
            if tx.is_closed() {
                tracing::debug!("Beam search: receiver dropped before evaluation, aborting");
                return;
            }

            let mut node = ThoughtNode {
                id: node_id.clone(),
                thought: request.thought.clone(),
                depth: request.thought_number - 1,
                score: 0.0,
                children: vec![],
                parent_id: request.parent_id.clone(),
                is_complete: !request.next_thought_needed,
                is_synthetic: false,
            };

            // Evaluate and score the node
            let score = self_clone
                .base
                .evaluate_thought(&node, parent_node.as_ref())
                .await;
            node.set_score_or_default(score, 0.5);
            if let Err(e) = self_clone.base.save_node_with_retry(node.clone(), None).await {
                tracing::error!("Fatal: Failed to save node {}: {}", node.id, e);
                BaseStrategy::log_channel_send_error(
                    tx.send(Err(crate::strategies::base::ReasoningError::Fatal(format!(
                        "Failed to save node after retries: {}", e
                    )))).await,
                    "node save error"
                );
                return;
            }

            // Update parent if exists
            if let Some(mut parent) = parent_node {
                parent.children.push(node.id.clone());
                if let Err(e) = self_clone.base.save_node_with_retry(parent, None).await {
                    tracing::warn!("Failed to update parent node: {}", e);
                    // Non-fatal: continue processing even if parent update fails
                }
            }

            // Manage beam at current depth
            let mut beams = self_clone.beams.lock().await;
            let current_beam = beams.entry(node.depth).or_insert_with(Vec::new);
            current_beam.push(node.clone());

            // Sort with secondary key for determinism
            current_beam.sort_by(|a, b| {
                match b.score.partial_cmp(&a.score) {
                    Some(std::cmp::Ordering::Equal) => {
                        // Break ties by node ID for determinism
                        a.id.cmp(&b.id)
                    }
                    Some(order) => order,
                    None => {
                        // Should never happen after filtering
                        tracing::error!("Unexpected NaN in beam sort after filtering");
                        std::cmp::Ordering::Equal
                    }
                }
            });

            // Prune beam to maintain beam width
            if current_beam.len() > self_clone.beam_width {
                *current_beam = current_beam[0..self_clone.beam_width].to_vec();
            }

            // Calculate best score across all beams (while holding lock)
            let best_beam_score = beams
                .values()
                .flat_map(|nodes| nodes.iter().map(|n| n.score))
                .fold(f64::NEG_INFINITY, f64::max);

            // Calculate possible paths (inlined from calculate_possible_paths, while holding lock)
            let possible_paths = {
                let depths: Vec<usize> = beams.keys().copied().collect();
                let mut total_paths = 0;
                
                for depth in &depths {
                    let beam = match beams.get(depth) {
                        Some(beam) => beam,
                        None => {
                            tracing::error!("Beam at depth {} not found during path calculation", depth);
                            continue;
                        }
                    };
                    let next_beam = beams.get(&(depth + 1));

                    if let Some(next_beam) = next_beam {
                        total_paths += beam.len() * next_beam.len();
                    } else {
                        total_paths += beam.len();
                    }
                }
                
                total_paths
            };

            // Release lock - all beam-dependent data captured
            drop(beams);

            // CHECK 2: Before path calculations
            if tx.is_closed() {
                tracing::debug!("Beam search: receiver dropped after beam management, aborting");
                return;
            }

            // Calculate path statistics (uses StateManager - safe outside lock)
            let current_path = self_clone.base.state_manager.get_path(&node_id).await;
            let path_score = if current_path.is_empty() {
                0.0
            } else {
                current_path.iter().map(|n| n.score).sum::<f64>() / current_path.len() as f64
            };

            let response = ReasoningResponse {
                node_id: node.id,
                thought: node.thought,
                score: node.score,
                depth: node.depth,
                is_complete: node.is_complete,
                next_thought_needed: request.next_thought_needed,
                possible_paths: Some(possible_paths),
                best_score: Some(path_score.max(best_beam_score)),
                strategy_used: None, // Will be set by reasoner
                thought_number: 0, // Will be set by Tool layer
                total_thoughts: 0, // Will be set by Tool layer
                stats: crate::types::ReasoningStats {
                    total_nodes: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    branching_factor: 0.0,
                    strategy_metrics: std::collections::HashMap::new(),
                }, // Will be set by Tool layer
            };

            BaseStrategy::log_channel_send_error(
                tx.send(Ok(response)).await,
                "beam search response"
            );
        });

        TaskStream::new(rx)
    }

    fn get_best_path(&self) -> AsyncPath {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let beams = self_clone.beams.lock().await;

            // Find the deepest beam
            let max_depth = beams.keys().max().copied();

            if let Some(depth) = max_depth
                && let Some(deepest_beam) = beams.get(&depth)
                && !deepest_beam.is_empty()
            {
                // Get the best scoring node from deepest beam
                // Find best with deterministic tie-breaking
                let best_node_id = deepest_beam
                    .iter()
                    .max_by(|a, b| {
                        a.score.partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.id.cmp(&b.id))
                    })
                    // APPROVED BY DAVID MAPLE 09/30/2025: Panic is appropriate for logic invariant violation
                    .expect("Deepest beam should contain at least one element")
                    .id
                    .clone();

                drop(beams);
                let path = self_clone.base.state_manager.get_path(&best_node_id).await;
                let _ = tx.send(Ok(path));
                return;
            }

            let _ = tx.send(Ok(vec![]));
        });

        AsyncTask::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        let self_clone = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(8);

        tokio::spawn(async move {
            let base_metrics = self_clone
                .base
                .get_base_metrics()
                .await
                .unwrap_or_else(|_| StrategyMetrics {
                    name: String::from("Beam Search"),
                    nodes_explored: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    active: None,
                    extra: Default::default(),
                });

            let mut metrics = base_metrics;

            let beams = self_clone.beams.lock().await;
            let active_beams = beams.len();
            let total_beam_nodes = beams.values().map(|beam| beam.len()).sum::<usize>();
            drop(beams);

            metrics.name = "Beam Search".to_string();
            metrics
                .extra
                .insert("beam_width".to_string(), self_clone.beam_width.into());
            metrics
                .extra
                .insert("active_beams".to_string(), active_beams.into());
            metrics
                .extra
                .insert("total_beam_nodes".to_string(), total_beam_nodes.into());

            if (tx.send(Ok(metrics)).await).is_err() {
                // Channel closed, receiver dropped
            }
        });

        TaskStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let mut beams = self_clone.beams.lock().await;
            beams.clear();
            let _ = tx.send(Ok(()));
        });

        AsyncTask::new(rx)
    }
}

// Add Clone implementation for BeamSearchStrategy
impl Clone for BeamSearchStrategy {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            beam_width: self.beam_width,
            beams: Arc::clone(&self.beams),
        }
    }
}
