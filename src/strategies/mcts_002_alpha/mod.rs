mod metrics;
mod scoring;
mod tree_ops;
mod types;

pub use types::PolicyGuidedNode;

use crate::state::StateManager;
use crate::strategies::base::{
    AsyncPath, BaseStrategy, ClearedSignal, Metric, MetricStream, Reasoning, Strategy,
};
use crate::strategies::mcts::MonteCarloTreeSearchStrategy;
use crate::types::{ReasoningRequest, ReasoningResponse, ThoughtNode};
use futures::StreamExt;
use metrics::MetricsTracker;
use scoring::ScoringEngine;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tree_ops::TreeOperations;
// Tracing is imported for error logging in case of future extensions
#[allow(unused_imports)]
use tracing;
use uuid::Uuid;

// Note: Text embedding functionality requires VoyageAI API (VOYAGE_API_KEY env var)
// Removed WASM host function - native implementation uses BaseStrategy methods

pub struct MCTS002AlphaStrategy {
    base: BaseStrategy,
    inner_mcts: Arc<MonteCarloTreeSearchStrategy>,
    tree_ops: Arc<TreeOperations>,
    metrics_tracker: Arc<MetricsTracker>,
}

impl MCTS002AlphaStrategy {
    pub fn new(state_manager: Arc<StateManager>, num_simulations: Option<usize>) -> Self {
        let num_simulations = num_simulations.unwrap_or(crate::types::CONFIG.num_simulations);
        
        let base = BaseStrategy::new(Arc::clone(&state_manager));
        let scoring = ScoringEngine::new(base.clone());
        
        let temperature = 1.0;
        let exploration_rate = 2.0_f64.sqrt();
        let learning_rate = 0.1;
        let novelty_bonus = 0.2;

        let metrics_tracker = Arc::new(MetricsTracker::new(
            temperature,
            exploration_rate,
            learning_rate,
            novelty_bonus,
        ));

        let tree_ops = Arc::new(TreeOperations::new(scoring, learning_rate));

        Self {
            base: BaseStrategy::new(Arc::clone(&state_manager)),
            inner_mcts: Arc::new(MonteCarloTreeSearchStrategy::new(
                Arc::clone(&state_manager),
                Some(num_simulations),
            )),
            tree_ops,
            metrics_tracker,
        }
    }

    /// Public helper to convert ThoughtNode to PolicyGuidedNode
    pub async fn thought_to_policy(
        &self,
        node: ThoughtNode,
    ) -> Result<PolicyGuidedNode, Box<dyn std::error::Error + Send + Sync>> {
        self.tree_ops.thought_to_policy(node).await
    }
}

// Add Clone implementation for MCTS002AlphaStrategy
impl Clone for MCTS002AlphaStrategy {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            inner_mcts: Arc::clone(&self.inner_mcts),
            tree_ops: Arc::clone(&self.tree_ops),
            metrics_tracker: Arc::clone(&self.metrics_tracker),
        }
    }
}

impl Strategy for MCTS002AlphaStrategy {
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning {
        let (tx, rx) = mpsc::channel(1);
        let self_clone = self.clone();

        tokio::spawn(async move {
            // Use persistent inner_mcts field
            let mut mcts_reasoning = self_clone.inner_mcts.process_thought(request.clone());
            let base_response = match mcts_reasoning.next().await {
                Some(Ok(response)) => response,
                _ => {
                    let _ = tx
                        .send(Err(crate::strategies::base::ReasoningError::Other(
                            "Failed to get base MCTS response".into(),
                        )))
                        .await;
                    return;
                }
            };

            let node_id = Uuid::new_v4().to_string();
            let parent_node = match &request.parent_id {
                Some(parent_id) => {
                    if let Ok(Some(node)) = self_clone.base.get_node(parent_id).await {
                        (self_clone.tree_ops.thought_to_policy(node).await).ok()
                    } else {
                        None
                    }
                }
                None => None,
            };

            let mut base_node = ThoughtNode {
                id: node_id.clone(),
                thought: request.thought.clone(),
                depth: request.thought_number - 1,
                score: 0.0,
                children: vec![],
                parent_id: request.parent_id.clone(),
                is_complete: !request.next_thought_needed,
                is_synthetic: false,
            };

            // Create thought identifier history from parent or initialize new one
            let thought_identifier = self_clone.tree_ops.scoring.get_thought_identifier(&request.thought);
            let action_history = match &parent_node {
                // Reusing field name
                Some(parent) => {
                    let mut history = parent.action_history.clone().unwrap_or_default();
                    history.push(thought_identifier);
                    Some(history)
                }
                None => Some(vec![thought_identifier]),
            };

            // Initialize PolicyGuidedNode
            let mut node = PolicyGuidedNode {
                base: base_node.clone(),
                visits: 1,
                total_reward: 0.0,
                untried_actions: Some(vec![]),
                policy_score: 0.0,
                value_estimate: 0.0,
                prior_action_probs: std::collections::HashMap::new(),
                puct: None,
                action_history,
                novelty_score: None,
            };

            // Initialize node with policy guidance
            let score = self_clone
                .base
                .evaluate_thought(&node.base, parent_node.as_ref().map(|p| &p.base))
                .await;
            node.base.set_score_or_default(score, 0.5);
            node.visits = 1;
            node.total_reward = node.base.score;
            // Calculate policy score and value estimate
            node.policy_score = self_clone.tree_ops.scoring
                .calculate_policy_score(&node, parent_node.as_ref())
                .await;
            node.value_estimate = self_clone.tree_ops.scoring.estimate_value(&node).await;
            node.novelty_score = Some(self_clone.tree_ops.scoring.calculate_novelty_v2(&node));
            base_node.set_score_or_default(node.base.score, 0.5);

            // Save the node
            if let Err(e) = self_clone.base.save_node_with_retry(base_node.clone(), None).await {
                tracing::error!("Fatal: Failed to save base node: {}", e);
                BaseStrategy::log_channel_send_error(
                    tx.send(Err(crate::strategies::base::ReasoningError::Fatal(format!(
                        "Failed to save node: {}", e
                    )))).await,
                    "node save error"
                );
                return;
            }

            // Update parent if exists
            if let Some(mut parent) = parent_node {
                parent.base.children.push(node.base.id.clone());
                if let Err(e) = self_clone.base.save_node_with_retry(parent.base.clone(), None).await {
                    tracing::warn!("Failed to update parent node: {}", e);
                    // Non-fatal: child node is saved, parent reference may be stale
                }
                let _ = self_clone.metrics_tracker.update_policy_metrics(&node, &parent).await;
            }

            // Run policy-guided search
            if !node.base.is_complete {
                let metrics_tracker_clone = Arc::clone(&self_clone.metrics_tracker);
                let adapt_fn = move |n: &PolicyGuidedNode| {
                    let mt = Arc::clone(&metrics_tracker_clone);
                    let node_clone = n.clone();
                    async move {
                        mt.adapt_exploration_rate(&node_clone).await;
                    }
                };
                
                let _ = self_clone.tree_ops.run_policy_guided_search(
                    node.clone(),
                    crate::types::CONFIG.num_simulations,
                    adapt_fn,
                ).await;
            }

            // Calculate enhanced path statistics
            let current_path = self_clone.base.state_manager.get_path(&node_id).await;
            let enhanced_score = self_clone.tree_ops
                .calculate_policy_enhanced_score(&current_path)
                .await;

            let response = ReasoningResponse {
                node_id: base_response.node_id,
                thought: base_response.thought,
                score: enhanced_score,
                depth: base_response.depth,
                is_complete: base_response.is_complete,
                next_thought_needed: base_response.next_thought_needed,
                possible_paths: base_response.possible_paths,
                best_score: Some(base_response.best_score.unwrap_or(0.0).max(enhanced_score)),
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
                "mcts_002_alpha response"
            );
        });

        Reasoning::new(rx)
    }

    fn get_best_path(&self) -> AsyncPath {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Delegate to persistent inner_mcts field
            let path = self_clone.inner_mcts.get_best_path().await;
            let _ = tx.send(path);
        });

        AsyncPath::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        let self_clone = self.clone();
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            let base_metrics = match self_clone.base.get_base_metrics().await {
                Ok(metrics) => metrics,
                _ => Metric {
                    name: String::from("MCTS-002-Alpha"),
                    nodes_explored: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    active: None,
                    extra: Default::default(),
                },
            };

            let mut metrics = base_metrics;
            let policy_metrics = self_clone.metrics_tracker.policy_metrics.lock().await.clone();
            let exploration_rate = *self_clone.metrics_tracker.exploration_rate.lock().await;

            let policy_stats = serde_json::json!({
                "averages": {
                    "policy_score": policy_metrics.average_policy_score,
                    "value_estimate": policy_metrics.average_value_estimate,
                },
                "temperature": self_clone.metrics_tracker.temperature,
                "exploration_rate": exploration_rate,
                "learning_rate": self_clone.metrics_tracker.learning_rate,
                "novelty_bonus": self_clone.metrics_tracker.novelty_bonus,
                "policy_entropy": policy_metrics.convergence_metrics.policy_entropy,
                "value_stability": policy_metrics.convergence_metrics.value_stability,
            });

            metrics.name = "MCTS-002-Alpha (Policy Enhanced)".to_string();
            metrics
                .extra
                .insert("temperature".to_string(), self_clone.metrics_tracker.temperature.into());
            metrics
                .extra
                .insert("exploration_rate".to_string(), exploration_rate.into());
            metrics
                .extra
                .insert("learning_rate".to_string(), self_clone.metrics_tracker.learning_rate.into());
            metrics
                .extra
                .insert("policy_stats".to_string(), policy_stats);

            let _ = tx.send(Ok(metrics)).await;
        });

        MetricStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Reset metrics via MetricsTracker
            self_clone.metrics_tracker.reset_metrics().await;
            let _ = tx.send(Ok(()));
        });

        ClearedSignal::new(rx)
    }
}
