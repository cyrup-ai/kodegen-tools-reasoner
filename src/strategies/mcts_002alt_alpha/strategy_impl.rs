use crate::strategies::base::{
    AsyncPath, BaseStrategy, ClearedSignal, Metric, MetricStream, Reasoning, Strategy,
};
use crate::types::{ReasoningRequest, ReasoningResponse};
use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::core::MCTS002AltAlphaStrategy;
use super::types::{BidirectionalPolicyNode, BidirectionalStats};

impl Strategy for MCTS002AltAlphaStrategy {
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning {
        let (tx, rx) = mpsc::channel(1);
        let self_clone = self.clone();

        tokio::spawn(async move {
            // First get the base response from MCTS002Alpha
            let mut mcts_reasoning = self_clone.inner_strategy.process_thought(request.clone());
            let base_response = match mcts_reasoning.next().await {
                Some(Ok(response)) => response,
                _ => {
                    let _ = tx
                        .send(Err(crate::strategies::base::ReasoningError::Other(
                            "Failed to get base response from MCTS002Alpha".into(),
                        )))
                        .await;
                    return;
                }
            };

            let _node_id = Uuid::new_v4().to_string();

            // Process the thought using standard MCTS002Alpha
            let policy_response = base_response.clone();

            // Track start and goal nodes for bidirectional search
            if request.parent_id.is_none() {
                // This is a start node
                if let Ok(Some(node)) = self_clone.base.get_node(&policy_response.node_id).await
                    && let Ok(policy_node) = self_clone.inner_strategy.thought_to_policy(node).await
                {
                    let mut bi_node = BidirectionalPolicyNode {
                        base: policy_node,
                        g: 0.0,
                        h: 0.0,
                        f: 0.0,
                        parent: None,
                        direction: Some("forward".to_string()),
                        search_depth: Some(0),
                        meeting_point: None,
                    };

                    // Calculate h and f
                    bi_node.h = 1.0 - bi_node.base.value_estimate;
                    bi_node.f = bi_node.g + bi_node.h;

                    let mut start_node = self_clone.start_node.lock().await;
                    *start_node = Some(bi_node);
                }
            }

            if !request.next_thought_needed {
                // This is a goal node
                if let Ok(Some(node)) = self_clone.base.get_node(&policy_response.node_id).await
                    && let Ok(policy_node) = self_clone.inner_strategy.thought_to_policy(node).await
                {
                    let mut bi_node = BidirectionalPolicyNode {
                        base: policy_node,
                        g: 0.0,
                        h: 0.0,
                        f: 0.0,
                        parent: None,
                        direction: Some("backward".to_string()),
                        search_depth: Some(0),
                        meeting_point: None,
                    };

                    // Calculate h and f
                    bi_node.h = 1.0 - bi_node.base.value_estimate;
                    bi_node.f = bi_node.g + bi_node.h;

                    let mut goal_node = self_clone.goal_node.lock().await;
                    *goal_node = Some(bi_node);
                }
            }

            // Run bidirectional search if we have both endpoints
            let mut path = Vec::new();
            {
                let start_node = self_clone.start_node.lock().await;
                let goal_node = self_clone.goal_node.lock().await;

                if let (Some(start), Some(goal)) = (start_node.clone(), goal_node.clone())
                    && let Ok(found_path) = self_clone.bidirectional_search(start, goal).await
                {
                    path = found_path;
                }
            }

            if !path.is_empty() {
                let _ = self_clone.update_path_with_policy_guidance(&path).await;
            }

            // Calculate enhanced path statistics
            let current_path = self_clone
                .base
                .state_manager
                .get_path(&policy_response.node_id)
                .await;
            let stats = self_clone.bidirectional_stats.lock().await.clone();
            let enhanced_score =
                self_clone.calculate_bidirectional_policy_score(&current_path, &stats);

            let response = ReasoningResponse {
                node_id: policy_response.node_id,
                thought: policy_response.thought,
                score: enhanced_score,
                depth: policy_response.depth,
                is_complete: policy_response.is_complete,
                next_thought_needed: policy_response.next_thought_needed,
                possible_paths: policy_response.possible_paths,
                best_score: Some(
                    policy_response
                        .best_score
                        .unwrap_or(0.0)
                        .max(enhanced_score),
                ),
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
                "mcts_002alt_alpha response"
            );
        });

        Reasoning::new(rx)
    }

    fn get_best_path(&self) -> AsyncPath {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Use inner strategy for best path calculation
            let inner_path = self_clone.inner_strategy.get_best_path().await;
            let _ = tx.send(inner_path);
        });

        AsyncPath::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        let self_clone = self.clone();
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            let mut inner_metrics_stream = self_clone.inner_strategy.get_metrics();

            // Get the metrics from the stream (should be just one item)
            let base_metrics = match inner_metrics_stream.next().await {
                Some(Ok(metrics)) => metrics,
                _ => Metric {
                    name: String::from("MCTS-002Alt-Alpha"),
                    nodes_explored: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    active: None,
                    extra: Default::default(),
                },
            };

            let mut metrics = base_metrics;
            let nodes = self_clone.base.state_manager.get_all_nodes().await;

            // Build approximated bidirectional metrics
            let stats = self_clone.bidirectional_stats.lock().await.clone();
            let start_node = self_clone.start_node.lock().await;
            let goal_node = self_clone.goal_node.lock().await;
            // Split nodes by direction (approximate since we don't store direction with nodes)
            // This remains an approximation based on depth parity.
            let (forward_nodes, backward_nodes): (Vec<&crate::types::ThoughtNode>, Vec<&crate::types::ThoughtNode>) =
                nodes.iter().partition(|n| n.depth % 2 == 0); // Approximation

            let bidirectional_metrics = serde_json::json!({
                "forward_search_approx": { // Explicitly indicate approximation
                    "nodes_explored": forward_nodes.len(),
                    "average_score": if forward_nodes.is_empty() {
                        0.0
                    } else {
                        forward_nodes.iter().map(|n| n.score).sum::<f64>() / forward_nodes.len() as f64
                    },
                    "exploration_rate": stats.forward_exploration_rate
                },
                "backward_search_approx": { // Explicitly indicate approximation
                    "nodes_explored": backward_nodes.len(),
                    "average_score": if backward_nodes.is_empty() {
                        0.0
                    } else {
                        backward_nodes.iter().map(|n| n.score).sum::<f64>() / backward_nodes.len() as f64
                    },
                    "exploration_rate": stats.backward_exploration_rate
                },
                "meeting_points": {
                    "count": stats.meeting_points,
                }, // Keep reported meeting points from stats
                "path_quality": stats.path_quality
            });

            metrics.name = "MCTS-002Alt-Alpha (Bidirectional + Policy Enhanced)".to_string();
            metrics
                .extra
                .insert("has_start_node".to_string(), start_node.is_some().into());
            metrics
                .extra
                .insert("has_goal_node".to_string(), goal_node.is_some().into());
            metrics
                .extra
                .insert("bidirectional_metrics".to_string(), bidirectional_metrics);

            let _ = tx.send(Ok(metrics)).await;
        });

        MetricStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Clear the inner strategy first
            let inner_clear = self_clone.inner_strategy.clear();
            let _ = inner_clear.await; // Wait for inner clear to complete

            // Reset bidirectional search state
            let mut start_node = self_clone.start_node.lock().await;
            *start_node = None;

            let mut goal_node = self_clone.goal_node.lock().await;
            *goal_node = None;

            let mut stats = self_clone.bidirectional_stats.lock().await;
            *stats = BidirectionalStats {
                forward_exploration_rate: 2.0_f64.sqrt(),
                backward_exploration_rate: 2.0_f64.sqrt(),
                meeting_points: 0,
                path_quality: 0.0,
            };

            let _ = tx.send(Ok(()));
        });

        ClearedSignal::new(rx)
    }
}
