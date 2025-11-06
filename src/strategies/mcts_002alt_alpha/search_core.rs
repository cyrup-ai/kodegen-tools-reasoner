use crate::types::ThoughtNode;
use std::collections::HashMap;
use tracing;

use super::core::MCTS002AltAlphaStrategy;
use super::types::{BidirectionalPolicyNode, Queue};

impl MCTS002AltAlphaStrategy {
    #[allow(dead_code)]
    pub(super) async fn create_bidirectional_node(
        &self,
        thought_node: ThoughtNode,
        direction: &str,
        search_depth: usize,
        parent_g: f64,
    ) -> Result<BidirectionalPolicyNode, Box<dyn std::error::Error + Send + Sync>> {
        // Convert to PolicyGuidedNode first
        let policy_node = self.inner_strategy.thought_to_policy(thought_node).await?;

        // Calculate value estimate directly (same as inner_strategy.estimate_value)
        // This avoids dealing with the async Result return type in estimate_value
        let immediate_value = policy_node.base.score;
        let depth_potential =
            1.0 - (policy_node.base.depth as f64 / crate::types::CONFIG.max_depth as f64);
        let novelty_value = policy_node.novelty_score.unwrap_or(0.0);

        // Same weights as in the original function
        let value_estimate = 0.5 * immediate_value + 0.3 * depth_potential + 0.2 * novelty_value;

        Ok(BidirectionalPolicyNode {
            base: policy_node,
            g: parent_g + 1.0,
            h: 1.0 - value_estimate, // Heuristic is inverse of value
            f: parent_g + 1.0 + (1.0 - value_estimate),
            parent: None,
            direction: Some(direction.to_string()),
            search_depth: Some(search_depth),
            meeting_point: None,
        })
    }

    pub(super) async fn search_level(
        &self,
        queue: &mut Queue<BidirectionalPolicyNode>,
        visited: &mut HashMap<String, BidirectionalPolicyNode>,
        other_visited: &HashMap<String, BidirectionalPolicyNode>,
        direction: &str,
    ) -> Option<BidirectionalPolicyNode> {
        let level_size = queue.size();

        for _ in 0..level_size {
            if let Some(mut current) = queue.dequeue() {
                // Check if we've found a meeting point
                if other_visited.contains_key(&current.base.base.id) {
                    current.meeting_point = Some(true);
                    let mut stats = self.bidirectional_stats.lock().await;
                    stats.meeting_points += 1;

                    // Save the updated node
                    if (self.base.save_node(current.base.base.clone()).await).is_err() {
                        return Some(current); // Return meeting point despite error, just log it
                    }

                    return Some(current);
                }

                // Get neighbors based on direction and policy scores
                let mut neighbors = Vec::new();
                if direction == "forward" {
                    // Forward direction: get children
                    for id in &current.base.base.children {
                        if let Ok(Some(child_node)) = self.base.get_node(id).await
                            && let Ok(policy_child) =
                                self.inner_strategy.thought_to_policy(child_node).await
                        {
                            let search_depth = current.search_depth.unwrap_or(0) + 1;
                            let parent_g = current.g;

                            let mut bi_child = BidirectionalPolicyNode {
                                base: policy_child,
                                g: parent_g + 1.0,
                                h: 0.0, // Will calculate below
                                f: 0.0, // Will calculate below
                                parent: Some(current.base.base.id.clone()),
                                direction: Some(direction.to_string()),
                                search_depth: Some(search_depth),
                                meeting_point: None,
                            };

                            // Calculate h and f
                            bi_child.h = 1.0 - bi_child.base.value_estimate;
                            bi_child.f = bi_child.g + bi_child.h;

                            neighbors.push(bi_child);
                        }
                    }
                } else {
                    // Backward direction: get parent
                    if let Some(parent_id) = &current.base.base.parent_id
                        && let Ok(Some(parent_node)) = self.base.get_node(parent_id).await
                        && let Ok(policy_parent) =
                            self.inner_strategy.thought_to_policy(parent_node).await
                    {
                        let search_depth = current.search_depth.unwrap_or(0) + 1;
                        let parent_g = current.g;

                        let mut bi_parent = BidirectionalPolicyNode {
                            base: policy_parent,
                            g: parent_g + 1.0,
                            h: 0.0, // Will calculate below
                            f: 0.0, // Will calculate below
                            parent: Some(current.base.base.id.clone()),
                            direction: Some(direction.to_string()),
                            search_depth: Some(search_depth),
                            meeting_point: None,
                        };

                        // Calculate h and f
                        bi_parent.h = 1.0 - bi_parent.base.value_estimate;
                        bi_parent.f = bi_parent.g + bi_parent.h;

                        neighbors.push(bi_parent);
                    }
                }

                // Sort neighbors by policy score with deterministic tie-breaking
                neighbors.sort_by(|a, b| {
                    match b.base.policy_score.partial_cmp(&a.base.policy_score) {
                        Some(std::cmp::Ordering::Equal) => {
                            // Break ties by node ID
                            a.base.base.id.cmp(&b.base.base.id)
                        }
                        Some(order) => order,
                        None => {
                            tracing::error!("Unexpected NaN in neighbor sort after filtering");
                            std::cmp::Ordering::Equal
                        }
                    }
                });

                for neighbor in neighbors {
                    if !visited.contains_key(&neighbor.base.base.id) {
                        visited.insert(neighbor.base.base.id.clone(), neighbor.clone());

                        // Save updated node
                        if let Err(e) = self.base.save_node(neighbor.base.base.clone()).await {
                            tracing::error!("Error saving neighbor node: {}", e);
                            continue; // Skip this neighbor if we can't save it
                        }

                        queue.enqueue(neighbor);
                    }
                }
            }
        }

        None
    }

    pub(super) async fn bidirectional_search(
        &self,
        start: BidirectionalPolicyNode,
        goal: BidirectionalPolicyNode,
    ) -> Result<Vec<BidirectionalPolicyNode>, Box<dyn std::error::Error + Send + Sync>> {
        let mut forward_queue = Queue::new();
        let mut backward_queue = Queue::new();
        let mut forward_visited = HashMap::new();
        let mut backward_visited = HashMap::new();

        forward_queue.enqueue(start.clone());
        backward_queue.enqueue(goal.clone());
        forward_visited.insert(start.base.base.id.clone(), start);
        backward_visited.insert(goal.base.base.id.clone(), goal);

        while !forward_queue.is_empty() && !backward_queue.is_empty() {
            // Search from both directions with policy guidance
            if let Some(meeting_point) = self
                .search_level(
                    &mut forward_queue,
                    &mut forward_visited,
                    &backward_visited,
                    "forward",
                )
                .await
            {
                let path =
                    self.reconstruct_path(meeting_point, &forward_visited, &backward_visited);
                self.update_bidirectional_stats(&path).await;
                return Ok(path);
            }

            if let Some(back_meeting_point) = self
                .search_level(
                    &mut backward_queue,
                    &mut backward_visited,
                    &forward_visited,
                    "backward",
                )
                .await
            {
                let path =
                    self.reconstruct_path(back_meeting_point, &forward_visited, &backward_visited);
                self.update_bidirectional_stats(&path).await;
                return Ok(path);
            }

            // Adapt exploration rates based on progress
            self.adapt_bidirectional_exploration(&forward_visited, &backward_visited)
                .await;
        }

        Ok(vec![])
    }

    pub(super) async fn adapt_bidirectional_exploration(
        &self,
        forward_visited: &HashMap<String, BidirectionalPolicyNode>,
        backward_visited: &HashMap<String, BidirectionalPolicyNode>,
    ) {
        // Skip if either map is empty
        if forward_visited.is_empty() || backward_visited.is_empty() {
            return;
        }

        // Collect policy scores for averaging
        let forward_scores: Vec<f64> = forward_visited
            .values()
            .map(|node| node.base.policy_score)
            .collect();
        
        let backward_scores: Vec<f64> = backward_visited
            .values()
            .map(|node| node.base.policy_score)
            .collect();
        
        // Guard against empty collections
        if forward_scores.is_empty() || backward_scores.is_empty() {
            tracing::warn!(
                "Cannot compute bidirectional stats: empty node collections (forward: {}, backward: {})",
                forward_scores.len(), backward_scores.len()
            );
            return;
        }
        
        // Calculate averages (safe because we checked non-empty above)
        let forward_progress = forward_scores.iter().sum::<f64>() / forward_scores.len() as f64;
        let backward_progress = backward_scores.iter().sum::<f64>() / backward_scores.len() as f64;
        
        // Validate results (should never fail after embedding validation)
        if forward_progress.is_nan() || backward_progress.is_nan() {
            panic!(
                "FATAL: Progress calculation produced NaN (forward: {}, backward: {}). \
                 This should never happen after embedding validation. \
                 Check if scores have NaN before averaging.",
                forward_progress, backward_progress
            );
        }

        // Adjust exploration rates with clamping
        use super::types::{MIN_EXPLORATION_RATE, MAX_EXPLORATION_RATE};
        let mut stats = self.bidirectional_stats.lock().await;
        if forward_progress > backward_progress {
            stats.backward_exploration_rate = (stats.backward_exploration_rate * 1.05)
                .clamp(MIN_EXPLORATION_RATE, MAX_EXPLORATION_RATE);
            stats.forward_exploration_rate = (stats.forward_exploration_rate * 0.95)
                .clamp(MIN_EXPLORATION_RATE, MAX_EXPLORATION_RATE);
        } else {
            stats.forward_exploration_rate = (stats.forward_exploration_rate * 1.05)
                .clamp(MIN_EXPLORATION_RATE, MAX_EXPLORATION_RATE);
            stats.backward_exploration_rate = (stats.backward_exploration_rate * 0.95)
                .clamp(MIN_EXPLORATION_RATE, MAX_EXPLORATION_RATE);
        }
    }

    pub(super) async fn update_bidirectional_stats(&self, path: &[BidirectionalPolicyNode]) {
        if path.is_empty() {
            return;
        }

        let forward_nodes: Vec<&BidirectionalPolicyNode> = path
            .iter()
            .filter(|n| n.direction.as_deref() == Some("forward"))
            .collect();

        let backward_nodes: Vec<&BidirectionalPolicyNode> = path
            .iter()
            .filter(|n| n.direction.as_deref() == Some("backward"))
            .collect();

        // Skip if either direction is missing
        if forward_nodes.is_empty() || backward_nodes.is_empty() {
            return;
        }

        // Collect policy scores for averaging
        let forward_scores: Vec<f64> = forward_nodes
            .iter()
            .map(|n| n.base.policy_score)
            .collect();
        
        let backward_scores: Vec<f64> = backward_nodes
            .iter()
            .map(|n| n.base.policy_score)
            .collect();
        
        // Guard against empty collections
        if forward_scores.is_empty() || backward_scores.is_empty() {
            tracing::warn!(
                "Cannot compute bidirectional stats: empty node collections (forward: {}, backward: {})",
                forward_scores.len(), backward_scores.len()
            );
            return;
        }

        // Calculate quality metrics (safe because we checked non-empty)
        let forward_quality = forward_scores.iter().sum::<f64>() / forward_scores.len() as f64;
        let backward_quality = backward_scores.iter().sum::<f64>() / backward_scores.len() as f64;
        
        // Validate before storing (should never fail after embedding validation)
        if forward_quality.is_nan() || backward_quality.is_nan() {
            panic!(
                "FATAL: Quality calculation produced NaN (forward: {}, backward: {}). \
                 This should never happen after embedding validation. \
                 Check if value_estimates have NaN before averaging.",
                forward_quality, backward_quality
            );
        }

        let mut stats = self.bidirectional_stats.lock().await;
        stats.path_quality = (forward_quality + backward_quality) / 2.0;
    }
}
