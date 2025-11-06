use std::collections::{HashMap, HashSet};
use tracing;

use super::core::MCTS002AltAlphaStrategy;
use super::types::BidirectionalPolicyNode;

impl MCTS002AltAlphaStrategy {
    pub(super) fn reconstruct_path(
        &self,
        meeting_point: BidirectionalPolicyNode,
        forward_visited: &HashMap<String, BidirectionalPolicyNode>,
        backward_visited: &HashMap<String, BidirectionalPolicyNode>,
    ) -> Vec<BidirectionalPolicyNode> {
        const MAX_PATH_LENGTH: usize = 1000;  // Safety limit for pathological cases
        
        let mut path = vec![meeting_point.clone()];
        let mut visited_ids = HashSet::new();
        visited_ids.insert(meeting_point.base.base.id.clone());

        // Reconstruct forward path with dual protection
        let mut current = meeting_point.clone();
        let mut iterations = 0;
        
        while let Some(parent_id) = &current.parent {
            iterations += 1;
            
            // Protection 1: Iteration limit (prevents unbounded growth)
            if iterations > MAX_PATH_LENGTH {
                tracing::error!(
                    "Forward path reconstruction exceeded {} iterations at node {} (possible infinite loop)",
                    MAX_PATH_LENGTH,
                    parent_id
                );
                break;
            }
            
            // Protection 2: Cycle detection (catches circular references)
            if visited_ids.contains(parent_id) {
                tracing::error!(
                    "Cycle detected in forward path reconstruction at node {} (iteration {})",
                    parent_id,
                    iterations
                );
                break;
            }
            
            if let Some(parent) = forward_visited.get(parent_id) {
                visited_ids.insert(parent_id.clone());
                path.insert(0, parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }

        // Reconstruct backward path with dual protection
        current = meeting_point;
        visited_ids.clear();
        visited_ids.insert(current.base.base.id.clone());
        iterations = 0;
        
        while let Some(parent_id) = &current.parent {
            iterations += 1;
            
            // Protection 1: Iteration limit
            if iterations > MAX_PATH_LENGTH {
                tracing::error!(
                    "Backward path reconstruction exceeded {} iterations at node {} (possible infinite loop)",
                    MAX_PATH_LENGTH,
                    parent_id
                );
                break;
            }
            
            // Protection 2: Cycle detection
            if visited_ids.contains(parent_id) {
                tracing::error!(
                    "Cycle detected in backward path reconstruction at node {} (iteration {})",
                    parent_id,
                    iterations
                );
                break;
            }
            
            if let Some(parent) = backward_visited.get(parent_id) {
                visited_ids.insert(parent_id.clone());
                path.push(parent.clone());
                current = parent.clone();
            } else {
                break;
            }
        }

        path
    }

    pub(super) async fn update_path_with_policy_guidance(
        &self,
        path: &[BidirectionalPolicyNode],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path_bonus = 0.2;

        for node in path {
            let mut updated_node = node.clone();

            // Boost both policy and value estimates for nodes along the path
            updated_node.base.policy_score += path_bonus;
            updated_node.base.value_estimate = (updated_node.base.value_estimate + 1.0) / 2.0;

            // Update action probabilities with path information
            if let Some(parent_id) = &updated_node.base.base.parent_id
                && let Ok(Some(parent_node)) = self.base.get_node(parent_id).await
                && let Ok(mut policy_parent) = self
                    .inner_strategy
                    .thought_to_policy(parent_node.clone())
                    .await
            {
                let action_key = self.get_action_key(&updated_node.base.base.thought);
                let current_prob = *policy_parent
                    .prior_action_probs
                    .get(&action_key)
                    .unwrap_or(&0.0);
                let new_prob = current_prob.max(0.8); // Strong preference for path actions
                policy_parent
                    .prior_action_probs
                    .insert(action_key, new_prob);

                // Save updated parent node
                let updated_parent = policy_parent.base.clone();
                if let Err(e) = self.base.save_node(updated_parent).await {
                    tracing::error!("Error saving updated parent: {}", e);
                }
            }

            // Save updated node
            let base_node = updated_node.base.base.clone();
            if let Err(e) = self.base.save_node(base_node).await {
                tracing::error!("Error saving base node: {}", e);
            }
        }

        // Update path quality metric
        if !path.is_empty() {
            let path_quality = path
                .iter()
                .map(|node| node.base.policy_score + node.base.value_estimate)
                .sum::<f64>()
                / (path.len() as f64 * 2.0);

            let mut stats = self.bidirectional_stats.lock().await;
            stats.path_quality = path_quality;
        }

        Ok(())
    }
}
