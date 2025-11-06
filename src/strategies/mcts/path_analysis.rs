use super::types::{MCTSNode, MonteCarloTreeSearchStrategy};
use crate::types::ThoughtNode;
use std::collections::{HashSet, VecDeque};
use tracing;

impl MonteCarloTreeSearchStrategy {
    pub fn calculate_path_score(&self, path: &[ThoughtNode]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        path.iter().map(|node| node.score).sum::<f64>() / path.len() as f64
    }

    pub async fn calculate_possible_paths(&self, node: &MCTSNode) -> usize {
        // Count actual paths explored down to simulation depth
        self.count_paths_iterative(&node.base.id, node.base.depth)
            .await
    }

    pub async fn count_paths_iterative(&self, node_id: &str, start_depth: usize) -> usize {
        // Check cache first (pattern from mcts_002_alpha.rs:145-165)
        {
            let cache = self.path_count_cache.lock().await;
            if let Some(&count) = cache.get(node_id) {
                return count;
            }
        }

        // Iterative BFS using VecDeque (pattern from mcts_002alt_alpha.rs:15-40)
        let mut queue = VecDeque::new();
        queue.push_back((node_id.to_string(), start_depth));
        
        let mut path_count = 0;
        
        // Cycle detection (pattern from state.rs:82-87)
        let mut visited = HashSet::new();
        
        // Iteration limit for safety (pattern from mcts_002alt_alpha.rs:359-379)
        const MAX_PATH_COUNT: usize = 1_000_000;
        let mut iterations = 0;
        
        while let Some((current_id, depth)) = queue.pop_front() {
            // Protection 1: Iteration limit
            iterations += 1;
            if iterations > MAX_PATH_COUNT {
                tracing::error!(
                    "Path counting exceeded {} iterations at node {} (possible infinite tree)",
                    MAX_PATH_COUNT,
                    current_id
                );
                break;
            }
            
            // Max depth check
            if depth >= self.simulation_depth {
                path_count += 1;
                continue;
            }
            
            // Protection 2: Cycle detection
            if !visited.insert(current_id.clone()) {
                tracing::warn!("Cycle detected during path counting at node {}", current_id);
                continue;
            }
            
            // Get children (already async, no recursion needed)
            let children = self.base.state_manager.get_children(&current_id).await;
            
            if children.is_empty() {
                path_count += 1;
            } else {
                for child in children {
                    if child.depth < self.simulation_depth {
                        queue.push_back((child.id.clone(), child.depth));
                    } else {
                        path_count += 1;
                    }
                }
            }
        }
        
        // Cache result (pattern from mcts_002_alpha.rs:160-165)
        {
            let mut cache = self.path_count_cache.lock().await;
            cache.insert(node_id.to_string(), path_count);
        }
        
        path_count
    }
}
