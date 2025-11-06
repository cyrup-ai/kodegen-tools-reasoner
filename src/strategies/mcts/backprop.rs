use super::types::{MCTSNode, MonteCarloTreeSearchStrategy};
use crate::atomics::AtomicF64;
use crate::types::ThoughtNode;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing;

impl MonteCarloTreeSearchStrategy {
    pub async fn backpropagate(
        &self,
        node: MCTSNode,
        reward: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // NO LOCK NEEDED - atomics handle concurrency
        
        // Load entire path into memory
        let mut path = vec![node.clone()];
        let mut current_id = node.base.parent_id.clone();
        let mut visited = HashSet::new();
        visited.insert(node.base.id.clone());
        
        // Pre-load all ancestors
        while let Some(parent_id) = current_id {
            if !visited.insert(parent_id.clone()) {
                tracing::error!(
                    "Cycle detected during backpropagation at node {}",
                    parent_id
                );
                break;
            }
            
            if let Ok(Some(parent_node)) = self.base.get_node(&parent_id).await {
                let mcts_parent = self.thought_to_mcts(parent_node.clone()).await?;
                current_id = parent_node.parent_id.clone();
                path.push(mcts_parent);
            } else {
                break;
            }
        }
        
        // Update all nodes atomically - lock-free!
        for node in path {
            // Atomic increment - thread-safe, no races
            node.visits.fetch_add(1, Ordering::Relaxed);
            node.total_reward.fetch_add(reward, Ordering::Relaxed);
            
            // Note: We still save ThoughtNode, but without MCTS stats
            // The stats live in memory via stats_registry
            let updated_node = ThoughtNode {
                id: node.base.id.clone(),
                thought: node.base.thought.clone(),
                score: node.base.score,
                depth: node.base.depth,
                children: node.base.children.clone(),
                parent_id: node.base.parent_id.clone(),
                is_complete: node.base.is_complete,
                is_synthetic: node.base.is_synthetic,
            };
            
            if let Err(e) = self.base.save_node(updated_node).await {
                return Err(Box::new(e));
            }
        }
        
        Ok(())
    }

    pub fn select_best_uct(&self, nodes: Vec<MCTSNode>, parent_visits: usize) -> MCTSNode {
        nodes
            .into_iter()
            .fold(None, |best: Option<MCTSNode>, node| {
                // Load atomic values
                let visits = node.visits.load(Ordering::Relaxed);
                let total_reward = node.total_reward.load(Ordering::Relaxed);
                
                // Guard for unvisited nodes
                if visits == 0 {
                    tracing::warn!(
                        "Skipping node {} with 0 visits in UCT selection",
                        node.base.id
                    );
                    return best;
                }
                
                // Calculate UCT score
                let exploitation = total_reward / visits as f64;
                let exploration = ((parent_visits as f64).ln() / visits as f64).sqrt();
                let uct = exploitation + self.exploration_constant * exploration;

                match best {
                    None => Some(node),
                    Some(best_node) => {
                        let best_visits = best_node.visits.load(Ordering::Relaxed);
                        if best_visits == 0 {
                            return Some(node);
                        }
                        
                        let best_total_reward = best_node.total_reward.load(Ordering::Relaxed);
                        let best_exploitation = best_total_reward / best_visits as f64;
                        let best_exploration =
                            ((parent_visits as f64).ln() / best_visits as f64).sqrt();
                        let best_uct =
                            best_exploitation + self.exploration_constant * best_exploration;

                        if uct > best_uct {
                            Some(node)
                        } else {
                            Some(best_node)
                        }
                    }
                }
            })
            .unwrap_or_else(|| {
                // If no nodes provided, return a default MCTSNode
                MCTSNode {
                    base: ThoughtNode {
                        id: "default".to_string(),
                        thought: "Default selection".to_string(),
                        depth: 0,
                        score: 0.0,
                        children: vec![],
                        parent_id: None,
                        is_complete: false,
                        is_synthetic: true,
                    },
                    visits: Arc::new(AtomicUsize::new(1)),
                    total_reward: Arc::new(AtomicF64::new(0.0)),
                    untried_actions: Some(vec![]),
                }
            })
    }
}
