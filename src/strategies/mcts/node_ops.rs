use super::types::{MCTSNode, MonteCarloTreeSearchStrategy};
use crate::atomics::AtomicF64;
use crate::types::ThoughtNode;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

impl MonteCarloTreeSearchStrategy {
    /// Convert a ThoughtNode to MCTSNode, using shared atomic statistics from registry
    pub async fn thought_to_mcts(
        &self,
        node: ThoughtNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        let node_id = node.id.clone();
        
        // Get or create shared atomic statistics for this node
        let (visits, total_reward) = {
            let mut registry = self.stats_registry.lock().await;
            registry.entry(node_id.clone())
                .or_insert_with(|| {
                    // First time seeing this node - initialize with 1 visit
                    (Arc::new(AtomicUsize::new(1)), Arc::new(AtomicF64::new(0.0)))
                })
                .clone()  // Clone the Arc, not the atomics
        };
        
        Ok(MCTSNode {
            base: node,
            visits,
            total_reward,
            untried_actions: Some(vec![]),
        })
    }
}
