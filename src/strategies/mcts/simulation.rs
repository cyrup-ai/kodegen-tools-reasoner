use super::types::{MCTSNode, MonteCarloTreeSearchStrategy, SYNTHETIC_EXPANSION};
use crate::strategies::base::BaseStrategy;
use crate::types::ThoughtNode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

impl MonteCarloTreeSearchStrategy {
    pub async fn run_simulations(
        &self,
        node: MCTSNode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for _ in 0..self.num_simulations {
            let selected_node = self.select(node.clone()).await?;
            let expanded_node = self.expand(selected_node).await?;
            let reward = self.simulate(&expanded_node).await?;
            self.backpropagate(expanded_node, reward).await?;
        }
        Ok(())
    }

    pub async fn select(
        &self,
        node: MCTSNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node;

        while !current.base.children.is_empty()
            && current
                .untried_actions
                .as_ref()
                .is_none_or(|a| a.is_empty())
        {
            let mut children = Vec::new();
            for id in &current.base.children {
                if let Ok(Some(child_node)) = self.base.get_node(id).await {
                    // Convert ThoughtNode to MCTSNode
                    let mcts_child = self.thought_to_mcts(child_node).await?;
                    children.push(mcts_child);
                }
            }

            if children.is_empty() {
                break;
            }

            let parent_visits = current.visits.load(Ordering::Relaxed);
            current = self.select_best_uct(children, parent_visits);
        }

        Ok(current)
    }

    pub async fn expand(
        &self,
        node: MCTSNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        if node.base.is_complete || node.base.depth >= self.simulation_depth {
            return Ok(node);
        }

        // Create a new thought node as expansion
        let new_node_id = Uuid::new_v4().to_string();
        let new_thought = SYNTHETIC_EXPANSION.to_string();
        let mut new_node = ThoughtNode {
            id: new_node_id.clone(),
            thought: new_thought,
            depth: node.base.depth + 1,
            score: 0.0,
            children: vec![],
            parent_id: Some(node.base.id.clone()),
            is_complete: false,
            is_synthetic: true,
        };

        let score = self
            .base
            .evaluate_thought(&new_node, Some(&node.base))
            .await;
        new_node.set_score_or_default(score, 0.5);

        // Save the new node
        if let Err(e) = self.base.save_node(new_node.clone()).await {
            return Err(Box::new(e));
        }

        // Update parent
        let mut parent = node.clone();
        parent.base.children.push(new_node_id.clone());

        // Extract base node and save it
        let parent_base = parent.base.clone();
        if let Err(e) = self.base.save_node(parent_base).await {
            return Err(Box::new(e));
        }

        // Convert to MCTSNode and return (will use registry in thought_to_mcts)
        let mcts_node = self.thought_to_mcts(new_node).await?;

        Ok(mcts_node)
    }

    pub async fn simulate(
        &self,
        node: &MCTSNode,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node.clone();
        let mut total_score = current.base.score;
        let mut depth = current.base.depth;

        while depth < self.simulation_depth && !current.base.is_complete {
            let simulated_node_id = Uuid::new_v4().to_string();
            let simulated_thought = SYNTHETIC_EXPANSION.to_string();
            let mut simulated_node = ThoughtNode {
                id: simulated_node_id,
                thought: simulated_thought,
                depth: depth + 1,
                score: 0.0, // Score will be evaluated below
                children: vec![],
                parent_id: Some(current.base.id.clone()),
                is_complete: depth + 1 >= self.simulation_depth,
                is_synthetic: true,
            };

            let score = self
                .base
                .evaluate_thought(&simulated_node, Some(&current.base))
                .await;
            simulated_node.set_score_or_default(score, 0.5);
            total_score += simulated_node.score;

            // Update current to the simulated node (create new atomic counters)
            current = MCTSNode {
                base: simulated_node,
                visits: Arc::new(AtomicUsize::new(1)),
                total_reward: Arc::new(crate::atomics::AtomicF64::new(0.0)),
                untried_actions: Some(vec![]),
            };

            depth += 1;
        }

        // Calculate steps with protection against edge cases
        let steps = (depth.saturating_sub(node.base.depth) + 1).max(1) as f64;
        Ok(BaseStrategy::safe_divide(total_score, steps, node.base.score))
    }
}
