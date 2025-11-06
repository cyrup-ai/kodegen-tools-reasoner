use crate::state::StateManager;
use crate::strategies::base::BaseStrategy;
use crate::strategies::mcts_002_alpha::MCTS002AlphaStrategy;
use crate::types::ThoughtNode;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::types::{BidirectionalPolicyNode, BidirectionalStats};

pub struct MCTS002AltAlphaStrategy {
    pub(super) base: BaseStrategy,
    pub(super) inner_strategy: MCTS002AlphaStrategy,
    pub(super) start_node: Arc<Mutex<Option<BidirectionalPolicyNode>>>,
    pub(super) goal_node: Arc<Mutex<Option<BidirectionalPolicyNode>>>,
    pub(super) bidirectional_stats: Arc<Mutex<BidirectionalStats>>,
    #[allow(dead_code)]
    pub(super) simulation_count: usize,
}

impl MCTS002AltAlphaStrategy {
    pub fn new(state_manager: Arc<StateManager>, num_simulations: Option<usize>) -> Self {
        let num_simulations = num_simulations.unwrap_or(crate::types::CONFIG.num_simulations);

        let bidirectional_stats = BidirectionalStats {
            forward_exploration_rate: 2.0_f64.sqrt(),
            backward_exploration_rate: 2.0_f64.sqrt(),
            meeting_points: 0,
            path_quality: 0.0,
        };

        Self {
            base: BaseStrategy::new(Arc::clone(&state_manager)),
            inner_strategy: MCTS002AlphaStrategy::new(
                Arc::clone(&state_manager),
                Some(num_simulations),
            ),
            start_node: Arc::new(Mutex::new(None)),
            goal_node: Arc::new(Mutex::new(None)),
            bidirectional_stats: Arc::new(Mutex::new(bidirectional_stats)),
            simulation_count: num_simulations,
        }
    }

    pub(super) fn get_action_key(&self, thought: &str) -> String {
        // Same as the extract_action method in MCTS002AlphaStrategy
        thought
            .split_whitespace()
            .take(3)
            .collect::<Vec<&str>>()
            .join("_")
            .to_lowercase()
    }

    pub(super) fn calculate_bidirectional_policy_score(
        &self,
        path: &[ThoughtNode],
        stats: &BidirectionalStats,
    ) -> f64 {
        if path.is_empty() {
            return 0.0; // No path, no score
        }

        let mut total_score = 0.0;

        for node in path {
            // Use the node's evaluated score
            let node_score = node.score;

            // Add a small bonus based on estimated direction exploration rate
            let direction_bonus = if node.depth % 2 == 0 {
                stats.forward_exploration_rate * 0.1
            } else {
                stats.backward_exploration_rate * 0.1
            };
            total_score += node_score + direction_bonus;
        }

        total_score / path.len() as f64
    }
}

impl Clone for MCTS002AltAlphaStrategy {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            inner_strategy: self.inner_strategy.clone(),
            start_node: Arc::clone(&self.start_node),
            goal_node: Arc::clone(&self.goal_node),
            bidirectional_stats: Arc::clone(&self.bidirectional_stats),
            simulation_count: self.simulation_count,
        }
    }
}
