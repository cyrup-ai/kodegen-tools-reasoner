//! Thought evaluation and scoring logic
//!
//! Contains methods for evaluating thought quality based on logical structure,
//! depth, and semantic coherence.

use super::BaseStrategy;
use super::types::{LOGICAL_CONNECTORS, MATH_EXPRESSIONS};
use crate::types::ThoughtNode;

impl BaseStrategy {
    pub async fn evaluate_thought(&self, node: &ThoughtNode, parent: Option<&ThoughtNode>) -> f64 {
        // Skip expensive evaluation for MCTS-generated synthetic nodes
        if node.is_synthetic {
            return 0.5; // Neutral score - won't bias tree search
        }

        // Base evaluation logic - Semantic coherence is now handled async by strategies
        let logical_score = self.calculate_logical_score(node, parent).await;
        let depth_penalty = self.calculate_depth_penalty(node);
        let completion_bonus = if node.is_complete { 0.2 } else { 0.0 };

        let result = Self::safe_divide(
            logical_score + depth_penalty + completion_bonus,
            3.0,
            0.5,
        );

        if result.is_nan() {
            tracing::error!("evaluate_thought produced NaN for node {}", node.id);
            return 0.5;
        }

        result
    }

    pub(super) async fn calculate_logical_score(
        &self,
        node: &ThoughtNode,
        parent: Option<&ThoughtNode>,
    ) -> f64 {
        let mut score = 0.0;

        // Length and complexity
        score += (node.thought.len() as f64 / 200.0).min(0.3);

        // Logical connectors (compiled once via lazy_static)
        if LOGICAL_CONNECTORS.is_match(&node.thought) {
            score += 0.2;
        }

        // Mathematical/logical expressions (compiled once via lazy_static)
        if MATH_EXPRESSIONS.is_match(&node.thought) {
            score += 0.2;
        }

        // Parent-child semantic coherence using Stella embeddings
        if let Some(parent_node) = parent {
            let coherence = self
                .calculate_semantic_coherence(&parent_node.thought, &node.thought)
                .await
                .unwrap_or(0.5); // Fallback only on embedding error
            score += coherence * 0.3; // CORRECT WEIGHT: 0.3 (matches TypeScript)
        }

        // Ensure score is within a reasonable range (e.g., 0 to 1) before returning
        let result = score.clamp(0.0, 1.0);

        if result.is_nan() {
            tracing::error!("calculate_logical_score produced NaN for node {}", node.id);
            return 0.5;
        }

        result
    }

    pub(super) fn calculate_depth_penalty(&self, node: &ThoughtNode) -> f64 {
        // Penalize deeper thoughts slightly less aggressively
        (1.0 - (node.depth as f64 / crate::types::CONFIG.max_depth as f64) * 0.2).max(0.0)
    }
}
