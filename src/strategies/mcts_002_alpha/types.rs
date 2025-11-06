use crate::types::ThoughtNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Synthetic thoughts are never evaluated (see base.rs:402-404)
/// so we use a constant to avoid allocation overhead
pub const SYNTHETIC_EXPANSION: &str = "[synthetic]";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyGuidedNode {
    #[serde(flatten)]
    pub base: ThoughtNode,
    pub visits: usize,
    #[serde(rename = "totalReward")]
    pub total_reward: f64,
    #[serde(rename = "untriedActions")]
    pub untried_actions: Option<Vec<String>>,
    #[serde(rename = "policyScore")]
    pub policy_score: f64, // Policy network prediction
    #[serde(rename = "valueEstimate")]
    pub value_estimate: f64, // Value network estimate
    #[serde(rename = "priorActionProbs")]
    pub prior_action_probs: HashMap<String, f64>, // Action probabilities
    pub puct: Option<f64>, // PUCT score for selection
    #[serde(rename = "actionHistory")]
    pub action_history: Option<Vec<String>>, // Track sequence of actions
    #[serde(rename = "noveltyScore")]
    pub novelty_score: Option<f64>, // Measure of thought novelty
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetrics {
    #[serde(rename = "averagePolicyScore")]
    pub average_policy_score: f64,
    #[serde(rename = "averageValueEstimate")]
    pub average_value_estimate: f64,
    #[serde(rename = "actionDistribution")]
    pub action_distribution: HashMap<u64, usize>,
    #[serde(rename = "explorationStats")]
    pub exploration_stats: ExplorationStats,
    #[serde(rename = "convergenceMetrics")]
    pub convergence_metrics: ConvergenceMetrics,
    #[serde(rename = "sampleCount")]
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStats {
    pub temperature: f64,
    #[serde(rename = "explorationRate")]
    pub exploration_rate: f64,
    #[serde(rename = "noveltyBonus")]
    pub novelty_bonus: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    #[serde(rename = "policyEntropy")]
    pub policy_entropy: f64,
    #[serde(rename = "valueStability")]
    pub value_stability: f64,
}

impl PolicyGuidedNode {
    /// Creates a default PolicyGuidedNode for fallback scenarios
    pub fn default_node() -> Self {
        Self {
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
            visits: 1,
            total_reward: 0.0,
            untried_actions: Some(vec![]),
            policy_score: 0.0,
            value_estimate: 0.0,
            prior_action_probs: HashMap::new(),
            puct: None,
            action_history: Some(vec![]),
            novelty_score: Some(0.0),
        }
    }
}

impl PolicyMetrics {
    /// Creates a new PolicyMetrics instance with default values
    pub fn new(temperature: f64, exploration_rate: f64, novelty_bonus: f64) -> Self {
        Self {
            average_policy_score: 0.0,
            average_value_estimate: 0.0,
            action_distribution: HashMap::new(),
            exploration_stats: ExplorationStats {
                temperature,
                exploration_rate,
                novelty_bonus,
            },
            convergence_metrics: ConvergenceMetrics {
                policy_entropy: 0.0,
                value_stability: 0.0,
            },
            sample_count: 0,
        }
    }
}
