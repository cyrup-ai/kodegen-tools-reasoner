use super::types::{PolicyGuidedNode, PolicyMetrics, ExplorationStats, ConvergenceMetrics};
use crate::strategies::base::ReasoningError;
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct MetricsTracker {
    pub policy_metrics: Arc<Mutex<PolicyMetrics>>,
    pub exploration_rate: Arc<Mutex<f64>>,
    pub temperature: f64,
    pub learning_rate: f64,
    pub novelty_bonus: f64,
}

impl MetricsTracker {
    pub fn new(
        temperature: f64,
        exploration_rate: f64,
        learning_rate: f64,
        novelty_bonus: f64,
    ) -> Self {
        let policy_metrics = PolicyMetrics::new(temperature, exploration_rate, novelty_bonus);

        Self {
            policy_metrics: Arc::new(Mutex::new(policy_metrics)),
            exploration_rate: Arc::new(Mutex::new(exploration_rate)),
            temperature,
            learning_rate,
            novelty_bonus,
        }
    }

    // Extract an action-like identifier from a thought
    fn extract_action(&self, thought: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        thought.hash(&mut hasher);
        hasher.finish()
    }

    pub async fn update_policy_metrics(
        &self,
        node: &PolicyGuidedNode,
        parent: &PolicyGuidedNode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut metrics = self.policy_metrics.lock().await;

        // Guard against NaN propagation (should never occur after embedding validation)
        if node.policy_score.is_nan() || node.value_estimate.is_nan() {
            return Err(Box::new(ReasoningError::Fatal(format!(
                "Metric update failed: node {} has NaN values (policy={}, value={}). \
                 Root cause: likely corrupted embedding. Check base.rs embedding validation.",
                node.base.id, node.policy_score, node.value_estimate
            ))));
        }

        // True running average: avg_new = (avg_old × n + value) / (n + 1)
        let n = metrics.sample_count as f64;
        metrics.average_policy_score = 
            (metrics.average_policy_score * n + node.policy_score) / (n + 1.0);
        metrics.average_value_estimate = 
            (metrics.average_value_estimate * n + node.value_estimate) / (n + 1.0);
        metrics.sample_count += 1;

        // Update action distribution
        let action = self.extract_action(&node.base.thought);
        let count = metrics.action_distribution.entry(action).or_insert(0);
        *count += 1;

        // Update exploration stats
        let exploration_rate = *self.exploration_rate.lock().await; // Read the current rate
        metrics.exploration_stats = ExplorationStats {
            temperature: self.temperature,
            exploration_rate,
            novelty_bonus: self.novelty_bonus,
        };

        // Calculate policy entropy and value stability
        let probs: Vec<f64> = parent.prior_action_probs.values().copied().collect();
        metrics.convergence_metrics = ConvergenceMetrics {
            policy_entropy: self.calculate_entropy(&probs),
            value_stability: (node.value_estimate - parent.value_estimate).abs(),
        };

        Ok(())
    }

    fn calculate_entropy(&self, probs: &[f64]) -> f64 {
        let sum: f64 = probs.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }

        -probs
            .iter()
            .map(|&p| {
                let norm = p / sum;
                if norm <= 0.0 {
                    0.0
                } else {
                    norm * (norm + 1e-10).log2()
                }
            })
            .sum::<f64>()
    }

    pub async fn adapt_exploration_rate(&self, node: &PolicyGuidedNode) {
        let success_rate = node.total_reward / node.visits as f64;
        let target_rate = 0.6;

        let mut exploration_rate = self.exploration_rate.lock().await;
        if success_rate > target_rate {
            // Reduce exploration when doing well
            *exploration_rate = (0.5f64).max(*exploration_rate * 0.95);
        } else {
            // Increase exploration when results are poor
            *exploration_rate = (2.0f64).min(*exploration_rate / 0.95);
        }
    }

    /// Reset all metrics to default values
    ///
    /// Called from Strategy::clear() (mod.rs:322) which is invoked via trait objects.
    /// Rust's dead code analysis cannot trace trait object method calls.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-11-07
    #[allow(dead_code)]
    pub async fn reset_metrics(&self) {
        let exploration_rate_val = *self.exploration_rate.lock().await;

        let mut metrics = self.policy_metrics.lock().await;
        *metrics = PolicyMetrics {
            average_policy_score: 0.0,
            average_value_estimate: 0.0,
            action_distribution: HashMap::new(),
            exploration_stats: ExplorationStats {
                temperature: self.temperature,
                exploration_rate: exploration_rate_val,
                novelty_bonus: self.novelty_bonus,
            },
            convergence_metrics: ConvergenceMetrics {
                policy_entropy: 0.0,
                value_stability: 0.0,
            },
            sample_count: 0,
        };

        // Reset exploration rate
        let mut exploration_rate = self.exploration_rate.lock().await;
        *exploration_rate = 2.0_f64.sqrt();
    }
}
