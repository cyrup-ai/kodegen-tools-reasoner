use crate::state::StateManager;
use crate::strategies::base::{
    AsyncPath, BaseStrategy, ClearedSignal, Metric, MetricStream, Reasoning, ReasoningError, Strategy,
    NOVELTY_MARKERS,
};
use crate::strategies::mcts::MonteCarloTreeSearchStrategy;
use crate::types::{ReasoningRequest, ReasoningResponse, ThoughtNode};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot};
// Tracing is imported for error logging in case of future extensions
#[allow(unused_imports)]
use tracing;
use uuid::Uuid;

/// Synthetic thoughts are never evaluated (see base.rs:402-404)
/// so we use a constant to avoid allocation overhead
const SYNTHETIC_EXPANSION: &str = "[synthetic]";

// Note: Text embedding functionality requires VoyageAI API (VOYAGE_API_KEY env var)
// Removed WASM host function - native implementation uses BaseStrategy methods

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

pub struct MCTS002AlphaStrategy {
    base: BaseStrategy,
    inner_mcts: Arc<MonteCarloTreeSearchStrategy>,
    temperature: f64,
    exploration_rate: Arc<Mutex<f64>>,
    learning_rate: f64,
    novelty_bonus: f64,
    policy_metrics: Arc<Mutex<PolicyMetrics>>,
    simulation_count: usize,

    // Cache for semantic coherence scores (avoids redundant similarity calculations)
    coherence_cache: Arc<Mutex<HashMap<String, f64>>>,

    // Lock to serialize backpropagation and prevent race conditions
    backprop_lock: Arc<Mutex<()>>,
}

impl MCTS002AlphaStrategy {
    pub fn new(state_manager: Arc<StateManager>, num_simulations: Option<usize>) -> Self {
        let num_simulations = num_simulations.unwrap_or(crate::types::CONFIG.num_simulations);

        let policy_metrics = PolicyMetrics {
            average_policy_score: 0.0,
            average_value_estimate: 0.0,
            action_distribution: HashMap::new(),
            exploration_stats: ExplorationStats {
                temperature: 1.0,
                exploration_rate: 2.0_f64.sqrt(),
                novelty_bonus: 0.2,
            },
            convergence_metrics: ConvergenceMetrics {
                policy_entropy: 0.0,
                value_stability: 0.0,
            },
            sample_count: 0,
        };

        Self {
            base: BaseStrategy::new(Arc::clone(&state_manager)),
            inner_mcts: Arc::new(MonteCarloTreeSearchStrategy::new(
                Arc::clone(&state_manager),
                Some(num_simulations),
            )),
            temperature: 1.0,
            exploration_rate: Arc::new(Mutex::new(2.0_f64.sqrt())),
            learning_rate: 0.1,
            novelty_bonus: 0.2,
            policy_metrics: Arc::new(Mutex::new(policy_metrics)),
            simulation_count: num_simulations,

            // Cache for semantic coherence scores
            coherence_cache: Arc::new(Mutex::new(HashMap::new())),

            // Initialize lock for backpropagation serialization
            backprop_lock: Arc::new(Mutex::new(())),
        }
    }

    // Renamed for clarity, now uses hashing for better uniqueness representation
    pub fn get_thought_identifier(&self, thought: &str) -> String {
        // Use a hash of the thought content as a more robust identifier
        // than just the first few words.
        let mut hasher = DefaultHasher::new();
        thought.hash(&mut hasher);
        hasher.finish().to_string()
    }

    /// Creates a collision-resistant, symmetric cache key for thought pairs.
    /// Uses hash-based key generation to prevent:
    /// 1. Delimiter collisions (thoughts containing "||")
    /// 2. Asymmetry waste (coherence(A,B) = coherence(B,A))
    /// 3. Memory bloat from long thoughts
    ///
    /// Returns: String representation of u64 hash (8 bytes as string)
    fn create_coherence_cache_key(thought1: &str, thought2: &str) -> String {
        // Sort thoughts lexicographically to ensure symmetry
        // coherence(A, B) must have same key as coherence(B, A)
        let (first, second) = if thought1 < thought2 {
            (thought1, thought2)
        } else {
            (thought2, thought1)
        };
        
        // Hash both thoughts with separator to prevent concatenation collisions
        // Example: hash("AB", "C") must differ from hash("A", "BC")
        let mut hasher = DefaultHasher::new();
        first.hash(&mut hasher);
        0u8.hash(&mut hasher);  // Separator prevents "AB"+"C" = "A"+"BC"
        second.hash(&mut hasher);
        
        hasher.finish().to_string()
    }

    /// Compute semantic coherence using VoyageAI embeddings (via BaseStrategy)
    async fn thought_coherence(
        &self,
        thought1: &str,
        thought2: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Create collision-resistant, symmetric cache key
        let cache_key = Self::create_coherence_cache_key(thought1, thought2);

        // Check cache first
        {
            let cache = self.coherence_cache.lock().await;
            if let Some(&score) = cache.get(&cache_key) {
                return Ok(score);
            }
        }

        // Cache miss - call BaseStrategy method
        let score = self
            .base
            .calculate_semantic_coherence(thought1, thought2)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Store in cache
        let mut cache = self.coherence_cache.lock().await;
        cache.insert(cache_key, score);

        Ok(score)
    }

    // Async policy score calculation with semantic coherence
    pub async fn calculate_policy_score(
        &self,
        node: &PolicyGuidedNode,
        parent: Option<&PolicyGuidedNode>,
    ) -> f64 {
        // Combine multiple policy factors
        let depth_factor = (-0.1 * node.base.depth as f64).exp();

        // Use semantic coherence from VoyageAI embeddings
        let parent_alignment = if let Some(p) = parent {
            self.thought_coherence(&p.base.thought, &node.base.thought)
                .await
                .unwrap_or(0.5) // Fallback to neutral on error
        } else {
            0.5 // No parent, neutral alignment
        };

        // Combine factors: depth, semantic coherence, policy score
        let combined = depth_factor * 0.3 + parent_alignment * 0.4 + node.policy_score * 0.3;
        let result = combined.clamp(0.0, 1.0);
        
        if result.is_nan() {
            tracing::error!(
                "calculate_policy_score produced NaN for node {}",
                node.base.id
            );
            return 0.5;
        }
        
        result
    }

    // Now async (though not strictly needed if policy_score already calculated coherence)
    pub async fn estimate_value(&self, node: &PolicyGuidedNode) -> f64 {
        let weights = &crate::types::VALUE_WEIGHTS;

        // 1. Immediate value
        let immediate = node.base.score;

        // 2. Completion potential
        let completion = if node.base.is_complete {
            1.0
        } else {
            0.5 + 0.5 * (1.0 - node.base.depth as f64 / crate::types::CONFIG.max_depth as f64)
        };

        // 3. Goal coherence
        let coherence = self.calculate_goal_alignment(node).await;

        // 4. Novelty value
        let novelty = node.novelty_score.unwrap_or(0.0);

        let result = BaseStrategy::safe_weighted_sum(&[
            (immediate, weights.immediate_weight),
            (completion, weights.completion_weight),
            (coherence, weights.coherence_weight),
            (novelty, weights.novelty_weight),
        ]);
        
        if result.is_nan() {
            tracing::error!("estimate_value produced NaN for node {}", node.base.id);
            return 0.5;
        }
        
        result.clamp(0.0, 1.0)
    }

    pub fn calculate_novelty(&self, node: &PolicyGuidedNode) -> f64 {
        // Measure thought novelty based on thought identifier history
        let thought_history = match &node.action_history {
            // Reusing action_history field for thought identifiers
            Some(history) => history,
            None => return 0.0, // No history, no novelty score
        };

        if thought_history.is_empty() {
            return 0.0;
        }

        let unique_thoughts = thought_history.iter().collect::<HashSet<_>>().len();
        let history_length = thought_history.len();
        // Avoid division by zero if history_length is somehow 0 despite check
        let uniqueness_ratio = if history_length > 0 {
            unique_thoughts as f64 / history_length as f64
        } else {
            0.0
        };

        // Combine with linguistic novelty
        let complexity_score = NOVELTY_MARKERS
            .find_iter(&node.base.thought)
            .count() as f64 / 10.0; // Simple complexity heuristic

        // Weights are heuristic
        0.7 * uniqueness_ratio + 0.3 * complexity_score
    }

    pub fn calculate_novelty_v2(&self, node: &PolicyGuidedNode) -> f64 {
        // Reuse existing uniqueness calculation (0.5 weight)
        let thought_history = match &node.action_history {
            Some(history) => history,
            None => return 0.0,
        };
        
        let unique_thoughts = thought_history.iter().collect::<HashSet<_>>().len();
        let uniqueness_ratio = unique_thoughts as f64 / thought_history.len().max(1) as f64;

        // Add lexical diversity (0.3 weight)
        let words: Vec<_> = node.base.thought.split_whitespace().collect();
        let unique_words: HashSet<_> = words.iter().collect();
        let lexical_diversity = unique_words.len() as f64 / words.len().max(1) as f64;

        // Add reasoning depth (0.2 weight)
        let reasoning_markers = ["therefore", "because", "however", "although",
                                 "if", "then", "consequently", "thus", "hence"];
        let thought_lower = node.base.thought.to_lowercase();
        let reasoning_depth = reasoning_markers.iter()
            .filter(|&&m| thought_lower.contains(m))
            .count() as f64 / 5.0;

        let result = 0.5 * uniqueness_ratio + 0.3 * lexical_diversity + 0.2 * reasoning_depth.min(1.0);
        
        if result.is_nan() {
            tracing::warn!("calculate_novelty_v2 produced NaN for node {}", node.base.id);
            return 0.0;
        }
        
        result
    }

    async fn calculate_goal_alignment(&self, node: &PolicyGuidedNode) -> f64 {
        // Get path to root using StateManager.get_path()
        let root_path = self.base.state_manager.get_path(&node.base.id).await;
        if let Some(root) = root_path.first() {
            // Use existing semantic coherence
            self.thought_coherence(&root.thought, &node.base.thought)
                .await
                .unwrap_or(0.5)
        } else {
            0.5  // No root, neutral alignment
        }
    }

    async fn run_policy_guided_search(
        &self,
        node: PolicyGuidedNode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for _ in 0..self.simulation_count {
            let selected_node = self.select_with_puct(node.clone()).await?;
            let expanded_node = self.expand_with_policy(selected_node).await?;
            let reward = self.simulate_with_value_guidance(&expanded_node).await?;
            let expanded_node_clone = expanded_node.clone(); // Clone before moving
            self.backpropagate_with_policy_update(expanded_node, reward)
                .await?;

            // Adapt exploration rate
            self.adapt_exploration_rate(&expanded_node_clone).await;
        }

        Ok(())
    }

    async fn select_with_puct(
        &self,
        root: PolicyGuidedNode,
    ) -> Result<PolicyGuidedNode, Box<dyn std::error::Error + Send + Sync>> {
        let mut node = root;

        while !node.base.children.is_empty() {
            let mut children = Vec::new();
            for id in &node.base.children {
                if let Ok(Some(child_node)) = self.base.get_node(id).await
                    && let Ok(policy_child) = self.thought_to_policy(child_node).await
                {
                    children.push(policy_child);
                }
            }

            if children.is_empty() {
                break;
            }

            node = self.select_best_puct_child(children, node.visits).await;
        }

        Ok(node)
    }

    async fn select_best_puct_child(&self, nodes: Vec<PolicyGuidedNode>, parent_visits: usize) -> PolicyGuidedNode {
        nodes
            .into_iter()
            .map(|mut node| {
                // Unvisited nodes get infinite priority
                if node.visits == 0 {
                    return (f64::INFINITY, node);
                }

                // Correct exploitation: Q/N
                let q_over_n = node.total_reward / node.visits as f64;

                let score = if crate::types::CONFIG.use_puct_formula {
                    // PUCT: Q/N + c*P*sqrt(N_parent)/(1+N)
                    let c = crate::types::CONFIG.puct_exploration_constant;
                    q_over_n + c * node.policy_score * (parent_visits as f64).sqrt() / (1.0 + node.visits as f64)
                } else {
                    // UCB1: Q/N + c*sqrt(ln(N_parent)/N)
                    let c = crate::types::CONFIG.ucb_exploration_constant;
                    q_over_n + c * ((parent_visits as f64).ln() / node.visits as f64).sqrt()
                };

                node.puct = Some(score);
                (score, node)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, node)| node)
            .unwrap_or_else(|| {
                // If no nodes provided, return a default PolicyGuidedNode
                PolicyGuidedNode {
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
            })
    }

    async fn expand_with_policy(
        &self,
        node: PolicyGuidedNode,
    ) -> Result<PolicyGuidedNode, Box<dyn std::error::Error + Send + Sync>> {
        if node.base.is_complete {
            return Ok(node);
        }

        let new_node_id = Uuid::new_v4().to_string();
        let new_thought = SYNTHETIC_EXPANSION.to_string();

        let base_node = ThoughtNode {
            id: new_node_id.clone(),
            thought: new_thought.clone(),
            depth: node.base.depth + 1,
            score: 0.0, // Will be evaluated later
            children: vec![],
            parent_id: Some(node.base.id.clone()),
            is_complete: false,
            is_synthetic: true,
        };

        // Update history with the new thought's identifier
        let thought_identifier = self.get_thought_identifier(&new_thought);
        let action_history = match &node.action_history {
            // Reusing field name
            Some(history) => {
                let mut new_history = history.clone();
                new_history.push(thought_identifier);
                Some(new_history)
            }
            None => Some(vec![thought_identifier]),
        };

        let mut new_node = PolicyGuidedNode {
            base: base_node.clone(),
            visits: 1,
            total_reward: 0.0,
            untried_actions: Some(vec![]),
            policy_score: 0.0,
            value_estimate: 0.0,
            prior_action_probs: HashMap::new(),
            puct: None,
            action_history,
            novelty_score: None,
        };

        new_node.novelty_score = Some(self.calculate_novelty_v2(&new_node));
        // Calculate policy score with semantic coherence
        new_node.policy_score = self.calculate_policy_score(&new_node, Some(&node)).await;
        let score = self
            .base
            .evaluate_thought(&new_node.base, Some(&node.base))
            .await;
        new_node.base.set_score_or_default(score, 0.5);
        // Await the async calculation
        new_node.value_estimate = self.estimate_value(&new_node).await;

        // Save the base node
        if let Err(e) = self.base.save_node_with_retry(base_node, None).await {
            tracing::error!("Failed to save expanded node: {}", e);
            return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
        }

        Ok(new_node)
    }

    async fn simulate_with_value_guidance(
        &self,
        node: &PolicyGuidedNode,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node.clone();
        let mut total_reward = 0.0;
        let mut depth = 0;

        while !current.base.is_complete && depth < crate::types::CONFIG.max_depth {
            // Await the async value estimate
            let reward = self.estimate_value(&current).await;
            total_reward += reward;

            // Expansion uses heuristic generation
            if let Ok(expanded) = self.expand_with_policy(current).await {
                current = expanded;
                depth += 1;
            } else {
                break;
            }
        }

        if depth == 0 {
            return Ok(node.value_estimate);
        }

        Ok(total_reward / depth as f64)
    }

    async fn backpropagate_with_policy_update(
        &self,
        node: PolicyGuidedNode,
        reward: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // STEP 1: Acquire lock
        let _guard = self.backprop_lock.lock().await;
        
        // STEP 2: Pre-load entire path
        let mut path = vec![node.clone()];
        let mut current_id = node.base.parent_id.clone();
        let mut visited = HashSet::new();
        visited.insert(node.base.id.clone());
        
        while let Some(parent_id) = current_id {
            if !visited.insert(parent_id.clone()) {
                tracing::error!(
                    "Cycle detected during policy backpropagation at node {}",
                    parent_id
                );
                break;
            }
            
            if let Ok(Some(parent_node)) = self.base.get_node(&parent_id).await {
                if let Ok(policy_parent) = self.thought_to_policy(parent_node.clone()).await {
                    current_id = parent_node.parent_id.clone();
                    path.push(policy_parent);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        // STEP 3: Apply updates to entire path
        for i in 0..path.len() {
            let mut current = path[i].clone();
            
            // Update node stats
            current.visits += 1;
            current.total_reward += reward;
            
            // Update value estimate with temporal difference
            let _current_value_estimate = self.estimate_value(&current).await;
            let new_value = (1.0 - self.learning_rate) * current.value_estimate
                + self.learning_rate * reward;
            current.value_estimate = new_value;
            
            // Update parent's action probabilities if not root
            if i + 1 < path.len() {
                let mut parent = path[i + 1].clone();
                let thought_key = self.get_thought_identifier(&current.base.thought);
                let current_prob = *parent.prior_action_probs
                    .get(&thought_key)
                    .unwrap_or(&0.0);
                let new_prob = current_prob + self.learning_rate * (reward - current_prob);
                parent.prior_action_probs.insert(thought_key, new_prob);
                
                // Save updated parent (with retry for transient failures)
                if let Err(e) = self.base.save_node_with_retry(parent.base.clone(), None).await {
                    tracing::warn!(
                        "Failed to save parent node {} during backprop: {}",
                        parent.base.id,
                        e
                    );
                    // Continue - other updates still valuable
                }
                
                // Update parent in path for next iteration
                path[i + 1] = parent;
            }
            
            // Save current node (with retry)
            if let Err(e) = self.base.save_node_with_retry(current.base.clone(), None).await {
                tracing::warn!(
                    "Failed to save node {} during backprop: {}",
                    current.base.id,
                    e
                );
            }
            
            // Update current in path
            path[i] = current;
        }
        
        Ok(())
    }

    async fn adapt_exploration_rate(&self, node: &PolicyGuidedNode) {
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

    // Extract an action-like identifier from a thought
    fn extract_action(&self, thought: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        thought.hash(&mut hasher);
        hasher.finish()
    }

    async fn update_policy_metrics(
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

    async fn calculate_policy_enhanced_score(&self, path: &[ThoughtNode]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut parent_policy_node: Option<PolicyGuidedNode> = None;

        for node in path {
            // Convert current node to PolicyGuidedNode (initially with defaults)
            let mut policy_node = match self.thought_to_policy(node.clone()).await {
                Ok(pn) => pn,
                Err(_) => {
                    // Fallback to base score if conversion fails
                    total_score += node.score;
                    continue;
                }
            };

            // Calculate ACTUAL scores using existing methods
            policy_node.policy_score = self
                .calculate_policy_score(&policy_node, parent_policy_node.as_ref())
                .await;
            
            policy_node.value_estimate = self.estimate_value(&policy_node).await;
            
            policy_node.novelty_score = Some(self.calculate_novelty(&policy_node));

            // Use weighted sum with CONFIG weights (NOT simple /4.0 average)
            let base_score = node.score;
            let weighted_score = 
                crate::types::CONFIG.base_score_weight * base_score +
                crate::types::CONFIG.policy_weight * policy_node.policy_score +
                crate::types::CONFIG.value_weight * policy_node.value_estimate +
                crate::types::CONFIG.novelty_weight * policy_node.novelty_score.unwrap_or(0.0);

            total_score += weighted_score;
            
            // Update parent reference for next iteration
            parent_policy_node = Some(policy_node);
        }

        total_score / path.len() as f64
    }

    pub async fn thought_to_policy(
        &self,
        node: ThoughtNode,
    ) -> Result<PolicyGuidedNode, Box<dyn std::error::Error + Send + Sync>> {
        // Convert ThoughtNode to PolicyGuidedNode with default initial values
        Ok(PolicyGuidedNode {
            base: node,
            visits: 1,
            total_reward: 0.0,
            untried_actions: Some(vec![]),
            policy_score: 0.5,   // Default value
            value_estimate: 0.5, // Default value
            prior_action_probs: HashMap::new(),
            puct: None,
            action_history: Some(vec![]),
            novelty_score: Some(0.5), // Default value
        })
    }
}

// Add Clone implementation for MCTS002AlphaStrategy
impl Clone for MCTS002AlphaStrategy {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            inner_mcts: Arc::clone(&self.inner_mcts),
            temperature: self.temperature,
            exploration_rate: Arc::clone(&self.exploration_rate),
            learning_rate: self.learning_rate,
            novelty_bonus: self.novelty_bonus,
            policy_metrics: Arc::clone(&self.policy_metrics),
            simulation_count: self.simulation_count,
            coherence_cache: Arc::clone(&self.coherence_cache),
            backprop_lock: Arc::clone(&self.backprop_lock),
        }
    }
}

impl Strategy for MCTS002AlphaStrategy {
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning {
        let (tx, rx) = mpsc::channel(1);
        let self_clone = self.clone();

        tokio::spawn(async move {
            // Use persistent inner_mcts field
            let mut mcts_reasoning = self_clone.inner_mcts.process_thought(request.clone());
            let base_response = match mcts_reasoning.next().await {
                Some(Ok(response)) => response,
                _ => {
                    let _ = tx
                        .send(Err(crate::strategies::base::ReasoningError::Other(
                            "Failed to get base MCTS response".into(),
                        )))
                        .await;
                    return;
                }
            };

            let node_id = Uuid::new_v4().to_string();
            let parent_node = match &request.parent_id {
                Some(parent_id) => {
                    if let Ok(Some(node)) = self_clone.base.get_node(parent_id).await {
                        (self_clone.thought_to_policy(node).await).ok()
                    } else {
                        None
                    }
                }
                None => None,
            };

            let mut base_node = ThoughtNode {
                id: node_id.clone(),
                thought: request.thought.clone(),
                depth: request.thought_number - 1,
                score: 0.0,
                children: vec![],
                parent_id: request.parent_id.clone(),
                is_complete: !request.next_thought_needed,
                is_synthetic: false,
            };

            // Create thought identifier history from parent or initialize new one
            let thought_identifier = self_clone.get_thought_identifier(&request.thought);
            let action_history = match &parent_node {
                // Reusing field name
                Some(parent) => {
                    let mut history = parent.action_history.clone().unwrap_or_default();
                    history.push(thought_identifier);
                    Some(history)
                }
                None => Some(vec![thought_identifier]),
            };

            // Initialize PolicyGuidedNode
            let mut node = PolicyGuidedNode {
                base: base_node.clone(),
                visits: 1,
                total_reward: 0.0,
                untried_actions: Some(vec![]),
                policy_score: 0.0,
                value_estimate: 0.0,
                prior_action_probs: HashMap::new(),
                puct: None,
                action_history,
                novelty_score: None,
            };

            // Initialize node with policy guidance
            let score = self_clone
                .base
                .evaluate_thought(&node.base, parent_node.as_ref().map(|p| &p.base))
                .await;
            node.base.set_score_or_default(score, 0.5);
            node.visits = 1;
            node.total_reward = node.base.score;
            // Calculate policy score and value estimate
            node.policy_score = self_clone
                .calculate_policy_score(&node, parent_node.as_ref())
                .await;
            node.value_estimate = self_clone.estimate_value(&node).await;
            node.novelty_score = Some(self_clone.calculate_novelty_v2(&node));
            base_node.set_score_or_default(node.base.score, 0.5);

            // Save the node
            if let Err(e) = self_clone.base.save_node_with_retry(base_node.clone(), None).await {
                tracing::error!("Fatal: Failed to save base node: {}", e);
                BaseStrategy::log_channel_send_error(
                    tx.send(Err(crate::strategies::base::ReasoningError::Fatal(format!(
                        "Failed to save node: {}", e
                    )))).await,
                    "node save error"
                );
                return;
            }

            // Update parent if exists
            if let Some(mut parent) = parent_node {
                parent.base.children.push(node.base.id.clone());
                if let Err(e) = self_clone.base.save_node_with_retry(parent.base.clone(), None).await {
                    tracing::warn!("Failed to update parent node: {}", e);
                    // Non-fatal: child node is saved, parent reference may be stale
                }
                let _ = self_clone.update_policy_metrics(&node, &parent).await;
            }

            // Run policy-guided search
            if !node.base.is_complete {
                let _ = self_clone.run_policy_guided_search(node.clone()).await;
            }

            // Calculate enhanced path statistics
            let current_path = self_clone.base.state_manager.get_path(&node_id).await;
            let enhanced_score = self_clone
                .calculate_policy_enhanced_score(&current_path)
                .await;

            let response = ReasoningResponse {
                node_id: base_response.node_id,
                thought: base_response.thought,
                score: enhanced_score,
                depth: base_response.depth,
                is_complete: base_response.is_complete,
                next_thought_needed: base_response.next_thought_needed,
                possible_paths: base_response.possible_paths,
                best_score: Some(base_response.best_score.unwrap_or(0.0).max(enhanced_score)),
                strategy_used: None, // Will be set by reasoner
                thought_number: 0, // Will be set by Tool layer
                total_thoughts: 0, // Will be set by Tool layer
                stats: crate::types::ReasoningStats {
                    total_nodes: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    branching_factor: 0.0,
                    strategy_metrics: std::collections::HashMap::new(),
                }, // Will be set by Tool layer
            };

            BaseStrategy::log_channel_send_error(
                tx.send(Ok(response)).await,
                "mcts_002_alpha response"
            );
        });

        Reasoning::new(rx)
    }

    fn get_best_path(&self) -> AsyncPath {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Delegate to persistent inner_mcts field
            let path = self_clone.inner_mcts.get_best_path().await;
            let _ = tx.send(path);
        });

        AsyncPath::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        let self_clone = self.clone();
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            let base_metrics = match self_clone.base.get_base_metrics().await {
                Ok(metrics) => metrics,
                _ => Metric {
                    name: String::from("MCTS-002-Alpha"),
                    nodes_explored: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    active: None,
                    extra: Default::default(),
                },
            };

            let mut metrics = base_metrics;
            let policy_metrics = self_clone.policy_metrics.lock().await.clone();
            let exploration_rate = *self_clone.exploration_rate.lock().await;

            let policy_stats = serde_json::json!({
                "averages": {
                    "policy_score": policy_metrics.average_policy_score,
                    "value_estimate": policy_metrics.average_value_estimate,
                },
                "temperature": self_clone.temperature,
                "exploration_rate": exploration_rate,
                "learning_rate": self_clone.learning_rate,
                "novelty_bonus": self_clone.novelty_bonus,
                "policy_entropy": policy_metrics.convergence_metrics.policy_entropy,
                "value_stability": policy_metrics.convergence_metrics.value_stability,
            });

            metrics.name = "MCTS-002-Alpha (Policy Enhanced)".to_string();
            metrics
                .extra
                .insert("temperature".to_string(), self_clone.temperature.into());
            metrics
                .extra
                .insert("exploration_rate".to_string(), exploration_rate.into());
            metrics
                .extra
                .insert("learning_rate".to_string(), self_clone.learning_rate.into());
            metrics
                .extra
                .insert("policy_stats".to_string(), policy_stats);

            let _ = tx.send(Ok(metrics)).await;
        });

        MetricStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Reset exploration rate and policy metrics
            let mut exploration_rate = self_clone.exploration_rate.lock().await;
            *exploration_rate = 2.0_f64.sqrt();

            let mut metrics = self_clone.policy_metrics.lock().await;
            *metrics = PolicyMetrics {
                average_policy_score: 0.0,
                average_value_estimate: 0.0,
                action_distribution: HashMap::new(),
                exploration_stats: ExplorationStats {
                    temperature: self_clone.temperature,
                    exploration_rate: *exploration_rate,
                    novelty_bonus: self_clone.novelty_bonus,
                },
                convergence_metrics: ConvergenceMetrics {
                    policy_entropy: 0.0,
                    value_stability: 0.0,
                },
                sample_count: 0,
            };

            let _ = tx.send(Ok(()));
        });

        ClearedSignal::new(rx)
    }
}
