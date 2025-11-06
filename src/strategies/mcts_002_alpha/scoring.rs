use super::types::PolicyGuidedNode;
use crate::strategies::base::{BaseStrategy, NOVELTY_MARKERS};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use tokio::sync::Mutex;
use std::sync::Arc;

pub struct ScoringEngine {
    pub base: BaseStrategy,
    pub coherence_cache: Arc<Mutex<HashMap<String, f64>>>,
}

impl ScoringEngine {
    pub fn new(base: BaseStrategy) -> Self {
        Self {
            base,
            coherence_cache: Arc::new(Mutex::new(HashMap::new())),
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
    pub async fn thought_coherence(
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

    pub async fn calculate_goal_alignment(&self, node: &PolicyGuidedNode) -> f64 {
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
}
