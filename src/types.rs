use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ThoughtNode {
    pub id: String,
    pub thought: String,
    pub score: f64,
    pub depth: usize,
    pub children: Vec<String>, // Store child IDs
    pub parent_id: Option<String>, // Store parent ID
    pub is_complete: bool,
    // Marks MCTS-generated expansion nodes (true) vs real user thoughts (false)
    #[serde(default)]
    pub is_synthetic: bool,
}

impl ThoughtNode {
    /// Sets the score with validation to prevent NaN and infinite values.
    /// Returns an error if the score is invalid.
    pub fn set_score(&mut self, score: f64) -> Result<(), String> {
        if score.is_nan() {
            return Err(format!(
                "Cannot set NaN score on node {} (thought: '{}')",
                self.id,
                self.thought.chars().take(50).collect::<String>()
            ));
        }
        if score.is_infinite() {
            return Err(format!(
                "Cannot set infinite score ({}) on node {}",
                score, self.id
            ));
        }
        self.score = score.clamp(0.0, 1.0);
        Ok(())
    }

    /// Safe score setter with fallback for non-critical cases.
    /// If score is NaN or infinite, uses the default value instead.
    pub fn set_score_or_default(&mut self, score: f64, default: f64) {
        if score.is_nan() || score.is_infinite() {
            tracing::warn!(
                "Invalid score ({}) for node {}, using default {}",
                score, self.id, default
            );
            self.score = default.clamp(0.0, 1.0);
        } else {
            self.score = score.clamp(0.0, 1.0);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningRequest {
    pub thought: String,
    pub thought_number: usize,
    pub total_thoughts: usize,
    pub next_thought_needed: bool,
    pub parent_id: Option<String>, // For branching thoughts
    pub strategy_type: Option<String>, // Strategy to use for reasoning
    pub beam_width: Option<usize>, // Number of top paths to maintain (n-sampling)
    pub num_simulations: Option<usize>, // Number of MCTS simulations to run
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningResponse {
    pub node_id: String,
    pub thought: String,
    pub score: f64,
    pub depth: usize,
    pub is_complete: bool,
    pub next_thought_needed: bool,
    pub possible_paths: Option<usize>,
    pub best_score: Option<f64>,
    pub strategy_used: Option<String>,

    // Echo input fields for client convenience
    pub thought_number: usize,
    pub total_thoughts: usize,

    // Embedded stats (computed on every response)
    pub stats: ReasoningStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningStats {
    pub total_nodes: usize,
    pub average_score: f64,
    pub max_depth: usize,
    pub branching_factor: f64,
    pub strategy_metrics: HashMap<String, StrategyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StrategyMetrics {
    pub name: String,
    pub nodes_explored: usize,
    pub average_score: f64,
    pub max_depth: usize,
    pub active: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Telemetry for embedding cache performance and memory usage
#[derive(Debug)]
pub struct EmbeddingCacheStats {
    /// Number of cache hits (embedding found in cache)
    pub hits: AtomicUsize,
    
    /// Number of cache misses (embedding had to be generated)
    pub misses: AtomicUsize,
    
    /// Number of entries evicted from cache (LRU eviction)
    pub evictions: AtomicUsize,
    
    /// Current estimated memory usage in bytes
    pub current_size_bytes: AtomicUsize,
}

impl EmbeddingCacheStats {
    pub fn new() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            evictions: AtomicUsize::new(0),
            current_size_bytes: AtomicUsize::new(0),
        }
    }
    
    /// Increment cache hits counter
    #[inline]
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment cache misses counter
    #[inline]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record cache eviction and update size estimate
    #[inline]
    pub fn record_eviction(&self, entry_size_bytes: usize) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.current_size_bytes.fetch_sub(entry_size_bytes, Ordering::Relaxed);
    }
    
    /// Add to current size when inserting entry
    #[inline]
    pub fn add_size(&self, bytes: usize) {
        self.current_size_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    
    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// Get current stats snapshot for logging
    pub fn snapshot(&self) -> EmbeddingCacheSnapshot {
        EmbeddingCacheSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            size_bytes: self.current_size_bytes.load(Ordering::Relaxed),
        }
    }
}

impl Default for EmbeddingCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of cache stats at a point in time (for logging/serialization)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingCacheSnapshot {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub size_bytes: usize,
}

pub struct ValueEstimationWeights {
    pub immediate_weight: f64,      // Default 0.5
    pub completion_weight: f64,     // Default 0.2
    pub coherence_weight: f64,      // Default 0.2
    pub novelty_weight: f64,        // Default 0.1
}

pub const VALUE_WEIGHTS: ValueEstimationWeights = ValueEstimationWeights {
    immediate_weight: 0.5,
    completion_weight: 0.2,
    coherence_weight: 0.2,
    novelty_weight: 0.1,
};

pub struct Config {
    pub beam_width: usize,
    pub max_depth: usize,
    pub min_score: f64,
    pub temperature: f64,
    pub cache_size: usize,
    pub default_strategy: &'static str,
    pub num_simulations: usize,
    pub policy_weight: f64,      // Weight for policy network guidance
    pub value_weight: f64,       // Weight for value estimation
    pub novelty_weight: f64,     // Weight for exploration novelty
    pub base_score_weight: f64,  // Weight for base evaluation score
    pub use_puct_formula: bool,        // true=PUCT, false=UCB1
    pub ucb_exploration_constant: f64, // Default: 1.414 (√2)
    pub puct_exploration_constant: f64, // Default: 1.0
    pub max_retry_attempts: u32,
    pub max_retry_delay_ms: u64,
    pub initial_retry_delay_ms: u64,
}

pub const CONFIG: Config = Config {
    beam_width: 3,                   // Keep top 3 paths
    max_depth: 5,                    // Reasonable depth limit
    min_score: 0.5,                  // Threshold for path viability
    temperature: 0.7,                // For thought diversity
    cache_size: 1000,                // LRU cache size
    default_strategy: "beam_search", // Match TypeScript reference default strategy
    num_simulations: 50,             // Default number of MCTS simulations
    policy_weight: 0.3,              // Policy guidance important but not dominant
    value_weight: 0.4,               // Value estimation weighted highest (AlphaZero pattern)
    novelty_weight: 0.1,             // Exploration bonus, smaller weight
    base_score_weight: 0.2,          // Base score from evaluate_thought
    use_puct_formula: true,          // Use AlphaZero PUCT by default
    ucb_exploration_constant: 1.414, // √2 for UCB1
    puct_exploration_constant: 1.0,  // Standard PUCT constant
    max_retry_attempts: 3,           // Default 3 retries (preserves current behavior)
    max_retry_delay_ms: 1000,        // Cap at 1 second
    initial_retry_delay_ms: 10,      // Start at 10ms
};
