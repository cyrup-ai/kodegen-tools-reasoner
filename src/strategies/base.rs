use crate::state::StateManager;
use crate::types::{ReasoningRequest, ReasoningResponse, StrategyMetrics, ThoughtNode};
use futures::Stream;
use kodegen_candle_agent::prelude::{Embedding, EmbeddingBuilder};
use lazy_static::lazy_static;
use lru::LruCache;
use regex::Regex;
#[allow(unused_imports)]
use serde_json::json;
use std::collections::HashSet;
use std::fmt;
use std::future::Future;
#[allow(unused_imports)]
use std::hash::Hash;
#[allow(unused_imports)]
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};
use std::time::Duration;
use sysinfo::System;
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio_stream::wrappers::ReceiverStream;

lazy_static! {
    /// Regex for logical connectors (therefore, because, if, then, thus, hence, so)
    /// Used in calculate_logical_score() for thought quality evaluation.
    static ref LOGICAL_CONNECTORS: Regex =
        Regex::new(r"\b(therefore|because|if|then|thus|hence|so)\b")
            .expect("Failed to compile logical connectors regex");

    /// Regex for mathematical/logical expressions (+, -, *, /, =, <, >)
    /// Used in calculate_logical_score() for thought quality evaluation.
    static ref MATH_EXPRESSIONS: Regex =
        Regex::new(r"[+\-*/=<>]")
            .expect("Failed to compile mathematical expressions regex");

    /// Regex for novelty calculation (sentence terminators and reasoning markers)
    /// Used in calculate_novelty() for linguistic complexity scoring.
    /// Made public for use in mcts_002_alpha strategy.
    pub static ref NOVELTY_MARKERS: Regex =
        Regex::new(r"[.!?;]|therefore|because|if|then")
            .expect("Failed to compile novelty markers regex");
}

/// Expected embedding dimension for Stella 400M model
const EXPECTED_EMBEDDING_DIM: usize = 400;

/// Memory pressure threshold: trigger aggressive eviction if available memory < 15%
const MEMORY_PRESSURE_THRESHOLD: f64 = 0.15;

/// Target cache size reduction during memory pressure (reduce to 25% of max)
const MEMORY_PRESSURE_CACHE_REDUCTION: f64 = 0.25;

/// Error type for reasoning operations with severity categorization
#[derive(Debug, Clone)]
pub enum ReasoningError {
    /// Fatal error - cannot continue processing
    Fatal(String),
    
    /// Recoverable error - can continue with degraded functionality
    Recoverable(String),
    
    /// Generic error (preserved for backward compatibility)
    Other(String),
}

impl ReasoningError {
    pub fn is_fatal(&self) -> bool {
        matches!(self, ReasoningError::Fatal(_))
    }
    
    pub fn is_recoverable(&self) -> bool {
        matches!(self, ReasoningError::Recoverable(_))
    }
}

impl fmt::Display for ReasoningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(msg) => write!(f, "Fatal reasoning error: {}", msg),
            Self::Recoverable(msg) => write!(f, "Recoverable reasoning error: {}", msg),
            Self::Other(msg) => write!(f, "Reasoning error: {}", msg),
        }
    }
}

impl std::error::Error for ReasoningError {}

/// A convenience type alias for reasoning results
pub type ReasoningResult<T> = Result<T, ReasoningError>;

//==============================================================================
// AsyncTask - Generic awaitable type for all single-value operations
//==============================================================================

/// Generic awaitable future for any operation that returns a single value
pub struct AsyncTask<T> {
    rx: oneshot::Receiver<ReasoningResult<T>>,
}

impl<T> AsyncTask<T> {
    /// Creates a new AsyncTask from a receiver
    pub fn new(rx: oneshot::Receiver<ReasoningResult<T>>) -> Self {
        Self { rx }
    }

    /// Creates an AsyncTask from a direct value
    ///
    /// Used in BaseStrategy::clear() default implementation (line 921 in this file).
    /// Rust's dead code analysis cannot trace usage in default trait implementations.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-10-27
    #[allow(dead_code)]
    pub fn from_value(value: T) -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(value));
        Self { rx }
    }
}

impl<T> Future for AsyncTask<T> {
    type Output = ReasoningResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(ReasoningError::Other("Task failed".into()))),
            Poll::Pending => Poll::Pending,
        }
    }
}

//==============================================================================
// TaskStream - Generic stream type for all multi-value operations
//==============================================================================

/// Generic stream for any operation that returns multiple values
pub struct TaskStream<T> {
    inner: ReceiverStream<ReasoningResult<T>>,
}

impl<T> TaskStream<T> {
    /// Creates a new stream from a receiver
    pub fn new(rx: mpsc::Receiver<ReasoningResult<T>>) -> Self {
        Self {
            inner: ReceiverStream::new(rx),
        }
    }

    /// Creates a stream that produces an error
    pub fn from_error(error: ReasoningError) -> Self {
        let (tx, rx) = mpsc::channel(1);
        let _ = tx.try_send(Err(error));
        Self::new(rx)
    }
}

impl<T> Stream for TaskStream<T> {
    type Item = ReasoningResult<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

// Type aliases for backward compatibility with existing code
pub type AsyncPath = AsyncTask<Vec<ThoughtNode>>;

/// Type alias for clear operation results.
///
/// Return type for Strategy::clear() trait method (line 205) and all implementations.
/// Rust's dead code analysis cannot trace trait method return types.
///
/// APPROVED BY DAVID MAPLE on 2025-10-27
#[allow(dead_code)]
pub type ClearedSignal = AsyncTask<()>;

pub type MetricStream = TaskStream<StrategyMetrics>;
pub type Reasoning = TaskStream<ReasoningResponse>;
pub type Metric = StrategyMetrics;

/// Strategy trait without async_trait
pub trait Strategy: Send + Sync {
    /// Process a thought with the selected strategy
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning;

    /// Get the best reasoning path found by this strategy
    fn get_best_path(&self) -> AsyncPath;

    /// Get strategy metrics
    fn get_metrics(&self) -> MetricStream;

    /// Clear strategy state
    ///
    /// Called via trait objects in wrapper strategies (mcts_002alt_alpha.rs:859).
    /// Rust's dead code analysis cannot trace trait object method calls.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-10-27
    #[allow(dead_code)]
    fn clear(&self) -> ClearedSignal;
}

/// Base strategy implementation that provides common functionality
pub struct BaseStrategy {
    pub state_manager: Arc<StateManager>,
    
    // LRU cache for embedding vectors (text -> Vec<f32>)
    // Size configurable via CONFIG.cache_size (~1.6MB for 1000x400-dim embeddings)
    embedding_cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,
    
    // Telemetry for cache performance and memory monitoring
    embedding_stats: Arc<crate::types::EmbeddingCacheStats>,
}

impl Clone for BaseStrategy {
    fn clone(&self) -> Self {
        Self {
            state_manager: Arc::clone(&self.state_manager),
            embedding_cache: Arc::clone(&self.embedding_cache),
            embedding_stats: Arc::clone(&self.embedding_stats),
        }
    }
}

/// Default cache size for embedding vectors.
/// APPROVED PANIC: Compile-time constant with non-zero value
/// - Evaluated at compile time, will fail during build if invalid
/// - Value 1000 is clearly non-zero in source code
/// - No runtime failure possible
const DEFAULT_EMBEDDING_CACHE_SIZE: NonZeroUsize = NonZeroUsize::new(1000).unwrap();

impl BaseStrategy {
    pub fn new(state_manager: Arc<StateManager>) -> Self {
        Self {
            state_manager,
            embedding_cache: Arc::new(RwLock::new(
                LruCache::new(DEFAULT_EMBEDDING_CACHE_SIZE)
            )),
            embedding_stats: Arc::new(crate::types::EmbeddingCacheStats::new()),
        }
    }

    /// Safe division that returns fallback on NaN/infinite/divide-by-zero
    #[inline]
    pub(crate) fn safe_divide(numerator: f64, denominator: f64, fallback: f64) -> f64 {
        if denominator == 0.0 || denominator.is_nan() || numerator.is_nan() {
            return fallback;
        }
        let result = numerator / denominator;
        if result.is_nan() || result.is_infinite() {
            fallback
        } else {
            result
        }
    }

    /// Safe weighted sum with NaN checking for each component.
    /// Skips NaN components rather than propagating them.
    #[inline]
    pub(crate) fn safe_weighted_sum(components: &[(f64, f64)]) -> f64 {
        let mut sum = 0.0;
        let mut valid_weight_sum = 0.0;
        
        for (value, weight) in components {
            if value.is_nan() || weight.is_nan() {
                tracing::warn!(
                    "NaN detected in weighted sum component: value={}, weight={}",
                    value, weight
                );
                continue;
            }
            sum += value * weight;
            valid_weight_sum += weight;
        }
        
        if valid_weight_sum == 0.0 {
            return 0.5;
        }
        
        let result = sum / valid_weight_sum;
        if result.is_nan() {
            0.5
        } else {
            result
        }
    }

    pub fn get_node(&self, id: &str) -> AsyncTask<Option<ThoughtNode>> {
        let state_manager = Arc::clone(&self.state_manager);
        let id = id.to_string();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = state_manager.get_node(&id).await;
            let _ = tx.send(Ok(result));
        });

        AsyncTask::new(rx)
    }

    pub fn save_node(&self, node: ThoughtNode) -> AsyncTask<()> {
        let state_manager = Arc::clone(&self.state_manager);
        let node = node.clone();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            state_manager.save_node(node).await;
            let _ = tx.send(Ok(()));
        });

        AsyncTask::new(rx)
    }

    /// Save node with exponential backoff retry logic
    pub async fn save_node_with_retry(
        &self,
        node: ThoughtNode,
        max_retries: Option<u32>,
    ) -> Result<(), ReasoningError> {
        let max_retries = max_retries.unwrap_or(crate::types::CONFIG.max_retry_attempts);
        let mut attempt = 0;
        let mut delay = Duration::from_millis(crate::types::CONFIG.initial_retry_delay_ms);
        let max_delay = Duration::from_millis(crate::types::CONFIG.max_retry_delay_ms);
        
        loop {
            match self.save_node(node.clone()).await {
                Ok(_) => return Ok(()),
                Err(e) if e.is_fatal() => {
                    // Fatal errors should not be retried
                    return Err(e);
                }
                Err(e) if attempt < max_retries => {
                    tracing::warn!(
                        "Save node {} failed (attempt {}/{}): {}",
                        node.id,
                        attempt + 1,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(max_delay);  // Capped exponential backoff
                    attempt += 1;
                }
                Err(e) => {
                    return Err(ReasoningError::Recoverable(format!(
                        "Failed to save node {} after {} attempts: {}",
                        node.id,
                        max_retries,
                        e
                    )));
                }
            }
        }
    }

    /// Get node with retry logic for transient failures
    pub async fn get_node_with_retry(
        &self,
        node_id: &str,
        max_retries: Option<u32>,
    ) -> Result<Option<ThoughtNode>, ReasoningError> {
        let max_retries = max_retries.unwrap_or(crate::types::CONFIG.max_retry_attempts);
        let mut attempt = 0;
        let mut delay = Duration::from_millis(crate::types::CONFIG.initial_retry_delay_ms);
        let max_delay = Duration::from_millis(crate::types::CONFIG.max_retry_delay_ms);
        
        loop {
            match self.get_node(node_id).await {
                Ok(result) => return Ok(result),
                Err(e) if e.is_fatal() => {
                    // Fatal errors should not be retried
                    return Err(e);
                }
                Err(e) if e.is_recoverable() && attempt < max_retries => {
                    tracing::warn!(
                        "Get node {} failed (attempt {}/{}): {}",
                        node_id,
                        attempt + 1,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(max_delay);  // Capped exponential backoff
                    attempt += 1;
                }
                Err(e) => {
                    return Err(ReasoningError::Recoverable(format!(
                        "Failed to get node {} after {} attempts: {}",
                        node_id,
                        max_retries,
                        e
                    )));
                }
            }
        }
    }

    /// Helper to log and handle channel send failures
    pub fn log_channel_send_error<T>(
        result: Result<(), tokio::sync::mpsc::error::SendError<T>>,
        context: &str,
    ) {
        if result.is_err() {
            tracing::error!("Failed to send {} (receiver dropped)", context);
        }
    }

    /// Calculates cosine similarity between two vectors.
    fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
        if v1.len() != v2.len() || v1.is_empty() {
            return 0.0; // Return 0 if vectors are different lengths or empty
        }

        // Check v1 for invalid values
        if let Some((idx, val)) = v1.iter().enumerate()
            .find(|(_, v)| v.is_nan() || v.is_infinite()) 
        {
            tracing::error!(
                "cosine_similarity: v1 contains {} at position {} - this indicates embedding validation failure",
                if val.is_nan() { "NaN" } else { "Inf" },
                idx
            );
            return 0.0;
        }
        
        // Check v2 for invalid values
        if let Some((idx, val)) = v2.iter().enumerate()
            .find(|(_, v)| v.is_nan() || v.is_infinite()) 
        {
            tracing::error!(
                "cosine_similarity: v2 contains {} at position {} - this indicates embedding validation failure",
                if val.is_nan() { "NaN" } else { "Inf" },
                idx
            );
            return 0.0;
        }

        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let magnitude1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return 0.0; // Avoid division by zero
        }

        let result = (dot_product / (magnitude1 * magnitude2)) as f64;
        
        if result.is_nan() {
            tracing::error!(
                "cosine_similarity produced NaN despite input validation - check for numeric overflow"
            );
            return 0.0;
        }
        
        result
    }

    pub async fn evaluate_thought(&self, node: &ThoughtNode, parent: Option<&ThoughtNode>) -> f64 {
        // Skip expensive evaluation for MCTS-generated synthetic nodes
        if node.is_synthetic {
            return 0.5;  // Neutral score - won't bias tree search
        }
        
        // Base evaluation logic - Semantic coherence is now handled async by strategies
        let logical_score = self.calculate_logical_score(node, parent).await;
        let depth_penalty = self.calculate_depth_penalty(node);
        let completion_bonus = if node.is_complete { 0.2 } else { 0.0 };

        let result = Self::safe_divide(logical_score + depth_penalty + completion_bonus, 3.0, 0.5);
        
        if result.is_nan() {
            tracing::error!("evaluate_thought produced NaN for node {}", node.id);
            return 0.5;
        }
        
        result
    }

    async fn calculate_logical_score(
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

    fn calculate_depth_penalty(&self, node: &ThoughtNode) -> f64 {
        // Penalize deeper thoughts slightly less aggressively
        (1.0 - (node.depth as f64 / crate::types::CONFIG.max_depth as f64) * 0.2).max(0.0)
    }

    /// Check if system is under memory pressure
    /// Returns true if available memory is below MEMORY_PRESSURE_THRESHOLD
    fn check_memory_pressure() -> bool {
        let mut sys = System::new_all();
        sys.refresh_memory();
        
        let available = sys.available_memory() as f64;
        let total = sys.total_memory() as f64;
        
        if total == 0.0 {
            return false; // Safety check
        }
        
        let available_ratio = available / total;
        available_ratio < MEMORY_PRESSURE_THRESHOLD
    }
    
    /// Aggressively evict cache entries when under memory pressure
    /// Reduces cache to target_ratio of max size (e.g., 25% = 250 entries from 1000)
    async fn aggressive_cache_evict(
        cache: &Arc<RwLock<LruCache<String, Vec<f32>>>>,
        stats: &Arc<crate::types::EmbeddingCacheStats>,
        target_ratio: f64,
    ) {
        let mut cache_lock = cache.write().await;
        let current_len = cache_lock.len();
        let max_size = cache_lock.cap().get();
        let target_size = (max_size as f64 * target_ratio).ceil() as usize;
        
        if current_len <= target_size {
            return; // Already below target
        }
        
        let to_evict = current_len - target_size;
        
        tracing::warn!(
            "Memory pressure detected! Evicting {} cache entries ({}% reduction)",
            to_evict,
            ((to_evict as f64 / current_len as f64) * 100.0) as usize
        );
        
        // LRU cache pop_lru() removes least recently used entries
        for _ in 0..to_evict {
            if let Some((_, embedding)) = cache_lock.pop_lru() {
                let entry_size = estimate_entry_size(&embedding);
                stats.record_eviction(entry_size);
            }
        }
        
        tracing::info!(
            "Cache reduced from {} to {} entries",
            current_len,
            cache_lock.len()
        );
    }
    
    /// Validate embedding has expected dimensions
    /// Returns true if embedding is valid for caching
    fn validate_embedding_size(embedding: &[f32]) -> bool {
        let dim = embedding.len();
        
        if dim != EXPECTED_EMBEDDING_DIM {
            tracing::warn!(
                "Embedding dimension mismatch: expected {}, got {}",
                EXPECTED_EMBEDDING_DIM,
                dim
            );
            return false;
        }
        
        true
    }
    
    /// Get current cache statistics (for logging/monitoring)
    pub fn get_cache_stats(&self) -> crate::types::EmbeddingCacheSnapshot {
        self.embedding_stats.snapshot()
    }

    /// Calculates semantic coherence using cached Stella embeddings.
    /// Returns an AsyncTask with cosine similarity score [0.0, 1.0].
    pub fn calculate_semantic_coherence(
        &self,
        parent_thought: &str,
        child_thought: &str,
    ) -> AsyncTask<f64> {
        let parent_thought = parent_thought.to_string();
        let child_thought = child_thought.to_string();
        let embedding_cache = Arc::clone(&self.embedding_cache);
        let embedding_stats = Arc::clone(&self.embedding_stats);
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Helper closure to get or compute embedding with cache access and telemetry
            let get_embedding = |text: String, 
                                  cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,
                                  stats: Arc<crate::types::EmbeddingCacheStats>| async move {
                // Fast path: check cache first
                {
                    let cache_read = cache.read().await;
                    if let Some(embedding) = cache_read.peek(&text) {
                        stats.record_hit();
                        return Ok(embedding.clone());
                    }
                }
                
                // Cache miss recorded
                stats.record_miss();
                
                // Check memory pressure before generating new embedding
                if Self::check_memory_pressure() {
                    tracing::warn!("Memory pressure detected before embedding generation");
                    
                    // Aggressively evict cache entries
                    Self::aggressive_cache_evict(
                        &cache,
                        &stats,
                        MEMORY_PRESSURE_CACHE_REDUCTION,
                    ).await;
                }
                
                // Generate embedding
                let embedding_result = Embedding::from_document(&text)
                    .model("dunzhang/stella_en_400M_v5")
                    .embed()
                    .await;
                
                let embedding = match embedding_result {
                    Ok(Ok(emb)) => match emb.as_vec() {
                        Some(vec) => vec.clone(),
                        None => {
                            return Err(ReasoningError::Other(
                                "Embedding vector is empty".into(),
                            ));
                        }
                    },
                    Ok(Err(e)) => {
                        return Err(ReasoningError::Other(format!(
                            "Failed to generate embedding: {}",
                            e
                        )));
                    }
                    Err(e) => {
                        return Err(ReasoningError::Other(format!(
                            "Task join error for embedding: {}",
                            e
                        )));
                    }
                };
                
                // Validate embedding contains no NaN or Infinite values
                if let Some((idx, val)) = embedding.iter().enumerate()
                    .find(|(_, v)| v.is_nan() || v.is_infinite()) 
                {
                    return Err(ReasoningError::Other(format!(
                        "Embedding contains {} at position {} for text: '{}' (model: dunzhang/stella_en_400M_v5)",
                        if val.is_nan() { "NaN" } else { "Inf" },
                        idx,
                        &text.chars().take(100).collect::<String>()
                    )));
                }
                
                // Validate embedding size before caching
                if !Self::validate_embedding_size(&embedding) {
                    return Err(ReasoningError::Other(format!(
                        "Invalid embedding dimension: expected {}, got {}",
                        EXPECTED_EMBEDDING_DIM,
                        embedding.len()
                    )));
                }
                
                // Store in cache with size tracking
                {
                    let mut cache_write = cache.write().await;
                    
                    // DOUBLE-CHECK: Another thread may have inserted while we generated
                    if let Some(embedding) = cache_write.get(&text) {
                        stats.record_hit();  // Count as hit since we avoided duplicate work
                        return Ok(embedding.clone());
                    }
                    
                    let entry_size = estimate_entry_size_with_key(&text, &embedding);
                    
                    // Check if insertion will cause eviction
                    let will_evict = cache_write.len() >= cache_write.cap().get();
                    
                    if will_evict {
                        // Estimate size of evicted entry (we don't know which will be evicted)
                        // Use average embedding size as approximation
                        stats.record_eviction(EXPECTED_EMBEDDING_DIM * 4 + 200); // 200 = avg key size
                    }
                    
                    cache_write.put(text.clone(), embedding.clone());
                    stats.add_size(entry_size);
                    
                    // Log cache stats periodically (every 100 misses)
                    let misses = stats.misses.load(Ordering::Relaxed);
                    if misses.is_multiple_of(100) {
                        let snapshot = stats.snapshot();
                        tracing::info!(
                            "Embedding cache stats: hit_rate={:.2}%, entries={}, size={}KB, evictions={}",
                            stats.hit_rate() * 100.0,
                            cache_write.len(),
                            snapshot.size_bytes / 1024,
                            snapshot.evictions
                        );
                    }
                }
                
                Ok::<Vec<f32>, ReasoningError>(embedding)
            };

            // Get or compute parent embedding (cache hit if seen before)
            let parent_embedding = match get_embedding(
                parent_thought,
                Arc::clone(&embedding_cache),
                Arc::clone(&embedding_stats),
            ).await {
                Ok(emb) => emb,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Get or compute child embedding (cache hit if seen before)
            let child_embedding = match get_embedding(
                child_thought,
                embedding_cache,
                embedding_stats,
            ).await {
                Ok(emb) => emb,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Calculate cosine similarity
            let similarity = Self::cosine_similarity(&parent_embedding, &child_embedding);

            // Scale similarity from [-1, 1] to [0, 1] for scoring consistency
            let scaled_similarity = (similarity + 1.0) / 2.0;

            let _ = tx.send(Ok(scaled_similarity));
        });

        AsyncTask::new(rx)
    }

    // Original word overlap coherence function (kept for reference or fallback if needed)
    #[allow(dead_code)]
    fn calculate_word_overlap_coherence(&self, parent_thought: &str, child_thought: &str) -> f64 {
        let parent_terms: HashSet<String> = parent_thought
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let child_terms: Vec<String> = child_thought
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let shared_terms = child_terms
            .iter()
            .filter(|term| parent_terms.contains(*term))
            .count();

        if child_terms.is_empty() {
            return 0.0;
        }

        (shared_terms as f64 / child_terms.len() as f64).min(1.0)
    }

    /// Get base metrics
    pub fn get_base_metrics(&self) -> AsyncTask<StrategyMetrics> {
        let state_manager = Arc::clone(&self.state_manager);
        let cache_snapshot = self.get_cache_stats();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let nodes = state_manager.get_all_nodes().await;

            let avg_score = if nodes.is_empty() {
                0.0
            } else {
                nodes.iter().map(|n| n.score).sum::<f64>() / nodes.len() as f64
            };

            let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);

            let mut extra = std::collections::HashMap::new();
            extra.insert("cache_hits".to_string(), cache_snapshot.hits.into());
            extra.insert("cache_misses".to_string(), cache_snapshot.misses.into());
            extra.insert("cache_evictions".to_string(), cache_snapshot.evictions.into());
            extra.insert("cache_size_bytes".to_string(), cache_snapshot.size_bytes.into());

            let metrics = StrategyMetrics {
                name: String::from("BaseStrategy"),
                nodes_explored: nodes.len(),
                average_score: avg_score,
                max_depth,
                active: None,
                extra,
            };

            let _ = tx.send(Ok(metrics));
        });

        AsyncTask::new(rx)
    }
}

/// Estimate memory size of a cache entry (key + value)
/// Embedding: dimension * 4 bytes (f32)
/// String key: length * 1 byte (approximate)
fn estimate_entry_size(embedding: &[f32]) -> usize {
    std::mem::size_of_val(embedding)
}

/// Estimate memory size including string key
fn estimate_entry_size_with_key(key: &str, embedding: &[f32]) -> usize {
    key.len() + estimate_entry_size(embedding)
}

/// Default implementation of Strategy for BaseStrategy
impl Strategy for BaseStrategy {
    fn process_thought(&self, _request: ReasoningRequest) -> Reasoning {
        TaskStream::from_error(ReasoningError::Other(
            "Base strategy does not implement process_thought".into(),
        ))
    }

    fn get_best_path(&self) -> AsyncPath {
        let state_manager = Arc::clone(&self.state_manager);

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let nodes = state_manager.get_all_nodes().await;
            if nodes.is_empty() {
                let _ = tx.send(Ok(vec![]));
                return;
            }

            // Find highest scoring complete path
            let mut completed_nodes: Vec<ThoughtNode> =
                nodes.into_iter().filter(|n| n.is_complete).collect();

            if completed_nodes.is_empty() {
                let _ = tx.send(Ok(vec![]));
                return;
            }

            // Sort with deterministic tie-breaking
            completed_nodes.sort_by(|a, b| {
                b.score.partial_cmp(&a.score)
                    .unwrap_or_else(|| {
                        // Should never happen with preventive validation
                        tracing::error!(
                            "Unexpected NaN in completed nodes sort - validation failure at assignment"
                        );
                        std::cmp::Ordering::Equal
                    })
                    .then_with(|| a.id.cmp(&b.id))  // Deterministic tie-breaking
            });

            let path = state_manager.get_path(&completed_nodes[0].id).await;
            let _ = tx.send(Ok(path));
        });

        AsyncTask::new(rx)
    }

    fn get_metrics(&self) -> MetricStream {
        // Convert AsyncTask to TaskStream
        let async_metrics = self.get_base_metrics();
        let (tx, rx) = mpsc::channel(1);

        tokio::spawn(async move {
            match async_metrics.await {
                Ok(metrics) => {
                    let _ = tx.send(Ok(metrics)).await;
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                }
            }
        });

        TaskStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        AsyncTask::from_value(())
    }
}
