//! Semantic coherence calculation using embeddings
//!
//! Provides methods for calculating semantic similarity between thoughts
//! using cached embeddings from the Stella 400M model.

use super::BaseStrategy;
use super::cache::estimate_entry_size_with_key;
use super::types::{AsyncTask, ReasoningError, MEMORY_PRESSURE_CACHE_REDUCTION};
use kodegen_candle_agent::prelude::{Embedding, EmbeddingBuilder};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::oneshot;

impl BaseStrategy {
    /// Calculates cosine similarity between two vectors.
    fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
        if v1.len() != v2.len() || v1.is_empty() {
            return 0.0; // Return 0 if vectors are different lengths or empty
        }

        // Check v1 for invalid values
        if let Some((idx, val)) = v1
            .iter()
            .enumerate()
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
        if let Some((idx, val)) = v2
            .iter()
            .enumerate()
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
                                  cache: Arc<tokio::sync::RwLock<lru::LruCache<String, Vec<f32>>>>,
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
                if BaseStrategy::check_memory_pressure() {
                    tracing::warn!("Memory pressure detected before embedding generation");

                    // Aggressively evict cache entries
                    BaseStrategy::aggressive_cache_evict(
                        &cache,
                        &stats,
                        MEMORY_PRESSURE_CACHE_REDUCTION,
                    )
                    .await;
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
                if let Some((idx, val)) = embedding
                    .iter()
                    .enumerate()
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
                if !BaseStrategy::validate_embedding_size(&embedding) {
                    return Err(ReasoningError::Other(format!(
                        "Invalid embedding dimension: expected {}, got {}",
                        super::types::EXPECTED_EMBEDDING_DIM,
                        embedding.len()
                    )));
                }

                // Store in cache with size tracking
                {
                    let mut cache_write = cache.write().await;

                    // DOUBLE-CHECK: Another thread may have inserted while we generated
                    if let Some(embedding) = cache_write.get(&text) {
                        stats.record_hit(); // Count as hit since we avoided duplicate work
                        return Ok(embedding.clone());
                    }

                    let entry_size = estimate_entry_size_with_key(&text, &embedding);

                    // Check if insertion will cause eviction
                    let will_evict = cache_write.len() >= cache_write.cap().get();

                    if will_evict {
                        // Estimate size of evicted entry (we don't know which will be evicted)
                        // Use average embedding size as approximation
                        stats.record_eviction(super::types::EXPECTED_EMBEDDING_DIM * 4 + 200);
                        // 200 = avg key size
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
            )
            .await
            {
                Ok(emb) => emb,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Get or compute child embedding (cache hit if seen before)
            let child_embedding = match get_embedding(child_thought, embedding_cache, embedding_stats).await {
                Ok(emb) => emb,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            // Calculate cosine similarity
            let similarity = BaseStrategy::cosine_similarity(&parent_embedding, &child_embedding);

            // Scale similarity from [-1, 1] to [0, 1] for scoring consistency
            let scaled_similarity = (similarity + 1.0) / 2.0;

            let _ = tx.send(Ok(scaled_similarity));
        });

        AsyncTask::new(rx)
    }

    // Original word overlap coherence function (kept for reference or fallback if needed)
    #[allow(dead_code)]
    pub(super) fn calculate_word_overlap_coherence(
        &self,
        parent_thought: &str,
        child_thought: &str,
    ) -> f64 {
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
}
