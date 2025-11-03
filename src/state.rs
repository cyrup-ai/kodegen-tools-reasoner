use crate::types::ThoughtNode;
use lru::LruCache;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Minimum cache size fallback when cache_size parameter is 0.
/// APPROVED PANIC: Compile-time constant with non-zero value
/// - Evaluated at compile time, will fail during build if invalid
/// - Value 1 is clearly non-zero in source code
/// - No runtime failure possible
const MIN_CACHE_SIZE: NonZeroUsize = NonZeroUsize::new(1).unwrap();

/// Manages thought node storage with dual-layer caching architecture.
///
/// ## Lock Ordering Invariant
///
/// To prevent deadlock, ALL methods acquiring multiple locks MUST respect
/// this canonical ordering:
///
/// ```
/// 1. nodes (authoritative storage)
/// 2. cache (derived LRU cache)
/// ```
///
/// **CRITICAL**: Never hold `cache` lock while requesting `nodes` lock.
///
/// ## Concurrent Safety
///
/// Thread-safe for concurrent access via `Arc<Mutex<T>>`. The MCTS strategies
/// spawn multiple tokio tasks that call `get_node()` and `save_node()`
/// concurrently. Lock ordering prevents deadlock under high concurrency.
pub struct StateManager {
    cache: Arc<Mutex<LruCache<String, ThoughtNode>>>,
    nodes: Arc<Mutex<HashMap<String, ThoughtNode>>>,
}

impl StateManager {
    pub fn new(cache_size: usize) -> Self {
        let cache_size = NonZeroUsize::new(cache_size)
            .unwrap_or(MIN_CACHE_SIZE);
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            nodes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Retrieves a thought node by ID with LRU caching.
    ///
    /// ## Performance Characteristics
    ///
    /// - **Cache hit**: O(1) with single lock acquisition
    /// - **Cache miss**: O(1) with sequential lock acquisition (nodes → cache)
    ///
    /// ## Deadlock Safety
    ///
    /// Uses sequential lock acquisition to prevent deadlock:
    /// 1. Check cache (acquire → release)
    /// 2. If miss, read nodes (acquire → release)
    /// 3. Update cache (acquire → release)
    ///
    /// At no point are both locks held simultaneously, eliminating circular wait.
    pub async fn get_node(&self, id: &str) -> Option<ThoughtNode> {
        // Fast path: Check cache without holding lock longer than needed
        // This optimizes the common case (cache hit) while avoiding deadlock
        {
            let mut cache = self.cache.lock().await;
            if let Some(node) = cache.get(id) {
                return Some(node.clone());
            }
        }
        // Lock released here - no locks held between cache check and nodes acquisition

        // Cache miss: Use sequential lock acquisition (no overlap)
        let nodes = self.nodes.lock().await;
        let node_opt = nodes.get(id).cloned();
        drop(nodes);  // ✅ Explicit release before next lock

        if let Some(ref node) = node_opt {
            let mut cache = self.cache.lock().await;
            cache.put(id.to_string(), node.clone());
        }

        node_opt
    }

    pub async fn save_node(&self, node: ThoughtNode) {
        let node_id = node.id.clone();

        // Atomic update: lock both in consistent order
        let mut nodes = self.nodes.lock().await;
        let mut cache = self.cache.lock().await;

        nodes.insert(node_id.clone(), node.clone());
        cache.put(node_id, node);
    }

    pub async fn get_children(&self, node_id: &str) -> Vec<ThoughtNode> {
        let node = match self.get_node(node_id).await {
            Some(n) => n,
            None => return vec![],
        };

        let mut children = vec![];
        for id in &node.children {
            if let Some(child) = self.get_node(id).await {
                children.push(child);
            }
        }

        children
    }

    pub async fn get_path(&self, node_id: &str) -> Vec<ThoughtNode> {
        let mut path = Vec::new();
        let mut current_id = node_id.to_string();
        let mut visited = HashSet::new();

        while !current_id.is_empty() {
            // Cycle detection: insert returns false if already present
            if !visited.insert(current_id.clone()) {
                tracing::error!(
                    "Cycle detected in node chain at {}, returning partial path",
                    current_id
                );
                break;
            }

            match self.get_node(&current_id).await {
                Some(node) => {
                    let parent_id = node.parent_id.clone().unwrap_or_default();
                    path.push(node);
                    current_id = parent_id;
                }
                None => break,
            }
        }

        path.reverse();
        path
    }

    pub async fn get_all_nodes(&self) -> Vec<ThoughtNode> {
        let nodes = self.nodes.lock().await;
        nodes.values().cloned().collect()
    }

    pub async fn clear(&self) {
        // Atomic clear: lock both in consistent order
        let mut nodes = self.nodes.lock().await;
        let mut cache = self.cache.lock().await;

        nodes.clear();
        cache.clear();
    }
}
