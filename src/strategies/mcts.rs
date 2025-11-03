use crate::atomics::AtomicF64;
use crate::state::StateManager;
use crate::strategies::base::{
    AsyncPath, BaseStrategy, ClearedSignal, Metric, MetricStream, Reasoning, Strategy,
};
use crate::types::{CONFIG, ReasoningRequest, ReasoningResponse, ThoughtNode};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing;
use uuid::Uuid;

/// Synthetic thoughts are never evaluated (see base.rs:402-404)
/// so we use a constant to avoid allocation overhead
const SYNTHETIC_EXPANSION: &str = "[synthetic]";

/// Type alias for node statistics registry storing atomic counters
type StatsRegistry = Arc<Mutex<HashMap<String, (Arc<AtomicUsize>, Arc<AtomicF64>)>>>;

#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub base: ThoughtNode,
    
    // Atomic counters - shared across all clones via Arc
    pub visits: Arc<AtomicUsize>,
    pub total_reward: Arc<AtomicF64>,
    
    pub untried_actions: Option<Vec<String>>,
}

pub struct MonteCarloTreeSearchStrategy {
    base: BaseStrategy,
    exploration_constant: f64,
    simulation_depth: usize,
    num_simulations: usize,
    root: Arc<Mutex<Option<MCTSNode>>>,

    // Registry mapping node IDs to shared atomic statistics
    stats_registry: StatsRegistry,

    // Cache for path counts to avoid redundant calculations
    path_count_cache: Arc<Mutex<HashMap<String, usize>>>,
}

impl MonteCarloTreeSearchStrategy {
    pub fn new(state_manager: Arc<StateManager>, num_simulations: Option<usize>) -> Self {
        Self {
            base: BaseStrategy::new(state_manager),
            exploration_constant: 2.0_f64.sqrt(),
            simulation_depth: CONFIG.max_depth,
            num_simulations: num_simulations
                .unwrap_or(CONFIG.num_simulations)
                .clamp(1, 150),
            root: Arc::new(Mutex::new(None)),
            
            // Initialize statistics registry
            stats_registry: Arc::new(Mutex::new(HashMap::new())),
            
            // Initialize cache for path counting
            path_count_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn run_simulations(
        &self,
        node: MCTSNode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for _ in 0..self.num_simulations {
            let selected_node = self.select(node.clone()).await?;
            let expanded_node = self.expand(selected_node).await?;
            let reward = self.simulate(&expanded_node).await?;
            self.backpropagate(expanded_node, reward).await?;
        }
        Ok(())
    }

    async fn select(
        &self,
        node: MCTSNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node;

        while !current.base.children.is_empty()
            && current
                .untried_actions
                .as_ref()
                .is_none_or(|a| a.is_empty())
        {
            let mut children = Vec::new();
            for id in &current.base.children {
                if let Ok(Some(child_node)) = self.base.get_node(id).await {
                    // Convert ThoughtNode to MCTSNode
                    let mcts_child = self.thought_to_mcts(child_node).await?;
                    children.push(mcts_child);
                }
            }

            if children.is_empty() {
                break;
            }

            let parent_visits = current.visits.load(Ordering::Relaxed);
            current = self.select_best_uct(children, parent_visits);
        }

        Ok(current)
    }

    async fn expand(
        &self,
        node: MCTSNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        if node.base.is_complete || node.base.depth >= self.simulation_depth {
            return Ok(node);
        }

        // Create a new thought node as expansion
        let new_node_id = Uuid::new_v4().to_string();
        let new_thought = SYNTHETIC_EXPANSION.to_string();
        let mut new_node = ThoughtNode {
            id: new_node_id.clone(),
            thought: new_thought,
            depth: node.base.depth + 1,
            score: 0.0,
            children: vec![],
            parent_id: Some(node.base.id.clone()),
            is_complete: false,
            is_synthetic: true,
        };

        let score = self
            .base
            .evaluate_thought(&new_node, Some(&node.base))
            .await;
        new_node.set_score_or_default(score, 0.5);

        // Save the new node
        if let Err(e) = self.base.save_node(new_node.clone()).await {
            return Err(Box::new(e));
        }

        // Update parent
        let mut parent = node.clone();
        parent.base.children.push(new_node_id.clone());

        // Extract base node and save it
        let parent_base = parent.base.clone();
        if let Err(e) = self.base.save_node(parent_base).await {
            return Err(Box::new(e));
        }

        // Convert to MCTSNode and return (will use registry in thought_to_mcts)
        let mcts_node = self.thought_to_mcts(new_node).await?;

        Ok(mcts_node)
    }

    async fn simulate(
        &self,
        node: &MCTSNode,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node.clone();
        let mut total_score = current.base.score;
        let mut depth = current.base.depth;

        while depth < self.simulation_depth && !current.base.is_complete {
            let simulated_node_id = Uuid::new_v4().to_string();
            let simulated_thought = SYNTHETIC_EXPANSION.to_string();
            let mut simulated_node = ThoughtNode {
                id: simulated_node_id,
                thought: simulated_thought,
                depth: depth + 1,
                score: 0.0, // Score will be evaluated below
                children: vec![],
                parent_id: Some(current.base.id.clone()),
                is_complete: depth + 1 >= self.simulation_depth,
                is_synthetic: true,
            };

            let score = self
                .base
                .evaluate_thought(&simulated_node, Some(&current.base))
                .await;
            simulated_node.set_score_or_default(score, 0.5);
            total_score += simulated_node.score;

            // Update current to the simulated node (create new atomic counters)
            current = MCTSNode {
                base: simulated_node,
                visits: Arc::new(AtomicUsize::new(1)),
                total_reward: Arc::new(AtomicF64::new(0.0)),
                untried_actions: Some(vec![]),
            };

            depth += 1;
        }

        // Calculate steps with protection against edge cases
        let steps = (depth.saturating_sub(node.base.depth) + 1).max(1) as f64;
        Ok(BaseStrategy::safe_divide(total_score, steps, node.base.score))
    }

    async fn backpropagate(
        &self,
        node: MCTSNode,
        reward: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // NO LOCK NEEDED - atomics handle concurrency
        
        // Load entire path into memory
        let mut path = vec![node.clone()];
        let mut current_id = node.base.parent_id.clone();
        let mut visited = HashSet::new();
        visited.insert(node.base.id.clone());
        
        // Pre-load all ancestors
        while let Some(parent_id) = current_id {
            if !visited.insert(parent_id.clone()) {
                tracing::error!(
                    "Cycle detected during backpropagation at node {}",
                    parent_id
                );
                break;
            }
            
            if let Ok(Some(parent_node)) = self.base.get_node(&parent_id).await {
                let mcts_parent = self.thought_to_mcts(parent_node.clone()).await?;
                current_id = parent_node.parent_id.clone();
                path.push(mcts_parent);
            } else {
                break;
            }
        }
        
        // Update all nodes atomically - lock-free!
        for node in path {
            // Atomic increment - thread-safe, no races
            node.visits.fetch_add(1, Ordering::Relaxed);
            node.total_reward.fetch_add(reward, Ordering::Relaxed);
            
            // Note: We still save ThoughtNode, but without MCTS stats
            // The stats live in memory via stats_registry
            let updated_node = ThoughtNode {
                id: node.base.id.clone(),
                thought: node.base.thought.clone(),
                score: node.base.score,
                depth: node.base.depth,
                children: node.base.children.clone(),
                parent_id: node.base.parent_id.clone(),
                is_complete: node.base.is_complete,
                is_synthetic: node.base.is_synthetic,
            };
            
            if let Err(e) = self.base.save_node(updated_node).await {
                return Err(Box::new(e));
            }
        }
        
        Ok(())
    }

    fn select_best_uct(&self, nodes: Vec<MCTSNode>, parent_visits: usize) -> MCTSNode {
        nodes
            .into_iter()
            .fold(None, |best: Option<MCTSNode>, node| {
                // Load atomic values
                let visits = node.visits.load(Ordering::Relaxed);
                let total_reward = node.total_reward.load(Ordering::Relaxed);
                
                // Guard for unvisited nodes
                if visits == 0 {
                    tracing::warn!(
                        "Skipping node {} with 0 visits in UCT selection",
                        node.base.id
                    );
                    return best;
                }
                
                // Calculate UCT score
                let exploitation = total_reward / visits as f64;
                let exploration = ((parent_visits as f64).ln() / visits as f64).sqrt();
                let uct = exploitation + self.exploration_constant * exploration;

                match best {
                    None => Some(node),
                    Some(best_node) => {
                        let best_visits = best_node.visits.load(Ordering::Relaxed);
                        if best_visits == 0 {
                            return Some(node);
                        }
                        
                        let best_total_reward = best_node.total_reward.load(Ordering::Relaxed);
                        let best_exploitation = best_total_reward / best_visits as f64;
                        let best_exploration =
                            ((parent_visits as f64).ln() / best_visits as f64).sqrt();
                        let best_uct =
                            best_exploitation + self.exploration_constant * best_exploration;

                        if uct > best_uct {
                            Some(node)
                        } else {
                            Some(best_node)
                        }
                    }
                }
            })
            .unwrap_or_else(|| {
                // If no nodes provided, return a default MCTSNode
                MCTSNode {
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
                    visits: Arc::new(AtomicUsize::new(1)),
                    total_reward: Arc::new(AtomicF64::new(0.0)),
                    untried_actions: Some(vec![]),
                }
            })
    }

    fn calculate_path_score(&self, path: &[ThoughtNode]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        path.iter().map(|node| node.score).sum::<f64>() / path.len() as f64
    }

    async fn calculate_possible_paths(&self, node: &MCTSNode) -> usize {
        // Count actual paths explored down to simulation depth
        self.count_paths_iterative(&node.base.id, node.base.depth)
            .await
    }

    async fn count_paths_iterative(&self, node_id: &str, start_depth: usize) -> usize {
        // Check cache first (pattern from mcts_002_alpha.rs:145-165)
        {
            let cache = self.path_count_cache.lock().await;
            if let Some(&count) = cache.get(node_id) {
                return count;
            }
        }

        // Iterative BFS using VecDeque (pattern from mcts_002alt_alpha.rs:15-40)
        let mut queue = VecDeque::new();
        queue.push_back((node_id.to_string(), start_depth));
        
        let mut path_count = 0;
        
        // Cycle detection (pattern from state.rs:82-87)
        let mut visited = HashSet::new();
        
        // Iteration limit for safety (pattern from mcts_002alt_alpha.rs:359-379)
        const MAX_PATH_COUNT: usize = 1_000_000;
        let mut iterations = 0;
        
        while let Some((current_id, depth)) = queue.pop_front() {
            // Protection 1: Iteration limit
            iterations += 1;
            if iterations > MAX_PATH_COUNT {
                tracing::error!(
                    "Path counting exceeded {} iterations at node {} (possible infinite tree)",
                    MAX_PATH_COUNT,
                    current_id
                );
                break;
            }
            
            // Max depth check
            if depth >= self.simulation_depth {
                path_count += 1;
                continue;
            }
            
            // Protection 2: Cycle detection
            if !visited.insert(current_id.clone()) {
                tracing::warn!("Cycle detected during path counting at node {}", current_id);
                continue;
            }
            
            // Get children (already async, no recursion needed)
            let children = self.base.state_manager.get_children(&current_id).await;
            
            if children.is_empty() {
                path_count += 1;
            } else {
                for child in children {
                    if child.depth < self.simulation_depth {
                        queue.push_back((child.id.clone(), child.depth));
                    } else {
                        path_count += 1;
                    }
                }
            }
        }
        
        // Cache result (pattern from mcts_002_alpha.rs:160-165)
        {
            let mut cache = self.path_count_cache.lock().await;
            cache.insert(node_id.to_string(), path_count);
        }
        
        path_count
    }

    async fn thought_to_mcts(
        &self,
        node: ThoughtNode,
    ) -> Result<MCTSNode, Box<dyn std::error::Error + Send + Sync>> {
        let node_id = node.id.clone();
        
        // Get or create shared atomic statistics for this node
        let (visits, total_reward) = {
            let mut registry = self.stats_registry.lock().await;
            registry.entry(node_id.clone())
                .or_insert_with(|| {
                    // First time seeing this node - initialize with 1 visit
                    (Arc::new(AtomicUsize::new(1)), Arc::new(AtomicF64::new(0.0)))
                })
                .clone()  // Clone the Arc, not the atomics
        };
        
        Ok(MCTSNode {
            base: node,
            visits,
            total_reward,
            untried_actions: Some(vec![]),
        })
    }
}

// Add Clone implementation for MonteCarloTreeSearchStrategy
impl Clone for MonteCarloTreeSearchStrategy {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            exploration_constant: self.exploration_constant,
            simulation_depth: self.simulation_depth,
            num_simulations: self.num_simulations,
            root: Arc::clone(&self.root),
            stats_registry: Arc::clone(&self.stats_registry),
            path_count_cache: Arc::clone(&self.path_count_cache),
        }
    }
}

impl Strategy for MonteCarloTreeSearchStrategy {
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning {
        let (tx, rx) = mpsc::channel(1);
        let self_clone = self.clone();

        tokio::spawn(async move {
            let node_id = Uuid::new_v4().to_string();
            let parent_node = match &request.parent_id {
                Some(parent_id) => self_clone.base.get_node(parent_id).await.unwrap_or(None),
                None => None,
            };

            // CHECK 1: Before expensive evaluation
            if tx.is_closed() {
                tracing::debug!("MCTS: receiver dropped before evaluation, aborting");
                return;
            }

            let mut node = ThoughtNode {
                id: node_id.clone(),
                thought: request.thought.clone(),
                depth: request.thought_number - 1,
                score: 0.0,
                children: vec![],
                parent_id: request.parent_id.clone(),
                is_complete: !request.next_thought_needed,
                is_synthetic: false,
            };

            // Initialize node
            let score = self_clone
                .base
                .evaluate_thought(&node, parent_node.as_ref())
                .await;
            node.set_score_or_default(score, 0.5);
            if let Err(e) = self_clone.base.save_node_with_retry(node.clone(), None).await {
                tracing::error!("Failed to save node {} after retries: {}", node.id, e);
                // Continue with in-memory state
            }

            // Update parent if exists
            if let Some(mut parent) = parent_node {
                parent.children.push(node.id.clone());
                if let Err(e) = self_clone.base.save_node_with_retry(parent, None).await {
                    tracing::warn!("Failed to save parent node: {}", e);
                }
            }

            // Create MCTS node with atomic counters
            let visits = Arc::new(AtomicUsize::new(1));
            let total_reward = Arc::new(AtomicF64::new(node.score));
            
            let mcts_node = MCTSNode {
                base: node.clone(),
                visits: Arc::clone(&visits),
                total_reward: Arc::clone(&total_reward),
                untried_actions: Some(vec![]),
            };

            // Store in registry so thought_to_mcts finds it
            {
                let mut registry = self_clone.stats_registry.lock().await;
                registry.insert(
                    node.id.clone(),
                    (Arc::clone(&visits), Arc::clone(&total_reward))
                );
            }

            // If this is a root node, store it
            if node.parent_id.is_none() {
                let mut root = self_clone.root.lock().await;
                *root = Some(mcts_node.clone());
            }

            // CHECK 2: Before expensive simulations
            if tx.is_closed() {
                tracing::debug!("MCTS: receiver dropped before simulations, aborting");
                return;
            }

            // Run MCTS simulations
            if !node.is_complete
                && let Err(e) = self_clone.run_simulations(mcts_node).await
            {
                tracing::warn!("Simulations completed with errors: {}", e);
                // Continue - partial results still useful
            }

            // Calculate path statistics
            let current_path = self_clone.base.state_manager.get_path(&node_id).await;
            let path_score = self_clone.calculate_path_score(&current_path);

            // CHECK 3: Before calculating possible paths
            if tx.is_closed() {
                tracing::debug!("MCTS: receiver dropped before possible_paths, aborting");
                return;
            }

            // Calculate possible paths
            let mcts_node_for_paths = MCTSNode {
                base: node.clone(),
                visits: Arc::new(AtomicUsize::new(1)),
                total_reward: Arc::new(AtomicF64::new(node.score)),
                untried_actions: Some(vec![]),
            };
            let possible_paths = self_clone
                .calculate_possible_paths(&mcts_node_for_paths)
                .await;

            let response = ReasoningResponse {
                node_id: node.id,
                thought: node.thought,
                score: node.score,
                depth: node.depth,
                is_complete: node.is_complete,
                next_thought_needed: request.next_thought_needed,
                possible_paths: Some(possible_paths),
                best_score: Some(path_score),
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
                "mcts response"
            );
        });

        Reasoning::new(rx)
    }

    fn get_best_path(&self) -> AsyncPath {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let root_opt = self_clone.root.lock().await.clone();

            if let Some(root) = root_opt {
                let children = self_clone
                    .base
                    .state_manager
                    .get_children(&root.base.id)
                    .await;
                if children.is_empty() {
                    let _ = tx.send(Ok(vec![]));
                    return;
                }

                let mut best_child: Option<ThoughtNode> = None;
                let mut max_visits = 0;

                for child in children {
                    let child_id = child.id.clone();
                    if let Ok(Some(child_node)) = self_clone.base.get_node(&child_id).await {
                        let mcts_child = match self_clone.thought_to_mcts(child_node).await {
                            Ok(node) => node,
                            Err(_) => continue,
                        };

                        // Load visits atomically
                        let visits = mcts_child.visits.load(Ordering::Relaxed);
                        if visits > max_visits {
                            max_visits = visits;
                            best_child = Some(mcts_child.base);
                        }
                    }
                }

                if let Some(best) = best_child {
                    let path = self_clone.base.state_manager.get_path(&best.id).await;
                    let _ = tx.send(Ok(path));
                    return;
                }
            }

            let _ = tx.send(Ok(vec![]));
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
                    name: String::from("Monte Carlo Tree Search"),
                    nodes_explored: 0,
                    average_score: 0.0,
                    max_depth: 0,
                    active: None,
                    extra: Default::default(),
                },
            };

            let mut metrics = base_metrics;

            let root_visits = match &*self_clone.root.lock().await {
                Some(root) => root.visits.load(Ordering::Relaxed),
                None => 0,
            };

            metrics.name = "Monte Carlo Tree Search".to_string();
            metrics.extra.insert(
                "simulation_depth".to_string(),
                self_clone.simulation_depth.into(),
            );
            metrics.extra.insert(
                "num_simulations".to_string(),
                self_clone.num_simulations.into(),
            );
            metrics.extra.insert(
                "exploration_constant".to_string(),
                self_clone.exploration_constant.into(),
            );
            metrics
                .extra
                .insert("total_simulations".to_string(), root_visits.into());

            let _ = tx.send(Ok(metrics)).await;
        });

        MetricStream::new(rx)
    }

    fn clear(&self) -> ClearedSignal {
        let self_clone = self.clone();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            // Clear root node
            let mut root = self_clone.root.lock().await;
            *root = None;
            
            // Reset all atomic statistics in registry
            let mut registry = self_clone.stats_registry.lock().await;
            for (visits, total_reward) in registry.values() {
                visits.store(0, std::sync::atomic::Ordering::Relaxed);
                total_reward.store(0.0, std::sync::atomic::Ordering::Relaxed);
            }
            registry.clear();
            
            // Clear path count cache
            let mut cache = self_clone.path_count_cache.lock().await;
            cache.clear();
            
            let _ = tx.send(Ok(()));
        });

        ClearedSignal::new(rx)
    }
}
