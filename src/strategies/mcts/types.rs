use crate::atomics::AtomicF64;
use crate::state::StateManager;
use crate::strategies::base::BaseStrategy;
use crate::types::{CONFIG, ThoughtNode};
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Synthetic thoughts are never evaluated (see base.rs:402-404)
/// so we use a constant to avoid allocation overhead
pub const SYNTHETIC_EXPANSION: &str = "[synthetic]";

/// Type alias for node statistics registry storing atomic counters
pub type StatsRegistry = Arc<Mutex<HashMap<String, (Arc<AtomicUsize>, Arc<AtomicF64>)>>>;

#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub base: ThoughtNode,
    
    // Atomic counters - shared across all clones via Arc
    pub visits: Arc<AtomicUsize>,
    pub total_reward: Arc<AtomicF64>,
    
    pub untried_actions: Option<Vec<String>>,
}

pub struct MonteCarloTreeSearchStrategy {
    pub base: BaseStrategy,
    pub exploration_constant: f64,
    pub simulation_depth: usize,
    pub num_simulations: usize,
    pub root: Arc<Mutex<Option<MCTSNode>>>,

    // Registry mapping node IDs to shared atomic statistics
    pub stats_registry: StatsRegistry,

    // Cache for path counts to avoid redundant calculations
    pub path_count_cache: Arc<Mutex<HashMap<String, usize>>>,
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
