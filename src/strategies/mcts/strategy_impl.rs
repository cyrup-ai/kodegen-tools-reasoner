use super::types::{MCTSNode, MonteCarloTreeSearchStrategy};
use crate::atomics::AtomicF64;
use crate::strategies::base::{
    AsyncPath, BaseStrategy, ClearedSignal, Metric, MetricStream, Reasoning, Strategy,
};
use crate::types::{ReasoningRequest, ReasoningResponse};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing;
use uuid::Uuid;

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

            let mut node = crate::types::ThoughtNode {
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

                let mut best_child: Option<crate::types::ThoughtNode> = None;
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
