use super::scoring::ScoringEngine;
use super::types::{PolicyGuidedNode, SYNTHETIC_EXPANSION};
use crate::types::ThoughtNode;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

pub struct TreeOperations {
    pub scoring: ScoringEngine,
    pub backprop_lock: Arc<Mutex<()>>,
    pub learning_rate: f64,
}

impl TreeOperations {
    pub fn new(scoring: ScoringEngine, learning_rate: f64) -> Self {
        Self {
            scoring,
            backprop_lock: Arc::new(Mutex::new(())),
            learning_rate,
        }
    }

    pub async fn run_policy_guided_search<F, Fut>(
        &self,
        node: PolicyGuidedNode,
        simulation_count: usize,
        adapt_fn: F,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: Fn(&PolicyGuidedNode) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = ()> + Send,
    {
        for _ in 0..simulation_count {
            let selected_node = self.select_with_puct(node.clone()).await?;
            let expanded_node = self.expand_with_policy(selected_node).await?;
            let reward = self.simulate_with_value_guidance(&expanded_node).await?;
            let expanded_node_clone = expanded_node.clone(); // Clone before moving
            self.backpropagate_with_policy_update(expanded_node, reward)
                .await?;

            // Adapt exploration rate
            adapt_fn(&expanded_node_clone).await;
        }

        Ok(())
    }

    pub async fn select_with_puct(
        &self,
        root: PolicyGuidedNode,
    ) -> Result<PolicyGuidedNode, Box<dyn std::error::Error + Send + Sync>> {
        let mut node = root;

        while !node.base.children.is_empty() {
            let mut children = Vec::new();
            for id in &node.base.children {
                if let Ok(Some(child_node)) = self.scoring.base.get_node(id).await
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

    pub async fn select_best_puct_child(&self, nodes: Vec<PolicyGuidedNode>, parent_visits: usize) -> PolicyGuidedNode {
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
            .unwrap_or_else(PolicyGuidedNode::default_node)
    }

    pub async fn expand_with_policy(
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
        let thought_identifier = self.scoring.get_thought_identifier(&new_thought);
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
            prior_action_probs: std::collections::HashMap::new(),
            puct: None,
            action_history,
            novelty_score: None,
        };

        new_node.novelty_score = Some(self.scoring.calculate_novelty_v2(&new_node));
        // Calculate policy score with semantic coherence
        new_node.policy_score = self.scoring.calculate_policy_score(&new_node, Some(&node)).await;
        let score = self.scoring
            .base
            .evaluate_thought(&new_node.base, Some(&node.base))
            .await;
        new_node.base.set_score_or_default(score, 0.5);
        // Await the async calculation
        new_node.value_estimate = self.scoring.estimate_value(&new_node).await;

        // Save the base node
        if let Err(e) = self.scoring.base.save_node_with_retry(base_node, None).await {
            tracing::error!("Failed to save expanded node: {}", e);
            return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
        }

        Ok(new_node)
    }

    pub async fn simulate_with_value_guidance(
        &self,
        node: &PolicyGuidedNode,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut current = node.clone();
        let mut total_reward = 0.0;
        let mut depth = 0;

        while !current.base.is_complete && depth < crate::types::CONFIG.max_depth {
            // Await the async value estimate
            let reward = self.scoring.estimate_value(&current).await;
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

    pub async fn backpropagate_with_policy_update(
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
            
            if let Ok(Some(parent_node)) = self.scoring.base.get_node(&parent_id).await {
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
            let _current_value_estimate = self.scoring.estimate_value(&current).await;
            let new_value = (1.0 - self.learning_rate) * current.value_estimate
                + self.learning_rate * reward;
            current.value_estimate = new_value;
            
            // Update parent's action probabilities if not root
            if i + 1 < path.len() {
                let mut parent = path[i + 1].clone();
                let thought_key = self.scoring.get_thought_identifier(&current.base.thought);
                let current_prob = *parent.prior_action_probs
                    .get(&thought_key)
                    .unwrap_or(&0.0);
                let new_prob = current_prob + self.learning_rate * (reward - current_prob);
                parent.prior_action_probs.insert(thought_key, new_prob);
                
                // Save updated parent (with retry for transient failures)
                if let Err(e) = self.scoring.base.save_node_with_retry(parent.base.clone(), None).await {
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
            if let Err(e) = self.scoring.base.save_node_with_retry(current.base.clone(), None).await {
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

    pub async fn calculate_policy_enhanced_score(&self, path: &[ThoughtNode]) -> f64 {
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
            policy_node.policy_score = self.scoring
                .calculate_policy_score(&policy_node, parent_policy_node.as_ref())
                .await;
            
            policy_node.value_estimate = self.scoring.estimate_value(&policy_node).await;
            
            policy_node.novelty_score = Some(self.scoring.calculate_novelty(&policy_node));

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
            prior_action_probs: std::collections::HashMap::new(),
            puct: None,
            action_history: Some(vec![]),
            novelty_score: Some(0.5), // Default value
        })
    }
}
