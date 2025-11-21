//! Advanced reasoning tool with multiple strategies
//!
//! Provides Beam Search, MCTS, and experimental MCTS variants for
//! complex problem-solving with branching and revision support.

mod atomics;
mod reasoner;
mod state;
mod strategies;
mod types;

pub use reasoner::Reasoner;
pub use types::*;

use kodegen_mcp_tool::{Tool, ToolExecutionContext, error::McpError};
use kodegen_mcp_schema::reasoning::{ReasonerArgs, ReasonerPromptArgs};
use rmcp::model::{Content, PromptArgument, PromptMessage, PromptMessageContent, PromptMessageRole};
use std::sync::Arc;

// ============================================================================
// TOOL IMPLEMENTATION
// ============================================================================

/// Advanced reasoning tool with multiple strategies
#[derive(Clone)]
pub struct ReasonerTool {
    reasoner: Arc<Reasoner>,
}

impl ReasonerTool {
    /// Create new reasoner tool with optional cache size
    pub fn new(cache_size: Option<usize>) -> Self {
        Self {
            reasoner: Arc::new(Reasoner::new(cache_size)),
        }
    }
}

impl Tool for ReasonerTool {
    type Args = ReasonerArgs;
    type PromptArgs = ReasonerPromptArgs;

    fn name() -> &'static str {
        kodegen_mcp_schema::reasoning::REASONER
    }

    fn description() -> &'static str {
        "Advanced reasoning tool with multiple strategies (Beam Search, MCTS). \
         Processes thoughts step-by-step, supports branching and revision, \
         and tracks best reasoning paths. Use for complex problem-solving \
         that requires exploration of multiple solution approaches.\n\n\
         Strategies:\n\
         - beam_search: Breadth-first exploration (default)\n\
         - mcts: Monte Carlo Tree Search with UCB1\n\
         - mcts_002_alpha: High exploration MCTS variant\n\
         - mcts_002alt_alpha: Length-rewarding MCTS variant\n\n\
         Optional VoyageAI Embedding Integration:\n\
         Set VOYAGE_API_KEY environment variable to enable semantic coherence scoring."
    }

    fn read_only() -> bool {
        true // Only tracks reasoning state, no external modifications
    }

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<Vec<Content>, McpError> {
        // Convert args to internal ReasoningRequest
        let request = ReasoningRequest {
            thought: args.thought,
            thought_number: args.thought_number,
            total_thoughts: args.total_thoughts,
            next_thought_needed: args.next_thought_needed,
            parent_id: args.parent_id,
            strategy_type: args.strategy_type,
            beam_width: args.beam_width,
            num_simulations: args.num_simulations,
        };

        // Process thought via reasoner (already async)
        let mut response = self.reasoner.process_thought(request).await;

        // Echo input fields for client convenience
        response.thought_number = args.thought_number;
        response.total_thoughts = args.total_thoughts;

        // Compute and embed stats for all strategies
        let stats = self.reasoner.get_stats(vec![
            "beam_search",
            "mcts",
            "mcts_002_alpha",
            "mcts_002alt_alpha",
        ]).await;
        response.stats = stats;

        // Build Vec<Content> with two items
        let mut contents = Vec::new();

        // ========================================
        // Content[0]: Terminal-Formatted Summary
        // ========================================
        let strategy = response.strategy_used.as_deref().unwrap_or("unknown");
        let summary = format!(
            "\x1b[35m Reasoning Node: {}\x1b[0m\n\
              Score: {:.3} · Depth: {} · Strategy: {}\n\
              Complete: {} · Next needed: {}",
            response.node_id,
            response.score,
            response.depth,
            strategy,
            response.is_complete,
            response.next_thought_needed
        );
        contents.push(Content::text(summary));

        // ========================================
        // Content[1]: Machine-Parseable JSON
        // ========================================
        let metadata = serde_json::to_value(&response)
            .map_err(|e| McpError::Other(anyhow::anyhow!("Serialization failed: {}", e)))?;
        let json_str = serde_json::to_string_pretty(&metadata)
            .unwrap_or_else(|_| "{}".to_string());
        contents.push(Content::text(json_str));

        Ok(contents)
    }

    fn prompt_arguments() -> Vec<PromptArgument> {
        vec![]
    }

    async fn prompt(&self, _args: Self::PromptArgs) -> Result<Vec<PromptMessage>, McpError> {
        Ok(vec![
            PromptMessage {
                role: PromptMessageRole::User,
                content: PromptMessageContent::text(
                    "How do I use the reasoner tool with different strategies?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "The reasoner tool supports 4 reasoning strategies:\n\n\
                     1. **beam_search** (default): Explores top N paths simultaneously\n\
                        - Use for breadth-first exploration\n\
                        - Set beamWidth to control path count (default: 3)\n\n\
                     2. **mcts**: Monte Carlo Tree Search with UCB1\n\
                        - Use for exploration-exploitation balance\n\
                        - Set numSimulations to control search depth (default: 50)\n\n\
                     3. **mcts_002_alpha**: MCTS with higher exploration\n\
                        - Use for creative problem-solving\n\
                        - 10% exploration boost\n\n\
                     4. **mcts_002alt_alpha**: MCTS with length bonus\n\
                        - Use for detailed reasoning\n\
                        - Rewards thorough explanations\n\n\
                     Example usage:\n\
                     ```json\n\
                     {\n\
                       \"thought\": \"Analyzing algorithm complexity\",\n\
                       \"thoughtNumber\": 1,\n\
                       \"totalThoughts\": 5,\n\
                       \"nextThoughtNeeded\": true,\n\
                       \"strategyType\": \"mcts\",\n\
                       \"numSimulations\": 100\n\
                     }\n\
                     ```\n\n\
                     The tool returns:\n\
                     - nodeId: Unique identifier for this thought\n\
                     - score: Quality score (0.0-1.0)\n\
                     - depth: Current reasoning depth\n\
                     - strategyUsed: Which strategy was applied\n\
                     - bestScore: Highest score in current path\n\n\
                     Optional: Set VOYAGE_API_KEY environment variable to enable \n\
                     semantic coherence scoring using VoyageAI embeddings.",
                ),
            },
        ])
    }
}

// ============================================================================
// EMBEDDED SERVER FUNCTION
// ============================================================================

/// Start the reasoner HTTP server programmatically
///
/// Returns a ServerHandle for graceful shutdown control.
/// This function is non-blocking - the server runs in background tasks.
///
/// # Arguments
/// * `addr` - Socket address to bind to
/// * `tls_cert` - Optional path to TLS certificate file
/// * `tls_key` - Optional path to TLS private key file
///
/// # Returns
/// ServerHandle for graceful shutdown, or error if startup fails
pub async fn start_server(
    addr: std::net::SocketAddr,
    tls_cert: Option<std::path::PathBuf>,
    tls_key: Option<std::path::PathBuf>,
) -> anyhow::Result<kodegen_server_http::ServerHandle> {
    use kodegen_server_http::{create_http_server, Managers, RouterSet, register_tool};
    use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};
    use std::time::Duration;

    let tls_config = match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => Some((cert, key)),
        _ => None,
    };

    let shutdown_timeout = Duration::from_secs(30);
    let session_keep_alive = Duration::ZERO;

    create_http_server("reasoner", addr, tls_config, shutdown_timeout, session_keep_alive, |_config, _tracker| {
        Box::pin(async move {
            let mut tool_router = ToolRouter::new();
            let mut prompt_router = PromptRouter::new();
            let managers = Managers::new();

            // Register reasoner tool (default cache size = None)
            (tool_router, prompt_router) = register_tool(
                tool_router,
                prompt_router,
                crate::ReasonerTool::new(None),
            );

            Ok(RouterSet::new(tool_router, prompt_router, managers))
        })
    }).await
}
