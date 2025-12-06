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

use kodegen_mcp_schema::{Tool, ToolExecutionContext, ToolResponse, McpError};
use kodegen_mcp_schema::reasoning::{ReasonerArgs, ReasonerOutput, ReasonerPrompts};
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
    type Prompts = ReasonerPrompts;

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

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_schema::ToolArgs>::Output>, McpError> {
        // Convert args to internal ReasoningRequest
        let thought_content = args.thought.clone();
        let request = ReasoningRequest {
            thought: args.thought,
            thought_number: args.thought_number,
            total_thoughts: args.total_thoughts,
            next_thought_needed: args.next_thought_needed,
            parent_id: args.parent_id.clone(),
            strategy_type: args.strategy_type.clone(),
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

        // Build terminal summary
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

        // Build typed output
        let output = ReasonerOutput {
            session_id: response.node_id.clone(),
            thought_number: args.thought_number,
            total_thoughts: args.total_thoughts,
            thought: thought_content,
            strategy: strategy.to_string(),
            next_thought_needed: response.next_thought_needed,
            best_path_score: response.best_score,
            branches: response.possible_paths,
            history_length: response.stats.total_nodes,
            score: response.score,
            depth: response.depth,
            is_complete: response.is_complete,
        };

        Ok(ToolResponse::new(summary, output))
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

/// Start reasoner HTTP server using pre-bound listener (TOCTOU-safe)
///
/// This variant is used by kodegend to eliminate TOCTOU race conditions
/// during port cleanup. The listener is already bound to a port.
///
/// # Arguments
/// * `listener` - Pre-bound TcpListener (port already reserved)
/// * `tls_config` - Optional (cert_path, key_path) for HTTPS
///
/// # Returns
/// ServerHandle for graceful shutdown, or error if startup fails
pub async fn start_server_with_listener(
    listener: tokio::net::TcpListener,
    tls_config: Option<(std::path::PathBuf, std::path::PathBuf)>,
) -> anyhow::Result<kodegen_server_http::ServerHandle> {
    use kodegen_server_http::{create_http_server_with_listener, Managers, RouterSet, register_tool};
    use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};
    use std::time::Duration;

    let shutdown_timeout = Duration::from_secs(30);
    let session_keep_alive = Duration::ZERO;

    create_http_server_with_listener("reasoner", listener, tls_config, shutdown_timeout, session_keep_alive, |_config, _tracker| {
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
