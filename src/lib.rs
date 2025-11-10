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

use kodegen_mcp_tool::{Tool, error::McpError};
use kodegen_mcp_schema::reasoning::{ReasonerArgs, ReasonerPromptArgs};
use rmcp::model::{PromptArgument, PromptMessage, PromptMessageContent, PromptMessageRole};
use serde_json::Value;
use std::sync::Arc;

// ============================================================================
// TOOL IMPLEMENTATION
// ============================================================================

/// Advanced reasoning tool with multiple strategies
#[derive(Clone)]
pub struct SequentialThinkingReasonerTool {
    reasoner: Arc<Reasoner>,
}

impl SequentialThinkingReasonerTool {
    /// Create new reasoner tool with optional cache size
    pub fn new(cache_size: Option<usize>) -> Self {
        Self {
            reasoner: Arc::new(Reasoner::new(cache_size)),
        }
    }
}

impl Tool for SequentialThinkingReasonerTool {
    type Args = ReasonerArgs;
    type PromptArgs = ReasonerPromptArgs;

    fn name() -> &'static str {
        "sequential_thinking_reasoner"
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

    async fn execute(&self, args: Self::Args) -> Result<Value, McpError> {
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

        // Convert to JSON
        serde_json::to_value(&response)
            .map_err(|e| McpError::Other(anyhow::anyhow!("Serialization failed: {}", e)))
    }

    fn prompt_arguments() -> Vec<PromptArgument> {
        vec![]
    }

    async fn prompt(&self, _args: Self::PromptArgs) -> Result<Vec<PromptMessage>, McpError> {
        Ok(vec![
            PromptMessage {
                role: PromptMessageRole::User,
                content: PromptMessageContent::text(
                    "How do I use the sequential_thinking_reasoner tool with different strategies?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "The sequential_thinking_reasoner tool supports 4 reasoning strategies:\n\n\
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

/// Start the reasoner HTTP server programmatically for embedded mode
pub async fn start_server(
    addr: std::net::SocketAddr,
    tls_cert: Option<std::path::PathBuf>,
    tls_key: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    use kodegen_server_http::{Managers, RouterSet, register_tool};
    use kodegen_tools_config::ConfigManager;
    use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};
    use std::sync::Arc;

    let _ = env_logger::try_init();

    if rustls::crypto::ring::default_provider().install_default().is_err() {
        log::debug!("rustls crypto provider already installed");
    }

    let config = ConfigManager::new();
    config.init().await?;

    let timestamp = chrono::Utc::now();
    let pid = std::process::id();
    let instance_id = format!("{}-{}", timestamp.format("%Y%m%d-%H%M%S-%9f"), pid);
    let usage_tracker = kodegen_utils::usage_tracker::UsageTracker::new(
        format!("reasoner-{}", instance_id)
    );

    kodegen_mcp_tool::tool_history::init_global_history(instance_id.clone()).await;

    let mut tool_router = ToolRouter::new();
    let mut prompt_router = PromptRouter::new();
    let managers = Managers::new();

    // Register reasoner tool (default cache size = None)
    (tool_router, prompt_router) = register_tool(
        tool_router,
        prompt_router,
        crate::SequentialThinkingReasonerTool::new(None),
    );

    let router_set = RouterSet::new(tool_router, prompt_router, managers);

    let session_config = rmcp::transport::streamable_http_server::session::local::SessionConfig {
        channel_capacity: 16,
        keep_alive: Some(std::time::Duration::from_secs(3600)),
    };
    let session_manager = Arc::new(
        rmcp::transport::streamable_http_server::session::local::LocalSessionManager {
            sessions: Default::default(),
            session_config,
        }
    );

    let server = kodegen_server_http::HttpServer::new(
        router_set.tool_router,
        router_set.prompt_router,
        usage_tracker,
        config,
        router_set.managers,
        session_manager,
    );

    let shutdown_timeout = std::time::Duration::from_secs(30);
    let tls_config = tls_cert.zip(tls_key);
    let handle = server.serve_with_tls(addr, tls_config, shutdown_timeout).await?;

    handle.wait_for_completion(shutdown_timeout).await
        .map_err(|e| anyhow::anyhow!("Server shutdown error: {}", e))?;

    Ok(())
}
