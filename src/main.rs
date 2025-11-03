// Category HTTP Server: Reasoner Tools
//
// This binary serves advanced reasoning tools over HTTP/HTTPS transport.
// Managed by kodegend daemon, typically running on port 30453.

use anyhow::Result;
use kodegen_server_http::{run_http_server, Managers, RouterSet, register_tool};
use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};

#[tokio::main]
async fn main() -> Result<()> {
    run_http_server("reasoner", |_config, _tracker| {
        let mut tool_router = ToolRouter::new();
        let mut prompt_router = PromptRouter::new();
        let managers = Managers::new();

        // Register reasoner tool (uses default cache size)
        (tool_router, prompt_router) = register_tool(
            tool_router,
            prompt_router,
            kodegen_tools_reasoner::SequentialThinkingReasonerTool::new(None),
        );

        Ok(RouterSet::new(tool_router, prompt_router, managers))
    })
    .await
}
