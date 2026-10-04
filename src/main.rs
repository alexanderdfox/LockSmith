//! # Multi-Agent Secure SSH Key Manager
//! Safety-critical implementation featuring multi-ai provider routing, 
//! strict file permissions, an advanced Ratatui TUI dashboard with API key updating, and an Axum Web Server mode.

#![deny(warnings)]
#![deny(clippy::all)]
#![warn(missing_docs)]

use axum::{
    routing::post,
    extract::State,
    Json, Router,
};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use zeroize::Zeroizing;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

/// Maximum allowed length for user intent strings to prevent resource exhaustion.
const MAX_INTENT_LENGTH: usize = 256;

/// Timeout limit for external API communications in seconds.
const API_TIMEOUT_SECS: u64 = 10;

/// System keyring service identifier namespace.
const KEYRING_SERVICE_NAME: &str = "keepr-agent";

// ---------------------------------------------------------------------------
// AI Provider Selection
// ---------------------------------------------------------------------------

/// Supported AI provider options for semantic routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiProvider {
    /// xAI Grok provider.
    Grok,
    /// OpenAI GPT-4o provider.
    OpenAi,
    /// Anthropic Claude provider.
    Anthropic,
    /// Google Gemini provider.
    Gemini,
}

impl AiProvider {
    /// Cycles to the next available AI provider.
    pub const fn next(self) -> Self {
        match self {
            Self::Grok => Self::OpenAi,
            Self::OpenAi => Self::Anthropic,
            Self::Anthropic => Self::Gemini,
            Self::Gemini => Self::Grok,
        }
    }

    /// Returns the associated environment variable name for fallback.
    pub const fn env_var_name(self) -> &'static str {
        match self {
            Self::Grok => "XAI_API_KEY",
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
        }
    }

    /// Returns the secure keyring username/key identifier string.
    pub const fn keyring_key(self) -> &'static str {
        match self {
            Self::Grok => "grok_api_key",
            Self::OpenAi => "openai_api_key",
            Self::Anthropic => "anthropic_api_key",
            Self::Gemini => "gemini_api_key",
        }
    }
}

// ---------------------------------------------------------------------------
// Rigorous Error Handling
// ---------------------------------------------------------------------------

/// Exhaustive error types for secure operations.
#[derive(Debug, Serialize)]
pub enum SshAgentError {
    /// No matching key tool found for the given intent.
    NoSuitableKey(String),
    /// Target cryptographic key file does not exist.
    KeyNotFound,
    /// Key file permissions exceed safety thresholds.
    InsecureKeyPermissions,
    /// External process execution failed.
    ExecutionFailed,
    /// Remote AI communication failed.
    ApiError,
    /// User input failed validation bounds.
    ValidationError(String),
    /// Key ring contains no operational assets.
    EmptyRing,
}

impl fmt::Display for SshAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuitableKey(intent) => {
                write!(f, "Access denied or no matching key found for intent: '{}'", intent)
            }
            Self::KeyNotFound => {
                write!(f, "Security policy violation: Key file does not exist or is inaccessible.")
            }
            Self::InsecureKeyPermissions => {
                write!(f, "CRITICAL SECURITY ERROR: Private key permissions exceed safe boundaries (strictly 0600).")
            }
            Self::ExecutionFailed => {
                write!(f, "System command execution failed securely.")
            }
            Self::ApiError => {
                write!(f, "Remote intelligence service communication failed securely.")
            }
            Self::ValidationError(msg) => {
                write!(f, "Input validation failed: {}", msg)
            }
            Self::EmptyRing => {
                write!(f, "The secure key ring contains no registered operational assets.")
            }
        }
    }
}

impl std::error::Error for SshAgentError {}

// ---------------------------------------------------------------------------
// Core Tool Abstraction
// ---------------------------------------------------------------------------

/// Trait defining a secure, executable cryptographic tool asset.
pub trait SshKeyTool: Send + Sync {
    /// Returns the unique identifier name of the tool.
    fn name(&self) -> &str;
    /// Returns a human-readable description of the tool's purpose.
    fn description(&self) -> &str;
    /// Returns the file path associated with the cryptographic asset.
    fn key_path(&self) -> &PathBuf;

    /// Executes the key loading process under rigorous security validations.
    fn execute(&self) -> Result<String, SshAgentError> {
        let path = self.key_path();
        
        if !path.exists() {
            return Err(SshAgentError::KeyNotFound);
        }

        #[cfg(unix)]
        {
            let metadata = std::fs::metadata(path)
                .map_err(|_| SshAgentError::KeyNotFound)?;
            let mode = metadata.permissions().mode();
            if mode & 0o077 != 0 {
                return Err(SshAgentError::InsecureKeyPermissions);
            }
        }

        let output = Command::new("ssh-add")
            .arg(path)
            .output()
            .map_err(|_| SshAgentError::ExecutionFailed)?;

        if output.status.success() {
            Ok(format!("Securely loaded verified SSH asset '{}'", self.name()))
        } else {
            Err(SshAgentError::ExecutionFailed)
        }
    }
}

/// Concrete GitHub authentication key asset wrapper.
pub struct GithubKey {
    path: PathBuf,
}

impl GithubKey {
    /// Instantiates a new GitHub key wrapper.
    pub const fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl SshKeyTool for GithubKey {
    fn name(&self) -> &str { "github" }
    fn description(&self) -> &str { "GitHub Authentication Asset" }
    fn key_path(&self) -> &PathBuf { &self.path }
}

/// Concrete Production Server authentication key asset wrapper.
pub struct ProductionServerKey {
    path: PathBuf,
}

impl ProductionServerKey {
    /// Instantiates a new Production key wrapper.
    pub const fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl SshKeyTool for ProductionServerKey {
    fn name(&self) -> &str { "production" }
    fn description(&self) -> &str { "Production Infrastructure Asset" }
    fn key_path(&self) -> &PathBuf { &self.path }
}

// ---------------------------------------------------------------------------
// Multi-Provider AI Routing Integration
// ---------------------------------------------------------------------------

async fn route_via_ai(intent: &str, provider: AiProvider, api_key: &str) -> Result<String, SshAgentError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(API_TIMEOUT_SECS))
        .build()
        .map_err(|_| SshAgentError::ApiError)?;

    let choice = match provider {
        AiProvider::Grok => {
            #[derive(Serialize)]
            struct Req { model: String, messages: Vec<Msg> }
            #[derive(Serialize, Deserialize)]
            struct Msg { role: String, content: String }

            let payload = Req {
                model: "grok-4.7".to_string(),
                messages: vec![
                    Msg { role: "system".to_string(), content: "Classify intent into exactly one word: 'github' or 'production'. If ambiguous, output 'none'.".to_string() },
                    Msg { role: "user".to_string(), content: intent.to_string() },
                ],
            };
            let res = client.post("https://api.x.ai/v1/chat/completions")
                .bearer_auth(api_key).json(&payload).send().await
                .map_err(|_| SshAgentError::ApiError)?
                .json::<serde_json::Value>().await
                .map_err(|_| SshAgentError::ApiError)?;
            res["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_lowercase()
        }
        AiProvider::OpenAi => {
            #[derive(Serialize)]
            struct Req { model: String, messages: Vec<Msg> }
            #[derive(Serialize, Deserialize)]
            struct Msg { role: String, content: String }

            let payload = Req {
                model: "gpt-4o".to_string(),
                messages: vec![
                    Msg { role: "system".to_string(), content: "Classify intent into exactly one word: 'github' or 'production'. If ambiguous, output 'none'.".to_string() },
                    Msg { role: "user".to_string(), content: intent.to_string() },
                ],
            };
            let res = client.post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(api_key).json(&payload).send().await
                .map_err(|_| SshAgentError::ApiError)?
                .json::<serde_json::Value>().await
                .map_err(|_| SshAgentError::ApiError)?;
            res["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_lowercase()
        }
        AiProvider::Anthropic => {
            #[derive(Serialize)]
            struct Req { model: String, max_tokens: u32, system: String, messages: Vec<Msg> }
            #[derive(Serialize, Deserialize)]
            struct Msg { role: String, content: String }

            let payload = Req {
                model: "claude-3-5-sonnet-20241022".to_string(),
                max_tokens: 10,
                system: "Classify intent into exactly one word: 'github' or 'production'. If ambiguous, output 'none'.".to_string(),
                messages: vec![Msg { role: "user".to_string(), content: intent.to_string() }],
            };
            let res = client.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&payload).send().await
                .map_err(|_| SshAgentError::ApiError)?
                .json::<serde_json::Value>().await
                .map_err(|_| SshAgentError::ApiError)?;
            res["content"][0]["text"].as_str().unwrap_or("").trim().to_lowercase()
        }
        AiProvider::Gemini => {
            #[derive(Serialize)]
            struct Req { contents: Vec<Content> }
            #[derive(Serialize)]
            struct Content { parts: Vec<Part> }
            #[derive(Serialize)]
            struct Part { text: String }

            let payload = Req {
                contents: vec![Content { parts: vec![Part { text: format!("Classify this intent into exactly one word ('github' or 'production'): {}", intent) }] }],
            };
            let url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={}", api_key);
            let res = client.post(&url)
                .json(&payload).send().await
                .map_err(|_| SshAgentError::ApiError)?
                .json::<serde_json::Value>().await
                .map_err(|_| SshAgentError::ApiError)?;
            res["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("").trim().to_lowercase()
        }
    };

    if choice.contains("github") {
        Ok("github".to_string())
    } else if choice.contains("production") {
        Ok("production".to_string())
    } else {
        Err(SshAgentError::NoSuitableKey(intent.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Secure Agent Registry
// ---------------------------------------------------------------------------

/// Manages the registry of secure key tools and coordinates processing.
pub struct SshKeyAgent {
    keys: HashMap<String, Arc<dyn SshKeyTool>>,
    api_key: Zeroizing<String>,
}

impl SshKeyAgent {
    /// Initializes a new agent registry with a securely bound token.
    pub fn new(api_key: Zeroizing<String>) -> Self {
        Self { keys: HashMap::new(), api_key }
    }

    /// Registers a secure key tool into the agent's collection.
    pub fn register(mut self, tool: Arc<dyn SshKeyTool>) -> Self {
        self.keys.insert(tool.name().to_string(), tool);
        self
    }

    /// Returns a list of all registered key names and paths.
    pub fn list_keys(&self) -> Vec<(String, PathBuf)> {
        self.keys.values().map(|t| (t.name().to_string(), t.key_path().clone())).collect()
    }

    /// Updates the internal API key securely.
    pub fn update_api_key(&mut self, new_key: Zeroizing<String>) {
        self.api_key = new_key;
    }

    /// Validates input and processes user intent securely via the selected AI.
    pub async fn process_intent(&self, intent: &str, provider: AiProvider) -> Result<String, SshAgentError> {
        if intent.is_empty() || intent.len() > MAX_INTENT_LENGTH {
            return Err(SshAgentError::ValidationError(format!(
                "Intent length must be between 1 and {} characters.",
                MAX_INTENT_LENGTH
            )));
        }

        if self.keys.is_empty() {
            return Err(SshAgentError::EmptyRing);
        }

        let tool_name = route_via_ai(intent, provider, &self.api_key).await?;

        let tool = self.keys
            .get(&tool_name)
            .ok_or_else(|| SshAgentError::NoSuitableKey(intent.to_string()))?;

        tool.execute()
    }
}

// ---------------------------------------------------------------------------
// Web API Payload Structures
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WebIntentRequest {
    intent: String,
    provider: Option<AiProvider>,
}

#[derive(Serialize)]
struct WebIntentResponse {
    success: bool,
    message: String,
}

async fn handle_web_intent(
    State((agent, default_provider)): State<(Arc<Mutex<SshKeyAgent>>, AiProvider)>,
    Json(payload): Json<WebIntentRequest>,
) -> Json<WebIntentResponse> {
    let agent_lock = agent.lock().await;
    let provider = payload.provider.unwrap_or(default_provider);
    match agent_lock.process_intent(&payload.intent, provider).await {
        Ok(msg) => Json(WebIntentResponse { success: true, message: msg }),
        Err(e) => Json(WebIntentResponse { success: false, message: e.to_string() }),
    }
}

/// Helper utility to center popup rectangles on screen.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ---------------------------------------------------------------------------
// Entry Point & Mode Selector
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("══════════════════════════════════════════════════════════════");
    println!(" 🚀 Multi-AI Secure SSH Key Agent Dashboard");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("Select Operation Mode:");
    println!("  [1] Advanced Interactive TUI Dashboard");
    println!("  [2] Local Web Server (Browser HTML UI)\n");

    print!("Enter choice [1-2]: ");
    io::stdout().flush()?;

    let mut mode_input = String::new();
    io::stdin().read_line(&mut mode_input)?;
    let mode = mode_input.trim().to_string();

    println!("\nSelect default AI routing brain:");
    println!("  [1] xAI Grok");
    println!("  [2] OpenAI GPT-4o");
    println!("  [3] Anthropic Claude");
    println!("  [4] Google Gemini\n");

    print!("Enter choice [1-4]: ");
    io::stdout().flush()?;

    let mut menu_input = String::new();
    io::stdin().read_line(&mut menu_input)?;
    let initial_provider = match menu_input.trim() {
        "1" => AiProvider::Grok,
        "2" => AiProvider::OpenAi,
        "3" => AiProvider::Anthropic,
        "4" => AiProvider::Gemini,
        _ => AiProvider::Grok,
    };

    // --- SECURE KEYRING RETRIEVAL / FALLBACK LOGIC ---
    let entry = keyring::Entry::new(KEYRING_SERVICE_NAME, initial_provider.keyring_key())?;
    
    let api_key_string = match entry.get_password() {
        Ok(key) if !key.is_empty() => {
            println!("🔒 Retrieved stored API key securely from OS Keyring.");
            key
        }
        _ => {
            let env_var = initial_provider.env_var_name();
            let key = match std::env::var(env_var) {
                Ok(k) if !k.is_empty() => k,
                _ => rpassword::prompt_password("\n🔑 Enter API key (will be securely saved to OS Keyring): ")?,
            };

            let trimmed = key.trim().to_string();
            if trimmed.is_empty() {
                eprintln!("❌ API key cannot be empty.");
                std::process::exit(1);
            }

            if let Err(e) = entry.set_password(&trimmed) {
                eprintln!("⚠️ Warning: Failed to save API key to OS keyring: {}", e);
            } else {
                println!("✅ API key successfully saved to OS Keyring.");
            }
            trimmed
        }
    };

    let secure_api_key = Zeroizing::new(api_key_string);
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    let agent = Arc::new(Mutex::new(
        SshKeyAgent::new(secure_api_key)
            .register(Arc::new(GithubKey::new(home.join(".ssh/id_ed25519"))))
            .register(Arc::new(ProductionServerKey::new(home.join(".ssh/id_rsa"))))
    ));

    if mode == "2" {
        // --- WEB SERVER MODE ---
        let shared_state = (Arc::clone(&agent), initial_provider);
        let app = Router::new()
            .route("/api/intent", post(handle_web_intent))
            .fallback_service(ServeDir::new("."))
            .with_state(shared_state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
        println!("\n🌐 Secure Web Server running at http://127.0.0.1:3000");
        println!("Serving files (including index.html) directly from current directory!\n");
        axum::serve(listener, app).await?;
    } else {
        // --- RATATUI TUI MODE ---
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut input_buffer = String::new();
        let mut status_message = "Ready. Type your intent and press Enter.".to_string();
        let mut status_color = Color::Green;
        let mut current_provider = initial_provider;
        let key_list = agent.lock().await.list_keys();

        // API Key editing popup state
        let mut editing_api_key = false;
        let mut api_key_buffer = String::new();

        loop {
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),  // Header
                        Constraint::Min(6),    // Split Body (Key Ring + Log)
                        Constraint::Length(3),  // Input Box
                        Constraint::Length(1),  // Shortcut Footer
                    ])
                    .split(f.area());

                // Header
                let header = Paragraph::new(Line::from(vec![
                    Span::styled(" 🚀 Secure SSH Key Agent TUI ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(format!(" | Active AI: {:?}", current_provider)),
                ]))
                .block(Block::default().borders(Borders::ALL).title("Status Dashboard"));
                f.render_widget(header, chunks[0]);

                // Middle Split Body (Left: Key Registry, Right: Log)
                let body_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
                    .split(chunks[1]);

                // Left Panel: Key Ring
                let items: Vec<ListItem> = key_list
                    .iter()
                    .map(|(name, path)| {
                        let exists = path.exists();
                        let status_symbol = if exists { "✅" } else { "❌" };
                        ListItem::new(format!("{} {} ({})", status_symbol, name, path.display()))
                    })
                    .collect();
                let keys_widget = List::new(items)
                    .block(Block::default().borders(Borders::ALL).title("Key Ring Assets"));
                f.render_widget(keys_widget, body_chunks[0]);

                // Right Panel: Logs
                let log_widget = Paragraph::new(status_message.clone())
                    .style(Style::default().fg(status_color))
                    .block(Block::default().borders(Borders::ALL).title("Execution Log & Output"))
                    .wrap(Wrap { trim: true });
                f.render_widget(log_widget, body_chunks[1]);

                // Input Box
                let input_widget = Paragraph::new(input_buffer.as_str())
                    .style(Style::default().fg(Color::Yellow))
                    .block(Block::default().borders(Borders::ALL).title("Intent Input"));
                f.render_widget(input_widget, chunks[2]);

                // Footer Shortcuts
                let footer = Paragraph::new(Span::styled(
                    " [F2] Switch AI | [F3] Change API Key | [Enter] Execute | [Esc] Quit ",
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                ));
                f.render_widget(footer, chunks[3]);

                // --- API KEY POPUP OVERLAY ---
                if editing_api_key {
                    let popup_area = centered_rect(60, 20, f.area());
                    f.render_widget(Clear, popup_area);

                    let popup_content = Paragraph::new(api_key_buffer.as_str())
                        .style(Style::default().fg(Color::LightMagenta))
                        .block(Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" Update API Key for {:?} (Press Enter to Save, Esc to Cancel) ", current_provider))
                            .border_style(Style::default().fg(Color::Magenta)));
                    f.render_widget(popup_content, popup_area);
                }
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if editing_api_key {
                        match key.code {
                            KeyCode::Esc => {
                                editing_api_key = false;
                                api_key_buffer.clear();
                            }
                            KeyCode::Enter => {
                                let new_key_trimmed = api_key_buffer.trim().to_string();
                                if !new_key_trimmed.is_empty() {
                                    match keyring::Entry::new(KEYRING_SERVICE_NAME, current_provider.keyring_key()) {
                                        Ok(entry) => {
                                            if let Err(e) = entry.set_password(&new_key_trimmed) {
                                                status_message = format!("❌ Keyring error: {}", e);
                                                status_color = Color::Red;
                                            } else {
                                                let mut agent_lock = agent.lock().await;
                                                agent_lock.update_api_key(Zeroizing::new(new_key_trimmed));
                                                status_message = format!("✅ Successfully updated and saved API key for {:?}!", current_provider);
                                                status_color = Color::Green;
                                            }
                                        }
                                        Err(e) => {
                                            status_message = format!("❌ Failed to access keyring entry: {}", e);
                                            status_color = Color::Red;
                                        }
                                    }
                                }
                                editing_api_key = false;
                                api_key_buffer.clear();
                            }
                            KeyCode::Backspace => {
                                api_key_buffer.pop();
                            }
                            KeyCode::Char(c) => {
                                api_key_buffer.push(c);
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::Esc => break,
                            KeyCode::F(2) => {
                                current_provider = current_provider.next();
                                status_message = format!("Switched active AI provider to: {:?}", current_provider);
                                status_color = Color::Cyan;
                            }
                            KeyCode::F(3) => {
                                editing_api_key = true;
                                api_key_buffer.clear();
                                status_message = format!("Editing API key for {:?}...", current_provider);
                                status_color = Color::Magenta;
                            }
                            KeyCode::Enter => {
                                if input_buffer.eq_ignore_ascii_case("exit") || input_buffer.eq_ignore_ascii_case("quit") {
                                    break;
                                }
                                if !input_buffer.trim().is_empty() {
                                    let intent = input_buffer.trim().to_string();
                                    input_buffer.clear();
                                    
                                    let agent_ref = Arc::clone(&agent);
                                    let res = tokio::task::block_in_place(|| {
                                        tokio::runtime::Handle::current().block_on(async {
                                            let agent_lock = agent_ref.lock().await;
                                            agent_lock.process_intent(&intent, current_provider).await
                                        })
                                    });

                                    match res {
                                        Ok(msg) => {
                                            status_message = format!("✅ SUCCESS: {}", msg);
                                            status_color = Color::Green;
                                        }
                                        Err(e) => {
                                            status_message = format!("❌ BLOCKED: {}", e);
                                            status_color = Color::Red;
                                        }
                                    }
                                }
                            }
                            KeyCode::Backspace => {
                                input_buffer.pop();
                            }
                            KeyCode::Char(c) => {
                                if input_buffer.len() < MAX_INTENT_LENGTH {
                                    input_buffer.push(c);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
    }

    Ok(())
}