use genai_rs::{Client, Content, Step, request::InteractionInput, tools as genai_tool};
use serde_json::json;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::process::Command;

const SYSTEM_PROMPT: &str = "You are a coding assistant running in a terminal. You have three tools: bash, read_file and write_file. Be concise.";

#[tokio::main]
async fn main() -> Result<(), genai_rs::GenaiError> {
    let tools = get_tools();

    let client = Client::new(std::env::var("GEMINI_API_KEY").unwrap());

    let mut inputs: Vec<Step> = Vec::new();

    let stdin = io::stdin();
    let reader = BufReader::with_capacity(64 * 1024, stdin.lock());
    let mut lines = reader.lines();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let line = match lines.next() {
            Some(Ok(l)) => l,
            Some(Err(e)) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
            None => break,
        };

        let user_input = line.trim().to_string();
        if user_input.is_empty() {
            continue;
        }

        inputs.push(Step::user_input(vec![Content::text(user_input)]));

        let current_inputs = std::mem::take(&mut inputs);
        inputs = agent_loop(&client, current_inputs, &tools).await?;
    }

    Ok(())
}

async fn agent_loop(
    client: &Client,
    mut inputs: Vec<Step>,
    tools: &Vec<genai_tool::Tool>,
) -> Result<Vec<Step>, genai_rs::GenaiError> {
    loop {
        let mut response = client
            .interaction()
            .with_system_instruction(SYSTEM_PROMPT)
            .with_model("gemini-3.5-flash-lite")
            .with_generation_config(genai_rs::GenerationConfig {
                max_output_tokens: Some(8192),
                ..Default::default()
            })
            .with_input(InteractionInput::Steps(inputs.clone()))
            .with_tools(tools.to_vec())
            .create()
            .await?;

        let mut tool_results = Vec::new();
        for step in response.steps.iter() {
            match step {
                Step::ModelOutput { content, error: _ } => {
                    println!("{}", content[0].as_text().unwrap_or_default())
                }
                Step::FunctionCall {
                    id,
                    name,
                    arguments,
                    ..
                } => {
                    let (result, is_err) = execute_tool(name, arguments);
                    let result_step = match is_err {
                        true => Step::function_result(name, id, result),
                        false => Step::function_result_error(name, id, result),
                    };
                    tool_results.push(result_step);
                }
                _ => continue,
            }
        }

        let got_result = matches!(response.steps.last(), Some(Step::ModelOutput { .. }));

        inputs.append(&mut response.steps);
        inputs.append(&mut tool_results);

        if got_result {
            return Ok(inputs);
        }
    }
}

fn execute_tool(name: &String, arguments: &serde_json::Value) -> (String, bool) {
    println!("[Tool] {} {}", name, arguments.to_string());

    match name.as_str() {
        "bash" => {
            let command = match arguments.get("command") {
                Some(c) => c,
                None => return ("Missing 'command' argument.".into(), true),
            };

            let cmd_str = match command.as_str() {
                Some(s) => s,
                None => return ("'command' argument must be a string".into(), true),
            };

            let out = Command::new("cmd.exe").arg("/C").arg(cmd_str).output();

            let (result, is_err) = match out {
                Ok(output) => {
                    let mut combined = output.stdout;
                    combined.extend(output.stderr);

                    let text = String::from_utf8_lossy(&combined).to_string();

                    let failed = !output.status.success();

                    (text, failed)
                }
                Err(err) => (format!("[Exit Error: {}]", err), true),
            };

            return (result, is_err);
        }
        "read_file" => {
            let path = match arguments.get("path") {
                Some(c) => c,
                None => return ("Missing 'path' argument.".into(), true),
            };

            let path_str = match path.as_str() {
                Some(s) => s,
                None => return ("'path' argument must be a string".into(), true),
            };

            let data = match fs::read_to_string(&path_str) {
                Ok(text) => text,
                Err(err) => {
                    return (format!("Failed to read file '{}': {}", path, err), true);
                }
            };

            return (data, false);
        }
        "write_file" => {
            let path = arguments.get("path").and_then(|v| v.as_str());
            let content = arguments.get("content").and_then(|v| v.as_str());

            let (path, content) = match (path, content) {
                (Some(p), Some(c)) => (p, c),
                _ => {
                    return (
                        "Missing or invalid 'path' or 'content' arguments.".into(),
                        true,
                    );
                }
            };

            if let Err(err) = fs::write(&path, content.as_bytes()) {
                return (err.to_string(), true);
            }

            return (format!("Wrote {}", path), false);
        }
        _ => return (format!("Unknown Tool: {}", name), true),
    }
}

fn get_tools() -> Vec<genai_tool::Tool> {
    let tools = vec![
        genai_tool::FunctionDeclarationBuilder::new("bash")
            .with_description("Run a shell command and return its combined stdout/stderr.")
            .add_parameter(
                "command",
                json!({
                    "type": "string",
                    "description": "The command to run."
                }),
            )
            .with_required(vec!["command".into()])
            .build()
            .into_tool(),
        genai_tool::FunctionDeclarationBuilder::new("read_file")
            .with_description("Read the contents of a file at a given path.")
            .add_parameter(
                "path",
                json!({
                    "type": "string",
                    "description": "Filesystem path to read."
                }),
            )
            .with_required(vec!["path".into()])
            .build()
            .into_tool(),
        genai_tool::FunctionDeclarationBuilder::new("write_file")
            .with_description("Write a content into a file (creating or overwriting it).")
            .add_parameter(
                "path",
                json!({
                    "type": "string",
                    "description": "Filesystem path to write."
                }),
            )
            .add_parameter(
                "content",
                json!({
                    "type": "string",
                    "description": "The bytes to write."
                }),
            )
            .with_required(vec!["path".into(), "content".into()])
            .build()
            .into_tool(),
    ];

    return tools;
}
