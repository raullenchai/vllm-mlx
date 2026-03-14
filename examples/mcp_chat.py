#!/usr/bin/env python3
"""
Interactive chat with MCP tools.

The LLM can use MCP tools (filesystem, etc.) to perform actions.

Usage:
    python examples/mcp_chat.py

Example prompts:
    - "Create a file at /tmp/test.txt with content hello world"
    - "List files in /tmp"
    - "Read the file /tmp/test.txt"
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def get_mcp_tools():
    """Get MCP tools in OpenAI format."""
    response = requests.get(f"{BASE_URL}/v1/mcp/tools").json()
    tools = []
    for tool in response.get("tools", []):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )
    return tools


def execute_tool(tool_name: str, arguments: dict):
    """Execute an MCP tool."""
    response = requests.post(
        f"{BASE_URL}/v1/mcp/execute",
        json={"tool_name": tool_name, "arguments": arguments},
    ).json()
    return response


def chat(messages: list, tools: list):
    """Send message to LLM with tools."""
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "tools": tools,
                "max_tokens": 1000,
            },
            timeout=120,
        )
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}


def main():
    print("=" * 60)
    print("MCP Chat - LLM can use filesystem tools")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end\n")

    # Get MCP tools
    tools = get_mcp_tools()
    if not tools:
        print("ERROR: No MCP tools available")
        print("Make sure to start the server with --mcp-config")
        return

    print(f"Available tools: {len(tools)}")
    for t in tools[:5]:
        print(f"  - {t['function']['name']}")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more\n")

    # Build tools description for system prompt
    tools_desc = "\n".join(
        [
            f"- {t['function']['name']}: {t['function']['description'][:100]}"
            for t in tools[:10]
        ]
    )

    system_prompt = f"""You are an assistant with access to filesystem tools.

IMPORTANT: When the user asks for file operations, you MUST use the available tools via function calls. Do NOT suggest bash commands or code. USE the tools directly.

Available tools:
{tools_desc}

To create a file, use filesystem__write_file with path and content parameters.
To read a file, use filesystem__read_file or filesystem__read_text_file.
To list directories, use filesystem__list_directory.

ALWAYS respond with tool_calls when you need to perform file operations."""

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Send to LLM
        response = chat(messages, tools)

        if "error" in response:
            print(f"Error: {response['error']}")
            messages.pop()  # Remove failed message
            continue

        choice = response.get("choices", [{}])[0]
        assistant_message = choice.get("message", {})

        # Check for tool_calls
        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            print(f"\nAssistant: [Using {len(tool_calls)} tool(s)...]")

            # Add assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            # Execute each tool call
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])

                print(f"  -> Executing: {func_name}")
                print(f"     Args: {func_args}")

                result = execute_tool(func_name, func_args)

                if result.get("is_error"):
                    tool_result = f"Error: {result.get('error_message')}"
                else:
                    tool_result = str(result.get("content", ""))

                print(
                    f"     Result: {tool_result[:100]}{'...' if len(tool_result) > 100 else ''}"
                )

                # Add tool result
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": tool_result}
                )

            # Get final LLM response
            response = chat(messages, tools)
            choice = response.get("choices", [{}])[0]
            assistant_message = choice.get("message", {})

        # Show response
        content = assistant_message.get("content", "")
        if content:
            print(f"\nAssistant: {content}")
            messages.append({"role": "assistant", "content": content})
        else:
            print("\nAssistant: [No response]")


if __name__ == "__main__":
    main()
