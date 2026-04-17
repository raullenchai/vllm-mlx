# Tool Calling

rapid-mlx supports OpenAI-compatible tool calling (function calling) with automatic parsing for many popular model families.

## Quick Start

Enable tool calling by adding the `--enable-auto-tool-choice` flag when starting the server:

```bash
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Then use tools with the standard OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## Supported Parsers

Use `--tool-call-parser` to select a parser for your model family:

| Parser | Aliases | Models | Format |
|--------|---------|--------|--------|
| `auto` | | Any model | Auto-detects format (tries all parsers) |
| `mistral` | | Mistral, Devstral | `[TOOL_CALLS]` JSON array |
| `qwen` | `qwen3` | Qwen, Qwen3 | `<tool_call>` XML or `[Calling tool:]` |
| `llama` | `llama3`, `llama4` | Llama 3.x, 4.x | `<function=name>` tags |
| `hermes` | `nous` | Hermes, NousResearch | `<tool_call>` JSON in XML |
| `deepseek` | `deepseek_v3`, `deepseek_r1` | DeepSeek V3, R1 | Unicode delimiters |
| `kimi` | `kimi_k2`, `moonshot` | Kimi K2, Moonshot | `<\|tool_call_begin\|>` tokens |
| `granite` | `granite3` | IBM Granite 3.x, 4.x | `<\|tool_call\|>` or `<tool_call>` |
| `nemotron` | `nemotron3` | NVIDIA Nemotron | `<tool_call><function=...><parameter=...>` |
| `xlam` | | Salesforce xLAM | JSON with `tool_calls` array |
| `functionary` | `meetkai` | MeetKai Functionary | Multiple function blocks |
| `glm47` | `glm4` | GLM-4.7, GLM-4.7-Flash | `<tool_call>` with `<arg_key>`/`<arg_value>` XML |

## Model Examples

### Mistral / Devstral

```bash
# Devstral Small (optimized for coding and tool use)
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Mistral Instruct
rapid-mlx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

### Qwen

```bash
# Qwen3
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen
```

### Llama

```bash
# Llama 3.2
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser llama
```

### DeepSeek

```bash
# DeepSeek V3
rapid-mlx serve mlx-community/DeepSeek-V3-0324-4bit \
  --enable-auto-tool-choice --tool-call-parser deepseek
```

### IBM Granite

```bash
# Granite 4.0
rapid-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite
```

### NVIDIA Nemotron

```bash
# Nemotron 3 Nano
rapid-mlx serve mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit \
  --enable-auto-tool-choice --tool-call-parser nemotron
```

### GLM-4.7

```bash
# GLM-4.7 Flash
rapid-mlx serve lmstudio-community/GLM-4.7-Flash-MLX-8bit \
  --enable-auto-tool-choice --tool-call-parser glm47
```

### Kimi K2

```bash
# Kimi K2
rapid-mlx serve mlx-community/Kimi-K2-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser kimi
```

### Salesforce xLAM

```bash
# xLAM
rapid-mlx serve mlx-community/xLAM-2-fc-r-4bit \
  --enable-auto-tool-choice --tool-call-parser xlam
```

## Auto Parser

If you're not sure which parser to use, the `auto` parser tries to detect the format automatically:

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser auto
```

The auto parser tries formats in this order:
1. Mistral (`[TOOL_CALLS]`)
2. Qwen bracket (`[Calling tool:]`)
3. Nemotron (`<tool_call><function=...><parameter=...>`)
4. Qwen/Hermes XML (`<tool_call>{...}</tool_call>`)
5. Llama (`<function=name>{...}</function>`)
6. Raw JSON

## Streaming Tool Calls

Tool calls work with streaming. The tool call information is sent when the model finishes generating:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## Handling Tool Results

After receiving a tool call, execute the function and send the result back:

```python
import json

# First request - model decides to call a tool
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Get the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute the function (your implementation)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Send result back to model
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think Tag Handling

Models that produce `<think>...</think>` reasoning tags (like DeepSeek-R1, Qwen3, GLM-4.7) are handled automatically. The parser strips thinking content before extracting tool calls, so reasoning tags never interfere with tool call parsing.

This works even when `<think>` was injected in the prompt (implicit think tags with only a closing `</think>`).

## CLI Reference

| Option | Description |
|--------|-------------|
| `--enable-auto-tool-choice` | Enable automatic tool calling |
| `--tool-call-parser` | Select parser (see table above) |

See [CLI Reference](../reference/cli.md) for all options.
