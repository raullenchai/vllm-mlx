# Agentic Runtime Design — Glue Layer Architecture

> Internal design doc. Do NOT commit to GitHub.
> Last updated: 2026-04-12

## Vision

Rapid-MLX 的核心价值不是推理引擎本身（上游也能做），而是 **Agent 和推理引擎之间的 Glue Layer**。

```
┌──────────────────────────────────────────────┐
│  Agent Layer                                  │
│  Hermes · Goose · OpenClaude · Aider · Cline │
│  Codex CLI · OpenHands · SWE-Agent · ...     │
└──────────────┬───────────────────────────────┘
               │  OpenAI-compatible API
┌──────────────▼───────────────────────────────┐
│  Glue Layer  ← 我们的核心价值                  │
│  Agent Profiles · Pipeline Filters ·          │
│  Tool Parsers · Streaming · Auto-Detection    │
└──────────────┬───────────────────────────────┘
               │  mlx-lm / mlx-vlm
┌──────────────▼───────────────────────────────┐
│  Inference Layer                              │
│  MLX · Apple Silicon · Unified Memory         │
└──────────────────────────────────────────────┘
```

---

## Agent Landscape Survey (2026-04)

### Tier 1: Giants (40K+ stars)

| Agent | Stars | 语言 | LLM 接入方式 | Tool 格式 | 本地模型支持 | 多模型 | 我们的机会 |
|---|---|---|---|---|---|---|---|
| Claw Code | 182K | Rust | Anthropic | structured tool_use | ? | No | 待研究 |
| Claude Code | 113K | TS | Anthropic SDK | structured tool_use | No (用 fork) | No | 通过 OpenClaude |
| Gemini CLI | 101K | TS | Gemini API | Gemini FC | No (有需求) | No | 等他们开放 |
| Codex CLI | 75K | Rust | OpenAI Responses API | OpenAI FC | buggy | No | Responses API 不标准 |
| OpenHands | 71K | Python | LiteLLM | 文本 action 格式 | Yes | No | LLM_BASE_URL |
| **Hermes** | **64K** | Python | **OpenAI-compat** | **OpenAI FC** | **Yes** | **Yes (双模型路由)** | **已接入** |
| Cline | 60K | TS | 多 provider | XML 文本解析 | Yes | No | 已 discuss |
| Aider | 43K | Python | LiteLLM | 文本 edit format (无FC) | **最好** | Optional (architect) | 已接入 |
| Goose | 41K | Rust | 原生多 provider | OpenAI FC + adapter | Yes (原生 Ollama) | No | 已接入 |

### Tier 2: Established (10K-40K)

| Agent | Stars | 语言 | LLM 接入方式 | Tool 格式 | 本地模型 | 多模型 | 机会 |
|---|---|---|---|---|---|---|---|
| Continue | 33K | TS | Configurable | IDE actions | Yes | No | PR 被拒 |
| Crush/OpenCode | 23K | Go | Direct SDKs | Standard | Yes | No | LOCAL_ENDPOINT |
| OpenClaude | 21K | TS/Bun | OpenAI-compat | tool_use adapted | Yes | Yes (agent routing) | **已接入** |
| SWE-Agent | 19K | Python | LiteLLM | 文本 ACI | Yes | No | 学术型 |
| Plandex | 15K | Go | OpenRouter | 文本 diff | Indirect | Yes (model packs) | 间接 |

### 关键发现

#### 1. 双模型/多模型是趋势
- **Hermes**: `smart_model_routing.py` — cheap model 处理简单消息，strong model 处理复杂任务
- **Aider**: Architect 模式 — 一个模型规划，一个模型编辑
- **OpenClaude**: `agentModels` — 不同 agent 路由到不同模型
- **Plandex**: Model packs — 组合多个 provider 的最优模型

→ **我们需要支持：多模型同时加载 + 请求级别模型切换**

#### 2. Tool 格式分三派
- **OpenAI Function Calling**: Hermes, Goose, Codex CLI, OpenClaude (适配后) → 我们的强项
- **文本解析 (无 FC)**: Aider (edit format), Cline (XML), SWE-Agent (ACI), OpenHands (action format) → 不需要 tool parser
- **Anthropic tool_use**: Claude Code, Claw Code → 我们有 Anthropic adapter

→ **文本解析派的 agent 其实更容易接入**，因为不需要 tool calling 支持

#### 3. 共同的痛点
| 痛点 | 哪些 agent 受影响 | 我们能做什么 |
|---|---|---|
| **模型加载超时** (30-120s) | Goose (600s timeout), Hermes, Codex | 模型预加载 + health probe |
| **小模型 FC 质量差** | Hermes #4505, Codex #14743 | tool logits bias, tool injection |
| **Context window 未知** | Hermes (descending probe), Cline | `/v1/models` 返回 context_length |
| **Ollama 慢** | Hermes #7800, 所有用 Ollama 的 | **我们的 4x 加速** |
| **Responses API 不兼容** | Codex CLI #12669, #14743 | 实现 /v1/responses 端点？ |
| **SSH/远程断连** | Cline #10147, Hermes #7905 | 连接保活 + 超时检测 |

#### 4. MCP 成为标准
- Hermes, Goose, Claude Code, OpenClaude, Cline 都支持 MCP
- 我们已有 MCP 集成 (`vllm_mlx/mcp/`)

---

## Glue Layer 模块分析 — 用 Survey 结果对齐

### 模块 1: Tool Parser (当前 20 个 parser)
- **Survey 结论**: 只有 ~50% 的 agent 需要 OpenAI FC 格式。Aider, Cline, SWE-Agent 用文本格式
- **优先级**: ✅ 已经很好。继续按需加新 parser
- **缺口**: 无

### 模块 2: Streaming Filter (当前硬编码 _TOOL_CALL_TAGS)
- **Survey 结论**: 所有 agent 都需要 SSE streaming。tag 泄漏是致命 bug
- **优先级**: 🔴 高 — 需要插件化，让 agent profile 声明自己的 tags
- **缺口**: 硬编码列表、无法通过配置扩展

### 模块 3: Reasoning Router (当前 7 个 parser)
- **Survey 结论**: Claude Code, Codex CLI, Hermes 都需要 thinking blocks。Qwen3.5 和 DeepSeek-R1 是主要模型
- **优先级**: ✅ 已经很好
- **缺口**: 无

### 模块 4: Tool Injection (大量 tools 的性能)
- **Survey 结论**: Hermes 注入 62 个 tools (~5000 prompt tokens)。Goose 通过 MCP 可能更多
- **优先级**: 🟡 中 — 需要优化大量 tools 的 prompt 效率
- **缺口**: 工具 schema 缓存、prompt 压缩

### 模块 5: Config Bridge (agent → 我们的 API)
- **Survey 结论**: 每个 agent 都有不同的 config 格式（env/yaml/json/toml）
- **优先级**: 🔴 高 — Agent Profile 系统
- **缺口**: 完全手动，无自动化

### 模块 6: Model Selection (推荐机制)
- **Survey 结论**: Hermes 用 Qwen3.5；Aider 用 DeepSeek；不同 agent 最优模型不同
- **优先级**: 🟡 中 — 每个 profile 推荐模型列表
- **缺口**: 没有 per-agent 推荐

### 模块 7: Output Sanitization
- **Survey 结论**: 所有 agent 都会被 special token 泄漏搞崩
- **优先级**: ✅ 已经很好。agent profile 可加自定义 pattern
- **缺口**: 小

### 🔴 新发现的模块 8: Multi-Model Support (双模型路由)
- **Survey 结论**: Hermes (cheap+strong), Aider (architect), OpenClaude (agent routing), Plandex (model packs) 都需要
- **优先级**: 🔴 高 — 这是新的核心需求
- **具体需求**:
  - 同时加载 2 个模型（小模型 4B + 大模型 27B）
  - 请求级别模型切换 (`model` 字段选择)
  - 或者：一个模型处理 prefill，另一个处理 decode（投机解码的变体）
- **当前状态**: SimpleEngine 只支持单模型。需要 model registry + 动态切换

### 🟡 新发现的模块 9: Responses API Compatibility
- **Survey 结论**: Codex CLI (75K stars) 用 OpenAI Responses API，不是 Chat Completions
- **优先级**: 🟡 中 — 75K stars 的 agent，但 API 格式差异大
- **具体需求**: `/v1/responses` 端点，与 Chat Completions 类似但有 `conversation` 概念

### 🟡 新发现的模块 10: Health & Capability Discovery
- **Survey 结论**: Agent 需要知道：模型是否加载完成、context window 多大、支持哪些功能
- **优先级**: 🟡 中
- **具体需求**:
  - `/v1/models` 返回 `context_length`, `supports_tools`, `supports_vision`
  - `/health` 返回模型加载状态（loading/ready/error）
  - Agent 可以 probe 能力而不是猜测

---

## 优先级排序

### P0: 必须做 (直接影响 agent 接入速度)

1. **Agent Profile 系统** — YAML 声明式配置，加新 agent = 加一个文件
   - 包含：config bridge, model recommendation, streaming tags, known issues
   - CLI: `rapid-mlx agents list/setup/test`

2. **Streaming Filter 插件化** — `_TOOL_CALL_TAGS` 从硬编码 → profile 驱动
   - Agent profile 声明 `extra_tool_tags`
   - 运行时动态注入

3. **Multi-Model Support** — 同时加载 2+ 模型，请求级切换
   - `/v1/models` 返回所有已加载模型
   - 请求中 `model` 字段选择
   - 对 Hermes dual-routing, Aider architect, OpenClaude agent-routing 都有用

### P1: 应该做 (提升体验)

4. **Health & Capability API 增强**
   - `/v1/models` 返回 context_length, supports_tools, supports_vision
   - `/health` 返回 model loading status

5. **Pipeline Filter Chain** — 重构 streaming 为 composable filters
   - 让 agent 的 streaming quirks 成为 filter 而不是 if/else

6. **server.py 拆分** — 代码组织（反 backport 的副作用）

### P2: 可以做 (长期)

7. **Responses API** — 给 Codex CLI 用
8. **Tool Schema 缓存** — 优化 60+ tools 的 prompt 效率
9. **Context Window Auto-Detection** — 模型加载时自动设置

---

## 实施计划

### Sprint 1 (本周): Agent Profile + Multi-Model
- 创建 `agents/` 目录 + AgentProfile dataclass
- 写 5 个 profile YAML (hermes, goose, opencode, aider, generic)
- CLI: `rapid-mlx agents`
- Multi-model: model registry + `/v1/models` 多模型 + 请求级切换

### Sprint 2 (下周): Streaming Pipeline + Filter Chain
- 创建 `pipeline/` 目录 + StreamFilter ABC
- 拆解 `stream_chat_completion()` 为 filter chain
- Agent profile 的 `extra_tool_tags` 自动注入到 filter

### Sprint 3: server.py 拆分 + Health API
- config/, routes/, middleware/, runtime/
- `/v1/models` 增强 (context_length, capabilities)
- `/health` 增强 (loading status)

---

## Appendix: Agent-Specific Integration Notes

### Hermes Agent (64K stars) — DONE
- Config: `~/.hermes/config.yaml`, provider: "custom"
- 62 tools injected per request
- Gemma 4 tool calling: only works with Rapid-MLX
- Dual model routing: `smart_model_routing.py`
- Known issue: `todo` tool conflict, `[Calling tool` mimicry

### Goose (41K stars) — Basic integration done
- Config: `GOOSE_PROVIDER=ollama`, `OLLAMA_HOST=http://localhost:8000`
- MCP-first tool architecture
- 600s timeout for model loading (we load faster)
- 30s timeout on complex multi-step

### OpenClaude (21K stars) — Done
- Config: `CLAUDE_CODE_USE_OPENAI=1`, `OPENAI_BASE_URL`
- Agent routing: different models for different agents
- Tool_use format adapted to OpenAI

### Aider (43K stars) — Works out of box
- Config: `--openai-api-base http://localhost:8000/v1`
- NO function calling needed (text edit formats)
- Architect mode: dual model (plan + edit)

### Cline (60K stars) — Untested deep integration
- VS Code settings: API provider = "OpenAI Compatible"
- XML text parsing (no FC needed)
- Multimodal: sends screenshots
- 60K stars — high priority target

### Codex CLI (75K stars) — Blocked
- Uses Responses API (non-standard)
- `openai_base_url` exists but buggy
- Would need `/v1/responses` endpoint

### OpenHands (71K stars) — Easy via LiteLLM
- Config: `LLM_BASE_URL=http://localhost:8000/v1`
- Text action format (no FC needed)
- Docker required for sandbox

### Gemini CLI (101K stars) — Blocked
- Gemini API only, no OpenAI-compat
- Open issues requesting local model support
- Watch for future OpenAI-compat endpoint
