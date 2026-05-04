# pi-cursor-provider

[![npm version](https://img.shields.io/npm/v/pi-cursor-provider.svg)](https://www.npmjs.com/package/pi-cursor-provider)

[Pi](https://github.com/badlogic/pi-mono) extension that provides access to [Cursor](https://cursor.com) models via OAuth authentication and a local OpenAI-compatible proxy.

## How it works

```
pi  →  openai-completions  →  localhost:PORT/v1/chat/completions
                                      ↓
                              proxy.ts (HTTP server)
                                      ↓
                              h2-bridge.mjs (Node HTTP/2)
                                      ↓
                              api2.cursor.sh gRPC
```

1. **PKCE OAuth** — browser-based login to Cursor, no client secret needed
2. **Model discovery** — queries Cursor's `GetUsableModels` gRPC endpoint
3. **Local proxy** — translates OpenAI `/v1/chat/completions` to Cursor's protobuf/HTTP2 Connect protocol using Cursor's newer `requestedModel` request field
4. **Tool routing** — rejects Cursor's native tools, exposes pi's tools via MCP

## Install

```bash
# Via pi install
pi install npm:pi-cursor-provider

# Or manually
git clone https://github.com/ndraiman/pi-cursor-provider ~/.pi/agent/extensions/cursor-provider
cd ~/.pi/agent/extensions/cursor-provider
npm install
```

## Usage

```
/login cursor     # authenticate via browser
/model            # select a Cursor model
```

## Model Mapping

Cursor exposes many model variants that encode **effort level** (`low`, `medium`, `high`, `xhigh`, `max`, `none`) and **speed** (`-fast`) or **thinking** (`-thinking`) in the model ID. This extension deduplicates them so pi's reasoning effort setting controls the effort level.

### How it works

Each raw Cursor model ID is parsed into components. Cursor has used both thinking/effort orders, and this extension preserves the exact raw ID returned by Cursor when dispatching requests:

```
{base}-{effort}[-thinking][-fast]
{base}-thinking-{effort}[-fast]
```

Examples:

| Raw Cursor ID | Base | Effort | Variant |
|---|---|---|---|
| `gpt-5.4-medium` | `gpt-5.4` | `medium` | — |
| `gpt-5.4-high-fast` | `gpt-5.4` | `high` | `-fast` |
| `claude-4.6-opus-max-thinking` | `claude-4.6-opus` | `max` | `-thinking` |
| `claude-opus-4-7-thinking-max` | `claude-opus-4-7` | `max` | `-thinking` |
| `gpt-5.1-codex-max-high` | `gpt-5.1-codex-max` | `high` | — |
| `composer-2` | `composer-2` | — | — |

Models sharing the same `(base, variant)` with **≥2 effort levels** and a sensible default (`medium` or no-suffix) are collapsed into a single entry with `supportsReasoningEffort: true`. Pi's thinking level maps to the effort suffix:

| Pi Level | Cursor Suffix |
|---|---|
| `minimal` | `none` (if available) or `low` |
| `low` | `low` |
| `medium` | `medium` or no suffix (default) |
| `high` | `high` |
| `xhigh` | `max` (Claude) or `xhigh` (GPT) |

### Parameterized Cursor models

Cursor CLI exposes some choices as model parameters rather than standalone model IDs. For example, GPT-5.5 has separate **Context** settings (`272K` and `1M`), **Reasoning** settings, and a **Fast** toggle for 272K variants. Pi's model picker cannot edit those Cursor-specific parameters directly, so this extension exposes them as separate selectable rows:

| Pi model | Cursor `requestedModel` |
|---|---|
| `gpt-5.5` | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "false"`, `maxMode: false` |
| `gpt-5.5-fast` | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "true"`, `maxMode: false` |
| `gpt-5.5-max` | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "false"`, `maxMode: true` |
| `gpt-5.5-max-fast` | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "true"`, `maxMode: true` |
| `gpt-5.5-1m` | `modelId: "gpt-5.5"`, `context: "1m"`, `fast: "false"`, `maxMode: true` |

Pi's thinking level supplies the Cursor `reasoning` parameter for those rows (`none`, `low`, `medium`, `high`, or `extra-high`). There is no separate `/max` toggle: Cursor-specific flags like `maxMode` and `fast` are determined by the selected model row. Cursor's own model metadata does not include any `context=1m` + `fast=true` GPT-5.5 variant; sending that invalid combination is sanitized by Cursor to the default 1M medium configuration, so this extension intentionally does not expose `gpt-5.5-1m-fast`.

For deduped models, the extension keeps an exact map from `(displayed model, effort)` back to the raw Cursor model ID or parameter set returned/derived from Cursor. That avoids guessing where the effort segment belongs:

```
pi selects: gpt-5.4-fast              + effort: high   → Cursor receives: gpt-5.4-high-fast
pi selects: gpt-5.4                   + effort: medium → Cursor receives: gpt-5.4-medium
pi selects: gpt-5.5-1m                + effort: high   → Cursor receives: gpt-5.5 + context=1m + reasoning=high
pi selects: claude-opus-4-7-thinking  + effort: max    → Cursor receives: claude-opus-4-7-thinking-max
pi selects: composer-2                + no effort      → Cursor receives: composer-2
```

When a group is **collapsed**, the proxy registers one model with `supportsReasoningEffort: true` and an internal effort map (see table above).

**Collapsed** when Cursor returns either:

- **Multiple** effort suffixes for the same `(base, -fast, -thinking)` group, or
- **A single** variant whose parsed effort suffix is **non-empty** (for example only `claude-4.5-opus-high` is listed). The suffix is removed from the displayed ID so Pi's reasoning-effort setting supplies it.

**Left as-is** (raw Cursor ID on that row, `supportsReasoningEffort: false`) when the group has **one** variant and the parsed effort suffix is **empty**—typically IDs with no effort segment, such as `composer-2`, `gemini-3.1-pro`, or `kimi-k2.5`.

### Disabling the mapping

To see all raw Cursor model variants without dedup:

```bash
PI_CURSOR_RAW_MODELS=1 pi
```

## Session Management

The proxy maintains conversation state per pi session, enabling multi-turn conversations with Cursor models while preserving forks, tool continuations, and interruptions correctly.

### How it works

- **Session tracking** — pi's session ID is injected into requests via a `before_provider_request` hook. The proxy keys bridge state and stored conversation state from that real session ID.
- **Checkpoints** — Cursor returns a conversation checkpoint after completed turns. The proxy stores that checkpoint, plus the completed-turn count and a fingerprint of the completed structured history, and reuses it only when the incoming history still matches.
- **Session-scoped state** — real pi session state is kept in memory until explicit cleanup or process restart. Anonymous fallback state can still be TTL-evicted.
- **Lifecycle cleanup** — session state is cleaned up on pi lifecycle events such as session switch, fork, `/tree`, and shutdown.

### Tool continuations

When Cursor pauses for a tool call, the proxy keeps the live upstream bridge open and waits for pi to send the tool result on the next request. That tool result is sent back into the same in-flight Cursor run, so the tool continuation stays part of the original user turn instead of inflating completed history.

### Interruptions

If the client disconnects or interrupts a turn mid-stream, the proxy cancels the upstream Cursor run and does **not** commit the pending checkpoint. Checkpoints are only committed after a turn finishes successfully.

### Session fork

When you navigate back in pi's session tree and branch from an earlier point, the proxy discards the stored checkpoint whenever the completed history no longer matches the stored checkpoint metadata. That includes both:

- completed turn count mismatches, and
- same-depth branch changes detected via completed-history fingerprint mismatch.

After discarding a stale checkpoint, the proxy reconstructs proper protobuf conversation turns from the message history pi sends, so Cursor sees the actual conversation structure at the fork point.

### Session resume

Conversation state is stored in memory. If the proxy restarts, checkpoints are lost. On the next request, pi sends the full conversation history, and the proxy reconstructs structured protobuf turns from that history instead of relying on an inline plaintext fallback.

That reconstruction preserves:

- assistant messages
- tool calls
- tool results
- final assistant text after tool results

## Requirements

- [Pi](https://github.com/badlogic/pi-mono)
- [Node.js](https://nodejs.org) >= 18
- Active [Cursor](https://cursor.com) subscription

## Development

```bash
npm install
npm test
```

## Debug log timeline

When `PI_CURSOR_PROVIDER_DEBUG=1` is enabled, the proxy writes timestamped JSONL logs to `os.tmpdir()` by default. You can turn a log into a compact human-readable timeline with:

```bash
npm run debug:timeline -- --latest
npm run debug:timeline -- /path/to/pi-cursor-provider-debug-2026-04-08T14-06-07-565Z-41184.log
```

Add `--json` if you want the parsed summary as JSON instead of formatted text.

## Credits

OAuth flow and gRPC proxy adapted from [opencode-cursor](https://github.com/ephraimduncan/opencode-cursor) by Ephraim Duncan.
