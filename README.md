# pi-cursor-provider

[![npm version](https://img.shields.io/npm/v/pi-cursor-provider.svg)](https://www.npmjs.com/package/pi-cursor-provider)

[Pi](https://github.com/earendil-works/pi-mono) extension that provides access to [Cursor](https://cursor.com) models via OAuth authentication and a native Pi `streamSimple` provider.

## How it works

```
pi  →  cursor-native streamSimple  →  Cursor protobuf/Connect frames
                                              ↓
                                      h2-bridge.mjs (Node HTTP/2)
                                              ↓
                                      api2.cursor.sh gRPC
```

1. **PKCE OAuth** — browser-based login to Cursor, no client secret needed
2. **Model discovery** — queries Cursor's `GetUsableModels` gRPC endpoint
3. **Native Pi provider** — translates Pi context/tools directly to Cursor's protobuf/HTTP2 Connect protocol using Cursor's newer `requestedModel` request field
4. **Image input** — forwards Pi image blocks as Cursor `SelectedImage` entries in `UserMessage.selectedContext`
5. **Tool-result images** — forwards image blocks returned by Pi tools as Cursor MCP image result content
6. **Tool routing** — rejects Cursor's native tools, exposes Pi's tools via MCP

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

| Raw Cursor ID                  | Base                | Effort   | Variant     |
| ------------------------------ | ------------------- | -------- | ----------- |
| `gpt-5.4-medium`               | `gpt-5.4`           | `medium` | —           |
| `gpt-5.4-high-fast`            | `gpt-5.4`           | `high`   | `-fast`     |
| `claude-4.6-opus-max-thinking` | `claude-4.6-opus`   | `max`    | `-thinking` |
| `claude-opus-4-7-thinking-max` | `claude-opus-4-7`   | `max`    | `-thinking` |
| `gpt-5.1-codex-max-high`       | `gpt-5.1-codex-max` | `high`   | —           |
| `composer-2`                   | `composer-2`        | —        | —           |

Models sharing the same `(base, variant)` with **≥2 effort levels** and a sensible default (`medium` or no-suffix) are collapsed into a single entry with a Pi `thinkingLevelMap`. Pi's thinking level maps to the effort suffix:

| Pi Level  | Cursor Suffix                   |
| --------- | ------------------------------- |
| `minimal` | `none` (if available) or `low`  |
| `low`     | `low`                           |
| `medium`  | `medium` or no suffix (default) |
| `high`    | `high`                          |
| `xhigh`   | `max` (Claude) or `xhigh` (GPT) |

### Parameterized Cursor models

Cursor exposes some choices as model parameters rather than standalone model IDs. The extension queries Cursor's `AiService.AvailableModels(useModelParameters=true)` endpoint when authenticated and generates rows for all advertised parameterized model variants; where Cursor marks a model as supporting Max Mode, rows expose normalized `-max` entries (not `-max-mode`, and never duplicated as `-max-max`) over the same advertised parameter sets. At startup, it attempts live discovery with Pi's stored Cursor OAuth credentials (or a `CURSOR_ACCESS_TOKEN` override for testing) so models are available in `pi --list-models` and the model picker without requiring a fresh `/login cursor`. It falls back to the bundled static list when live metadata is unavailable. For example, GPT-5.5 has separate **Context** settings (`272K` and `1M`), **Reasoning** settings, and a **Fast** toggle for 272K variants. Pi's model picker cannot edit those Cursor-specific parameters directly, so this extension exposes them as separate selectable rows:

| Pi model           | Cursor `requestedModel`                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| `gpt-5.5`          | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "false"`, `maxMode: false` |
| `gpt-5.5-fast`     | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "true"`, `maxMode: false`  |
| `gpt-5.5-max`      | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "false"`, `maxMode: true`  |
| `gpt-5.5-max-fast` | `modelId: "gpt-5.5"`, `context: "272k"`, `fast: "true"`, `maxMode: true`   |
| `gpt-5.5-1m`       | `modelId: "gpt-5.5"`, `context: "1m"`, `fast: "false"`, `maxMode: true`    |

Pi's thinking level supplies the Cursor `reasoning` parameter for those rows (`none`, `low`, `medium`, `high`, or `extra-high`). Pi's `minimal` level maps to Cursor `none` when available, otherwise to the lowest available Cursor effort; `minimal` is never sent to Cursor. Likewise, `max` is not sent as a Cursor reasoning parameter. There is no separate `/max` toggle: Cursor-specific flags like `maxMode` and `fast` are determined by the selected model row. Cursor's own model metadata does not include any `context=1m` + `fast=true` GPT-5.5 variant; sending that invalid combination is sanitized by Cursor to the default 1M medium configuration, so this extension intentionally does not expose `gpt-5.5-1m-fast`.

For deduped models, the extension keeps an exact map from `(displayed model, effort)` back to the raw Cursor model ID or parameter set returned/derived from Cursor. That avoids guessing where the effort segment belongs:

```
pi selects: gpt-5.4-fast              + effort: high   → Cursor receives: gpt-5.4-high-fast
pi selects: gpt-5.4                   + effort: medium → Cursor receives: gpt-5.4-medium
pi selects: gpt-5.5-max-fast          + effort: high   → Cursor receives: gpt-5.5 + context=272k + reasoning=high + fast=true + maxMode=true
pi selects: gpt-5.5-1m                + effort: high   → Cursor receives: gpt-5.5 + context=1m + reasoning=high + fast=false + maxMode=true
pi selects: claude-opus-4-7-thinking  + effort: xhigh  → Cursor receives: claude-opus-4-7 + thinking=true + context=300k + effort=xhigh
pi selects: composer-2-fast           + no effort      → Cursor receives: composer-2 + fast=true
pi selects: composer-2-max-fast       + no effort      → Cursor receives: composer-2 + fast=true + maxMode=true
```

When a group is **collapsed**, the provider registers one model with Pi thinking-level support and an internal effort map (see table above).

**Collapsed** when Cursor returns either:

- **Multiple** effort suffixes for the same `(base, -fast, -thinking)` group, or
- **A single** variant whose parsed effort suffix is **non-empty** (for example only `claude-4.5-opus-high` is listed). The suffix is removed from the displayed ID so Pi's reasoning-effort setting supplies it.

**Left as-is** (raw Cursor ID on that row, with no effort `thinkingLevelMap`) when the group has **one** variant and the parsed effort suffix is **empty**—typically IDs with no effort segment, such as `composer-2`, `gemini-3.1-pro`, or `kimi-k2.5`.

### Disabling the mapping

To see all raw Cursor model variants without dedup:

```bash
PI_CURSOR_RAW_MODELS=1 pi
```

### Live Cursor metadata verification

Normal tests use fixtures and do not require Cursor credentials. To verify the live Cursor parameterized metadata path, provide an access token and run:

```bash
LIVE_CURSOR_METADATA=1 CURSOR_ACCESS_TOKEN=... npm run verify:cursor-live
```

### Cursor client version header

The HTTP/2 bridge sends Cursor's CLI client-version header. Override it when testing against a different installed Cursor Agent build:

```bash
PI_CURSOR_CLIENT_VERSION=<cursor-cli-version> pi
```

## Image Support

Drag-and-dropped/user-attached images are forwarded to Cursor as selected images. Image blocks returned by pi tools are forwarded as MCP image result content, enabling screenshot-driven debugging, browser/tool visual feedback, generated image review, and visual regression analysis.

Image constraints are grounded in the Cursor CLI local-image path:

- Supported formats are detected by magic bytes: jpeg, png, gif, or webp.
- Maximum processed image payload is 5,242,880 bytes.
- Cursor CLI downscales/compresses local files to fit that cap; this provider receives inline image bytes from Pi, so oversized inline images are rejected rather than resized.
- Remote image URLs are not fetched; attach the image through Pi so it is available as an inline image block. Plain text URLs remain plain text.

## Session Management

The native provider runtime maintains conversation state per Pi session, enabling multi-turn conversations with Cursor models while preserving forks, tool continuations, and interruptions correctly.

### How it works

- **Session tracking** — Pi passes the session ID through `streamSimple` options. The provider keys bridge state and stored conversation state from that real session ID.
- **Checkpoints** — Cursor returns a conversation checkpoint after completed turns. The provider stores that checkpoint, plus the completed-turn count and a fingerprint of the completed structured history, and reuses it only when the incoming history still matches.
- **Session-scoped state** — real pi session state is kept in memory until explicit cleanup or process restart. Anonymous fallback state can still be TTL-evicted.
- **Lifecycle cleanup** — session state is cleaned up on pi lifecycle events such as session switch, fork, `/tree`, and shutdown.

### Tool continuations

When Cursor pauses for a tool call, the provider keeps the live upstream bridge open and waits for Pi to send the tool result on the next stream request. That tool result is sent back into the same in-flight Cursor run, so the tool continuation stays part of the original user turn instead of inflating completed history.

If that live bridge is gone before the tool result arrives (for example because the provider process restarted or the upstream stream died), the provider returns an explicit `tool_continuation_lost` error instead of silently starting a new Cursor turn with the tool result as user text. Retry from before the tool call or start a new turn. Paused tool-call bridges are cancelled after a TTL (default 15 minutes; override with `PI_CURSOR_ACTIVE_BRIDGE_TTL_MS`).

`tool_choice: "none"` is honored by withholding MCP tools from Cursor when supplied through provider options/hooks. Other forced tool choices are rejected because Cursor's agent protocol does not expose an equivalent control. Cursor controls output budgeting server-side; unsupported sampling parameters such as `temperature` are rejected instead of silently ignored.

### Interruptions

If Pi interrupts a turn mid-stream, the provider cancels the upstream Cursor run and does **not** commit the pending checkpoint or its newly written blobs. Checkpoints are only committed after a turn finishes successfully.

### Session fork

When you navigate back in Pi's session tree and branch from an earlier point, the provider discards the stored checkpoint whenever the completed history no longer matches the stored checkpoint metadata. That includes both:

- completed turn count mismatches, and
- same-depth branch changes detected via completed-history fingerprint mismatch.

After discarding a stale checkpoint, the provider reconstructs proper protobuf conversation turns from the message history Pi sends, so Cursor sees the actual conversation structure at the fork point.

### Session resume

Conversation state is stored in memory. If the provider process restarts, checkpoints are lost. On the next request, Pi sends the full conversation history, and the provider reconstructs structured protobuf turns from that history instead of relying on an inline plaintext fallback.

That reconstruction preserves:

- user image attachments
- assistant messages
- tool calls
- tool results, including image result content
- final assistant text after tool results

## Requirements

- [Pi](https://github.com/earendil-works/pi-mono)
- [Node.js](https://nodejs.org) >= 20.6.0
- Active [Cursor](https://cursor.com) subscription

## Development

```bash
npm install
npm run typecheck
npm run lint
npm run format:check
npm test
npm run pack:check
```

Protocol notes, generated protobuf provenance, and reverse-engineered wire-field mappings are documented in [`docs/protocol.md`](docs/protocol.md).

## Debug log timeline

When `PI_CURSOR_PROVIDER_DEBUG=1` is enabled, the provider runtime writes timestamped JSONL logs to `os.tmpdir()` by default. You can turn a log into a compact human-readable timeline with:

```bash
npm run debug:timeline -- --latest
npm run debug:timeline -- /path/to/pi-cursor-provider-debug-2026-04-08T14-06-07-565Z-41184.log
```

Add `--json` if you want the parsed summary as JSON instead of formatted text.

## Credits

OAuth flow and gRPC protocol bridge adapted from [opencode-cursor](https://github.com/ephraimduncan/opencode-cursor) by Ephraim Duncan.
