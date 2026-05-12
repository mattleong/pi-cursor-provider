# Cursor protocol notes

This provider talks to Cursor's upstream HTTP/2 Connect/protobuf APIs. The generated schema in `proto/agent_pb.ts` is produced by `protoc-gen-es` for Cursor's `agent.v1` API and is intentionally committed so the extension can run without a build step inside Pi.

## Runtime RPCs

- `POST /agent.v1.AgentService/Run` — streaming agent run used for chat completions, tool calls, checkpoints, and blob exchange.
- `POST /agent.v1.AgentService/GetUsableModels` — unary model discovery using generated protobuf schemas.
- `POST /aiserver.v1.AiService/AvailableModels` — unary parameterized model discovery.

## Runtime liveness

The HTTP/2 bridge still sends Cursor `clientHeartbeat` frames so Cursor keeps long agent runs alive, but provider-level liveness is based on decoded Cursor server messages rather than those client heartbeat writes. Active runs have two watchdogs: an upstream-idle timer reset by non-heartbeat server messages, and a visible-idle timer reset only by text/thinking/tool-call output. The visible timer intentionally does not reset on `tokenDelta`, so token-only hidden generation cannot spin forever without surfacing progress to Pi. Both watchdogs are stopped while a Cursor tool call is paused for Pi to execute; paused tool bridges are governed by `PI_CURSOR_ACTIVE_BRIDGE_TTL_MS` instead.

`h2-bridge.mjs` is a small Node HTTP/2 child process. It exists because the extension can be loaded by runtimes whose `node:http2` compatibility is unreliable. The parent process speaks length-prefixed frames to the child; the child speaks HTTP/2 to Cursor.

## Generated protobuf

`proto/agent_pb.ts` should be regenerated whenever Cursor changes the `agent.v1` schema used by the CLI. When updating it:

1. Extract the matching Cursor CLI protobuf schema for the client version used by `h2-bridge.mjs`'s `x-cursor-client-version` header.
2. Regenerate with `protoc-gen-es` compatible with `@bufbuild/protobuf` v2.
3. Run `npm test`, `npm run typecheck`, and live metadata verification when credentials are available.
4. Update this file with any new protocol assumptions.

The repository currently commits the generated TypeScript only; the upstream `.proto` source is not redistributed here.

## Manual wire helpers

`cursor-wire.ts` and `proxy.ts` contain the remaining reverse-engineered wire helpers that do not yet have generated schemas in this repo:

- `AvailableModelsRequest` encoder:
  - field 5: `use_model_parameters = true`
  - field 7: `do_not_use_markdown = true`
- `AvailableModelsResponse` decoder:
  - response field 2: repeated model
  - model field 1: name
  - model field 10: supports images
  - model field 14: supports max mode
  - model field 15: context token limit
  - model field 16: max-mode context token limit
  - model field 17: client display name
  - model field 18: server model name
  - model field 19: supports non-max mode
  - model field 30: repeated parameterized variant
  - variant field 1: repeated `{ id, value }` parameter
  - variant field 2: display name
  - variant field 3: is max mode
  - variant field 4: is default max config
  - variant field 5: is default non-max config
  - variant field 8: display name outside picker
  - variant field 9: variant string representation
- MCP schema compatibility:
  - Cursor CLI's current `agent.v1.McpToolDefinition.input_schema` and `McpArgs.args` map values are `google.protobuf.Value` messages.
  - The committed generated `proto/agent_pb.ts` still exposes those length-delimited fields as `bytes`, so the provider writes and reads serialized `Value` bytes at those positions.
- `AgentRunRequest.pre_fetched_blobs` encoder in `proxy.ts`:
  - request field 17: repeated pre-fetched blob
  - pre-fetched blob field 1: blob id bytes
  - pre-fetched blob field 2: blob value bytes
- System prompts are sent as JSON root prompt blobs referenced by `ConversationStateStructure.root_prompt_messages_json`. The provider deliberately does **not** send `AgentRunRequest.custom_system_prompt` because live Cursor backends can reject it as the internal `--system-prompt` option.
- `UserMessageAction.conversation_history` encoder in `proxy.ts`:
  - user-message action field 7: `ConversationHistory`
  - conversation history field 1: repeated messages
  - message field 1/2/3: user/assistant/tool oneof
  - user and assistant content field 1: repeated content
  - text content field 1: text
  - image content fields 1/2: base64 data string / MIME type
  - assistant content field 4: tool call
  - tool call fields 1/2/3: tool call id / tool name / JSON args
  - tool message fields 1/2/3/4: tool call id / tool name / repeated result content / is error

Cursor CLI `2026.05.09-0afadcc` no longer treats `UserMessage` field 10 as `selectedContextBlob`; that field is `conversation_state_blob_id` in the installed agent bundle, and field 17 is `thread_id`. The provider therefore does not send the stale field-10 selected-context or field-17 correlation workaround. Normal Pi-history rebuilds use `UserMessageAction.conversation_history` with no duplicated turn history in `ConversationStateStructure.turns`; the state only carries root prompt blob references and other non-history scaffolding. Rebuild requests intentionally include a redundant inline Pi transcript safety copy in the current `UserMessage.text` by default so prior context remains visible if Cursor ignores the reverse-engineered native history field. Valid checkpoints may be reused when their Pi-history/system-prompt fingerprints still match; checkpoint-backed requests do not add the inline transcript, avoiding repeated checkpoint+transcript duplication.

Prefer replacing these helpers with generated schemas once the corresponding `.proto` definitions are available.
