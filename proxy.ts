/**
 * Cursor native provider runtime: translates Pi streamSimple context to Cursor's
 * protobuf/HTTP2 Connect protocol.
 *
 * Based on https://github.com/ephraimduncan/opencode-cursor by Ephraim Duncan.
 * Uses Node's http2 via a child process bridge (h2-bridge.mjs).
 */
import { create, fromBinary, fromJson, type JsonValue, toBinary, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type ImageContent as PiImageContent,
  type Message as PiMessage,
  type Model,
  type SimpleStreamOptions,
  type TextContent as PiTextContent,
  type Tool as PiTool,
  type ToolCall as PiToolCall,
} from "@earendil-works/pi-ai";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createHash } from "node:crypto";
import { appendFileSync } from "node:fs";
import { homedir, release as osRelease, tmpdir, type as osType } from "node:os";
import { isAbsolute, join as pathJoin, resolve as pathResolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  AgentClientMessageSchema,
  AgentServerMessageSchema,
  CancelActionSchema,
  ClientHeartbeatSchema,
  ConversationActionSchema,
  ConversationStateStructureSchema,
  BackgroundShellSpawnResultSchema,
  ComputerUseErrorSchema,
  ComputerUseResultSchema,
  DeleteResultSchema,
  DeleteRejectedSchema,
  DiagnosticsResultSchema,
  ExecClientMessageSchema,
  FetchErrorSchema,
  FetchResultSchema,
  GetBlobResultSchema,
  GrepErrorSchema,
  GrepResultSchema,
  KvClientMessageSchema,
  ListMcpResourcesExecResultSchema,
  ListMcpResourcesRejectedSchema,
  LsRejectedSchema,
  LsResultSchema,
  McpImageContentSchema,
  McpResultSchema,
  McpSuccessSchema,
  McpTextContentSchema,
  McpToolDefinitionSchema,
  McpToolErrorSchema,
  McpToolResultSchema,
  McpToolResultContentItemSchema,
  McpToolsSchema,
  RequestedModelSchema,
  RequestedModel_ModelParameterbytesSchema,
  ReadMcpResourceExecResultSchema,
  ReadMcpResourceRejectedSchema,
  ReadRejectedSchema,
  ReadResultSchema,
  RecordScreenFailureSchema,
  RecordScreenResultSchema,
  RequestContextEnvSchema,
  RequestContextResultSchema,
  RequestContextSchema,
  RequestContextSuccessSchema,
  SelectedContextSchema,
  SelectedImageSchema,
  SetBlobResultSchema,
  ShellRejectedSchema,
  ShellResultSchema,
  ShellStreamSchema,
  UserMessageSchema,
  WriteRejectedSchema,
  WriteResultSchema,
  WriteShellStdinErrorSchema,
  WriteShellStdinResultSchema,
  GetUsableModelsRequestSchema,
  GetUsableModelsResponseSchema,
  type AgentServerMessage,
  type ConversationStateStructure,
  type ExecServerMessage,
  type KvServerMessage,
  type McpToolDefinition,
  type McpTools,
  type RequestedModel,
  type UserMessage,
} from "./proto/agent_pb.js";
import {
  createConnectFrameParser,
  frameConnectMessage,
  parseConnectEndStream,
  spawnBridge,
  type BridgeFactory,
  type BridgeHandle,
} from "./bridge.js";
import {
  decodeAvailableModelsResponse,
  encodeAvailableModelsRequest,
  type CursorModelParameter,
  type CursorParameterizedModel,
} from "./cursor-wire.js";
import {
  isCursorProviderDebugEnabled,
  summarizeBase64ImageData,
  summarizeByteData,
  truncateDebugString,
} from "./debug.js";
import {
  applyCursorModelRouting,
  resolveCursorModelRouting,
  type CursorModelRouting,
  type CursorModelRoutingByEffort,
  type CursorToolResultImagePayload,
} from "./cursor-routing.js";
export type {
  CursorModelParameter,
  CursorParameterizedModel,
  CursorParameterizedVariant,
} from "./cursor-wire.js";

// Cursor CLI's local-image path scales/compresses images to <= 5 MiB
// and accepts only jpeg/png/gif/webp by magic bytes.
const CURSOR_CLI_MAX_IMAGE_BYTES = 5_242_880;
const CURSOR_SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);
const MAX_OPENAI_REQUEST_BODY_BYTES = 25 * 1024 * 1024;

// ── Types ──

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface ContentPart {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
  image_url?: { url?: string };
}

interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null | ContentPart[];
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
  isError?: boolean;
  is_error?: boolean;
}

interface OpenAIToolDef {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface ChatCompletionRequest {
  model: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  tools?: OpenAIToolDef[];
  tool_choice?: unknown;
  reasoning_effort?: string;
  user?: string;
  pi_session_id?: string;
  cursor_model_id?: string;
  cursor_model_parameters?: CursorModelParameter[];
  cursor_tool_result_images?: CursorToolResultImagePayload[];
  cursor_requires_max_mode?: boolean;
  cursor_model_max_mode?: boolean;
  cursor_workspace_path?: string;
}

interface CursorWorkspaceContext {
  workspacePath: string;
  workspaceUri: string;
  projectFolder: string;
  terminalsFolder: string;
  agentSharedNotesFolder: string;
  agentConversationNotesFolder: string;
  agentTranscriptsFolder: string;
}

interface CursorRequestPayload {
  requestBytes: Uint8Array;
  blobStore: Map<string, Uint8Array>;
  mcpTools: McpToolDefinition[];
}

interface PendingExec {
  execId: string;
  execMsgId: number;
  toolCallId: string;
  toolName: string;
  decodedArgs: string;
}

interface ActiveBridge {
  bridge: BridgeHandle;
  heartbeatTimer: ReturnType<typeof setInterval>;
  toolTimeoutTimer?: ReturnType<typeof setTimeout>;
  blobStore: Map<string, Uint8Array>;
  mcpTools: McpToolDefinition[];
  pendingExecs: PendingExec[];
  currentTurn: ParsedTurn;
  convKey?: string;
  workspaceContext?: CursorWorkspaceContext;
}

export interface StoredConversation {
  conversationId: string;
  checkpoint: Uint8Array | null;
  checkpointTurnCount?: number;
  checkpointHistoryFingerprint?: string;
  checkpointSystemPromptFingerprint?: string;
  sessionScoped: boolean;
  blobStore: Map<string, Uint8Array>;
  lastAccessMs: number;
}

interface StreamState {
  toolCallIndex: number;
  pendingExecs: PendingExec[];
  outputTokens: number;
  totalTokens: number;
}

interface ToolResultInfo {
  toolCallId: string;
  content: string;
  images?: ParsedImageContent[];
  isError?: boolean;
}

export interface ParsedToolResult {
  content: string;
  isError: boolean;
  images?: ParsedImageContent[];
}

export interface ParsedImageContent {
  data: Uint8Array;
  mimeType: string;
}

export interface ParsedAssistantTextStep {
  kind: "assistantText";
  text: string;
}

export interface ParsedToolCallStep {
  kind: "toolCall";
  toolCallId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  result?: ParsedToolResult;
}

export type ParsedTurnStep = ParsedAssistantTextStep | ParsedToolCallStep;

export interface ParsedTurn {
  userText: string;
  steps: ParsedTurnStep[];
  userImages?: ParsedImageContent[];
}

interface ParsedMessages {
  systemPrompt: string;
  userText: string;
  userImages: ParsedImageContent[];
  turns: ParsedTurn[];
  toolResults: ToolResultInfo[];
}

// ── State ──

const activeBridges = new Map<string, ActiveBridge>();
const conversationStates = new Map<string, StoredConversation>();
const sessionLocks = new Map<string, Promise<void>>();
const CONVERSATION_TTL_MS = 30 * 60 * 1000;
const DEFAULT_ACTIVE_BRIDGE_TTL_MS = 15 * 60 * 1000;
const DEFAULT_UPSTREAM_IDLE_TIMEOUT_MS = 120 * 1000;
const DEFAULT_VISIBLE_IDLE_TIMEOUT_MS = 180 * 1000;
const configuredActiveBridgeTtlMs = Number(
  process.env.PI_CURSOR_ACTIVE_BRIDGE_TTL_MS ?? DEFAULT_ACTIVE_BRIDGE_TTL_MS,
);
const ACTIVE_BRIDGE_TTL_MS = Number.isFinite(configuredActiveBridgeTtlMs)
  ? Math.max(1_000, configuredActiveBridgeTtlMs)
  : DEFAULT_ACTIVE_BRIDGE_TTL_MS;
const UPSTREAM_IDLE_TIMEOUT_MS = readTimeoutMsEnv(
  "PI_CURSOR_UPSTREAM_IDLE_TIMEOUT_MS",
  DEFAULT_UPSTREAM_IDLE_TIMEOUT_MS,
);
const VISIBLE_IDLE_TIMEOUT_MS = readTimeoutMsEnv(
  "PI_CURSOR_VISIBLE_IDLE_TIMEOUT_MS",
  DEFAULT_VISIBLE_IDLE_TIMEOUT_MS,
);
const DEFAULT_INLINE_HISTORY_MAX_CHARS = Number.POSITIVE_INFINITY;
const DEFAULT_INLINE_HISTORY_HEAD_MAX_CHARS = Number.POSITIVE_INFINITY;
const DEFAULT_INLINE_HISTORY_SEGMENT_MAX_CHARS = Number.POSITIVE_INFINITY;
const INLINE_HISTORY_MAX_CHARS = readNonNegativeIntegerEnv(
  "PI_CURSOR_INLINE_HISTORY_MAX_CHARS",
  DEFAULT_INLINE_HISTORY_MAX_CHARS,
);
const INLINE_HISTORY_HEAD_MAX_CHARS = readNonNegativeIntegerEnv(
  "PI_CURSOR_INLINE_HISTORY_HEAD_MAX_CHARS",
  DEFAULT_INLINE_HISTORY_HEAD_MAX_CHARS,
);
const INLINE_HISTORY_SEGMENT_MAX_CHARS = readNonNegativeIntegerEnv(
  "PI_CURSOR_INLINE_HISTORY_SEGMENT_MAX_CHARS",
  DEFAULT_INLINE_HISTORY_SEGMENT_MAX_CHARS,
);
const defaultBridgeFactory: BridgeFactory = (options) => spawnBridge(options, debugLog);
let bridgeFactory: BridgeFactory = defaultBridgeFactory;
let debugRequestCounter = 0;
let debugLogFilePath: string | undefined;

function readNonNegativeIntegerEnv(name: string, fallback: number): number {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return fallback;
  if (["full", "infinite", "infinity", "unbounded"].includes(raw)) {
    return Number.POSITIVE_INFINITY;
  }
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : fallback;
}

function readTimeoutMsEnv(name: string, fallback: number): number {
  const value = readNonNegativeIntegerEnv(name, fallback);
  if (!Number.isFinite(value) || value <= 0) return 0;
  return Math.max(1_000, value);
}

function isProxyDebugEnabled(): boolean {
  return isCursorProviderDebugEnabled();
}

function summarizeDebugImageUrl(url: string): unknown {
  const trimmed = url.trim();
  const match = trimmed.match(/^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/is);
  if (match) {
    return {
      mimeType: normalizeImageMimeType(match[1]!),
      ...summarizeBase64ImageData(match[2]!),
    };
  }
  return {
    url: trimmed.startsWith("data:image/")
      ? `<redacted data image ${trimmed.length} chars>`
      : truncateDebugString(trimmed),
  };
}

function summarizeDebugImageObject(value: Record<string, unknown>): unknown | undefined {
  const imageUrl = value.image_url;
  if (imageUrl && typeof imageUrl === "object") {
    const url = (imageUrl as Record<string, unknown>).url;
    if (typeof url === "string")
      return { type: value.type ?? "image_url", image_url: summarizeDebugImageUrl(url) };
  }

  const mimeType =
    typeof value.mimeType === "string" ? normalizeImageMimeType(value.mimeType) : undefined;
  if (!mimeType?.startsWith("image/")) return undefined;
  const data = value.data;
  if (typeof data === "string") {
    return { type: value.type, mimeType, ...summarizeBase64ImageData(data) };
  }
  if (data instanceof Uint8Array || Buffer.isBuffer(data)) {
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    return { type: value.type, mimeType, ...summarizeByteData(bytes) };
  }
  return undefined;
}

function sanitizeForDebug(value: unknown): unknown {
  if (value == null) return value;
  if (typeof value === "string") return truncateDebugString(value);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
    const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
    return {
      __type: value instanceof Uint8Array ? "Uint8Array" : "Buffer",
      ...summarizeByteData(bytes),
    };
  }
  if (Array.isArray(value)) return value.map((item) => sanitizeForDebug(item));
  if (value instanceof Map) {
    return {
      __type: "Map",
      size: value.size,
      entries: Array.from(value.entries())
        .slice(0, 20)
        .map(([k, v]) => [sanitizeForDebug(k), sanitizeForDebug(v)]),
    };
  }
  if (typeof value === "object") {
    const imageSummary = summarizeDebugImageObject(value as Record<string, unknown>);
    if (imageSummary) return imageSummary;
    const entries = Object.entries(value as Record<string, unknown>).map(([key, inner]) => {
      if (key === "accessToken") return [key, "<redacted>"] as const;
      if (key === "data" && typeof inner === "string")
        return [key, `<redacted base64 ${inner.length} chars>`] as const;
      if (key === "url" && typeof inner === "string" && inner.startsWith("data:image/")) {
        return [key, `<redacted data image ${inner.length} chars>`] as const;
      }
      return [key, sanitizeForDebug(inner)] as const;
    });
    return Object.fromEntries(entries);
  }
  return String(value);
}

function getDebugLogFilePath(): string {
  const configured = process.env.PI_CURSOR_PROVIDER_DEBUG_FILE?.trim();
  if (configured) return configured;
  if (debugLogFilePath) return debugLogFilePath;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  debugLogFilePath = pathJoin(tmpdir(), `pi-cursor-provider-debug-${stamp}-${process.pid}.log`);
  return debugLogFilePath;
}

function debugLog(event: string, data?: Record<string, unknown>): void {
  if (!isProxyDebugEnabled()) return;
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    pid: process.pid,
    event,
    ...(data ? (sanitizeForDebug(data) as Record<string, unknown>) : {}),
  });
  const file = getDebugLogFilePath();
  try {
    appendFileSync(file, `${line}\n`, "utf8");
  } catch (error) {
    console.error("[pi-cursor-provider] failed to write debug log", error);
    console.error(`[pi-cursor-provider] ${line}`);
  }
}

function nextDebugRequestId(): string {
  debugRequestCounter += 1;
  return `req-${debugRequestCounter}`;
}

export const __testInternals = {
  activeBridges,
  conversationStates,
  fingerprintCompletedTurns,
  fingerprintSystemPrompt,
  UPSTREAM_IDLE_TIMEOUT_MS,
  VISIBLE_IDLE_TIMEOUT_MS,
};

export function setBridgeFactoryForTests(factory?: BridgeFactory): void {
  bridgeFactory = factory ?? defaultBridgeFactory;
}

let proxyServer: ReturnType<typeof createServer> | undefined;
let proxyPort: number | undefined;
let proxyAccessTokenProvider: (() => Promise<string>) | undefined;

// ── Unary RPC (for model discovery) ──

export async function callCursorUnaryRpc(options: {
  accessToken: string;
  rpcPath: string;
  requestBody: Uint8Array;
  url?: string;
  timeoutMs?: number;
}): Promise<{ body: Uint8Array; exitCode: number; timedOut: boolean }> {
  const bridge = bridgeFactory({
    accessToken: options.accessToken,
    rpcPath: options.rpcPath,
    url: options.url,
    unary: true,
  });
  const chunks: Buffer[] = [];
  return new Promise((resolve) => {
    let timedOut = false;
    const timeoutMs = options.timeoutMs ?? 5_000;
    const timeout =
      timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            try {
              bridge.proc.kill();
            } catch {}
          }, timeoutMs)
        : undefined;

    bridge.onData((chunk) => {
      chunks.push(Buffer.from(chunk));
    });
    bridge.onClose((exitCode) => {
      if (timeout) clearTimeout(timeout);
      resolve({ body: Buffer.concat(chunks), exitCode, timedOut });
    });

    bridge.write(options.requestBody);
    bridge.end();
  });
}

// ── Model discovery ──

export interface CursorModel {
  id: string;
  name: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  requestedModelId?: string;
  parameters?: CursorModelParameter[];
  requiresMaxMode?: boolean;
  requestedMaxMode?: boolean;
  supportsImages?: boolean;
}

let cachedModels: { tokenHash: string; models: CursorModel[] } | null = null;
let cachedParameterizedModels: { tokenHash: string; models: CursorParameterizedModel[] } | null =
  null;

function tokenCacheHash(apiKey: string): string {
  return createHash("sha256").update(apiKey).digest("hex").slice(0, 16);
}

export async function getCursorModels(apiKey: string): Promise<CursorModel[]> {
  const tokenHash = tokenCacheHash(apiKey);
  if (cachedModels?.tokenHash === tokenHash) return cachedModels.models;
  try {
    const requestPayload = create(GetUsableModelsRequestSchema, {});
    const requestBody = toBinary(GetUsableModelsRequestSchema, requestPayload);
    const response = await callCursorUnaryRpc({
      accessToken: apiKey,
      rpcPath: "/agent.v1.AgentService/GetUsableModels",
      requestBody,
    });
    if (!response.timedOut && response.exitCode === 0 && response.body.length > 0) {
      let decoded: any = null;
      try {
        decoded = fromBinary(GetUsableModelsResponseSchema, response.body);
      } catch {
        // Try Connect framing
        const body = decodeConnectUnaryBody(response.body);
        if (body) {
          try {
            decoded = fromBinary(GetUsableModelsResponseSchema, body);
          } catch {}
        }
      }
      if (decoded?.models?.length) {
        const models = normalizeCursorModels(decoded.models);
        if (models.length > 0) {
          cachedModels = { tokenHash, models };
          return models;
        }
      }
    }
  } catch (err) {
    console.error(
      "[cursor-provider] Model discovery failed:",
      err instanceof Error ? err.message : err,
    );
  }
  console.warn("[cursor-provider] Model discovery returned no models");
  return [];
}

function decodeConnectUnaryBody(payload: Uint8Array): Uint8Array | null {
  if (payload.length < 5) return null;
  let offset = 0;
  while (offset + 5 <= payload.length) {
    const flags = payload[offset]!;
    const view = new DataView(
      payload.buffer,
      payload.byteOffset + offset,
      payload.byteLength - offset,
    );
    const messageLength = view.getUint32(1, false);
    const frameEnd = offset + 5 + messageLength;
    if (frameEnd > payload.length) return null;
    if ((flags & 0b0000_0001) !== 0) return null;
    if ((flags & 0b0000_0010) === 0) return payload.subarray(offset + 5, frameEnd);
    offset = frameEnd;
  }
  return null;
}

export async function getCursorParameterizedModels(
  apiKey: string,
): Promise<CursorParameterizedModel[]> {
  const tokenHash = tokenCacheHash(apiKey);
  if (cachedParameterizedModels?.tokenHash === tokenHash) return cachedParameterizedModels.models;
  try {
    const response = await callCursorUnaryRpc({
      accessToken: apiKey,
      rpcPath: "/aiserver.v1.AiService/AvailableModels",
      requestBody: encodeAvailableModelsRequest(),
    });
    if (response.timedOut || response.exitCode !== 0 || response.body.length === 0) return [];
    const body = decodeConnectUnaryBody(response.body) ?? response.body;
    const models = decodeAvailableModelsResponse(body);
    cachedParameterizedModels = { tokenHash, models };
    return models;
  } catch (err) {
    console.error(
      "[cursor-provider] Parameterized model discovery failed:",
      err instanceof Error ? err.message : err,
    );
    return [];
  }
}

export function inferCursorContextWindow(id: string, name: string): number {
  const text = `${id} ${name}`.toLowerCase();
  if (/\b1\s*m\b|(?:^|-)1m(?:-|$)/.test(text)) return 1_000_000;
  if (/\b272\s*k\b|(?:^|-)272k(?:-|$)/.test(text)) return 272_000;
  return 200_000;
}

function normalizeCursorModels(models: readonly unknown[]): CursorModel[] {
  const byId = new Map<string, CursorModel>();
  for (const model of models) {
    const m = model as any;
    const id = m?.modelId?.trim?.();
    if (!id) continue;
    const name = m.displayName || m.displayNameShort || m.displayModelId || id;
    byId.set(id, {
      id,
      name,
      reasoning: Boolean(m.thinkingDetails),
      contextWindow: inferCursorContextWindow(id, name),
      maxTokens: 64_000,
    });
  }
  return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
}

// ── Proxy server ──

export function getProxyPort(): number | undefined {
  return proxyPort;
}

export async function startProxy(getAccessToken: () => Promise<string>): Promise<number> {
  proxyAccessTokenProvider = getAccessToken;
  if (proxyServer && proxyPort) return proxyPort;

  return new Promise((resolve, reject) => {
    const server = createServer(async (req, res) => {
      const url = new URL(req.url ?? "/", `http://localhost`);
      const requestId = nextDebugRequestId();
      debugLog("http.request", {
        requestId,
        method: req.method,
        pathname: url.pathname,
        headers: req.headers,
      });

      if (req.method === "GET" && url.pathname === "/v1/models") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: [] }));
        return;
      }

      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        try {
          const body = await readBody(req);
          const parsed = JSON.parse(body) as ChatCompletionRequest;
          debugLog("http.chat.body", { requestId, body: parsed });
          if (!proxyAccessTokenProvider) throw new Error("No access token provider");
          const accessToken = await proxyAccessTokenProvider();
          await withSessionLock(deriveRequestLockKey(parsed), () =>
            handleChatCompletion(parsed, accessToken, req, res, requestId),
          );
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          const invalidRequest =
            err instanceof SyntaxError || message.startsWith("Request body exceeds ");
          debugLog("http.chat.error", {
            requestId,
            message,
            stack: err instanceof Error ? err.stack : undefined,
          });
          res.writeHead(invalidRequest ? 400 : 500, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: {
                message,
                type: invalidRequest ? "invalid_request_error" : "server_error",
                code: invalidRequest ? "invalid_request" : "internal_error",
              },
            }),
          );
        }
        return;
      }

      res.writeHead(404);
      res.end("Not Found");
    });

    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (typeof addr === "object" && addr) {
        proxyPort = addr.port;
        proxyServer = server;
        debugLog("proxy.start", {
          port: proxyPort,
          debugLogFile: isProxyDebugEnabled() ? getDebugLogFilePath() : undefined,
        });
        resolve(proxyPort);
      } else {
        reject(new Error("Failed to bind proxy"));
      }
    });
  });
}

function clearActiveBridgeToolTimeout(active: ActiveBridge | undefined): void {
  if (active?.toolTimeoutTimer) clearTimeout(active.toolTimeoutTimer);
}

function removeActiveBridge(bridgeKey: string): void {
  clearActiveBridgeToolTimeout(activeBridges.get(bridgeKey));
  activeBridges.delete(bridgeKey);
}

function createActiveBridgeToolTimeout(
  bridgeKey: string,
  active: Pick<ActiveBridge, "bridge" | "heartbeatTimer" | "convKey">,
): ReturnType<typeof setTimeout> {
  const toolTimeoutTimer = setTimeout(() => {
    debugLog("bridge.active_ttl_expired", {
      bridgeKey,
      convKey: active.convKey,
      ttlMs: ACTIVE_BRIDGE_TTL_MS,
    });
    cleanupBridge(active.bridge, active.heartbeatTimer, bridgeKey);
  }, ACTIVE_BRIDGE_TTL_MS);
  toolTimeoutTimer.unref?.();
  return toolTimeoutTimer;
}

function setActiveBridge(bridgeKey: string, active: Omit<ActiveBridge, "toolTimeoutTimer">): void {
  clearActiveBridgeToolTimeout(activeBridges.get(bridgeKey));
  const toolTimeoutTimer = createActiveBridgeToolTimeout(bridgeKey, active);
  activeBridges.set(bridgeKey, { ...active, toolTimeoutTimer });
}

export function touchActiveBridgeForSession(
  sessionId?: string,
  reason = "tool_execution",
): boolean {
  if (!sessionId) return false;
  const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
  const active = activeBridges.get(bridgeKey);
  if (!active) return false;
  clearActiveBridgeToolTimeout(active);
  active.toolTimeoutTimer = createActiveBridgeToolTimeout(bridgeKey, active);
  debugLog("bridge.active_ttl_refreshed", {
    sessionId,
    bridgeKey,
    convKey: active.convKey,
    ttlMs: ACTIVE_BRIDGE_TTL_MS,
    reason,
  });
  return true;
}

type CursorStreamTimeoutKind = "upstream_idle" | "visible_idle";

interface CursorStreamWatchdog {
  touchUpstream(reason: string): void;
  touchVisible(reason: string): void;
  stop(): void;
}

function describeTimeoutMs(timeoutMs: number): string {
  if (timeoutMs % 60_000 === 0) return `${timeoutMs / 60_000}m`;
  if (timeoutMs % 1000 === 0) return `${timeoutMs / 1000}s`;
  return `${timeoutMs}ms`;
}

function cursorStreamTimeoutMessage(kind: CursorStreamTimeoutKind, timeoutMs: number): string {
  const duration = describeTimeoutMs(timeoutMs);
  if (kind === "upstream_idle") {
    return `Cursor upstream stalled: no non-heartbeat server messages for ${duration}.`;
  }
  return `Cursor upstream stalled: no visible text, thinking, or tool call for ${duration}.`;
}

function createCursorStreamWatchdog(options: {
  eventPrefix: string;
  requestId?: string;
  bridgeKey: string;
  convKey: string;
  bridge: BridgeHandle;
  heartbeatTimer: ReturnType<typeof setInterval>;
  onTimeout(kind: CursorStreamTimeoutKind, message: string): void;
}): CursorStreamWatchdog {
  let stopped = false;
  let upstreamTimer: ReturnType<typeof setTimeout> | undefined;
  let visibleTimer: ReturnType<typeof setTimeout> | undefined;
  let lastUpstreamMs = Date.now();
  let lastVisibleMs = Date.now();
  let lastUpstreamReason = "stream_start";
  let lastVisibleReason = "stream_start";

  const clearTimers = () => {
    if (upstreamTimer) clearTimeout(upstreamTimer);
    if (visibleTimer) clearTimeout(visibleTimer);
    upstreamTimer = undefined;
    visibleTimer = undefined;
  };

  const fire = (kind: CursorStreamTimeoutKind, timeoutMs: number, lastMs: number) => {
    if (stopped) return;
    stopped = true;
    clearTimers();
    const message = cursorStreamTimeoutMessage(kind, timeoutMs);
    debugLog(`${options.eventPrefix}.${kind}_timeout`, {
      requestId: options.requestId,
      bridgeKey: options.bridgeKey,
      convKey: options.convKey,
      timeoutMs,
      idleMs: Math.max(0, Date.now() - lastMs),
      lastUpstreamReason,
      lastVisibleReason,
      visibleTimeoutMs: VISIBLE_IDLE_TIMEOUT_MS,
      upstreamTimeoutMs: UPSTREAM_IDLE_TIMEOUT_MS,
      message,
    });
    options.onTimeout(kind, message);
    cleanupBridge(options.bridge, options.heartbeatTimer, options.bridgeKey);
  };

  const armUpstream = () => {
    if (upstreamTimer) clearTimeout(upstreamTimer);
    if (stopped || UPSTREAM_IDLE_TIMEOUT_MS <= 0) return;
    upstreamTimer = setTimeout(
      () => fire("upstream_idle", UPSTREAM_IDLE_TIMEOUT_MS, lastUpstreamMs),
      UPSTREAM_IDLE_TIMEOUT_MS,
    );
    upstreamTimer.unref?.();
  };

  const armVisible = () => {
    if (visibleTimer) clearTimeout(visibleTimer);
    if (stopped || VISIBLE_IDLE_TIMEOUT_MS <= 0) return;
    visibleTimer = setTimeout(
      () => fire("visible_idle", VISIBLE_IDLE_TIMEOUT_MS, lastVisibleMs),
      VISIBLE_IDLE_TIMEOUT_MS,
    );
    visibleTimer.unref?.();
  };

  armUpstream();
  armVisible();

  return {
    touchUpstream(reason: string) {
      if (stopped) return;
      lastUpstreamMs = Date.now();
      lastUpstreamReason = reason;
      armUpstream();
    },
    touchVisible(reason: string) {
      if (stopped) return;
      lastVisibleMs = Date.now();
      lastVisibleReason = reason;
      armVisible();
    },
    stop() {
      if (stopped) return;
      stopped = true;
      clearTimers();
    },
  };
}

function cursorServerMessageLabel(msg: AgentServerMessage): string {
  const msgCase = msg.message.case ?? "unknown";
  if (msgCase === "interactionUpdate") {
    const updateCase = (msg.message.value as any).message?.case ?? "unknown";
    return `${msgCase}.${updateCase}`;
  }
  if (msgCase === "execServerMessage") {
    const execCase = (msg.message.value as any).message?.case ?? "unknown";
    return `${msgCase}.${execCase}`;
  }
  return msgCase;
}

function isCursorServerHeartbeat(msg: AgentServerMessage): boolean {
  return (
    msg.message.case === "interactionUpdate" &&
    (msg.message.value as any).message?.case === "heartbeat"
  );
}

export function cleanupAllSessionState(): void {
  debugLog("session.cleanup_all", {
    cleanupKind: "all",
    activeBridgeCount: activeBridges.size,
    conversationCount: conversationStates.size,
  });
  for (const [bridgeKey, active] of activeBridges) {
    cleanupBridge(active.bridge, active.heartbeatTimer, bridgeKey);
  }
  conversationStates.clear();
}

export function stopProxy(): void {
  debugLog("proxy.stop", { port: proxyPort });
  if (proxyServer) {
    proxyServer.close();
    proxyServer = undefined;
    proxyPort = undefined;
    proxyAccessTokenProvider = undefined;
  }
  cleanupAllSessionState();
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let rejected = false;
    req.on("data", (c: Buffer) => {
      if (rejected) return;
      total += c.length;
      if (total > MAX_OPENAI_REQUEST_BODY_BYTES) {
        rejected = true;
        reject(new Error(`Request body exceeds ${MAX_OPENAI_REQUEST_BODY_BYTES} byte limit`));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

// ── Native pi streamSimple provider ──

export type CursorNativeModelRouting = CursorModelRouting;

export interface CursorNativeStreamConfig {
  getAccessToken(): Promise<string>;
  getNoReasoningEffortByModelId?(): Map<string, string>;
  getRawModelRoutingByModelId?(): Map<string, CursorModelRoutingByEffort>;
}

type CursorNativeStreamOptions = SimpleStreamOptions & {
  toolChoice?: unknown;
};

type NativeBlockKind = "text" | "thinking";

interface NativeStreamWriter {
  output: AssistantMessage;
  closed: boolean;
  start(): void;
  text(delta: string): void;
  thinking(delta: string): void;
  toolCall(exec: PendingExec): void;
  done(reason: "stop" | "length" | "toolUse", state?: StreamState): void;
  error(message: string, reason: "error" | "aborted", state?: StreamState): void;
}

function emptyCursorUsage(totalTokens = 0): AssistantMessage["usage"] {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function usageContextTokens(usage: AssistantMessage["usage"]): number {
  return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

function estimateTextTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateContentTokens(content: unknown): number {
  if (typeof content === "string") return estimateTextTokens(content);
  if (!Array.isArray(content)) return 0;
  return content.reduce((total, block) => {
    if (!block || typeof block !== "object") return total;
    const typed = block as Record<string, unknown>;
    if (typeof typed.text === "string") return total + estimateTextTokens(typed.text);
    // Pi thinking blocks are not replayed into Cursor conversation history, so
    // counting them here makes the footer context estimate grow faster than the
    // actual prompt sent to Cursor.
    if (typed.type === "thinking") return total;
    if (typed.type === "toolCall") return total + estimateTextTokens(JSON.stringify(typed));
    if (typed.type === "image") return total + 85;
    return total;
  }, 0);
}

function estimateMessageTokens(message: PiMessage): number {
  const baseTokens = 4;
  if (message.role === "toolResult") {
    return (
      baseTokens + estimateTextTokens(message.toolName) + estimateContentTokens(message.content)
    );
  }
  return baseTokens + estimateContentTokens(message.content);
}

function assistantUsageForContext(message: PiMessage): AssistantMessage["usage"] | undefined {
  if (message.role !== "assistant") return undefined;
  if (message.stopReason === "aborted" || message.stopReason === "error") return undefined;
  return message.usage;
}

function estimateInFlightContextTokens(context: Context): number {
  const messages = context.messages ?? [];
  const systemPromptTokens = context.systemPrompt ? estimateTextTokens(context.systemPrompt) : 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const usage = assistantUsageForContext(messages[i]!);
    if (!usage) continue;
    const trailingTokens = messages
      .slice(i + 1)
      .reduce((total, message) => total + estimateMessageTokens(message), 0);
    return usageContextTokens(usage) + trailingTokens;
  }
  return (
    systemPromptTokens +
    messages.reduce((total, message) => total + estimateMessageTokens(message), 0)
  );
}

function tokenCost(tokens: number, ratePerMillion = 0): number {
  return (tokens * ratePerMillion) / 1_000_000;
}

function applyCursorUsage(
  output: AssistantMessage,
  model: Model<Api>,
  state?: StreamState,
  fallbackTotalTokens = 0,
): void {
  if (!state) return;
  const usage = computeUsage(state, fallbackTotalTokens);
  const costInput = tokenCost(usage.prompt_tokens, model.cost?.input);
  const costOutput = tokenCost(usage.completion_tokens, model.cost?.output);
  output.usage = {
    input: usage.prompt_tokens,
    output: usage.completion_tokens,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: usage.total_tokens,
    cost: {
      input: costInput,
      output: costOutput,
      cacheRead: 0,
      cacheWrite: 0,
      total: costInput + costOutput,
    },
  };
}

function createCursorAssistantMessage(
  model: Model<Api>,
  initialContextTokens = 0,
): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: emptyCursorUsage(initialContextTokens),
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function createNativeStreamWriter(
  stream: AssistantMessageEventStream,
  model: Model<Api>,
  initialContextTokens = 0,
): NativeStreamWriter {
  const output = createCursorAssistantMessage(model, initialContextTokens);
  let started = false;
  let closed = false;
  let active: { kind: NativeBlockKind; contentIndex: number; ended: boolean } | undefined;

  const ensureStarted = () => {
    if (started) return;
    started = true;
    stream.push({ type: "start", partial: output });
  };

  const endActiveBlock = () => {
    if (!active || active.ended) return;
    const block = output.content[active.contentIndex];
    if (active.kind === "text" && block?.type === "text") {
      stream.push({
        type: "text_end",
        contentIndex: active.contentIndex,
        content: block.text,
        partial: output,
      });
    } else if (active.kind === "thinking" && block?.type === "thinking") {
      stream.push({
        type: "thinking_end",
        contentIndex: active.contentIndex,
        content: block.thinking,
        partial: output,
      });
    }
    active.ended = true;
    active = undefined;
  };

  const ensureBlock = (kind: NativeBlockKind): number => {
    ensureStarted();
    if (active?.kind === kind && !active.ended) return active.contentIndex;
    endActiveBlock();
    const contentIndex = output.content.length;
    if (kind === "text") {
      output.content.push({ type: "text", text: "" });
      stream.push({ type: "text_start", contentIndex, partial: output });
    } else {
      output.content.push({ type: "thinking", thinking: "" });
      stream.push({ type: "thinking_start", contentIndex, partial: output });
    }
    active = { kind, contentIndex, ended: false };
    return contentIndex;
  };

  return {
    output,
    get closed() {
      return closed;
    },
    start: ensureStarted,
    text(delta: string) {
      if (closed || !delta) return;
      const contentIndex = ensureBlock("text");
      const block = output.content[contentIndex];
      if (block?.type !== "text") return;
      block.text += delta;
      stream.push({ type: "text_delta", contentIndex, delta, partial: output });
    },
    thinking(delta: string) {
      if (closed || !delta) return;
      const contentIndex = ensureBlock("thinking");
      const block = output.content[contentIndex];
      if (block?.type !== "thinking") return;
      block.thinking += delta;
      stream.push({ type: "thinking_delta", contentIndex, delta, partial: output });
    },
    toolCall(exec: PendingExec) {
      if (closed) return;
      ensureStarted();
      endActiveBlock();
      const contentIndex = output.content.length;
      const parsedArguments = parseToolCallArguments(exec.decodedArgs);
      const block = {
        type: "toolCall" as const,
        id: exec.toolCallId,
        name: exec.toolName,
        arguments: {},
      };
      output.content.push(block);
      stream.push({ type: "toolcall_start", contentIndex, partial: output });
      block.arguments = parsedArguments;
      stream.push({
        type: "toolcall_delta",
        contentIndex,
        delta: exec.decodedArgs,
        partial: output,
      });
      stream.push({
        type: "toolcall_end",
        contentIndex,
        toolCall: {
          type: "toolCall",
          id: exec.toolCallId,
          name: exec.toolName,
          arguments: parsedArguments,
        },
        partial: output,
      });
    },
    done(reason: "stop" | "length" | "toolUse", state?: StreamState) {
      if (closed) return;
      ensureStarted();
      endActiveBlock();
      applyCursorUsage(output, model, state, initialContextTokens);
      output.stopReason = reason;
      stream.push({ type: "done", reason, message: output });
      closed = true;
      stream.end(output);
    },
    error(message: string, reason: "error" | "aborted", state?: StreamState) {
      if (closed) return;
      ensureStarted();
      endActiveBlock();
      applyCursorUsage(output, model, state, initialContextTokens);
      output.stopReason = reason;
      output.errorMessage = message;
      stream.push({ type: "error", reason, error: output });
      closed = true;
      stream.end(output);
    },
  };
}

function isPiTextContent(block: unknown): block is PiTextContent {
  return !!block && typeof block === "object" && (block as { type?: unknown }).type === "text";
}

function isPiImageContent(block: unknown): block is PiImageContent {
  return !!block && typeof block === "object" && (block as { type?: unknown }).type === "image";
}

function isPiToolCall(block: unknown): block is PiToolCall {
  return !!block && typeof block === "object" && (block as { type?: unknown }).type === "toolCall";
}

function piContentToOpenAIContent(
  content: string | PiMessage["content"],
): OpenAIMessage["content"] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (isPiTextContent(block)) {
      parts.push({ type: "text", text: block.text });
    } else if (isPiImageContent(block)) {
      parts.push({ type: "image", data: block.data, mimeType: block.mimeType });
    }
  }
  return parts.length > 0 ? parts : "";
}

function assistantTextFromPiContent(content: AssistantMessage["content"]): string {
  return content
    .filter((block): block is PiTextContent => isPiTextContent(block))
    .map((block) => block.text)
    .join("\n");
}

function cursorAssistantTextFromPiMessage(message: AssistantMessage): string {
  const text = assistantTextFromPiContent(message.content);
  if (message.stopReason !== "aborted" || !text.trim()) return text;
  const reason = message.errorMessage?.trim() || "Operation aborted";
  return `${text}\n\n[Interrupted: the user aborted this assistant response before it completed (${reason}). Do not treat it as a completed answer.]`;
}

function assistantToolCallsFromPiContent(content: AssistantMessage["content"]): OpenAIToolCall[] {
  return content.filter(isPiToolCall).map((block) => ({
    id: block.id,
    type: "function" as const,
    function: {
      name: block.name,
      arguments: JSON.stringify(block.arguments ?? {}),
    },
  }));
}

function piToolToOpenAI(tool: PiTool): OpenAIToolDef {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters as unknown as Record<string, unknown>,
    },
  };
}

function resolveNativeReasoningEffort(
  model: Model<Api>,
  options: CursorNativeStreamOptions | undefined,
  noReasoningEffortByModelId?: Map<string, string>,
): string | undefined {
  const requested = options?.reasoning;
  if (requested) {
    const mapped = model.thinkingLevelMap?.[requested];
    return typeof mapped === "string" ? mapped : requested;
  }
  const offMapped = model.thinkingLevelMap?.off;
  if (typeof offMapped === "string") return offMapped;
  return noReasoningEffortByModelId?.get(model.id);
}

function applyNativeCursorRouting(
  body: ChatCompletionRequest,
  rawRoutingByModelId?: Map<string, CursorModelRoutingByEffort>,
): void {
  const payload = body as unknown as Record<string, unknown>;
  const directRouting = rawRoutingByModelId
    ? resolveCursorModelRouting(payload, rawRoutingByModelId)
    : undefined;
  const fallbackRouting = rawRoutingByModelId?.get(body.model)?.[""];
  applyCursorModelRouting(payload, directRouting ?? fallbackRouting);
}

function contextToCursorChatCompletionRequest(
  model: Model<Api>,
  context: Context,
  options: CursorNativeStreamOptions | undefined,
  config: CursorNativeStreamConfig,
): ChatCompletionRequest {
  const messages: OpenAIMessage[] = [];
  if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });

  for (const message of context.messages) {
    if (message.role === "user") {
      messages.push({ role: "user", content: piContentToOpenAIContent(message.content) });
      continue;
    }

    if (message.role === "assistant") {
      const tool_calls = assistantToolCallsFromPiContent(message.content);
      messages.push({
        role: "assistant",
        content: cursorAssistantTextFromPiMessage(message),
        ...(tool_calls.length > 0 ? { tool_calls } : {}),
      });
      continue;
    }

    if (message.role === "toolResult") {
      messages.push({
        role: "tool",
        tool_call_id: message.toolCallId,
        content: piContentToOpenAIContent(message.content),
        ...(message.isError ? { isError: true } : {}),
      });
    }
  }

  const body: ChatCompletionRequest = {
    model: model.id,
    messages,
    stream: true,
    tools: (context.tools ?? []).map(piToolToOpenAI),
    tool_choice: options?.toolChoice,
    reasoning_effort: resolveNativeReasoningEffort(
      model,
      options,
      config.getNoReasoningEffortByModelId?.(),
    ),
    pi_session_id: options?.sessionId,
    user: options?.sessionId,
    temperature: options?.temperature,
    max_tokens: options?.maxTokens,
  };

  applyNativeCursorRouting(body, config.getRawModelRoutingByModelId?.());
  return body;
}

function nativeRequestParameterError(body: ChatCompletionRequest): string | undefined {
  if (body.temperature !== undefined)
    return "Unsupported Cursor provider parameter(s): temperature";
  return undefined;
}

interface PreparedCursorRequestContext {
  systemPromptFingerprint: string;
  modelId: string;
  maxMode: boolean;
  sessionId: string | undefined;
  bridgeKey: string;
  convKey: string;
  workspaceContext: CursorWorkspaceContext;
  activeBridge: ActiveBridge | undefined;
}

function prepareCursorRequestContext(
  body: ChatCompletionRequest,
  parsedMessages: Pick<ParsedMessages, "systemPrompt">,
): PreparedCursorRequestContext {
  const sessionId = derivePiSessionId(body);
  const bridgeKey = deriveBridgeKey(body.messages, sessionId);
  return {
    systemPromptFingerprint: fingerprintSystemPrompt(parsedMessages.systemPrompt),
    modelId: resolveRequestedModelId(body.model, body.reasoning_effort, body.cursor_model_id),
    maxMode:
      typeof body.cursor_model_max_mode === "boolean"
        ? body.cursor_model_max_mode
        : body.cursor_requires_max_mode === true,
    sessionId,
    bridgeKey,
    convKey: deriveConversationKey(body.messages, sessionId),
    workspaceContext: createWorkspaceContext(body.cursor_workspace_path),
    activeBridge: activeBridges.get(bridgeKey),
  };
}

function lostToolContinuationMessage(): string {
  return "Cursor tool continuation was lost because the live upstream bridge is no longer available. Retry from before the tool call or start a new turn.";
}

function estimateInlineHistoryPromptTokens(body: ChatCompletionRequest): number {
  try {
    const parsed = parseMessages(body.messages, body.cursor_tool_result_images, false);
    const sessionId = derivePiSessionId(body);
    const convKey = deriveConversationKey(body.messages, sessionId);
    const stored = conversationStates.get(convKey);
    if (
      stored?.checkpoint &&
      stored.checkpointTurnCount === parsed.turns.length &&
      stored.checkpointHistoryFingerprint === fingerprintCompletedTurns(parsed.turns) &&
      stored.checkpointSystemPromptFingerprint === fingerprintSystemPrompt(parsed.systemPrompt)
    ) {
      return 0;
    }
    const historyPrompt = inlineHistoryPrompt(parsed.turns, parsed.userText);
    return historyPrompt ? estimateTextTokens(historyPrompt) : 0;
  } catch {
    return 0;
  }
}

export function createCursorNativeStream(
  config: CursorNativeStreamConfig,
): (
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
) => AssistantMessageEventStream {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    let body = contextToCursorChatCompletionRequest(
      model,
      context,
      options as CursorNativeStreamOptions | undefined,
      config,
    );
    const initialContextTokens =
      estimateInFlightContextTokens(context) + estimateInlineHistoryPromptTokens(body);
    const writer = createNativeStreamWriter(stream, model, initialContextTokens);
    writer.start();

    (async () => {
      if (options?.onPayload) {
        const replacement = await options.onPayload(body, model);
        if (replacement && typeof replacement === "object")
          body = replacement as ChatCompletionRequest;
      }

      await withSessionLock(deriveRequestLockKey(body), async () => {
        if (writer.closed) return;
        const accessToken = await config.getAccessToken();
        await handleCursorNativeRequest(
          body,
          accessToken,
          model,
          options as CursorNativeStreamOptions | undefined,
          writer,
          nextDebugRequestId(),
        );
      });
    })().catch((error) => {
      writer.error(error instanceof Error ? error.message : String(error), "error");
    });

    return stream;
  };
}

async function handleCursorNativeRequest(
  body: ChatCompletionRequest,
  accessToken: string,
  model: Model<Api>,
  options: CursorNativeStreamOptions | undefined,
  writer: NativeStreamWriter,
  requestId: string,
): Promise<void> {
  let parsedMessages: ParsedMessages;
  try {
    parsedMessages = parseMessages(body.messages, body.cursor_tool_result_images);
  } catch (error) {
    writer.error(error instanceof Error ? error.message : String(error), "error");
    return;
  }

  const parameterError = nativeRequestParameterError(body);
  if (parameterError) {
    debugLog("native.unsupported_parameters", { requestId, message: parameterError });
    writer.error(parameterError, "error");
    return;
  }

  const toolResolution = resolveToolsForToolChoice(body.tools ?? [], body.tool_choice);
  if ("error" in toolResolution) {
    debugLog("native.unsupported_tool_choice", { requestId, tool_choice: body.tool_choice });
    writer.error(toolResolution.error, "error");
    return;
  }

  const { systemPrompt, userText, userImages, turns, toolResults } = parsedMessages;
  const {
    systemPromptFingerprint,
    modelId,
    maxMode,
    sessionId,
    bridgeKey,
    convKey,
    workspaceContext,
    activeBridge,
  } = prepareCursorRequestContext(body, parsedMessages);

  if (options?.signal?.aborted) {
    debugLog("native.request.aborted_before_dispatch", {
      requestId,
      sessionId,
      bridgeKey,
      convKey,
      hasActiveBridge: !!activeBridge,
    });
    if (activeBridge) cleanupBridge(activeBridge.bridge, activeBridge.heartbeatTimer, bridgeKey);
    writer.error("Aborted", "aborted");
    return;
  }

  debugLog("native.request", {
    requestId,
    sessionId,
    bridgeKey,
    convKey,
    workspaceContext,
    model: body.model,
    resolvedModelId: modelId,
    cursorModelId: body.cursor_model_id,
    cursorModelParameters: body.cursor_model_parameters,
    cursorRequiresMaxMode: body.cursor_requires_max_mode,
    cursorModelMaxMode: body.cursor_model_max_mode,
    maxMode,
    messageCount: body.messages.length,
    turnCount: turns.length,
    userText,
    toolResults,
    hasActiveBridge: !!activeBridge,
  });

  if (!userText && userImages.length === 0 && toolResults.length === 0) {
    writer.error("No user message found", "error");
    return;
  }

  if (toolResults.length > 0) {
    if (activeBridge) {
      removeActiveBridge(bridgeKey);
      if (activeBridge.bridge.alive) {
        handleNativeToolResultResume(
          activeBridge,
          toolResults,
          model,
          modelId,
          bridgeKey,
          convKey,
          turns,
          workspaceContext,
          systemPromptFingerprint,
          writer,
          options,
          requestId,
        );
        return;
      }
      clearInterval(activeBridge.heartbeatTimer);
      activeBridge.bridge.end();
    }
    const message = lostToolContinuationMessage();
    debugLog("native.lost_tool_continuation", {
      requestId,
      bridgeKey,
      convKey,
      toolResults,
      message,
    });
    writer.error(message, "error");
    return;
  }

  if (activeBridge && activeBridges.has(bridgeKey)) {
    clearInterval(activeBridge.heartbeatTimer);
    activeBridge.bridge.end();
    removeActiveBridge(bridgeKey);
  }

  let stored = conversationStates.get(convKey);
  if (!stored) {
    stored = {
      conversationId: deterministicConversationId(convKey),
      checkpoint: null,
      sessionScoped: !!sessionId,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    };
    conversationStates.set(convKey, stored);
  }
  stored.lastAccessMs = Date.now();
  evictStaleConversations();
  const requestCheckpoint = checkpointForRequest(
    stored,
    turns,
    systemPromptFingerprint,
    requestId,
    convKey,
  );

  const mcpTools = buildMcpToolDefinitions(toolResolution.tools);
  const effectiveUserText = userText;
  const effectiveUserImages = userText || userImages.length > 0 ? userImages : [];
  const requestConversationId = conversationIdForRequest(
    stored,
    convKey,
    turns,
    systemPromptFingerprint,
    effectiveUserText,
    effectiveUserImages,
    requestCheckpoint,
  );
  stored.conversationId = requestConversationId;
  const payload = buildCursorRequest(
    modelId,
    systemPrompt,
    effectiveUserText,
    turns,
    requestConversationId,
    requestCheckpoint,
    requestCheckpoint ? stored.blobStore : undefined,
    maxMode,
    body.cursor_model_parameters,
    mcpTools,
    effectiveUserImages,
    workspaceContext,
  );
  payload.mcpTools = mcpTools;

  const currentTurn: ParsedTurn = {
    userText: effectiveUserText,
    steps: [],
    ...(effectiveUserImages.length > 0 ? { userImages: effectiveUserImages } : {}),
  };

  debugLog("native.dispatch_stream", {
    requestId,
    bridgeKey,
    convKey,
    conversationId: requestConversationId,
    hasStoredCheckpoint: !!stored.checkpoint,
    hasRequestCheckpoint: !!requestCheckpoint,
    payload,
  });
  const { bridge, heartbeatTimer } = startBridge(accessToken, payload.requestBytes);
  writeNativeStream(
    bridge,
    heartbeatTimer,
    payload.blobStore,
    payload.mcpTools,
    model,
    modelId,
    bridgeKey,
    convKey,
    turns,
    currentTurn,
    workspaceContext,
    systemPromptFingerprint,
    writer,
    options,
    requestId,
  );
}

function createStreamState(): StreamState {
  return {
    toolCallIndex: 0,
    pendingExecs: [],
    outputTokens: 0,
    totalTokens: 0,
  };
}

function emitFilteredStreamText(
  tagFilter: ReturnType<typeof createThinkingTagFilter>,
  currentTurn: ParsedTurn,
  text: string,
  isThinking: boolean | undefined,
  emitThinking: (text: string) => void,
  emitText: (text: string) => void,
): void {
  if (isThinking) {
    emitThinking(text);
    return;
  }
  const { content, reasoning } = tagFilter.process(text);
  if (reasoning) emitThinking(reasoning);
  if (content) {
    appendAssistantTextToTurn(currentTurn, content);
    emitText(content);
  }
}

function flushFilteredStreamText(
  tagFilter: ReturnType<typeof createThinkingTagFilter>,
  currentTurn: ParsedTurn,
  emitThinking: (text: string) => void,
  emitText: (text: string) => void,
): void {
  const flushed = tagFilter.flush();
  if (flushed.reasoning) emitThinking(flushed.reasoning);
  if (flushed.content) {
    appendAssistantTextToTurn(currentTurn, flushed.content);
    emitText(flushed.content);
  }
}

function rememberToolCallPause(
  bridgeKey: string,
  bridge: BridgeHandle,
  heartbeatTimer: ReturnType<typeof setInterval>,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
  state: StreamState,
  currentTurn: ParsedTurn,
  convKey: string,
  workspaceContext: CursorWorkspaceContext,
  exec: PendingExec,
): void {
  state.pendingExecs.push(exec);
  currentTurn.steps.push({
    kind: "toolCall",
    toolCallId: exec.toolCallId,
    toolName: exec.toolName,
    arguments: parseToolCallArguments(exec.decodedArgs),
  });
  setActiveBridge(bridgeKey, {
    bridge,
    heartbeatTimer,
    blobStore,
    mcpTools,
    pendingExecs: state.pendingExecs,
    currentTurn,
    convKey,
    workspaceContext,
  });
}

function commitOrMergeStoredCheckpoint(
  stored: StoredConversation | undefined,
  latestCheckpoint: Uint8Array | null,
  blobStore: Map<string, Uint8Array>,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  systemPromptFingerprint: string,
): void {
  if (!stored) return;
  if (latestCheckpoint) {
    commitStoredCheckpoint(
      stored,
      latestCheckpoint,
      blobStore,
      completedTurns,
      currentTurn,
      systemPromptFingerprint,
    );
  } else {
    mergeBlobStore(stored, blobStore);
  }
}

function writeNativeStream(
  bridge: BridgeHandle,
  heartbeatTimer: ReturnType<typeof setInterval>,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
  model: Model<Api>,
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  writer: NativeStreamWriter,
  options?: CursorNativeStreamOptions,
  requestId?: string,
): void {
  debugLog("native.stream.start", { requestId, bridgeKey, convKey, modelId, workspaceContext });
  const state = createStreamState();
  const tagFilter = createThinkingTagFilter();
  let mcpExecReceived = false;
  let cancelled = false;
  let streamError: Error | null = null;
  let latestCheckpoint: Uint8Array | null = null;
  const watchdog = createCursorStreamWatchdog({
    eventPrefix: "native.stream",
    requestId,
    bridgeKey,
    convKey,
    bridge,
    heartbeatTimer,
    onTimeout: (_kind, message) => {
      streamError = new Error(message);
      if (!writer.closed) writer.error(message, "error", state);
    },
  });

  const abort = () => {
    if (cancelled) return;
    cancelled = true;
    watchdog.stop();
    debugLog("native.stream.abort", { requestId, bridgeKey, convKey, writerClosed: writer.closed });
    cleanupBridge(bridge, heartbeatTimer, bridgeKey);
    if (!writer.closed) writer.error("Aborted", "aborted", state);
  };
  if (options?.signal?.aborted) {
    abort();
    return;
  }
  options?.signal?.addEventListener("abort", abort, { once: true });

  const emitText = (text: string, isThinking?: boolean) => {
    if (writer.closed) return;
    emitFilteredStreamText(
      tagFilter,
      currentTurn,
      text,
      isThinking,
      (reasoning) => {
        watchdog.touchVisible("thinking_delta");
        writer.thinking(reasoning);
      },
      (content) => {
        watchdog.touchVisible("text_delta");
        writer.text(content);
      },
    );
  };

  const emitFlushed = () => {
    flushFilteredStreamText(
      tagFilter,
      currentTurn,
      (reasoning) => {
        watchdog.touchVisible("thinking_delta");
        writer.thinking(reasoning);
      },
      (content) => {
        watchdog.touchVisible("text_delta");
        writer.text(content);
      },
    );
  };

  const processChunk = createConnectFrameParser(
    (messageBytes) => {
      try {
        const serverMessage = fromBinary(AgentServerMessageSchema, messageBytes);
        if (!isCursorServerHeartbeat(serverMessage))
          watchdog.touchUpstream(cursorServerMessageLabel(serverMessage));
        processServerMessage(
          serverMessage,
          blobStore,
          mcpTools,
          workspaceContext,
          (data) => bridge.write(data),
          state,
          emitText,
          (exec) => {
            mcpExecReceived = true;
            watchdog.touchVisible("tool_call");
            watchdog.stop();
            emitFlushed();
            rememberToolCallPause(
              bridgeKey,
              bridge,
              heartbeatTimer,
              blobStore,
              mcpTools,
              state,
              currentTurn,
              convKey,
              workspaceContext,
              exec,
            );
            debugLog("native.stream.tool_call_pause", {
              requestId,
              bridgeKey,
              exec,
              pendingExecs: state.pendingExecs,
              currentTurn,
            });

            if (!writer.closed) {
              writer.toolCall(exec);
              writer.done("toolUse", state);
            }
          },
          (checkpointBytes) => {
            latestCheckpoint = checkpointBytes;
            debugLog("native.stream.checkpoint_buffered", { requestId, convKey, checkpointBytes });
          },
        );
      } catch (err) {
        streamError = err instanceof Error ? err : new Error(String(err));
        const message = streamError.message;
        debugLog("native.stream.process_error", { requestId, bridgeKey, convKey, message });
        watchdog.stop();
        if (!writer.closed) writer.error(message, "error", state);
        cleanupBridge(bridge, heartbeatTimer, bridgeKey);
      }
    },
    (endStreamBytes) => {
      const endError = parseConnectEndStream(endStreamBytes);
      if (endError) {
        streamError = endError;
        watchdog.stop();
        debugLog("native.stream.cursor_error", { requestId, modelId, message: endError.message });
        writer.error(endError.message, "error", state);
      }
    },
  );

  bridge.onData(processChunk);

  bridge.onClose((code) => {
    debugLog("native.stream.bridge_close", {
      requestId,
      bridgeKey,
      convKey,
      code,
      cancelled,
      mcpExecReceived,
      currentTurn,
      latestCheckpoint,
    });
    watchdog.stop();
    clearInterval(heartbeatTimer);
    options?.signal?.removeEventListener("abort", abort);

    if (cancelled) return;
    if (streamError) {
      removeActiveBridge(bridgeKey);
      return;
    }

    const stored = conversationStates.get(convKey);
    if (code !== 0) {
      writer.error("Bridge connection lost", "error", state);
      removeActiveBridge(bridgeKey);
      return;
    }

    if (!mcpExecReceived) {
      emitFlushed();
      commitOrMergeStoredCheckpoint(
        stored,
        latestCheckpoint,
        blobStore,
        completedTurns,
        currentTurn,
        systemPromptFingerprint,
      );
      if (stored && latestCheckpoint)
        debugLog("native.stream.checkpoint_committed", { requestId, convKey, stored });
      writer.done("stop", state);
    } else {
      removeActiveBridge(bridgeKey);
    }
  });
}

function applyToolResultsToTurn(currentTurn: ParsedTurn, toolResults: ToolResultInfo[]): void {
  for (const result of toolResults) {
    const turnToolStep = currentTurn.steps.find(
      (step): step is ParsedToolCallStep =>
        step.kind === "toolCall" && step.toolCallId === result.toolCallId,
    );
    if (turnToolStep) {
      turnToolStep.result = {
        content: result.content,
        images: result.images,
        isError: result.isError === true,
      };
    }
  }
}

function unresolvedPendingExecs(
  pendingExecs: PendingExec[],
  turnResults: Map<string, ParsedToolResult>,
): PendingExec[] {
  return pendingExecs.filter((exec) => !turnResults.has(exec.toolCallId));
}

function sendMcpResultsForPendingExecs(
  bridge: BridgeHandle,
  pendingExecs: PendingExec[],
  turnResults: Map<string, ParsedToolResult>,
  debugEvent: string,
  requestId?: string,
): void {
  for (const exec of pendingExecs) {
    const result = turnResults.get(exec.toolCallId);
    if (!result) continue;
    const mcpResult = mcpResultFromParsedToolResult(result);

    const execClientMessage = create(ExecClientMessageSchema, {
      id: exec.execMsgId,
      execId: exec.execId,
      message: { case: "mcpResult" as any, value: mcpResult as any },
    });
    const clientMessage = create(AgentClientMessageSchema, {
      message: { case: "execClientMessage", value: execClientMessage },
    });
    bridge.write(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
    debugLog(debugEvent, { requestId, exec, result });
  }
}

function handleNativeToolResultResume(
  active: ActiveBridge,
  toolResults: ToolResultInfo[],
  model: Model<Api>,
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  writer: NativeStreamWriter,
  options?: CursorNativeStreamOptions,
  requestId?: string,
): void {
  const { bridge, heartbeatTimer, blobStore, mcpTools, pendingExecs, currentTurn } = active;
  if (options?.signal?.aborted) {
    debugLog("native.tool_resume.aborted_before_result", { requestId, bridgeKey, convKey });
    cleanupBridge(bridge, heartbeatTimer, bridgeKey);
    writer.error("Aborted", "aborted");
    return;
  }
  debugLog("native.tool_resume.start", {
    requestId,
    bridgeKey,
    convKey,
    toolResults,
    pendingExecs,
    currentTurn,
  });

  applyToolResultsToTurn(currentTurn, toolResults);

  const turnResults = getTurnToolCallResults(currentTurn);
  const unresolvedExecs = unresolvedPendingExecs(pendingExecs, turnResults);
  if (unresolvedExecs.length > 0) {
    setActiveBridge(bridgeKey, {
      bridge,
      heartbeatTimer,
      blobStore,
      mcpTools,
      pendingExecs,
      currentTurn,
      convKey,
      workspaceContext,
    });
    debugLog("native.tool_resume.partial_wait", {
      requestId,
      bridgeKey,
      unresolvedExecs,
      currentTurn,
    });
    for (const exec of unresolvedExecs) writer.toolCall(exec);
    writer.done("toolUse");
    return;
  }

  sendMcpResultsForPendingExecs(
    bridge,
    pendingExecs,
    turnResults,
    "native.tool_resume.sent_result",
    requestId,
  );

  writeNativeStream(
    bridge,
    heartbeatTimer,
    blobStore,
    mcpTools,
    model,
    modelId,
    bridgeKey,
    convKey,
    completedTurns,
    currentTurn,
    workspaceContext,
    systemPromptFingerprint,
    writer,
    options,
    requestId,
  );
}

// ── Request handling ──

export function evictStaleConversations(now = Date.now()): void {
  for (const [key, stored] of conversationStates) {
    if (!stored.sessionScoped && now - stored.lastAccessMs > CONVERSATION_TTL_MS) {
      debugLog("conversation.evict", { key, stored, now });
      conversationStates.delete(key);
    }
  }
}

function normalizeWorkspacePath(raw: unknown): string {
  const text = typeof raw === "string" ? raw.trim() : "";
  if (!text) return process.cwd();
  return isAbsolute(text) ? text : pathResolve(text);
}

function cursorProjectSlug(workspacePath: string): string {
  const slug = workspacePath
    .replace(/\\/g, "/")
    .replace(/^\/+/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || "workspace";
}

function createWorkspaceContext(rawWorkspacePath?: unknown): CursorWorkspaceContext {
  const workspacePath = normalizeWorkspacePath(rawWorkspacePath);
  const projectFolder = pathJoin(
    homedir(),
    ".cursor",
    "projects",
    cursorProjectSlug(workspacePath),
  );
  return {
    workspacePath,
    workspaceUri: pathToFileURL(workspacePath).href,
    projectFolder,
    terminalsFolder: pathJoin(projectFolder, "terminals"),
    agentSharedNotesFolder: pathJoin(homedir(), ".cursor", "agent-shared-notes"),
    agentConversationNotesFolder: pathJoin(projectFolder, "conversation-notes"),
    agentTranscriptsFolder: pathJoin(projectFolder, "transcripts"),
  };
}

function stableNormalizeForHash(value: unknown): unknown {
  if (
    value == null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return value;
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
    const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
    return { __bytes: summarizeByteData(bytes) };
  }
  if (Array.isArray(value)) return value.map((item) => stableNormalizeForHash(item));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, inner]) => inner !== undefined)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, inner]) => [key, stableNormalizeForHash(inner)]),
    );
  }
  return String(value);
}

function fingerprintImage(image: ParsedImageContent): Record<string, unknown> {
  return {
    mimeType: image.mimeType,
    ...summarizeByteData(image.data),
  };
}

export function fingerprintSystemPrompt(systemPrompt: string): string {
  return createHash("sha256").update(systemPrompt).digest("hex");
}

export function fingerprintCompletedTurns(turns: ParsedTurn[]): string {
  const normalized = turns.map((turn) => ({
    userText: turn.userText,
    userImages: (turn.userImages ?? []).map(fingerprintImage),
    steps: turn.steps.map((step) => {
      if (step.kind === "assistantText") return { kind: step.kind, text: step.text };
      return {
        kind: step.kind,
        toolCallId: step.toolCallId,
        toolName: step.toolName,
        arguments: stableNormalizeForHash(step.arguments),
        result: step.result
          ? {
              content: step.result.content,
              isError: step.result.isError,
              images: (step.result.images ?? []).map(fingerprintImage),
            }
          : undefined,
      };
    }),
  }));
  return createHash("sha256").update(JSON.stringify(normalized)).digest("hex");
}

function clearStoredCheckpoint(stored: StoredConversation, clearBlobStore = false): void {
  stored.checkpoint = null;
  delete stored.checkpointTurnCount;
  delete stored.checkpointHistoryFingerprint;
  delete stored.checkpointSystemPromptFingerprint;
  if (clearBlobStore) stored.blobStore.clear();
}

function discardStaleCheckpointIfNeeded(
  stored: StoredConversation,
  turns: ParsedTurn[],
  systemPromptFingerprint: string,
  requestId: string,
  convKey: string,
): boolean {
  if (!stored.checkpoint) {
    debugLog("checkpoint.decision", {
      requestId,
      convKey,
      checkpointDecision: "none",
    });
    return false;
  }

  const currentTurnCount = turns.length;
  const currentHistoryFingerprint = fingerprintCompletedTurns(turns);
  const storedCheckpointTurnCount = stored.checkpointTurnCount;
  const storedCheckpointHistoryFingerprint = stored.checkpointHistoryFingerprint;
  const storedCheckpointSystemPromptFingerprint = stored.checkpointSystemPromptFingerprint;
  const reason =
    storedCheckpointTurnCount === undefined || !storedCheckpointHistoryFingerprint
      ? "missing_checkpoint_metadata"
      : !storedCheckpointSystemPromptFingerprint
        ? "missing_checkpoint_system_prompt_fingerprint"
        : storedCheckpointTurnCount !== currentTurnCount
          ? "completed_turn_count_mismatch"
          : storedCheckpointHistoryFingerprint !== currentHistoryFingerprint
            ? "completed_history_fingerprint_mismatch"
            : storedCheckpointSystemPromptFingerprint !== systemPromptFingerprint
              ? "system_prompt_fingerprint_mismatch"
              : undefined;

  if (!reason) {
    debugLog("checkpoint.decision", {
      requestId,
      convKey,
      checkpointDecision: "reuse",
      currentTurnCount,
      currentHistoryFingerprint,
      systemPromptFingerprint,
    });
    return true;
  }

  debugLog("chat.discard_checkpoint", {
    requestId,
    convKey,
    checkpointDecision: "discard",
    reason,
    storedCheckpointTurnCount,
    currentTurnCount,
    storedCheckpointHistoryFingerprint,
    currentHistoryFingerprint,
    storedCheckpointSystemPromptFingerprint,
    systemPromptFingerprint,
  });
  clearStoredCheckpoint(stored, true);
  return false;
}

function checkpointForRequest(
  stored: StoredConversation,
  turns: ParsedTurn[],
  systemPromptFingerprint: string,
  requestId: string,
  convKey: string,
): Uint8Array | null {
  const validCheckpoint = discardStaleCheckpointIfNeeded(
    stored,
    turns,
    systemPromptFingerprint,
    requestId,
    convKey,
  );
  if (!validCheckpoint || !stored.checkpoint) return null;

  debugLog("checkpoint.decision", {
    requestId,
    convKey,
    checkpointDecision: "reuse_valid_checkpoint",
    reason: "valid_checkpoint_preserves_cursor_context",
    currentTurnCount: turns.length,
  });
  return stored.checkpoint;
}

function conversationIdForRequest(
  stored: StoredConversation,
  convKey: string,
  turns: ParsedTurn[],
  systemPromptFingerprint: string,
  userText: string,
  userImages: ParsedImageContent[],
  requestCheckpoint: Uint8Array | null,
): string {
  if (requestCheckpoint) return stored.conversationId;
  const fingerprint = createHash("sha256")
    .update(
      JSON.stringify({
        convKey,
        systemPromptFingerprint,
        completedHistoryFingerprint: fingerprintCompletedTurns(turns),
        currentAction: {
          userText,
          userImages: userImages.map(fingerprintImage),
        },
      }),
    )
    .digest("hex");
  return deterministicConversationId(`pi-context:${fingerprint}`);
}

function mergeBlobStore(stored: StoredConversation, blobStore: Map<string, Uint8Array>): void {
  for (const [k, v] of blobStore) stored.blobStore.set(k, v);
  stored.lastAccessMs = Date.now();
}

function replaceBlobStore(stored: StoredConversation, blobStore: Map<string, Uint8Array>): void {
  stored.blobStore = new Map(blobStore);
  stored.lastAccessMs = Date.now();
}

function commitStoredCheckpoint(
  stored: StoredConversation,
  checkpointBytes: Uint8Array,
  blobStore: Map<string, Uint8Array>,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  systemPromptFingerprint: string,
): void {
  const completedHistory = [...completedTurns, currentTurn];
  replaceBlobStore(stored, blobStore);
  stored.checkpoint = checkpointBytes;
  stored.checkpointTurnCount = completedHistory.length;
  stored.checkpointHistoryFingerprint = fingerprintCompletedTurns(completedHistory);
  stored.checkpointSystemPromptFingerprint = systemPromptFingerprint;
}

/**
 * Insert reasoning effort into model ID, before -fast/-thinking suffix.
 * e.g. model="gpt-5.4" + effort="medium" → "gpt-5.4-medium"
 *      model="gpt-5.4-fast" + effort="high" → "gpt-5.4-high-fast"
 * If no effort provided, returns model as-is.
 */
export function resolveModelId(model: string, reasoningEffort?: string): string {
  if (!reasoningEffort) return model;

  let suffix = "";
  let base = model;
  if (base.endsWith("-fast")) {
    suffix = "-fast";
    base = base.slice(0, -5);
  } else if (base.endsWith("-thinking")) {
    suffix = "-thinking";
    base = base.slice(0, -9);
  }

  return `${base}-${reasoningEffort}${suffix}`;
}

export function resolveRequestedModelId(
  model: string,
  reasoningEffort?: string,
  cursorModelId?: string,
): string {
  const trimmedCursorModelId = cursorModelId?.trim();
  if (trimmedCursorModelId) return trimmedCursorModelId;
  return resolveModelId(model, reasoningEffort);
}

function deriveRequestLockKey(body: ChatCompletionRequest): string {
  const sessionId = derivePiSessionId(body);
  if (sessionId) return `session:${sessionId}`;
  return `anonymous:${deriveConversationKey(body.messages)}`;
}

async function withSessionLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const previous = sessionLocks.get(key) ?? Promise.resolve();
  let release!: () => void;
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  const chained = previous.catch(() => {}).then(() => current);
  sessionLocks.set(key, chained);
  await previous.catch(() => {});
  try {
    return await fn();
  } finally {
    release();
    if (sessionLocks.get(key) === chained) sessionLocks.delete(key);
  }
}

function writeJsonError(
  res: ServerResponse,
  statusCode: number,
  message: string,
  type: string,
  code?: string,
): void {
  res.writeHead(statusCode, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { message, type, ...(code ? { code } : {}) } }));
}

function rejectUnsupportedRequestParameters(
  body: ChatCompletionRequest,
  res: ServerResponse,
  requestId: string,
): boolean {
  const raw = body as unknown as Record<string, unknown>;
  // Pi's OpenAI-compatible provider sends max_tokens for normal Cursor requests.
  // Cursor's agent protocol controls output budgeting server-side here, so accept
  // max_tokens/max_completion_tokens as no-op compatibility fields rather than
  // breaking every request. Sampling controls remain rejected so users are not
  // misled into thinking they are honored.
  const unsupported = [["temperature", raw.temperature]].filter(([, value]) => value !== undefined);
  if (unsupported.length === 0) return false;

  debugLog("chat.unsupported_parameters", {
    requestId,
    parameters: unsupported.map(([name]) => name),
  });
  writeJsonError(
    res,
    400,
    `Unsupported Cursor proxy parameter(s): ${unsupported.map(([name]) => name).join(", ")}`,
    "invalid_request_error",
    "unsupported_parameter",
  );
  return true;
}

function resolveToolsForToolChoice(
  tools: OpenAIToolDef[],
  toolChoice: unknown,
): { tools: OpenAIToolDef[] } | { error: string } {
  if (toolChoice == null || toolChoice === "auto") return { tools };
  if (toolChoice === "none") return { tools: [] };
  if (
    typeof toolChoice === "object" &&
    toolChoice &&
    (toolChoice as Record<string, unknown>).type === "none"
  )
    return { tools: [] };
  return { error: "Only tool_choice 'auto' and 'none' are supported by pi-cursor-provider." };
}

async function handleChatCompletion(
  body: ChatCompletionRequest,
  accessToken: string,
  req: IncomingMessage,
  res: ServerResponse,
  requestId: string,
): Promise<void> {
  let parsedMessages: ParsedMessages;
  try {
    parsedMessages = parseMessages(body.messages, body.cursor_tool_result_images);
  } catch (error) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        error: {
          message: error instanceof Error ? error.message : String(error),
          type: "invalid_request_error",
        },
      }),
    );
    return;
  }
  const { systemPrompt, userText, userImages, turns, toolResults } = parsedMessages;
  const {
    systemPromptFingerprint,
    modelId,
    maxMode,
    sessionId,
    bridgeKey,
    convKey,
    workspaceContext,
    activeBridge,
  } = prepareCursorRequestContext(body, parsedMessages);
  if (body.reasoning_effort && !body.cursor_model_id && !body.cursor_model_parameters) {
    debugLog("model_routing.fallback_suffix_generation", {
      requestId,
      model: body.model,
      reasoning_effort: body.reasoning_effort,
      resolvedModelId: modelId,
    });
  }
  if (rejectUnsupportedRequestParameters(body, res, requestId)) return;
  const toolResolution = resolveToolsForToolChoice(body.tools ?? [], body.tool_choice);
  if ("error" in toolResolution) {
    debugLog("chat.unsupported_tool_choice", { requestId, tool_choice: body.tool_choice });
    writeJsonError(
      res,
      400,
      toolResolution.error,
      "invalid_request_error",
      "unsupported_tool_choice",
    );
    return;
  }
  const tools = toolResolution.tools;

  debugLog("chat.parsed_messages", {
    requestId,
    systemPrompt,
    userText,
    turns,
    toolResults,
    messageCount: body.messages.length,
    model: body.model,
    cursorModelId: body.cursor_model_id,
    cursorModelParameters: body.cursor_model_parameters,
    cursorRequiresMaxMode: body.cursor_requires_max_mode,
    cursorModelMaxMode: body.cursor_model_max_mode,
    resolvedModelId: modelId,
    stream: body.stream !== false,
    maxMode,
  });

  if (!userText && userImages.length === 0 && toolResults.length === 0) {
    debugLog("chat.no_user_message", { requestId, messages: body.messages });
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error: { message: "No user message found", type: "invalid_request_error" },
      }),
    );
    return;
  }

  if (body.stream === false && tools.length > 0) {
    debugLog("chat.nonstream_tools_unsupported", { requestId, toolCount: tools.length });
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error: {
          message:
            "stream:false with tools is not supported by pi-cursor-provider; use streaming tool calls instead.",
          type: "invalid_request_error",
          code: "nonstream_tools_unsupported",
        },
      }),
    );
    return;
  }

  debugLog("chat.session_keys", {
    requestId,
    sessionId,
    bridgeKey,
    convKey,
    workspaceContext,
    hasActiveBridge: !!activeBridge,
  });

  if (toolResults.length > 0) {
    if (activeBridge) {
      debugLog("chat.resume_tool_results", {
        requestId,
        bridgeKey,
        toolResults,
        pendingExecs: activeBridge.pendingExecs,
      });
      removeActiveBridge(bridgeKey);
      if (activeBridge.bridge.alive) {
        handleToolResultResume(
          activeBridge,
          toolResults,
          modelId,
          bridgeKey,
          convKey,
          turns,
          workspaceContext,
          systemPromptFingerprint,
          req,
          res,
          body.stream !== false,
          requestId,
        );
        return;
      }
      clearInterval(activeBridge.heartbeatTimer);
      activeBridge.bridge.end();
    }

    const message = lostToolContinuationMessage();
    debugLog("chat.lost_tool_continuation", {
      requestId,
      bridgeKey,
      convKey,
      toolResults,
      message,
    });
    res.writeHead(409, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error: {
          message,
          type: "invalid_state_error",
          code: "tool_continuation_lost",
        },
      }),
    );
    return;
  }

  if (activeBridge && activeBridges.has(bridgeKey)) {
    clearInterval(activeBridge.heartbeatTimer);
    activeBridge.bridge.end();
    removeActiveBridge(bridgeKey);
  }

  let stored = conversationStates.get(convKey);
  debugLog("chat.stored_state.before", { requestId, convKey, stored });
  if (!stored) {
    stored = {
      conversationId: deterministicConversationId(convKey),
      checkpoint: null,

      sessionScoped: !!sessionId,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    };
    conversationStates.set(convKey, stored);
  }
  stored.lastAccessMs = Date.now();
  evictStaleConversations();
  const requestCheckpoint = checkpointForRequest(
    stored,
    turns,
    systemPromptFingerprint,
    requestId,
    convKey,
  );

  const mcpTools = buildMcpToolDefinitions(tools);
  const effectiveUserText =
    userText || (toolResults.length > 0 ? toolResults.map((r) => r.content).join("\n") : "");
  const effectiveUserImages = userText || userImages.length > 0 ? userImages : [];
  const requestConversationId = conversationIdForRequest(
    stored,
    convKey,
    turns,
    systemPromptFingerprint,
    effectiveUserText,
    effectiveUserImages,
    requestCheckpoint,
  );
  stored.conversationId = requestConversationId;
  if (!requestCheckpoint) {
    debugLog("chat.no_request_checkpoint", {
      requestId,
      convKey,
      conversationId: requestConversationId,
      hasStoredCheckpoint: !!stored.checkpoint,
    });
  }
  const payload = buildCursorRequest(
    modelId,
    systemPrompt,
    effectiveUserText,
    turns,
    requestConversationId,
    requestCheckpoint,
    requestCheckpoint ? stored.blobStore : undefined,
    maxMode,
    body.cursor_model_parameters,
    mcpTools,
    effectiveUserImages,
    workspaceContext,
  );
  debugLog("chat.cursor_request", {
    requestId,
    conversationId: requestConversationId,
    effectiveUserText,
    turnCount: turns.length,
    hasStoredCheckpoint: !!stored.checkpoint,
    hasRequestCheckpoint: !!requestCheckpoint,
    payload,
  });
  payload.mcpTools = mcpTools;

  const currentTurn: ParsedTurn = {
    userText: effectiveUserText,
    steps: [],
    ...(effectiveUserImages.length > 0 ? { userImages: effectiveUserImages } : {}),
  };

  if (body.stream === false) {
    debugLog("chat.dispatch_nonstream", { requestId, convKey });
    await handleNonStreamingResponse(
      payload,
      accessToken,
      modelId,
      bridgeKey,
      convKey,
      turns,
      currentTurn,
      workspaceContext,
      systemPromptFingerprint,
      req,
      res,
      requestId,
    );
  } else {
    debugLog("chat.dispatch_stream", { requestId, bridgeKey, convKey });
    handleStreamingResponse(
      payload,
      accessToken,
      modelId,
      bridgeKey,
      convKey,
      turns,
      currentTurn,
      workspaceContext,
      systemPromptFingerprint,
      req,
      res,
      requestId,
    );
  }
}

// ── Message parsing ──

function textContent(content: OpenAIMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text!)
    .join("\n");
}

interface ImageDecodeOptions {
  enforceCursorCliLimits?: boolean;
}

function normalizeImageMimeType(mimeType: string): string {
  const normalized = mimeType.trim().toLowerCase();
  return normalized === "image/jpg" ? "image/jpeg" : normalized;
}

function sniffCursorImageMimeType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff)
    return "image/jpeg";
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  )
    return "image/png";
  if (bytes.length >= 3 && bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46)
    return "image/gif";
  if (
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  )
    return "image/webp";
  return undefined;
}

function validateCursorCliImageLimits(bytes: Uint8Array): string {
  if (bytes.length > CURSOR_CLI_MAX_IMAGE_BYTES) {
    throw new Error(
      `Image exceeds Cursor CLI's ${CURSOR_CLI_MAX_IMAGE_BYTES} byte limit after processing.`,
    );
  }
  const sniffedMimeType = sniffCursorImageMimeType(bytes);
  if (!sniffedMimeType || !CURSOR_SUPPORTED_IMAGE_MIME_TYPES.has(sniffedMimeType)) {
    throw new Error("Unsupported image type: supported formats are jpeg, png, gif, or webp.");
  }
  return sniffedMimeType;
}

function decodeBase64Image(
  data: string,
  mimeType: string,
  options: ImageDecodeOptions = {},
): ParsedImageContent | undefined {
  const normalizedMimeType = normalizeImageMimeType(mimeType);
  if (!normalizedMimeType.startsWith("image/")) return undefined;
  const base64 = data.replace(/\s/g, "");
  if (!base64) return undefined;
  const bytes = new Uint8Array(Buffer.from(base64, "base64"));
  if (bytes.length === 0) return undefined;
  const finalMimeType = options.enforceCursorCliLimits
    ? validateCursorCliImageLimits(bytes)
    : normalizedMimeType;
  return { data: bytes, mimeType: finalMimeType };
}

function parseImageDataUrl(
  url: string,
  options: ImageDecodeOptions = {},
): ParsedImageContent | undefined {
  const trimmed = url.trim();
  if (/^https?:\/\//i.test(trimmed)) {
    throw new Error(
      "Remote image URLs are not supported by pi-cursor-provider. Attach the image or send an inline data:image/...;base64,... URL.",
    );
  }
  if (!trimmed.startsWith("data:")) {
    throw new Error(
      "Only inline data:image/...;base64,... image_url values are supported by pi-cursor-provider.",
    );
  }
  const match = trimmed.match(/^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/is);
  if (!match) {
    throw new Error("Unsupported image_url format. Expected data:image/...;base64,...");
  }
  const image = decodeBase64Image(match[2]!, match[1]!, options);
  if (!image) {
    throw new Error("Unsupported image_url MIME type. Expected data:image/...;base64,...");
  }
  return image;
}

function contentHasImageParts(content: OpenAIMessage["content"]): boolean {
  return (
    Array.isArray(content) &&
    content.some(
      (part) =>
        (part.type === "image_url" && !!part.image_url?.url) ||
        (part.type === "image" && !!part.data && !!part.mimeType),
    )
  );
}

function imageContent(
  content: OpenAIMessage["content"],
  options: ImageDecodeOptions = {},
): ParsedImageContent[] {
  if (content == null || typeof content === "string") return [];
  const images: ParsedImageContent[] = [];
  for (const part of content) {
    if (part.type === "image_url" && part.image_url?.url) {
      const image = parseImageDataUrl(part.image_url.url, options);
      if (image) images.push(image);
    } else if (part.type === "image" && part.data && part.mimeType) {
      const image = decodeBase64Image(part.data, part.mimeType, options);
      if (image) images.push(image);
    }
  }
  return images;
}

function imageKey(image: ParsedImageContent): string {
  return `${image.mimeType}:${createHash("sha256").update(image.data).digest("hex")}`;
}

function mergeImages(
  ...groups: Array<ParsedImageContent[] | undefined>
): ParsedImageContent[] | undefined {
  const merged: ParsedImageContent[] = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const image of group ?? []) {
      const key = imageKey(image);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(image);
    }
  }
  return merged.length > 0 ? merged : undefined;
}

function parseToolResultImagePayloads(
  payloads: CursorToolResultImagePayload[] | undefined,
): Map<string, ParsedImageContent[]> {
  const byToolCallId = new Map<string, ParsedImageContent[]>();
  for (const payload of payloads ?? []) {
    if (!payload?.toolCallId || !Array.isArray(payload.images)) continue;
    const images = payload.images
      .map((image) =>
        decodeBase64Image(image.data, image.mimeType, { enforceCursorCliLimits: true }),
      )
      .filter((image): image is ParsedImageContent => !!image);
    if (images.length === 0) continue;
    byToolCallId.set(
      payload.toolCallId,
      mergeImages(byToolCallId.get(payload.toolCallId), images) ?? [],
    );
  }
  return byToolCallId;
}

function isSyntheticToolResultImageMessage(msg: OpenAIMessage): boolean {
  return (
    msg.role === "user" &&
    textContent(msg.content).trim() === "Attached image(s) from tool result:" &&
    contentHasImageParts(msg.content)
  );
}

function attachSyntheticToolResultImages(turn: ParsedTurn, images: ParsedImageContent[]): void {
  if (images.length === 0) return;
  const resultSteps = turn.steps
    .filter((step): step is ParsedToolCallStep => step.kind === "toolCall" && !!step.result)
    .filter((step) => !step.result!.images?.length);
  if (resultSteps.length === 0) return;

  const imageOnlySteps = resultSteps.filter(
    (step) => step.result!.content.trim() === "(see attached image)",
  );
  if (imageOnlySteps.length === images.length) {
    imageOnlySteps.forEach((step, index) => {
      step.result = { ...step.result!, content: "", images: [images[index]!] };
    });
    return;
  }

  const target = imageOnlySteps.length === 1 ? imageOnlySteps[0]! : resultSteps.at(-1)!;
  target.result = {
    ...target.result!,
    content: target.result!.content.trim() === "(see attached image)" ? "" : target.result!.content,
    images: mergeImages(target.result!.images, images),
  };
}

function normalizeToolResultText(
  content: string,
  images: ParsedImageContent[] | undefined,
): string {
  return images?.length && content.trim() === "(see attached image)" ? "" : content;
}

function parseToolCallArguments(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { value: parsed };
  } catch {
    return raw ? { __raw: raw } : {};
  }
}

function isToolCallStep(step: ParsedTurnStep): step is ParsedToolCallStep {
  return step.kind === "toolCall";
}

function getTurnToolCallResults(turn: ParsedTurn): Map<string, ParsedToolResult> {
  const results = new Map<string, ParsedToolResult>();
  for (const step of turn.steps) {
    if (step.kind === "toolCall" && step.result) results.set(step.toolCallId, step.result);
  }
  return results;
}

function appendAssistantTextToTurn(turn: ParsedTurn, text: string): void {
  if (!text) return;
  const last = turn.steps.at(-1);
  if (last?.kind === "assistantText") {
    last.text += text;
  } else {
    turn.steps.push({ kind: "assistantText", text });
  }
}

function stripTurnRuntimeState(
  turn: ParsedTurn & {
    toolCallById?: Map<string, ParsedToolCallStep>;
    sawToolResult?: boolean;
    sawAssistantAfterToolResult?: boolean;
  },
): ParsedTurn {
  return {
    userText: turn.userText,
    steps: turn.steps,
    ...(turn.userImages?.length ? { userImages: turn.userImages } : {}),
  };
}

export function parseMessages(
  messages: OpenAIMessage[],
  toolResultImagePayloads?: CursorToolResultImagePayload[],
  debug = true,
): ParsedMessages {
  let systemPrompt = "You are a helpful assistant.";
  const turns: ParsedTurn[] = [];
  const toolResultImagesById = parseToolResultImagePayloads(toolResultImagePayloads);

  if (debug) debugLog("parse_messages.start", { messages });

  const systemParts = messages
    .filter((m) => m.role === "system")
    .map((m) => textContent(m.content));
  if (systemParts.length > 0) systemPrompt = systemParts.join("\n");

  const nonSystem = messages.filter((m) => m.role !== "system");
  let currentTurn:
    | (ParsedTurn & {
        toolCallById: Map<string, ParsedToolCallStep>;
        sawToolResult: boolean;
        sawAssistantAfterToolResult: boolean;
      })
    | null = null;

  const finalizeCurrentTurn = () => {
    if (!currentTurn) return;
    turns.push(stripTurnRuntimeState(currentTurn));
    currentTurn = null;
  };

  for (const msg of nonSystem) {
    if (currentTurn && isSyntheticToolResultImageMessage(msg)) {
      const hasMetadataImages = currentTurn.steps.some(
        (step) => step.kind === "toolCall" && step.result?.images?.length,
      );
      if (!hasMetadataImages) {
        attachSyntheticToolResultImages(
          currentTurn,
          imageContent(msg.content, { enforceCursorCliLimits: true }),
        );
      }
      continue;
    }

    if (msg.role === "user") {
      const text = textContent(msg.content);
      const userImages = imageContent(msg.content, { enforceCursorCliLimits: true });
      if (currentTurn && currentTurn.steps.length === 0) {
        currentTurn.userText = [currentTurn.userText, text].filter(Boolean).join("\n\n");
        if (userImages.length > 0) {
          currentTurn.userImages = [...(currentTurn.userImages ?? []), ...userImages];
        }
        continue;
      }
      finalizeCurrentTurn();
      currentTurn = {
        userText: text,
        steps: [],
        ...(userImages.length > 0 ? { userImages } : {}),
        toolCallById: new Map(),
        sawToolResult: false,
        sawAssistantAfterToolResult: false,
      };
      continue;
    }

    if (!currentTurn) continue;

    if (msg.role === "assistant") {
      const text = textContent(msg.content);
      if (text) {
        if (currentTurn.sawToolResult) currentTurn.sawAssistantAfterToolResult = true;
        currentTurn.steps.push({ kind: "assistantText", text });
      }

      for (const toolCall of msg.tool_calls ?? []) {
        const step: ParsedToolCallStep = {
          kind: "toolCall",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          arguments: parseToolCallArguments(toolCall.function.arguments),
        };
        currentTurn.steps.push(step);
        currentTurn.toolCallById.set(step.toolCallId, step);
      }
      continue;
    }

    if (msg.role === "tool") {
      const toolCallId = msg.tool_call_id ?? "";
      const inlineImages = imageContent(msg.content, { enforceCursorCliLimits: true });
      const images = mergeImages(inlineImages, toolResultImagesById.get(toolCallId));
      const content = normalizeToolResultText(textContent(msg.content), images);
      const isError = msg.isError === true || msg.is_error === true;
      const existing = toolCallId ? currentTurn.toolCallById.get(toolCallId) : undefined;
      if (existing) {
        existing.result = { content, images, isError };
      } else {
        const step: ParsedToolCallStep = {
          kind: "toolCall",
          toolCallId,
          toolName: "",
          arguments: {},
          result: { content, images, isError },
        };
        currentTurn.steps.push(step);
        if (toolCallId) currentTurn.toolCallById.set(toolCallId, step);
      }
      currentTurn.sawToolResult = true;
    }
  }

  let userText = "";
  let userImages: ParsedImageContent[] = [];
  let toolResults: ToolResultInfo[] = [];

  if (currentTurn) {
    const toolCallSteps = currentTurn.steps.filter(isToolCallStep);
    const hasAnyToolResults = toolCallSteps.some((step) => step.result);
    const lastStep = currentTurn.steps.at(-1);
    const isToolContinuation = lastStep?.kind === "toolCall";

    if (currentTurn.steps.length === 0 || isToolContinuation) {
      userText = currentTurn.userText;
      userImages = currentTurn.userImages ?? [];
      if (hasAnyToolResults) {
        toolResults = toolCallSteps
          .filter((step) => step.result)
          .map((step) => ({
            toolCallId: step.toolCallId,
            content: step.result!.content,
            ...(step.result!.images?.length ? { images: step.result!.images } : {}),
            ...(step.result!.isError ? { isError: true } : {}),
          }));
      }
    } else {
      turns.push(stripTurnRuntimeState(currentTurn));
    }
  }

  const parsed = { systemPrompt, userText, userImages, turns, toolResults };
  if (debug) debugLog("parse_messages.end", parsed);
  return parsed;
}

// ── Tool definitions ──

function buildMcpToolDefinitions(tools: OpenAIToolDef[]): McpToolDefinition[] {
  return tools.map((t) => {
    const fn = t.function;
    const jsonSchema: JsonValue =
      fn.parameters && typeof fn.parameters === "object"
        ? (fn.parameters as JsonValue)
        : { type: "object", properties: {}, required: [] };
    // Cursor CLI's current schema uses google.protobuf.Value for
    // McpToolDefinition.input_schema. The committed generated schema still
    // exposes that field as bytes, but the outer wire encoding is identical
    // for bytes and message fields (length-delimited field #3), so place the
    // serialized Value bytes here.
    const inputSchema = toBinary(ValueSchema, fromJson(ValueSchema, jsonSchema));
    return create(McpToolDefinitionSchema, {
      name: fn.name,
      description: fn.description || "",
      providerIdentifier: "pi",
      toolName: fn.name,
      inputSchema,
    });
  });
}

function decodeMcpArgValue(value: Uint8Array): unknown {
  try {
    const parsed = fromBinary(ValueSchema, value);
    return toJson(ValueSchema, parsed);
  } catch {}
  return new TextDecoder().decode(value);
}

function decodeMcpArgsMap(args: Record<string, Uint8Array>): Record<string, unknown> {
  const decoded: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) decoded[key] = decodeMcpArgValue(value);
  return decoded;
}

// ── Build Cursor protobuf request ──

function createSelectedImages(images: ParsedImageContent[]) {
  // Matches Cursor CLI's ACP image path for inline image data:
  // new SelectedImage({ dataOrBlobId: { case: "data", value }, uuid, mimeType })
  return images.map((image) =>
    create(SelectedImageSchema, {
      uuid: crypto.randomUUID(),
      mimeType: image.mimeType,
      dataOrBlobId: { case: "data", value: image.data },
    }),
  );
}

function createUserMessage(text: string, images: ParsedImageContent[] = []): UserMessage {
  const messageId = crypto.randomUUID();
  // The committed generated schema is older than Cursor CLI 2026.05.09 here:
  // field 10 is conversation_state_blob_id, and field 17 is thread_id.
  // Leave both unset instead of using stale selectedContextBlob/correlationId names.
  return create(UserMessageSchema, {
    text,
    messageId,
    selectedContext: create(SelectedContextSchema, {
      selectedImages: createSelectedImages(images),
    }),
    mode: 1,
  });
}

function buildMcpSuccessContent(result: ParsedToolResult) {
  const content = [];
  if (result.content.length > 0) {
    content.push(
      create(McpToolResultContentItemSchema, {
        content: {
          case: "text",
          value: create(McpTextContentSchema, { text: result.content }),
        },
      }),
    );
  }
  for (const image of result.images ?? []) {
    content.push(
      create(McpToolResultContentItemSchema, {
        content: {
          case: "image",
          value: create(McpImageContentSchema, { data: image.data, mimeType: image.mimeType }),
        },
      }),
    );
  }
  if (content.length === 0) {
    content.push(
      create(McpToolResultContentItemSchema, {
        content: { case: "text", value: create(McpTextContentSchema, { text: "" }) },
      }),
    );
  }
  return content;
}

function mcpResultFromParsedToolResult(result: ParsedToolResult) {
  return create(McpToolResultSchema, {
    result: result.isError
      ? {
          case: "error",
          value: create(McpToolErrorSchema, { error: result.content }),
        }
      : {
          case: "success",
          value: create(McpSuccessSchema, {
            content: buildMcpSuccessContent(result),
            isError: false,
          }),
        },
  });
}

function storeAsBlob(data: Uint8Array, blobStore: Map<string, Uint8Array>): Uint8Array {
  const id = new Uint8Array(createHash("sha256").update(data).digest());
  blobStore.set(Buffer.from(id).toString("hex"), data);
  return id;
}

function transcriptImageSummary(images: ParsedImageContent[] | undefined): string {
  if (!images?.length) return "";
  return `\n[images: ${images.map((image) => `${image.mimeType}, ${image.data.length} bytes`).join("; ")}]`;
}

function truncateInlineHistoryText(text: string): string {
  if (INLINE_HISTORY_SEGMENT_MAX_CHARS === 0) return `[omitted ${text.length} chars]`;
  if (text.length <= INLINE_HISTORY_SEGMENT_MAX_CHARS) return text;
  return `${text.slice(0, INLINE_HISTORY_SEGMENT_MAX_CHARS)}\n[truncated ${text.length - INLINE_HISTORY_SEGMENT_MAX_CHARS} chars]`;
}

function transcriptStepText(step: ParsedTurnStep): string {
  if (step.kind === "assistantText") return `Assistant: ${truncateInlineHistoryText(step.text)}`;
  const argsText = Object.keys(step.arguments).length ? JSON.stringify(step.arguments) : "";
  const args = argsText ? ` ${truncateInlineHistoryText(argsText)}` : "";
  const result = step.result
    ? `\nTool result${step.result.isError ? " (error)" : ""}: ${truncateInlineHistoryText(step.result.content)}${transcriptImageSummary(step.result.images)}`
    : "";
  return `Tool call: ${step.toolName || "tool"}${args}${result}`;
}

function truncateInlineHistoryBlock(block: string, maxChars: number): string {
  if (block.length <= maxChars) return block;
  return `${block.slice(0, maxChars)}\n[truncated ${block.length - maxChars} chars]`;
}

function inlineHistorySearchTerms(currentUserText: string): Set<string> {
  const terms = new Set<string>();
  for (const match of currentUserText.matchAll(/[A-Za-z0-9_./@-]{4,}/g)) {
    const term = match[0]!.toLowerCase();
    if (
      [
        "what",
        "when",
        "where",
        "which",
        "continue",
        "context",
        "terms",
        "turns",
        "request",
        "first",
        "last",
        "previous",
      ].includes(term)
    )
      continue;
    terms.add(term);
  }
  return terms;
}

function scoreInlineHistoryBlock(block: string, terms: Set<string>): number {
  if (terms.size === 0) return 0;
  const lower = block.toLowerCase();
  let score = 0;
  for (const term of terms) {
    if (lower.includes(term)) score += term.includes("/") || term.includes(".") ? 3 : 1;
  }
  return score;
}

function inlineHistoryTurnPreview(block: string, index: number): string {
  const userLine = block.split("\n").find((line) => line.startsWith("User: ")) ?? "User: [no text]";
  return `Turn ${index + 1} preview: ${truncateInlineHistoryBlock(userLine, 240)}`;
}

function compressedInlineHistoryNotice(
  turnBlocks: string[],
  startIndex: number,
  endIndex: number,
): string {
  const count = endIndex - startIndex + 1;
  const previews: string[] = [];
  let previewChars = 0;
  for (let index = startIndex; index <= endIndex; index += 1) {
    const preview = inlineHistoryTurnPreview(turnBlocks[index]!, index);
    const nextLength = previewChars + (previews.length > 0 ? 1 : 0) + preview.length;
    if (nextLength > 6_000) {
      previews.push(`[${endIndex - index + 1} more compressed turn preview(s) omitted]`);
      break;
    }
    previews.push(preview);
    previewChars = nextLength;
  }
  return `[${count} turn(s) compressed to previews]\n${previews.join("\n")}`;
}

function selectedInlineHistoryBlocks(turnBlocks: string[], currentUserText: string): string[] {
  const selected = new Map<number, string>();
  let selectedLength = 0;
  const addBlock = (index: number, block: string, maxChars: number): boolean => {
    if (selected.has(index) || maxChars <= 0) return false;
    const value = truncateInlineHistoryBlock(block, maxChars);
    const separatorLength = selected.size > 0 ? 2 : 0;
    if (selectedLength + separatorLength + value.length > INLINE_HISTORY_MAX_CHARS) return false;
    selected.set(index, value);
    selectedLength += separatorLength + value.length;
    return true;
  };

  const headBudget = Number.isFinite(INLINE_HISTORY_MAX_CHARS)
    ? Math.min(INLINE_HISTORY_HEAD_MAX_CHARS, Math.floor(INLINE_HISTORY_MAX_CHARS / 2))
    : INLINE_HISTORY_HEAD_MAX_CHARS;
  let headLength = 0;
  if (headBudget > 0 && turnBlocks.length > 1) {
    for (let index = 0; index < turnBlocks.length; index += 1) {
      const block = turnBlocks[index]!;
      const separatorLength = headLength > 0 ? 2 : 0;
      const remainingHeadBudget = headBudget - headLength - separatorLength;
      if (remainingHeadBudget <= 0) break;
      const value = truncateInlineHistoryBlock(block, remainingHeadBudget);
      if (!addBlock(index, value, remainingHeadBudget)) break;
      headLength += separatorLength + value.length;
      if (value.length < block.length) break;
    }
  }

  const terms = inlineHistorySearchTerms(currentUserText);
  const scored = turnBlocks
    .map((block, index) => ({ index, score: scoreInlineHistoryBlock(block, terms) }))
    .filter((item) => item.score > 0 && !selected.has(item.index))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  for (const { index } of scored) {
    const remainingBudget = INLINE_HISTORY_MAX_CHARS - selectedLength - (selected.size > 0 ? 2 : 0);
    if (remainingBudget <= 0) break;
    addBlock(
      index,
      turnBlocks[index]!,
      Math.min(INLINE_HISTORY_SEGMENT_MAX_CHARS, remainingBudget),
    );
  }

  for (let index = turnBlocks.length - 1; index >= 0; index -= 1) {
    const remainingBudget = INLINE_HISTORY_MAX_CHARS - selectedLength - (selected.size > 0 ? 2 : 0);
    if (remainingBudget <= 0) break;
    addBlock(index, turnBlocks[index]!, remainingBudget);
  }

  if (selected.size === 0) {
    addBlock(turnBlocks.length - 1, turnBlocks.at(-1)!, INLINE_HISTORY_MAX_CHARS);
  }

  const sorted = [...selected.entries()].sort(([a], [b]) => a - b);
  const bodyBlocks: string[] = [];
  let previousIndex = -1;
  for (const [index, block] of sorted) {
    const omitted = index - previousIndex - 1;
    if (omitted > 0) {
      bodyBlocks.push(compressedInlineHistoryNotice(turnBlocks, previousIndex + 1, index - 1));
    }
    bodyBlocks.push(block);
    previousIndex = index;
  }
  const trailingOmitted = turnBlocks.length - previousIndex - 1;
  if (trailingOmitted > 0) {
    bodyBlocks.push(
      compressedInlineHistoryNotice(turnBlocks, previousIndex + 1, turnBlocks.length - 1),
    );
  }
  return bodyBlocks;
}

function inlineHistoryPrompt(turns: ParsedTurn[], currentUserText: string): string | undefined {
  if (turns.length === 0 || INLINE_HISTORY_MAX_CHARS === 0) return undefined;
  const turnBlocks = turns.map((turn, index) => {
    const lines = [
      `Turn ${index + 1}`,
      `User: ${truncateInlineHistoryText(turn.userText)}${transcriptImageSummary(turn.userImages)}`,
      ...turn.steps.map(transcriptStepText),
    ];
    return lines.join("\n");
  });

  const body = selectedInlineHistoryBlocks(turnBlocks, currentUserText).join("\n\n");
  return `Prior Pi conversation context follows. This is a redundant safety copy of the full prior Pi transcript. Use it to preserve session continuity and resolve references in the current user message; the current user message is separate and comes after this context.\n\n<pi_conversation_history>\n${body}\n</pi_conversation_history>`;
}

function userTextWithInlineHistory(userText: string, historyPrompt: string | undefined): string {
  if (!historyPrompt) return userText;
  const currentUserText = userText.trim()
    ? userText
    : "[The current user message has no text. Use any attached images as the current request.]";
  return `${historyPrompt}\n\nCurrent user message:\n<current_user_message>\n${currentUserText}\n</current_user_message>`;
}

function rootPromptBlobIds(systemPrompt: string, blobStore: Map<string, Uint8Array>): Uint8Array[] {
  if (!systemPrompt.trim()) return [];
  return [
    storeAsBlob(
      new TextEncoder().encode(JSON.stringify({ role: "system", content: systemPrompt })),
      blobStore,
    ),
  ];
}

function encodeWireVarint(value: number): Uint8Array {
  const out: number[] = [];
  let current = value >>> 0;
  while (current >= 0x80) {
    out.push((current & 0x7f) | 0x80);
    current >>>= 7;
  }
  out.push(current);
  return new Uint8Array(out);
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function encodeLengthDelimitedField(fieldNo: number, value: Uint8Array): Uint8Array {
  return concatBytes([encodeWireVarint((fieldNo << 3) | 2), encodeWireVarint(value.length), value]);
}

function encodePreFetchedBlob(id: Uint8Array, value: Uint8Array): Uint8Array {
  return concatBytes([encodeLengthDelimitedField(1, id), encodeLengthDelimitedField(2, value)]);
}

function appendPreFetchedBlobsToRunRequest(
  runRequestBytes: Uint8Array,
  blobStore: Map<string, Uint8Array>,
): Uint8Array {
  if (blobStore.size === 0) return runRequestBytes;
  const fields = [...blobStore.entries()].map(([hexId, value]) =>
    encodeLengthDelimitedField(17, encodePreFetchedBlob(Buffer.from(hexId, "hex"), value)),
  );
  return concatBytes([runRequestBytes, ...fields]);
}

function encodeAgentClientRunRequest(runRequestBytes: Uint8Array): Uint8Array {
  return encodeLengthDelimitedField(1, runRequestBytes);
}

function summarizeBlobStore(blobStore: Map<string, Uint8Array>) {
  const entries = [...blobStore.entries()].map(([id, bytes]) => ({ id, bytes: bytes.length }));
  entries.sort((a, b) => b.bytes - a.bytes);
  return {
    count: blobStore.size,
    bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0),
    largest: entries.slice(0, 5),
  };
}

function turnStats(turns: ParsedTurn[]) {
  return {
    count: turns.length,
    stepCount: turns.reduce((sum, turn) => sum + turn.steps.length, 0),
    toolStepCount: turns.reduce(
      (sum, turn) => sum + turn.steps.filter((step) => step.kind === "toolCall").length,
      0,
    ),
  };
}

interface EncodedConversationHistory {
  bytes?: Uint8Array;
  messageCount: number;
  userMessageCount: number;
  assistantMessageCount: number;
  toolMessageCount: number;
}

function encodeStringField(fieldNo: number, value: string): Uint8Array {
  return encodeLengthDelimitedField(fieldNo, new TextEncoder().encode(value));
}

function encodeBoolField(fieldNo: number, value: boolean): Uint8Array {
  return concatBytes([encodeWireVarint((fieldNo << 3) | 0), new Uint8Array([value ? 1 : 0])]);
}

function encodeConversationHistoryText(text: string): Uint8Array {
  return encodeStringField(1, text);
}

function encodeConversationHistoryImage(image: ParsedImageContent): Uint8Array {
  const fields = [encodeStringField(1, Buffer.from(image.data).toString("base64"))];
  if (image.mimeType) fields.push(encodeStringField(2, image.mimeType));
  return concatBytes(fields);
}

function encodeConversationHistoryUserContentText(text: string): Uint8Array {
  return encodeLengthDelimitedField(1, encodeConversationHistoryText(text));
}

function encodeConversationHistoryUserContentImage(image: ParsedImageContent): Uint8Array {
  return encodeLengthDelimitedField(2, encodeConversationHistoryImage(image));
}

function encodeConversationHistoryAssistantContentText(text: string): Uint8Array {
  return encodeLengthDelimitedField(1, encodeConversationHistoryText(text));
}

function encodeConversationHistoryToolCall(step: ParsedToolCallStep): Uint8Array {
  const fields = [
    encodeStringField(1, step.toolCallId),
    encodeStringField(2, step.toolName || "tool"),
    encodeStringField(3, JSON.stringify(step.arguments ?? {})),
  ];
  return concatBytes(fields);
}

function encodeConversationHistoryAssistantContentToolCall(step: ParsedToolCallStep): Uint8Array {
  return encodeLengthDelimitedField(4, encodeConversationHistoryToolCall(step));
}

function encodeConversationHistoryToolResultContentText(text: string): Uint8Array {
  return encodeLengthDelimitedField(1, encodeConversationHistoryText(text));
}

function encodeConversationHistoryToolResultContentImage(image: ParsedImageContent): Uint8Array {
  return encodeLengthDelimitedField(2, encodeConversationHistoryImage(image));
}

function encodeConversationHistoryUserMessage(turn: ParsedTurn): Uint8Array | undefined {
  const contents: Uint8Array[] = [];
  if (turn.userText.trim()) {
    contents.push(
      encodeLengthDelimitedField(1, encodeConversationHistoryUserContentText(turn.userText)),
    );
  }
  for (const image of turn.userImages ?? []) {
    contents.push(encodeLengthDelimitedField(1, encodeConversationHistoryUserContentImage(image)));
  }
  if (contents.length === 0) return undefined;
  return concatBytes(contents);
}

function encodeConversationHistoryAssistantTextMessage(text: string): Uint8Array | undefined {
  if (!text.trim()) return undefined;
  return encodeLengthDelimitedField(1, encodeConversationHistoryAssistantContentText(text));
}

function encodeConversationHistoryAssistantToolCallMessage(
  step: ParsedToolCallStep,
): Uint8Array | undefined {
  if (!step.toolCallId && !step.toolName) return undefined;
  return encodeLengthDelimitedField(1, encodeConversationHistoryAssistantContentToolCall(step));
}

function encodeConversationHistoryToolMessage(step: ParsedToolCallStep): Uint8Array | undefined {
  if (!step.result) return undefined;
  const fields = [
    encodeStringField(1, step.toolCallId),
    encodeStringField(2, step.toolName || "tool"),
  ];
  if (step.result.content.length > 0 || (step.result.images ?? []).length === 0) {
    fields.push(
      encodeLengthDelimitedField(
        3,
        encodeConversationHistoryToolResultContentText(step.result.content),
      ),
    );
  }
  for (const image of step.result.images ?? []) {
    fields.push(
      encodeLengthDelimitedField(3, encodeConversationHistoryToolResultContentImage(image)),
    );
  }
  if (step.result.isError) fields.push(encodeBoolField(4, true));
  return concatBytes(fields);
}

function encodeConversationHistoryMessage(fieldNo: 1 | 2 | 3, value: Uint8Array): Uint8Array {
  return encodeLengthDelimitedField(fieldNo, value);
}

function encodeConversationHistoryForTurns(turns: ParsedTurn[]): EncodedConversationHistory {
  const messages: Uint8Array[] = [];
  let userMessageCount = 0;
  let assistantMessageCount = 0;
  let toolMessageCount = 0;

  for (const turn of turns) {
    const userMessage = encodeConversationHistoryUserMessage(turn);
    if (userMessage) {
      messages.push(
        encodeLengthDelimitedField(1, encodeConversationHistoryMessage(1, userMessage)),
      );
      userMessageCount += 1;
    }

    for (const step of turn.steps) {
      if (step.kind === "assistantText") {
        const assistantMessage = encodeConversationHistoryAssistantTextMessage(step.text);
        if (assistantMessage) {
          messages.push(
            encodeLengthDelimitedField(1, encodeConversationHistoryMessage(2, assistantMessage)),
          );
          assistantMessageCount += 1;
        }
        continue;
      }

      const assistantToolCall = encodeConversationHistoryAssistantToolCallMessage(step);
      if (assistantToolCall) {
        messages.push(
          encodeLengthDelimitedField(1, encodeConversationHistoryMessage(2, assistantToolCall)),
        );
        assistantMessageCount += 1;
      }

      const toolMessage = encodeConversationHistoryToolMessage(step);
      if (toolMessage) {
        messages.push(
          encodeLengthDelimitedField(1, encodeConversationHistoryMessage(3, toolMessage)),
        );
        toolMessageCount += 1;
      }
    }
  }

  return {
    bytes: messages.length > 0 ? concatBytes(messages) : undefined,
    messageCount: messages.length,
    userMessageCount,
    assistantMessageCount,
    toolMessageCount,
  };
}

function encodeUserMessageAction(
  userMessage: UserMessage,
  conversationHistory: EncodedConversationHistory,
): Uint8Array {
  const fields = [encodeLengthDelimitedField(1, toBinary(UserMessageSchema, userMessage))];
  if (conversationHistory.bytes)
    fields.push(encodeLengthDelimitedField(7, conversationHistory.bytes));
  return concatBytes(fields);
}

function encodeUserMessageConversationAction(
  userMessage: UserMessage,
  conversationHistory: EncodedConversationHistory,
): Uint8Array {
  return encodeLengthDelimitedField(1, encodeUserMessageAction(userMessage, conversationHistory));
}

function encodeAgentRunRequest(params: {
  conversationState: ConversationStateStructure;
  userMessage: UserMessage;
  conversationHistory: EncodedConversationHistory;
  requestedModel: RequestedModel;
  mcpTools: McpTools;
  conversationId: string;
}): Uint8Array {
  const fields = [
    encodeLengthDelimitedField(
      1,
      toBinary(ConversationStateStructureSchema, params.conversationState),
    ),
    encodeLengthDelimitedField(
      2,
      encodeUserMessageConversationAction(params.userMessage, params.conversationHistory),
    ),
    encodeLengthDelimitedField(9, toBinary(RequestedModelSchema, params.requestedModel)),
    encodeLengthDelimitedField(4, toBinary(McpToolsSchema, params.mcpTools)),
  ];
  if (params.conversationId) fields.push(encodeStringField(5, params.conversationId));
  return concatBytes(fields);
}

export function buildCursorRequest(
  modelId: string,
  systemPrompt: string,
  userText: string,
  turns: ParsedTurn[],
  conversationId: string,
  checkpoint: Uint8Array | null,
  existingBlobStore?: Map<string, Uint8Array>,
  maxMode = false,
  cursorModelParameters: CursorModelParameter[] = [],
  mcpTools: McpToolDefinition[] = [],
  userImages: ParsedImageContent[] = [],
  workspaceContext: CursorWorkspaceContext = createWorkspaceContext(),
): CursorRequestPayload {
  debugLog("cursor_request.build.start", {
    modelId,
    systemPromptChars: systemPrompt.length,
    userTextChars: userText.length,
    turns: turnStats(turns),
    conversationId,
    checkpointBytes: checkpoint?.length ?? 0,
    existingBlobStore: existingBlobStore ? summarizeBlobStore(existingBlobStore) : undefined,
    maxMode,
    cursorModelParameters,
    mcpToolCount: mcpTools.length,
    userImageCount: userImages.length,
    workspaceContext,
  });
  const blobStore = new Map<string, Uint8Array>(existingBlobStore ?? []);
  const historyPrompt = checkpoint ? undefined : inlineHistoryPrompt(turns, userText);
  const rootPromptMessagesJson = checkpoint ? [] : rootPromptBlobIds(systemPrompt, blobStore);
  const conversationHistory = checkpoint
    ? {
        messageCount: 0,
        userMessageCount: 0,
        assistantMessageCount: 0,
        toolMessageCount: 0,
      }
    : encodeConversationHistoryForTurns(turns);

  const conversationState = checkpoint
    ? fromBinary(ConversationStateStructureSchema, checkpoint)
    : create(ConversationStateStructureSchema, {
        rootPromptMessagesJson,
        turns: [],
        todos: [],
        pendingToolCalls: [],
        previousWorkspaceUris: [workspaceContext.workspaceUri],
        mode: 1,
        fileStates: {},
        fileStatesV2: {},
        summaryArchives: [],
        turnTimings: [],
        subagentStates: {},
        selfSummaryCount: 0,
        readPaths: [],
        clientName: "pi",
      });

  const userMessage = createUserMessage(
    userTextWithInlineHistory(userText, historyPrompt),
    userImages,
  );
  // Cursor's newer request path uses requestedModel instead of legacy modelDetails.
  // Some Cursor models (for example GPT-5.5) use requestedModel.parameters
  // for context/reasoning/fast instead of encoding everything in the model ID.
  // Max Mode is routed from model metadata for parameterized variants.
  debugLog("cursor_request.requested_model", {
    modelId,
    maxMode,
    parameters: cursorModelParameters,
  });
  const parameters = cursorModelParameters.map((parameter) =>
    create(RequestedModel_ModelParameterbytesSchema, parameter),
  );
  const requestedModel = create(RequestedModelSchema, { modelId, maxMode, parameters });
  const mcpToolsMessage = create(McpToolsSchema, { mcpTools });
  const runRequestBytesWithoutPrefetch = encodeAgentRunRequest({
    conversationState,
    userMessage,
    conversationHistory,
    requestedModel,
    mcpTools: mcpToolsMessage,
    conversationId,
  });
  const runRequestBytes = appendPreFetchedBlobsToRunRequest(
    runRequestBytesWithoutPrefetch,
    blobStore,
  );

  const payload = {
    requestBytes: encodeAgentClientRunRequest(runRequestBytes),
    blobStore,
    mcpTools,
  };
  debugLog("cursor_request.build.end", {
    conversationId,
    checkpointUsed: Boolean(checkpoint),
    conversationTurns: (conversationState as any).turns?.length ?? 0,
    rootPromptMessages: (conversationState as any).rootPromptMessagesJson?.length ?? 0,
    conversationHistory,
    inlineHistoryChars: historyPrompt?.length ?? 0,
    requestBytes: payload.requestBytes.length,
    runRequestBytes: runRequestBytes.length,
    prefetchBytes: runRequestBytes.length - runRequestBytesWithoutPrefetch.length,
    blobs: summarizeBlobStore(blobStore),
    mcpToolCount: mcpTools.length,
  });
  return payload;
}

// ── Server message processing ──

function processServerMessage(
  msg: AgentServerMessage,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
  workspaceContext: CursorWorkspaceContext,
  sendFrame: (data: Uint8Array) => void,
  state: StreamState,
  onText: (text: string, isThinking?: boolean) => void,
  onMcpExec: (exec: PendingExec) => void,
  onCheckpoint?: (checkpointBytes: Uint8Array) => void,
): void {
  const msgCase = msg.message.case;
  debugLog("server_message", { msgCase, msg });

  if (msgCase === "interactionUpdate") {
    const update = msg.message.value as any;
    const updateCase = update.message?.case;
    if (updateCase === "textDelta") {
      const delta = update.message.value.text || "";
      if (delta) onText(delta, false);
    } else if (updateCase === "thinkingDelta") {
      const delta = update.message.value.text || "";
      if (delta) onText(delta, true);
    } else if (updateCase === "tokenDelta") {
      state.outputTokens += update.message.value.tokens ?? 0;
    } else if (
      updateCase &&
      ![
        "heartbeat",
        "thinkingCompleted",
        "summary",
        "summaryStarted",
        "summaryCompleted",
        "turnEnded",
        "userMessageAppended",
        "stepStarted",
        "stepCompleted",
      ].includes(updateCase)
    ) {
      debugLog("server_message.unhandled_interaction_update", { updateCase, update });
    }
  } else if (msgCase === "kvServerMessage") {
    handleKvMessage(msg.message.value as KvServerMessage, blobStore, sendFrame);
  } else if (msgCase === "execServerMessage") {
    handleExecMessage(
      msg.message.value as ExecServerMessage,
      mcpTools,
      workspaceContext,
      sendFrame,
      onMcpExec,
    );
  } else if (msgCase === "conversationCheckpointUpdate") {
    const stateStructure = msg.message.value as ConversationStateStructure;
    if ((stateStructure as any).tokenDetails) {
      state.totalTokens = (stateStructure as any).tokenDetails.usedTokens;
    }
    if (onCheckpoint) {
      onCheckpoint(toBinary(ConversationStateStructureSchema, stateStructure));
    }
  } else if (msgCase) {
    debugLog("server_message.unhandled", { msgCase, msg });
  }
}

function sendKvResponse(
  kvMsg: KvServerMessage,
  messageCase: string,
  value: unknown,
  sendFrame: (data: Uint8Array) => void,
): void {
  const response = create(KvClientMessageSchema, {
    id: (kvMsg as any).id,
    message: { case: messageCase as any, value: value as any },
  });
  const clientMsg = create(AgentClientMessageSchema, {
    message: { case: "kvClientMessage", value: response },
  });
  sendFrame(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMsg)));
}

function handleKvMessage(
  kvMsg: KvServerMessage,
  blobStore: Map<string, Uint8Array>,
  sendFrame: (data: Uint8Array) => void,
): void {
  const kvCase = (kvMsg as any).message.case;
  if (kvCase === "getBlobArgs") {
    const blobId = (kvMsg as any).message.value.blobId;
    const blobIdKey = Buffer.from(blobId).toString("hex");
    const blobData = blobStore.get(blobIdKey);
    sendKvResponse(
      kvMsg,
      "getBlobResult",
      create(GetBlobResultSchema, blobData ? { blobData } : {}),
      sendFrame,
    );
  } else if (kvCase === "setBlobArgs") {
    const { blobId, blobData } = (kvMsg as any).message.value;
    blobStore.set(Buffer.from(blobId).toString("hex"), blobData);
    sendKvResponse(kvMsg, "setBlobResult", create(SetBlobResultSchema, {}), sendFrame);
  }
}

const NATIVE_TOOL_REJECT_REASON =
  "Tool not available in this environment. Use the MCP tools provided instead.";

type NativeExecResult = { messageCase: string; value: unknown };
type NativeExecResultBuilder = (args: any) => NativeExecResult;

function shellRejected(args: any) {
  return create(ShellRejectedSchema, {
    command: args.command ?? "",
    workingDirectory: args.workingDirectory ?? "",
    reason: NATIVE_TOOL_REJECT_REASON,
    isReadonly: false,
  });
}

const NATIVE_EXEC_RESULT_BUILDERS: Record<string, NativeExecResultBuilder> = {
  readArgs: (args) => ({
    messageCase: "readResult",
    value: create(ReadResultSchema, {
      result: {
        case: "rejected",
        value: create(ReadRejectedSchema, { path: args.path, reason: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  lsArgs: (args) => ({
    messageCase: "lsResult",
    value: create(LsResultSchema, {
      result: {
        case: "rejected",
        value: create(LsRejectedSchema, { path: args.path, reason: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  grepArgs: () => ({
    messageCase: "grepResult",
    value: create(GrepResultSchema, {
      result: {
        case: "error",
        value: create(GrepErrorSchema, { error: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  writeArgs: (args) => ({
    messageCase: "writeResult",
    value: create(WriteResultSchema, {
      result: {
        case: "rejected",
        value: create(WriteRejectedSchema, { path: args.path, reason: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  deleteArgs: (args) => ({
    messageCase: "deleteResult",
    value: create(DeleteResultSchema, {
      result: {
        case: "rejected",
        value: create(DeleteRejectedSchema, { path: args.path, reason: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  shellArgs: (args) => ({
    messageCase: "shellResult",
    value: create(ShellResultSchema, {
      result: { case: "rejected", value: shellRejected(args) },
    }),
  }),
  shellStreamArgs: (args) => ({
    messageCase: "shellStream",
    value: create(ShellStreamSchema, {
      event: { case: "rejected", value: shellRejected(args) },
    }),
  }),
  backgroundShellSpawnArgs: (args) => ({
    messageCase: "backgroundShellSpawnResult",
    value: create(BackgroundShellSpawnResultSchema, {
      result: { case: "rejected", value: shellRejected(args) },
    }),
  }),
  writeShellStdinArgs: () => ({
    messageCase: "writeShellStdinResult",
    value: create(WriteShellStdinResultSchema, {
      result: {
        case: "error",
        value: create(WriteShellStdinErrorSchema, { error: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  fetchArgs: (args) => ({
    messageCase: "fetchResult",
    value: create(FetchResultSchema, {
      result: {
        case: "error",
        value: create(FetchErrorSchema, { url: args.url ?? "", error: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  diagnosticsArgs: () => ({
    messageCase: "diagnosticsResult",
    value: create(DiagnosticsResultSchema, {}),
  }),
  listMcpResourcesExecArgs: () => ({
    messageCase: "listMcpResourcesExecResult",
    value: create(ListMcpResourcesExecResultSchema, {
      result: {
        case: "rejected",
        value: create(ListMcpResourcesRejectedSchema, { reason: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  readMcpResourceExecArgs: (args) => ({
    messageCase: "readMcpResourceExecResult",
    value: create(ReadMcpResourceExecResultSchema, {
      result: {
        case: "rejected",
        value: create(ReadMcpResourceRejectedSchema, {
          uri: args.uri ?? "",
          reason: NATIVE_TOOL_REJECT_REASON,
        }),
      },
    }),
  }),
  recordScreenArgs: () => ({
    messageCase: "recordScreenResult",
    value: create(RecordScreenResultSchema, {
      result: {
        case: "failure",
        value: create(RecordScreenFailureSchema, { error: NATIVE_TOOL_REJECT_REASON }),
      },
    }),
  }),
  computerUseArgs: (args) => ({
    messageCase: "computerUseResult",
    value: create(ComputerUseResultSchema, {
      result: {
        case: "error",
        value: create(ComputerUseErrorSchema, {
          error: NATIVE_TOOL_REJECT_REASON,
          actionCount: Array.isArray(args.actions) ? args.actions.length : 0,
          durationMs: 0,
        }),
      },
    }),
  }),
};

function nativeExecResultFor(
  execCase: string,
  execMsg: ExecServerMessage,
): NativeExecResult | undefined {
  const builder = NATIVE_EXEC_RESULT_BUILDERS[execCase];
  if (!builder) return undefined;
  return builder((execMsg as any).message.value ?? {});
}

function handleExecMessage(
  execMsg: ExecServerMessage,
  mcpTools: McpToolDefinition[],
  workspaceContext: CursorWorkspaceContext,
  sendFrame: (data: Uint8Array) => void,
  onMcpExec: (exec: PendingExec) => void,
): void {
  const execCase = (execMsg as any).message.case;

  if (execCase === "requestContextArgs") {
    const requestContext = create(RequestContextSchema, {
      rules: [],
      env: create(RequestContextEnvSchema, {
        osVersion: `${osType()} ${osRelease()}`.trim(),
        workspacePaths: [workspaceContext.workspacePath],
        shell: process.env.SHELL ?? "",
        sandboxEnabled: false,
        terminalsFolder: workspaceContext.terminalsFolder,
        agentSharedNotesFolder: workspaceContext.agentSharedNotesFolder,
        agentConversationNotesFolder: workspaceContext.agentConversationNotesFolder,
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone ?? "",
        projectFolder: workspaceContext.projectFolder,
        agentTranscriptsFolder: workspaceContext.agentTranscriptsFolder,
      }),
      repositoryInfo: [],
      tools: mcpTools,
      gitRepos: [],
      projectLayouts: [],
      mcpInstructions: [],
      fileContents: {},
      customSubagents: [],
    });
    const result = create(RequestContextResultSchema, {
      result: { case: "success", value: create(RequestContextSuccessSchema, { requestContext }) },
    });
    sendExecResult(execMsg, "requestContextResult", result, sendFrame);
    return;
  }

  if (execCase === "mcpArgs") {
    const mcpArgs = (execMsg as any).message.value;
    const decoded = decodeMcpArgsMap(mcpArgs.args ?? {});
    onMcpExec({
      execId: (execMsg as any).execId,
      execMsgId: (execMsg as any).id,
      toolCallId: mcpArgs.toolCallId || crypto.randomUUID(),
      toolName: mcpArgs.toolName || mcpArgs.name,
      decodedArgs: JSON.stringify(decoded),
    });
    return;
  }

  // Reject native Cursor tools so model falls back to MCP tools.
  const nativeResult = nativeExecResultFor(execCase, execMsg);
  if (nativeResult) {
    sendExecResult(execMsg, nativeResult.messageCase, nativeResult.value, sendFrame);
    return;
  }

  // Catch-all: log and attempt a generic rejection so the bridge doesn't hang
  console.error(`[cursor-provider] UNHANDLED exec case: "${execCase}". Bridge may stall.`);
  // Try to derive the result case name from the args case name
  const guessedResult = (execCase as string)?.replace(/Args$/, "Result");
  if (guessedResult && guessedResult !== execCase) {
    sendExecResult(execMsg, guessedResult, create(McpResultSchema, {}), sendFrame);
  }
}

function sendExecResult(
  execMsg: ExecServerMessage,
  messageCase: string,
  value: unknown,
  sendFrame: (data: Uint8Array) => void,
): void {
  const execClientMessage = create(ExecClientMessageSchema, {
    id: (execMsg as any).id,
    execId: (execMsg as any).execId,
    message: { case: messageCase as any, value: value as any },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "execClientMessage", value: execClientMessage },
  });
  sendFrame(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
}

// ── Key derivation ──

export function derivePiSessionId(
  body: Pick<ChatCompletionRequest, "pi_session_id" | "user">,
): string | undefined {
  const raw = body.pi_session_id ?? body.user;
  if (typeof raw !== "string") return undefined;
  const trimmed = raw.trim();
  return trimmed ? trimmed : undefined;
}

export function deriveBridgeKeyFromSessionId(sessionId: string): string {
  return createHash("sha256").update(`bridge:${sessionId}`).digest("hex").slice(0, 16);
}

export function deriveConversationKeyFromSessionId(sessionId: string): string {
  return createHash("sha256").update(`conv:${sessionId}`).digest("hex").slice(0, 16);
}

export function deriveBridgeKey(messages: OpenAIMessage[], sessionId?: string): string {
  if (sessionId) return deriveBridgeKeyFromSessionId(sessionId);
  const firstUserMsg = messages.find((m) => m.role === "user");
  const firstUserText = firstUserMsg ? textContent(firstUserMsg.content) : "";
  return createHash("sha256")
    .update(`bridge:${firstUserText.slice(0, 200)}`)
    .digest("hex")
    .slice(0, 16);
}

export function deriveConversationKey(messages: OpenAIMessage[], sessionId?: string): string {
  if (sessionId) return deriveConversationKeyFromSessionId(sessionId);
  const firstUserMsg = messages.find((m) => m.role === "user");
  const firstUserText = firstUserMsg ? textContent(firstUserMsg.content) : "";
  return createHash("sha256")
    .update(`conv:${firstUserText.slice(0, 200)}`)
    .digest("hex")
    .slice(0, 16);
}

export function cleanupSessionActiveBridge(
  sessionId?: string,
  reason = "explicit",
  details: Record<string, unknown> = {},
): void {
  if (!sessionId) return;
  const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
  const convKey = deriveConversationKeyFromSessionId(sessionId);
  const active = activeBridges.get(bridgeKey);
  debugLog("session.cleanup", {
    cleanupKind: "activeBridge",
    reason,
    ...details,
    sessionId,
    bridgeKey,
    convKey,
    hadActiveBridge: !!active,
    hadConversation: conversationStates.has(convKey),
  });
  if (active) cleanupBridge(active.bridge, active.heartbeatTimer, bridgeKey);
}

export function cleanupSessionConversationState(
  sessionId?: string,
  reason = "explicit",
  details: Record<string, unknown> = {},
): void {
  if (!sessionId) return;
  const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
  const convKey = deriveConversationKeyFromSessionId(sessionId);
  const hadConversation = conversationStates.has(convKey);
  debugLog("session.cleanup", {
    cleanupKind: "conversationState",
    reason,
    ...details,
    sessionId,
    bridgeKey,
    convKey,
    hadActiveBridge: activeBridges.has(bridgeKey),
    hadConversation,
  });
  conversationStates.delete(convKey);
}

export function cleanupSessionState(
  sessionId?: string,
  reason = "explicit",
  details: Record<string, unknown> = {},
): void {
  if (!sessionId) return;
  debugLog("session.cleanup", {
    cleanupKind: "full",
    reason,
    ...details,
    sessionId,
    bridgeKey: deriveBridgeKeyFromSessionId(sessionId),
    convKey: deriveConversationKeyFromSessionId(sessionId),
  });
  cleanupSessionActiveBridge(sessionId, reason, details);
  cleanupSessionConversationState(sessionId, reason, details);
}

export function deterministicConversationId(convKey: string): string {
  const hex = createHash("sha256").update(`cursor-conv-id:${convKey}`).digest("hex").slice(0, 32);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    `4${hex.slice(13, 16)}`,
    `${(0x8 | (parseInt(hex[16], 16) & 0x3)).toString(16)}${hex.slice(17, 20)}`,
    hex.slice(20, 32),
  ].join("-");
}

// ── Thinking tag filter ──

const THINKING_TAG_NAMES = ["think", "thinking", "reasoning", "thought", "think_intent"];
const MAX_THINKING_TAG_LEN = 16;

function createThinkingTagFilter() {
  let buffer = "";
  let inThinking = false;
  return {
    process(text: string) {
      const input = buffer + text;
      buffer = "";
      let content = "";
      let reasoning = "";
      let lastIdx = 0;
      const re = new RegExp(`<(/?)(?:${THINKING_TAG_NAMES.join("|")})\\s*>`, "gi");
      let match: RegExpExecArray | null;
      while ((match = re.exec(input)) !== null) {
        const before = input.slice(lastIdx, match.index);
        if (inThinking) reasoning += before;
        else content += before;
        inThinking = match[1] !== "/";
        lastIdx = re.lastIndex;
      }
      const rest = input.slice(lastIdx);
      const ltPos = rest.lastIndexOf("<");
      if (
        ltPos >= 0 &&
        rest.length - ltPos < MAX_THINKING_TAG_LEN &&
        /^<\/?[a-z_]*$/i.test(rest.slice(ltPos))
      ) {
        buffer = rest.slice(ltPos);
        const before = rest.slice(0, ltPos);
        if (inThinking) reasoning += before;
        else content += before;
      } else {
        if (inThinking) reasoning += rest;
        else content += rest;
      }
      return { content, reasoning };
    },
    flush() {
      const b = buffer;
      buffer = "";
      if (!b) return { content: "", reasoning: "" };
      return inThinking ? { content: "", reasoning: b } : { content: b, reasoning: "" };
    },
  };
}

// ── Connect frame helpers ──

function makeHeartbeatBytes(): Uint8Array {
  const heartbeat = create(AgentClientMessageSchema, {
    message: { case: "clientHeartbeat", value: create(ClientHeartbeatSchema, {}) },
  });
  return frameConnectMessage(toBinary(AgentClientMessageSchema, heartbeat));
}

function computeUsage(state: StreamState, fallbackTotalTokens = 0) {
  const completion_tokens = state.outputTokens;
  const total_tokens = state.totalTokens || Math.max(fallbackTotalTokens, completion_tokens);
  const prompt_tokens = Math.max(0, total_tokens - completion_tokens);
  return { prompt_tokens, completion_tokens, total_tokens };
}

function respondWithPendingToolCalls(
  modelId: string,
  pendingExecs: PendingExec[],
  stream: boolean,
  res: ServerResponse,
): void {
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`;
  const created = Math.floor(Date.now() / 1000);
  const toolCalls = pendingExecs.map((exec, index) => ({
    index,
    id: exec.toolCallId,
    type: "function" as const,
    function: { name: exec.toolName, arguments: exec.decodedArgs },
  }));

  if (stream) {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    for (const toolCall of toolCalls) {
      res.write(
        `data: ${JSON.stringify({
          id: completionId,
          object: "chat.completion.chunk",
          created,
          model: modelId,
          choices: [{ index: 0, delta: { tool_calls: [toolCall] }, finish_reason: null }],
        })}\n\n`,
      );
    }
    res.write(
      `data: ${JSON.stringify({
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model: modelId,
        choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
      })}\n\n`,
    );
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(
    JSON.stringify({
      id: completionId,
      object: "chat.completion",
      created,
      model: modelId,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null, tool_calls: toolCalls },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    }),
  );
}

// ── Streaming response ──

function startBridge(accessToken: string, requestBytes: Uint8Array) {
  const bridge = bridgeFactory({ accessToken, rpcPath: "/agent.v1.AgentService/Run" });
  debugLog("bridge.start_run", { requestBytes });
  bridge.write(frameConnectMessage(requestBytes));
  const heartbeatTimer = setInterval(() => {
    if (!bridge.alive) return;
    try {
      bridge.write(makeHeartbeatBytes());
    } catch (error) {
      debugLog("bridge.heartbeat_failed", {
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }, 5_000);
  return { bridge, heartbeatTimer };
}

function handleStreamingResponse(
  payload: CursorRequestPayload,
  accessToken: string,
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  req: IncomingMessage,
  res: ServerResponse,
  requestId: string,
): void {
  debugLog("stream.start", { requestId, bridgeKey, convKey, modelId, workspaceContext });
  const { bridge, heartbeatTimer } = startBridge(accessToken, payload.requestBytes);
  writeSSEStream(
    bridge,
    heartbeatTimer,
    payload.blobStore,
    payload.mcpTools,
    modelId,
    bridgeKey,
    convKey,
    completedTurns,
    currentTurn,
    workspaceContext,
    systemPromptFingerprint,
    req,
    res,
    requestId,
  );
}

function sendCancelAction(bridge: BridgeHandle): void {
  debugLog("bridge.cancel_action", {});
  const action = create(ConversationActionSchema, {
    action: { case: "cancelAction", value: create(CancelActionSchema, {}) },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "conversationAction", value: action },
  });
  try {
    bridge.write(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
  } catch (error) {
    debugLog("bridge.cancel_action_failed", {
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

function cleanupBridge(
  bridge: BridgeHandle,
  heartbeatTimer: ReturnType<typeof setInterval>,
  bridgeKey: string,
): void {
  debugLog("bridge.cleanup", { bridgeKey, alive: bridge.alive });
  clearInterval(heartbeatTimer);
  clearActiveBridgeToolTimeout(activeBridges.get(bridgeKey));
  if (bridge.alive) {
    sendCancelAction(bridge);
    bridge.end();
  }
  activeBridges.delete(bridgeKey);
}

function writeSSEStream(
  bridge: BridgeHandle,
  heartbeatTimer: ReturnType<typeof setInterval>,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  req: IncomingMessage,
  res: ServerResponse,
  requestId?: string,
): void {
  debugLog("stream.writer_start", {
    requestId,
    bridgeKey,
    convKey,
    modelId,
    workspaceContext,
    completedTurnCount: completedTurns.length,
    currentTurn,
  });
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`;
  const created = Math.floor(Date.now() / 1000);

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  let closed = false;
  const sendSSE = (data: object) => {
    if (closed) return;
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };
  const sendDone = () => {
    if (closed) return;
    res.write("data: [DONE]\n\n");
  };
  const closeResponse = () => {
    if (closed) return;
    closed = true;
    res.end();
  };

  const makeChunk = (delta: Record<string, unknown>, finishReason: string | null = null) => ({
    id: completionId,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });

  const makeUsageChunk = () => {
    const { prompt_tokens, completion_tokens, total_tokens } = computeUsage(state);
    return {
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model: modelId,
      choices: [],
      usage: { prompt_tokens, completion_tokens, total_tokens },
    };
  };

  const state = createStreamState();
  const tagFilter = createThinkingTagFilter();
  let mcpExecReceived = false;
  let cancelled = false;
  let streamError: Error | null = null;
  let latestCheckpoint: Uint8Array | null = null;
  const watchdog = createCursorStreamWatchdog({
    eventPrefix: "stream",
    requestId,
    bridgeKey,
    convKey,
    bridge,
    heartbeatTimer,
    onTimeout: (_kind, message) => {
      streamError = new Error(message);
      sendSSE(makeChunk({ content: message }, "error"));
      sendSSE(makeUsageChunk());
      sendDone();
      closeResponse();
    },
  });

  // Detect client disconnect (e.g. user pressed Escape in pi)
  const onClientClose = () => {
    if (cancelled || closed) return;
    debugLog("stream.client_close", { requestId, bridgeKey, convKey });
    cancelled = true;
    watchdog.stop();
    cleanupBridge(bridge, heartbeatTimer, bridgeKey);
    closeResponse();
  };
  req.on("close", onClientClose);
  res.on("close", onClientClose);

  const processChunk = createConnectFrameParser(
    (messageBytes) => {
      try {
        const serverMessage = fromBinary(AgentServerMessageSchema, messageBytes);
        if (!isCursorServerHeartbeat(serverMessage))
          watchdog.touchUpstream(cursorServerMessageLabel(serverMessage));
        processServerMessage(
          serverMessage,
          blobStore,
          mcpTools,
          workspaceContext,
          (data) => bridge.write(data),
          state,
          (text, isThinking) => {
            emitFilteredStreamText(
              tagFilter,
              currentTurn,
              text,
              isThinking,
              (reasoning) => {
                watchdog.touchVisible("thinking_delta");
                sendSSE(makeChunk({ reasoning_content: reasoning }));
              },
              (content) => {
                watchdog.touchVisible("text_delta");
                sendSSE(makeChunk({ content }));
              },
            );
          },
          (exec) => {
            mcpExecReceived = true;
            watchdog.touchVisible("tool_call");
            watchdog.stop();

            flushFilteredStreamText(
              tagFilter,
              currentTurn,
              (reasoning) => sendSSE(makeChunk({ reasoning_content: reasoning })),
              (content) => sendSSE(makeChunk({ content })),
            );

            const toolCallIndex = state.toolCallIndex++;
            sendSSE(
              makeChunk({
                tool_calls: [
                  {
                    index: toolCallIndex,
                    id: exec.toolCallId,
                    type: "function",
                    function: { name: exec.toolName, arguments: exec.decodedArgs },
                  },
                ],
              }),
            );

            rememberToolCallPause(
              bridgeKey,
              bridge,
              heartbeatTimer,
              blobStore,
              mcpTools,
              state,
              currentTurn,
              convKey,
              workspaceContext,
              exec,
            );
            debugLog("stream.tool_call_pause", {
              requestId,
              bridgeKey,
              exec,
              pendingExecs: state.pendingExecs,
              currentTurn,
            });

            sendSSE(makeChunk({}, "tool_calls"));
            sendDone();
            closeResponse();
          },
          (checkpointBytes) => {
            latestCheckpoint = checkpointBytes;
            debugLog("stream.checkpoint_buffered", { requestId, convKey, checkpointBytes });
          },
        );
      } catch (err) {
        streamError = err instanceof Error ? err : new Error(String(err));
        console.error("[cursor-provider] Stream message processing error:", streamError.message);
        debugLog("stream.process_error", {
          requestId,
          bridgeKey,
          convKey,
          message: streamError.message,
        });
        sendSSE(makeChunk({ content: streamError.message }, "error"));
        sendSSE(makeUsageChunk());
        sendDone();
        closeResponse();
        watchdog.stop();
        cleanupBridge(bridge, heartbeatTimer, bridgeKey);
      }
    },
    (endStreamBytes) => {
      const endError = parseConnectEndStream(endStreamBytes);
      if (endError) {
        streamError = endError;
        watchdog.stop();
        console.error(`[cursor-provider] Cursor stream error (${modelId}):`, endError.message);
        sendSSE(makeChunk({ content: endError.message }, "error"));
        sendSSE(makeUsageChunk());
        sendDone();
        closeResponse();
      }
    },
  );

  bridge.onData(processChunk);

  bridge.onClose((code) => {
    debugLog("stream.bridge_close", {
      requestId,
      bridgeKey,
      convKey,
      code,
      cancelled,
      mcpExecReceived,
      currentTurn,
      latestCheckpoint,
    });
    watchdog.stop();
    clearInterval(heartbeatTimer);
    req.removeListener("close", onClientClose);
    res.removeListener("close", onClientClose);

    if (cancelled) return;
    if (streamError) {
      removeActiveBridge(bridgeKey);
      return;
    }

    const stored = conversationStates.get(convKey);
    if (code !== 0) {
      sendSSE(makeChunk({ content: "Bridge connection lost" }, "error"));
      sendSSE(makeUsageChunk());
      sendDone();
      closeResponse();
      removeActiveBridge(bridgeKey);
      return;
    }

    if (!mcpExecReceived) {
      flushFilteredStreamText(
        tagFilter,
        currentTurn,
        (reasoning) => sendSSE(makeChunk({ reasoning_content: reasoning })),
        (content) => sendSSE(makeChunk({ content })),
      );
      commitOrMergeStoredCheckpoint(
        stored,
        latestCheckpoint,
        blobStore,
        completedTurns,
        currentTurn,
        systemPromptFingerprint,
      );
      if (stored && latestCheckpoint)
        debugLog("stream.checkpoint_committed", { requestId, convKey, stored });
      sendSSE(makeChunk({}, "stop"));
      sendSSE(makeUsageChunk());
      sendDone();
      closeResponse();
    } else {
      removeActiveBridge(bridgeKey);
    }
  });
}

export function writeSSEStreamForTests(args: {
  bridge: BridgeHandle;
  heartbeatTimer: ReturnType<typeof setInterval>;
  blobStore?: Map<string, Uint8Array>;
  mcpTools?: McpToolDefinition[];
  modelId: string;
  bridgeKey: string;
  convKey: string;
  completedTurns: ParsedTurn[];
  currentTurn: ParsedTurn;
  workspaceContext?: CursorWorkspaceContext;
  systemPromptFingerprint?: string;
  req: IncomingMessage;
  res: ServerResponse;
  requestId?: string;
}): void {
  writeSSEStream(
    args.bridge,
    args.heartbeatTimer,
    args.blobStore ?? new Map(),
    args.mcpTools ?? [],
    args.modelId,
    args.bridgeKey,
    args.convKey,
    args.completedTurns,
    args.currentTurn,
    args.workspaceContext ?? createWorkspaceContext(),
    args.systemPromptFingerprint ?? fingerprintSystemPrompt(""),
    args.req,
    args.res,
    args.requestId,
  );
}

// ── Tool result resume ──

function handleToolResultResume(
  active: ActiveBridge,
  toolResults: ToolResultInfo[],
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  req: IncomingMessage,
  res: ServerResponse,
  stream: boolean,
  requestId?: string,
): void {
  const { bridge, heartbeatTimer, blobStore, mcpTools, pendingExecs, currentTurn } = active;
  debugLog("tool_resume.start", {
    requestId,
    bridgeKey,
    convKey,
    toolResults,
    pendingExecs,
    currentTurn,
  });

  applyToolResultsToTurn(currentTurn, toolResults);

  const turnResults = getTurnToolCallResults(currentTurn);
  const unresolvedExecs = unresolvedPendingExecs(pendingExecs, turnResults);
  if (unresolvedExecs.length > 0) {
    setActiveBridge(bridgeKey, {
      bridge,
      heartbeatTimer,
      blobStore,
      mcpTools,
      pendingExecs,
      currentTurn,
      convKey,
      workspaceContext,
    });
    debugLog("tool_resume.partial_wait", { requestId, bridgeKey, unresolvedExecs, currentTurn });
    respondWithPendingToolCalls(modelId, unresolvedExecs, stream, res);
    return;
  }

  sendMcpResultsForPendingExecs(
    bridge,
    pendingExecs,
    turnResults,
    "tool_resume.sent_result",
    requestId,
  );

  // Tool results belong to the same user turn that initiated the tool calls.
  // parseMessages keeps tool continuations out of completed history, so completedTurns
  // already reflects the correct history covered before this in-flight turn.
  writeSSEStream(
    bridge,
    heartbeatTimer,
    blobStore,
    mcpTools,
    modelId,
    bridgeKey,
    convKey,
    completedTurns,
    currentTurn,
    workspaceContext,
    systemPromptFingerprint,
    req,
    res,
    requestId,
  );
}

// ── Non-streaming response ──

async function handleNonStreamingResponse(
  payload: CursorRequestPayload,
  accessToken: string,
  modelId: string,
  bridgeKey: string,
  convKey: string,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  workspaceContext: CursorWorkspaceContext,
  systemPromptFingerprint: string,
  req: IncomingMessage,
  res: ServerResponse,
  requestId?: string,
): Promise<void> {
  debugLog("nonstream.start", {
    requestId,
    convKey,
    modelId,
    workspaceContext,
    currentTurn,
    completedTurnCount: completedTurns.length,
  });
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`;
  const created = Math.floor(Date.now() / 1000);

  const { bridge, heartbeatTimer } = startBridge(accessToken, payload.requestBytes);
  let cancelled = false;
  const state = createStreamState();
  const tagFilter = createThinkingTagFilter();
  let fullText = "";
  let nonStreamError: Error | null = null;
  let latestCheckpoint: Uint8Array | null = null;
  const watchdog = createCursorStreamWatchdog({
    eventPrefix: "nonstream",
    requestId,
    bridgeKey,
    convKey,
    bridge,
    heartbeatTimer,
    onTimeout: (_kind, message) => {
      nonStreamError = new Error(message);
    },
  });

  const onClientClose = () => {
    if (cancelled) return;
    debugLog("nonstream.client_close", { requestId, convKey });
    cancelled = true;
    watchdog.stop();
    clearInterval(heartbeatTimer);
    if (bridge.alive) {
      sendCancelAction(bridge);
      bridge.end();
    }
  };
  req.on("close", onClientClose);
  res.on("close", onClientClose);

  return new Promise((resolve) => {
    bridge.onData(
      createConnectFrameParser(
        (messageBytes) => {
          try {
            const serverMessage = fromBinary(AgentServerMessageSchema, messageBytes);
            if (!isCursorServerHeartbeat(serverMessage))
              watchdog.touchUpstream(cursorServerMessageLabel(serverMessage));
            processServerMessage(
              serverMessage,
              payload.blobStore,
              payload.mcpTools,
              workspaceContext,
              (data) => bridge.write(data),
              state,
              (text, isThinking) => {
                if (isThinking) return;
                const { content } = tagFilter.process(text);
                if (content) watchdog.touchVisible("text_delta");
                fullText += content;
                appendAssistantTextToTurn(currentTurn, content);
              },
              () => {},
              (checkpointBytes) => {
                latestCheckpoint = checkpointBytes;
                debugLog("nonstream.checkpoint_buffered", { requestId, convKey, checkpointBytes });
              },
            );
          } catch (err) {
            nonStreamError = err instanceof Error ? err : new Error(String(err));
            console.error(
              "[cursor-provider] Non-stream message processing error:",
              nonStreamError.message,
            );
            debugLog("nonstream.process_error", {
              requestId,
              convKey,
              message: nonStreamError.message,
            });
            watchdog.stop();
            clearInterval(heartbeatTimer);
            if (bridge.alive) {
              sendCancelAction(bridge);
              bridge.end();
            }
          }
        },
        (endStreamBytes) => {
          const endError = parseConnectEndStream(endStreamBytes);
          if (endError) {
            watchdog.stop();
            console.error(
              `[cursor-provider] Cursor non-stream error (${modelId}):`,
              endError.message,
            );
            nonStreamError = endError;
          }
        },
      ),
    );

    bridge.onClose((code) => {
      debugLog("nonstream.bridge_close", {
        requestId,
        convKey,
        code,
        cancelled,
        nonStreamError: nonStreamError?.message,
        currentTurn,
        latestCheckpoint,
      });
      watchdog.stop();
      clearInterval(heartbeatTimer);
      req.removeListener("close", onClientClose);
      res.removeListener("close", onClientClose);

      if (cancelled) {
        if (!res.headersSent) {
          res.writeHead(499, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: { message: "Client closed request", type: "aborted", code: "client_closed" },
            }),
          );
        }
        resolve();
        return;
      }

      if (code !== 0 && !nonStreamError) {
        nonStreamError = new Error("Bridge connection lost");
      }

      if (nonStreamError) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: {
              message: nonStreamError.message,
              type: "upstream_error",
              code: "cursor_error",
            },
          }),
        );
        resolve();
        return;
      }

      const flushed = tagFilter.flush();
      fullText += flushed.content;
      appendAssistantTextToTurn(currentTurn, flushed.content);
      const usage = computeUsage(state);
      const stored = conversationStates.get(convKey);
      if (stored) {
        if (latestCheckpoint) {
          commitStoredCheckpoint(
            stored,
            latestCheckpoint,
            payload.blobStore,
            completedTurns,
            currentTurn,
            systemPromptFingerprint,
          );
          debugLog("nonstream.checkpoint_committed", { requestId, convKey, stored });
        } else {
          mergeBlobStore(stored, payload.blobStore);
        }
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: completionId,
          object: "chat.completion",
          created,
          model: modelId,
          choices: [
            { index: 0, message: { role: "assistant", content: fullText }, finish_reason: "stop" },
          ],
          usage,
        }),
      );
      resolve();
    });
  });
}
