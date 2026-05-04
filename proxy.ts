/**
 * Local OpenAI-compatible proxy: translates /v1/chat/completions to Cursor's gRPC protocol.
 *
 * Based on https://github.com/ephraimduncan/opencode-cursor by Ephraim Duncan.
 * Uses Node's http2 via a child process bridge (h2-bridge.mjs).
 */
import { create, fromBinary, fromJson, type JsonValue, toBinary, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve as pathResolve, dirname, join as pathJoin } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AgentClientMessageSchema,
  AgentRunRequestSchema,
  AgentServerMessageSchema,
  CancelActionSchema,
  ClientHeartbeatSchema,
  ConversationActionSchema,
  ConversationStateStructureSchema,
  ConversationStepSchema,
  AgentConversationTurnStructureSchema,
  ConversationTurnStructureSchema,
  AssistantMessageSchema,

  BackgroundShellSpawnResultSchema,
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
  LsRejectedSchema,
  LsResultSchema,
  McpArgsSchema,
  McpErrorSchema,
  McpResultSchema,
  McpSuccessSchema,
  McpTextContentSchema,
  McpToolCallSchema,
  McpToolDefinitionSchema,
  McpToolResultContentItemSchema,
  McpToolsSchema,
  RequestedModelSchema,
  RequestedModel_ModelParameterbytesSchema,
  ReadRejectedSchema,
  ReadResultSchema,
  RequestContextResultSchema,
  RequestContextSchema,
  RequestContextSuccessSchema,
  SelectedContextSchema,
  SelectedImageSchema,
  SetBlobResultSchema,
  ShellRejectedSchema,
  ShellResultSchema,
  ShellStreamSchema,
  ToolCallSchema,
  UserMessageActionSchema,
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
  type UserMessage,
} from "./proto/agent_pb.js";

const CURSOR_API_URL = "https://api2.cursor.sh";
const CONNECT_END_STREAM_FLAG = 0b00000010;
// Use import.meta.url for bridge path resolution (jiti supports this)
const BRIDGE_PATH = pathResolve(dirname(fileURLToPath(import.meta.url)), "h2-bridge.mjs");

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
  tools?: OpenAIToolDef[];
  tool_choice?: unknown;
  reasoning_effort?: string;
  user?: string;
  pi_session_id?: string;
  cursor_model_id?: string;
  cursor_model_parameters?: CursorModelParameter[];
  cursor_requires_max_mode?: boolean;
  cursor_model_max_mode?: boolean;
}

export interface CursorModelParameter {
  id: string;
  value: string;
}

export interface CursorParameterizedVariant {
  parameters: CursorModelParameter[];
  isMaxMode: boolean;
  isDefaultMaxConfig?: boolean;
  isDefaultNonMaxConfig?: boolean;
  displayName?: string;
  displayNameOutsidePicker?: string;
  variantStringRepresentation?: string;
}

export interface CursorParameterizedModel {
  name: string;
  clientDisplayName?: string;
  serverModelName?: string;
  supportsMaxMode?: boolean;
  supportsNonMaxMode?: boolean;
  supportsImages?: boolean;
  contextTokenLimit?: number;
  contextTokenLimitForMaxMode?: number;
  variants: CursorParameterizedVariant[];
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

interface BridgeHandle {
  proc: Pick<ChildProcess, "kill">;
  readonly alive: boolean;
  write(data: Uint8Array): void;
  end(): void;
  onData(cb: (chunk: Buffer) => void): void;
  onClose(cb: (code: number) => void): void;
}

export type BridgeFactory = (options: SpawnBridgeOptions) => BridgeHandle;

interface ActiveBridge {
  bridge: BridgeHandle;
  heartbeatTimer: ReturnType<typeof setInterval>;
  blobStore: Map<string, Uint8Array>;
  mcpTools: McpToolDefinition[];
  pendingExecs: PendingExec[];
  currentTurn: ParsedTurn;
}

export interface StoredConversation {
  conversationId: string;
  checkpoint: Uint8Array | null;
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
}

export interface ParsedToolResult {
  content: string;
  isError: boolean;
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
const CONVERSATION_TTL_MS = 30 * 60 * 1000;
let bridgeFactory: BridgeFactory = spawnBridge;
let debugRequestCounter = 0;
let debugLogFilePath: string | undefined;

function isProxyDebugEnabled(): boolean {
  const raw = process.env.PI_CURSOR_PROVIDER_DEBUG?.trim().toLowerCase();
  return !!raw && raw !== "0" && raw !== "false" && raw !== "off";
}

function truncateDebugString(value: string, max = 4000): string {
  return value.length > max ? `${value.slice(0, max)}…<truncated ${value.length - max} chars>` : value;
}

function sanitizeForDebug(value: unknown): unknown {
  if (value == null) return value;
  if (typeof value === "string") return truncateDebugString(value);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
    const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
    return {
      __type: value instanceof Uint8Array ? "Uint8Array" : "Buffer",
      byteLength: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex").slice(0, 16),
    };
  }
  if (Array.isArray(value)) return value.map((item) => sanitizeForDebug(item));
  if (value instanceof Map) {
    return {
      __type: "Map",
      size: value.size,
      entries: Array.from(value.entries()).slice(0, 20).map(([k, v]) => [sanitizeForDebug(k), sanitizeForDebug(v)]),
    };
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).map(([key, inner]) => {
      if (key === "accessToken") return [key, "<redacted>"] as const;
      if (key === "data" && typeof inner === "string") return [key, `<redacted base64 ${inner.length} chars>`] as const;
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
    ...(data ? sanitizeForDebug(data) : {}),
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
};

export function setBridgeFactoryForTests(factory?: BridgeFactory): void {
  bridgeFactory = factory ?? spawnBridge;
}

let proxyServer: ReturnType<typeof createServer> | undefined;
let proxyPort: number | undefined;
let proxyAccessTokenProvider: (() => Promise<string>) | undefined;

// ── Bridge spawn ──

function lpEncode(data: Uint8Array): Buffer {
  const buf = Buffer.alloc(4 + data.length);
  buf.writeUInt32BE(data.length, 0);
  buf.set(data, 4);
  return buf;
}

function frameConnectMessage(data: Uint8Array, flags = 0): Buffer {
  const frame = Buffer.alloc(5 + data.length);
  frame[0] = flags;
  frame.writeUInt32BE(data.length, 1);
  frame.set(data, 5);
  return frame;
}

interface SpawnBridgeOptions {
  accessToken: string;
  rpcPath: string;
  url?: string;
  unary?: boolean;
}

function spawnBridge(options: SpawnBridgeOptions): BridgeHandle {
  debugLog("bridge.spawn", {
    rpcPath: options.rpcPath,
    url: options.url ?? CURSOR_API_URL,
    unary: options.unary ?? false,
    cursorClientVersion: process.env.PI_CURSOR_CLIENT_VERSION || "cli-2026.05.01-eea359f",
  });
  const proc = spawn("node", [BRIDGE_PATH], {
    stdio: ["pipe", "pipe", "ignore"],
  });

  const config = JSON.stringify({
    accessToken: options.accessToken,
    url: options.url ?? CURSOR_API_URL,
    path: options.rpcPath,
    unary: options.unary ?? false,
  });
  proc.stdin!.write(lpEncode(new TextEncoder().encode(config)));

  const cbs = {
    data: null as ((chunk: Buffer) => void) | null,
    close: null as ((code: number) => void) | null,
  };

  let exited = false;
  let exitCode = 1;

  let pending = Buffer.alloc(0);
  proc.stdout!.on("data", (chunk: Buffer) => {
    pending = Buffer.concat([pending, chunk]);
    while (pending.length >= 4) {
      const len = pending.readUInt32BE(0);
      if (pending.length < 4 + len) break;
      const payload = pending.subarray(4, 4 + len);
      pending = pending.subarray(4 + len);
      cbs.data?.(Buffer.from(payload));
    }
  });

  proc.on("exit", (code) => {
    exited = true;
    exitCode = code ?? 1;
    debugLog("bridge.exit", { rpcPath: options.rpcPath, exitCode });
    cbs.close?.(exitCode);
  });

  return {
    proc,
    get alive() { return !exited; },
    write(data: Uint8Array) {
      try { proc.stdin!.write(lpEncode(data)); } catch {}
    },
    end() {
      try {
        proc.stdin!.write(lpEncode(new Uint8Array(0)));
        proc.stdin!.end();
      } catch {}
    },
    onData(cb: (chunk: Buffer) => void) { cbs.data = cb; },
    onClose(cb: (code: number) => void) {
      if (exited) {
        queueMicrotask(() => cb(exitCode));
      } else {
        cbs.close = cb;
      }
    },
  };
}

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
    const timeout = timeoutMs > 0
      ? setTimeout(() => {
          timedOut = true;
          try { bridge.proc.kill(); } catch {}
        }, timeoutMs)
      : undefined;

    bridge.onData((chunk) => { chunks.push(Buffer.from(chunk)); });
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
let cachedParameterizedModels: { tokenHash: string; models: CursorParameterizedModel[] } | null = null;

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
          try { decoded = fromBinary(GetUsableModelsResponseSchema, body); } catch {}
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
    console.error("[cursor-provider] Model discovery failed:", err instanceof Error ? err.message : err);
  }
  console.warn("[cursor-provider] Model discovery returned no models");
  return [];
}

function decodeConnectUnaryBody(payload: Uint8Array): Uint8Array | null {
  if (payload.length < 5) return null;
  let offset = 0;
  while (offset + 5 <= payload.length) {
    const flags = payload[offset]!;
    const view = new DataView(payload.buffer, payload.byteOffset + offset, payload.byteLength - offset);
    const messageLength = view.getUint32(1, false);
    const frameEnd = offset + 5 + messageLength;
    if (frameEnd > payload.length) return null;
    if ((flags & 0b0000_0001) !== 0) return null;
    if ((flags & 0b0000_0010) === 0) return payload.subarray(offset + 5, frameEnd);
    offset = frameEnd;
  }
  return null;
}

function encodeVarint(value: number): number[] {
  const out: number[] = [];
  let v = value >>> 0;
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  out.push(v);
  return out;
}

function encodeBoolField(fieldNo: number, value: boolean): number[] {
  return [...encodeVarint(fieldNo << 3), value ? 1 : 0];
}

function encodeAvailableModelsRequest(): Uint8Array {
  // aiserver.v1.AvailableModelsRequest {
  //   optional bool use_model_parameters = 5;
  //   optional bool do_not_use_markdown = 7;
  // }
  return new Uint8Array([
    ...encodeBoolField(5, true),
    ...encodeBoolField(7, true),
  ]);
}

interface WireReader {
  bytes: Uint8Array;
  offset: number;
}

function readVarint(reader: WireReader): number {
  let result = 0;
  let shift = 0;
  while (reader.offset < reader.bytes.length) {
    const byte = reader.bytes[reader.offset++]!;
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return result >>> 0;
    shift += 7;
    if (shift > 35) throw new Error("varint too long");
  }
  throw new Error("unexpected EOF while reading varint");
}

function readLengthDelimited(reader: WireReader): Uint8Array {
  const length = readVarint(reader);
  const end = reader.offset + length;
  if (end > reader.bytes.length) throw new Error("length-delimited field exceeds buffer");
  const value = reader.bytes.subarray(reader.offset, end);
  reader.offset = end;
  return value;
}

function skipWireField(reader: WireReader, wireType: number): void {
  switch (wireType) {
    case 0:
      readVarint(reader);
      return;
    case 1:
      reader.offset += 8;
      return;
    case 2:
      readLengthDelimited(reader);
      return;
    case 5:
      reader.offset += 4;
      return;
    default:
      throw new Error(`unsupported wire type ${wireType}`);
  }
}

function decodeString(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

function decodeModelParameter(bytes: Uint8Array): CursorModelParameter {
  const reader: WireReader = { bytes, offset: 0 };
  const parameter: CursorModelParameter = { id: "", value: "" };
  while (reader.offset < bytes.length) {
    const tag = readVarint(reader);
    const fieldNo = tag >>> 3;
    const wireType = tag & 0x7;
    if (fieldNo === 1 && wireType === 2) parameter.id = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 2 && wireType === 2) parameter.value = decodeString(readLengthDelimited(reader));
    else skipWireField(reader, wireType);
  }
  return parameter;
}

function decodeParameterizedVariant(bytes: Uint8Array): CursorParameterizedVariant {
  const reader: WireReader = { bytes, offset: 0 };
  const variant: CursorParameterizedVariant = { parameters: [], isMaxMode: false };
  while (reader.offset < bytes.length) {
    const tag = readVarint(reader);
    const fieldNo = tag >>> 3;
    const wireType = tag & 0x7;
    if (fieldNo === 1 && wireType === 2) variant.parameters.push(decodeModelParameter(readLengthDelimited(reader)));
    else if (fieldNo === 2 && wireType === 2) variant.displayName = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 8 && wireType === 2) variant.displayNameOutsidePicker = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 3 && wireType === 0) variant.isMaxMode = readVarint(reader) !== 0;
    else if (fieldNo === 4 && wireType === 0) variant.isDefaultMaxConfig = readVarint(reader) !== 0;
    else if (fieldNo === 5 && wireType === 0) variant.isDefaultNonMaxConfig = readVarint(reader) !== 0;
    else if (fieldNo === 9 && wireType === 2) variant.variantStringRepresentation = decodeString(readLengthDelimited(reader));
    else skipWireField(reader, wireType);
  }
  return variant;
}

function decodeParameterizedModel(bytes: Uint8Array): CursorParameterizedModel {
  const reader: WireReader = { bytes, offset: 0 };
  const model: CursorParameterizedModel = { name: "", variants: [] };
  while (reader.offset < bytes.length) {
    const tag = readVarint(reader);
    const fieldNo = tag >>> 3;
    const wireType = tag & 0x7;
    if (fieldNo === 1 && wireType === 2) model.name = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 10 && wireType === 0) model.supportsImages = readVarint(reader) !== 0;
    else if (fieldNo === 14 && wireType === 0) model.supportsMaxMode = readVarint(reader) !== 0;
    else if (fieldNo === 19 && wireType === 0) model.supportsNonMaxMode = readVarint(reader) !== 0;
    else if (fieldNo === 15 && wireType === 0) model.contextTokenLimit = readVarint(reader);
    else if (fieldNo === 16 && wireType === 0) model.contextTokenLimitForMaxMode = readVarint(reader);
    else if (fieldNo === 17 && wireType === 2) model.clientDisplayName = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 18 && wireType === 2) model.serverModelName = decodeString(readLengthDelimited(reader));
    else if (fieldNo === 30 && wireType === 2) model.variants.push(decodeParameterizedVariant(readLengthDelimited(reader)));
    else skipWireField(reader, wireType);
  }
  return model;
}

function decodeAvailableModelsResponse(bytes: Uint8Array): CursorParameterizedModel[] {
  const reader: WireReader = { bytes, offset: 0 };
  const models: CursorParameterizedModel[] = [];
  while (reader.offset < bytes.length) {
    const tag = readVarint(reader);
    const fieldNo = tag >>> 3;
    const wireType = tag & 0x7;
    if (fieldNo === 2 && wireType === 2) {
      const model = decodeParameterizedModel(readLengthDelimited(reader));
      if (model.name) models.push(model);
    } else {
      skipWireField(reader, wireType);
    }
  }
  return models;
}

export async function getCursorParameterizedModels(apiKey: string): Promise<CursorParameterizedModel[]> {
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
    console.error("[cursor-provider] Parameterized model discovery failed:", err instanceof Error ? err.message : err);
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

export async function startProxy(
  getAccessToken: () => Promise<string>,
): Promise<number> {
  proxyAccessTokenProvider = getAccessToken;
  if (proxyServer && proxyPort) return proxyPort;

  return new Promise((resolve, reject) => {
    const server = createServer(async (req, res) => {
      const url = new URL(req.url ?? "/", `http://localhost`);
      const requestId = nextDebugRequestId();
      debugLog("http.request", { requestId, method: req.method, pathname: url.pathname, headers: req.headers });

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
          await handleChatCompletion(parsed, accessToken, req, res, requestId);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          debugLog("http.chat.error", { requestId, message, stack: err instanceof Error ? err.stack : undefined });
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message, type: "server_error", code: "internal_error" } }));
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
        debugLog("proxy.start", { port: proxyPort, debugLogFile: isProxyDebugEnabled() ? getDebugLogFilePath() : undefined });
        resolve(proxyPort);
      } else {
        reject(new Error("Failed to bind proxy"));
      }
    });
  });
}

export function cleanupAllSessionState(): void {
  debugLog("session.cleanup_all", { activeBridgeCount: activeBridges.size, conversationCount: conversationStates.size });
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
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
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

async function handleChatCompletion(
  body: ChatCompletionRequest,
  accessToken: string,
  req: IncomingMessage,
  res: ServerResponse,
  requestId: string,
): Promise<void> {
  const { systemPrompt, userText, userImages, turns, toolResults } = parseMessages(body.messages);
  const modelId = resolveRequestedModelId(body.model, body.reasoning_effort, body.cursor_model_id);
  if (body.reasoning_effort && !body.cursor_model_id && !body.cursor_model_parameters) {
    debugLog("model_routing.fallback_suffix_generation", {
      requestId,
      model: body.model,
      reasoning_effort: body.reasoning_effort,
      resolvedModelId: modelId,
    });
  }
  const maxMode = typeof body.cursor_model_max_mode === "boolean"
    ? body.cursor_model_max_mode
    : body.cursor_requires_max_mode === true;
  const tools = body.tools ?? [];

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

  if (!userText && toolResults.length === 0) {
    debugLog("chat.no_user_message", { requestId, messages: body.messages });
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: { message: "No user message found", type: "invalid_request_error" } }));
    return;
  }

  const sessionId = derivePiSessionId(body);
  const bridgeKey = deriveBridgeKey(body.messages, sessionId);
  const convKey = deriveConversationKey(body.messages, sessionId);
  const activeBridge = activeBridges.get(bridgeKey);
  debugLog("chat.session_keys", {
    requestId,
    sessionId,
    bridgeKey,
    convKey,
    hasActiveBridge: !!activeBridge,
  });

  if (activeBridge && toolResults.length > 0) {
    debugLog("chat.resume_tool_results", { requestId, bridgeKey, toolResults, pendingExecs: activeBridge.pendingExecs });
    activeBridges.delete(bridgeKey);
    if (activeBridge.bridge.alive) {
      handleToolResultResume(activeBridge, toolResults, modelId, bridgeKey, convKey, turns, req, res, body.stream !== false, requestId);
      return;
    }
    clearInterval(activeBridge.heartbeatTimer);
    activeBridge.bridge.end();
  }

  if (activeBridge && activeBridges.has(bridgeKey)) {
    clearInterval(activeBridge.heartbeatTimer);
    activeBridge.bridge.end();
    activeBridges.delete(bridgeKey);
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

  const mcpTools = buildMcpToolDefinitions(tools);
  const effectiveUserText = userText || (toolResults.length > 0
    ? toolResults.map((r) => r.content).join("\n")
    : "");
  const effectiveUserImages = userText || userImages.length > 0 ? userImages : [];
  if (!stored.checkpoint) {
    debugLog("chat.no_checkpoint", { requestId, convKey, conversationId: stored.conversationId });
  }
  const payload = buildCursorRequest(
    modelId, systemPrompt, effectiveUserText, turns,
    stored.conversationId, stored.checkpoint, stored.blobStore, maxMode, body.cursor_model_parameters, mcpTools, effectiveUserImages,
  );
  debugLog("chat.cursor_request", {
    requestId,
    conversationId: stored.conversationId,
    effectiveUserText,
    turnCount: turns.length,
    hasCheckpoint: !!stored.checkpoint,
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
    await handleNonStreamingResponse(payload, accessToken, modelId, convKey, turns, currentTurn, req, res, requestId);
  } else {
    debugLog("chat.dispatch_stream", { requestId, bridgeKey, convKey });
    handleStreamingResponse(payload, accessToken, modelId, bridgeKey, convKey, turns, currentTurn, req, res, requestId);
  }
}

// ── Message parsing ──

function textContent(content: OpenAIMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content.filter((p) => p.type === "text" && p.text).map((p) => p.text!).join("\n");
}

function decodeBase64Image(data: string, mimeType: string): ParsedImageContent | undefined {
  const normalizedMimeType = mimeType.trim().toLowerCase();
  if (!normalizedMimeType.startsWith("image/")) return undefined;
  const base64 = data.replace(/\s/g, "");
  if (!base64) return undefined;
  const bytes = Buffer.from(base64, "base64");
  if (bytes.length === 0) return undefined;
  return { data: new Uint8Array(bytes), mimeType: normalizedMimeType };
}

function parseImageDataUrl(url: string): ParsedImageContent | undefined {
  const match = url.match(/^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/is);
  if (!match) return undefined;
  return decodeBase64Image(match[2]!, match[1]!);
}

function imageContent(content: OpenAIMessage["content"]): ParsedImageContent[] {
  if (content == null || typeof content === "string") return [];
  const images: ParsedImageContent[] = [];
  for (const part of content) {
    if (part.type === "image_url" && part.image_url?.url) {
      const image = parseImageDataUrl(part.image_url.url);
      if (image) images.push(image);
    } else if (part.type === "image" && part.data && part.mimeType) {
      const image = decodeBase64Image(part.data, part.mimeType);
      if (image) images.push(image);
    }
  }
  return images;
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

function stripTurnRuntimeState(turn: ParsedTurn & {
  toolCallById?: Map<string, ParsedToolCallStep>;
  sawToolResult?: boolean;
  sawAssistantAfterToolResult?: boolean;
}): ParsedTurn {
  return {
    userText: turn.userText,
    steps: turn.steps,
    ...(turn.userImages?.length ? { userImages: turn.userImages } : {}),
  };
}

export function parseMessages(messages: OpenAIMessage[]): ParsedMessages {
  let systemPrompt = "You are a helpful assistant.";
  const turns: ParsedTurn[] = [];

  debugLog("parse_messages.start", { messages });

  const systemParts = messages.filter((m) => m.role === "system").map((m) => textContent(m.content));
  if (systemParts.length > 0) systemPrompt = systemParts.join("\n");

  const nonSystem = messages.filter((m) => m.role !== "system");
  let currentTurn: (ParsedTurn & {
    toolCallById: Map<string, ParsedToolCallStep>;
    sawToolResult: boolean;
    sawAssistantAfterToolResult: boolean;
  }) | null = null;

  const finalizeCurrentTurn = () => {
    if (!currentTurn) return;
    turns.push(stripTurnRuntimeState(currentTurn));
    currentTurn = null;
  };

  for (const msg of nonSystem) {
    if (msg.role === "user") {
      finalizeCurrentTurn();
      const userImages = imageContent(msg.content);
      currentTurn = {
        userText: textContent(msg.content),
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
      const content = textContent(msg.content);
      const existing = toolCallId ? currentTurn.toolCallById.get(toolCallId) : undefined;
      if (existing) {
        existing.result = { content, isError: false };
      } else {
        const step: ParsedToolCallStep = {
          kind: "toolCall",
          toolCallId,
          toolName: "",
          arguments: {},
          result: { content, isError: false },
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
          .map((step) => ({ toolCallId: step.toolCallId, content: step.result!.content }));
      }
    } else {
      turns.push(stripTurnRuntimeState(currentTurn));
    }
  }

  const parsed = { systemPrompt, userText, userImages, turns, toolResults };
  debugLog("parse_messages.end", parsed);
  return parsed;
}

// ── Tool definitions ──

function buildMcpToolDefinitions(tools: OpenAIToolDef[]): McpToolDefinition[] {
  return tools.map((t) => {
    const fn = t.function;
    const jsonSchema: JsonValue = fn.parameters && typeof fn.parameters === "object"
      ? (fn.parameters as JsonValue)
      : { type: "object", properties: {}, required: [] };
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

function encodeMcpArgValue(value: unknown): Uint8Array {
  try {
    return toBinary(ValueSchema, fromJson(ValueSchema, value as JsonValue));
  } catch {
    return new TextEncoder().encode(String(value));
  }
}

function encodeMcpArgsMap(args: Record<string, unknown>): Record<string, Uint8Array> {
  const encoded: Record<string, Uint8Array> = {};
  for (const [key, value] of Object.entries(args)) encoded[key] = encodeMcpArgValue(value);
  return encoded;
}

// No generated schema for selectedContextBlob; emit raw wire format for the two
// fields Cursor actually reads: field 1 (repeated bytes) rootPromptMessagesJson
// refs, field 22 (string) clientName. blobId.length < 128 (SHA256 = 32 bytes).
function buildSelectedContextBlob(rootPromptBlobIds: Uint8Array[], clientName: string): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const blobId of rootPromptBlobIds) {
    parts.push(new Uint8Array([0x0A, blobId.length, ...blobId]));
  }
  const clientBytes = new TextEncoder().encode(clientName);
  parts.push(new Uint8Array([0xB2, 0x01, clientBytes.length, ...clientBytes]));
  const total = parts.reduce((n, p) => n + p.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) { result.set(p, offset); offset += p.length; }
  return result;
}

function storeAsBlob(data: Uint8Array, blobStore: Map<string, Uint8Array>): Uint8Array {
  const id = new Uint8Array(createHash("sha256").update(data).digest());
  blobStore.set(Buffer.from(id).toString("hex"), data);
  return id;
}

function createSelectedImages(images: ParsedImageContent[]) {
  // Matches Cursor CLI's ACP image path for inline image data:
  // new SelectedImage({ dataOrBlobId: { case: "data", value }, uuid, mimeType })
  return images.map((image) => create(SelectedImageSchema, {
    uuid: crypto.randomUUID(),
    mimeType: image.mimeType,
    dataOrBlobId: { case: "data", value: image.data },
  }));
}

function createUserMessage(text: string, selectedContextBlob: Uint8Array, images: ParsedImageContent[] = []): UserMessage {
  const messageId = crypto.randomUUID();
  return create(UserMessageSchema, {
    text,
    messageId,
    selectedContext: create(SelectedContextSchema, {
      selectedImages: createSelectedImages(images),
    }),
    mode: 1,
    selectedContextBlob,
    correlationId: messageId,
  });
}

function buildTurnStepBytes(step: ParsedTurnStep): Uint8Array {
  if (step.kind === "assistantText") {
    return toBinary(ConversationStepSchema, create(ConversationStepSchema, {
      message: {
        case: "assistantMessage",
        value: create(AssistantMessageSchema, { text: step.text }),
      },
    }));
  }

  const toolName = step.toolName || "tool";
  const mcpToolCall = create(McpToolCallSchema, {
    args: create(McpArgsSchema, {
      name: toolName,
      args: encodeMcpArgsMap(step.arguments),
      toolCallId: step.toolCallId,
      providerIdentifier: "pi",
      toolName,
    }),
    ...(step.result && {
      result: create(McpResultSchema, {
        result: step.result.isError
          ? {
              case: "error",
              value: create(McpErrorSchema, { error: step.result.content }),
            }
          : {
              case: "success",
              value: create(McpSuccessSchema, {
                content: [
                  create(McpToolResultContentItemSchema, {
                    content: {
                      case: "text",
                      value: create(McpTextContentSchema, { text: step.result.content }),
                    },
                  }),
                ],
                isError: false,
              }),
            },
      }),
    }),
  });

  return toBinary(ConversationStepSchema, create(ConversationStepSchema, {
    message: {
      case: "toolCall",
      value: create(ToolCallSchema, {
        tool: {
          case: "mcpToolCall",
          value: mcpToolCall,
        },
      }),
    },
  }));
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
): CursorRequestPayload {
  debugLog("cursor_request.build.start", {
    modelId,
    systemPrompt,
    userText,
    turns,
    conversationId,
    checkpoint,
    existingBlobStore,
    maxMode,
    cursorModelParameters,
    mcpToolCount: mcpTools.length,
    userImageCount: userImages.length,
  });
  const blobStore = new Map<string, Uint8Array>(existingBlobStore ?? []);

  const systemBytes = new TextEncoder().encode(JSON.stringify({ role: "system", content: systemPrompt }));
  const systemBlobId = storeAsBlob(systemBytes, blobStore);
  const selectedCtxBlob = storeAsBlob(
    buildSelectedContextBlob([systemBlobId], "pi"), blobStore,
  );

  let conversationState;
  if (checkpoint) {
    conversationState = fromBinary(ConversationStateStructureSchema, checkpoint);
  } else {
    const turnBlobIds: Uint8Array[] = [];
    for (const turn of turns) {
      const userMsg = createUserMessage(turn.userText, selectedCtxBlob, turn.userImages ?? []);
      const userMsgBlobId = storeAsBlob(toBinary(UserMessageSchema, userMsg), blobStore);
      const stepBlobIds = turn.steps.map(s => storeAsBlob(buildTurnStepBytes(s), blobStore));

      const agentTurn = create(AgentConversationTurnStructureSchema, {
        userMessage: userMsgBlobId,
        steps: stepBlobIds,
        requestId: crypto.randomUUID(),
      });
      const turnStructure = create(ConversationTurnStructureSchema, {
        turn: { case: "agentConversationTurn", value: agentTurn },
      });
      turnBlobIds.push(storeAsBlob(toBinary(ConversationTurnStructureSchema, turnStructure), blobStore));
    }

    conversationState = create(ConversationStateStructureSchema, {
      rootPromptMessagesJson: [systemBlobId],
      turns: turnBlobIds,
      todos: [],
      pendingToolCalls: [],
      previousWorkspaceUris: [`file://${process.cwd()}`],
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
  }

  const userMessage = createUserMessage(userText, selectedCtxBlob, userImages);
  const action = create(ConversationActionSchema, {
    action: { case: "userMessageAction", value: create(UserMessageActionSchema, { userMessage }) },
  });
  // Cursor's newer request path uses requestedModel instead of legacy modelDetails.
  // Some Cursor models (for example GPT-5.5) use requestedModel.parameters
  // for context/reasoning/fast instead of encoding everything in the model ID.
  // Max Mode is routed from model metadata for parameterized variants.
  debugLog("cursor_request.requested_model", {
    modelId,
    maxMode,
    parameters: cursorModelParameters,
  });
  const parameters = cursorModelParameters.map((parameter) => create(RequestedModel_ModelParameterbytesSchema, parameter));
  const requestedModel = create(RequestedModelSchema, { modelId, maxMode, parameters });
  const runRequest = create(AgentRunRequestSchema, {
    conversationState,
    action,
    requestedModel,
    conversationId,
    mcpTools: create(McpToolsSchema, { mcpTools }),
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "runRequest", value: runRequest },
  });

  const payload = { requestBytes: toBinary(AgentClientMessageSchema, clientMessage), blobStore, mcpTools };
  debugLog("cursor_request.build.end", payload);
  return payload;
}

// ── Server message processing ──

function processServerMessage(
  msg: AgentServerMessage,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
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
    }
  } else if (msgCase === "kvServerMessage") {
    handleKvMessage(msg.message.value as KvServerMessage, blobStore, sendFrame);
  } else if (msgCase === "execServerMessage") {
    handleExecMessage(msg.message.value as ExecServerMessage, mcpTools, sendFrame, onMcpExec);
  } else if (msgCase === "conversationCheckpointUpdate") {
    const stateStructure = msg.message.value as ConversationStateStructure;
    if ((stateStructure as any).tokenDetails) {
      state.totalTokens = (stateStructure as any).tokenDetails.usedTokens;
    }
    if (onCheckpoint) {
      onCheckpoint(toBinary(ConversationStateStructureSchema, stateStructure));
    }
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
    sendKvResponse(kvMsg, "getBlobResult", create(GetBlobResultSchema, blobData ? { blobData } : {}), sendFrame);
  } else if (kvCase === "setBlobArgs") {
    const { blobId, blobData } = (kvMsg as any).message.value;
    blobStore.set(Buffer.from(blobId).toString("hex"), blobData);
    sendKvResponse(kvMsg, "setBlobResult", create(SetBlobResultSchema, {}), sendFrame);
  }
}

function handleExecMessage(
  execMsg: ExecServerMessage,
  mcpTools: McpToolDefinition[],
  sendFrame: (data: Uint8Array) => void,
  onMcpExec: (exec: PendingExec) => void,
): void {
  const execCase = (execMsg as any).message.case;
  const REJECT_REASON = "Tool not available in this environment. Use the MCP tools provided instead.";

  if (execCase === "requestContextArgs") {
    const requestContext = create(RequestContextSchema, {
      rules: [], repositoryInfo: [], tools: mcpTools, gitRepos: [],
      projectLayouts: [], mcpInstructions: [], fileContents: {}, customSubagents: [],
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

  // Reject native Cursor tools so model falls back to MCP tools
  if (execCase === "readArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "readResult", create(ReadResultSchema, {
      result: { case: "rejected", value: create(ReadRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "lsArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "lsResult", create(LsResultSchema, {
      result: { case: "rejected", value: create(LsRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "grepArgs") {
    sendExecResult(execMsg, "grepResult", create(GrepResultSchema, {
      result: { case: "error", value: create(GrepErrorSchema, { error: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "writeArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "writeResult", create(WriteResultSchema, {
      result: { case: "rejected", value: create(WriteRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "deleteArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "deleteResult", create(DeleteResultSchema, {
      result: { case: "rejected", value: create(DeleteRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "shellArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "shellResult", create(ShellResultSchema, {
      result: { case: "rejected", value: create(ShellRejectedSchema, {
        command: args.command ?? "", workingDirectory: args.workingDirectory ?? "",
        reason: REJECT_REASON, isReadonly: false,
      }) },
    }), sendFrame);
    return;
  }
  if (execCase === "shellStreamArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "shellStream", create(ShellStreamSchema, {
      event: { case: "rejected", value: create(ShellRejectedSchema, {
        command: args.command ?? "", workingDirectory: args.workingDirectory ?? "",
        reason: REJECT_REASON, isReadonly: false,
      }) },
    }), sendFrame);
    return;
  }
  if (execCase === "backgroundShellSpawnArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "backgroundShellSpawnResult", create(BackgroundShellSpawnResultSchema, {
      result: { case: "rejected", value: create(ShellRejectedSchema, {
        command: args.command ?? "", workingDirectory: args.workingDirectory ?? "",
        reason: REJECT_REASON, isReadonly: false,
      }) },
    }), sendFrame);
    return;
  }
  if (execCase === "writeShellStdinArgs") {
    sendExecResult(execMsg, "writeShellStdinResult", create(WriteShellStdinResultSchema, {
      result: { case: "error", value: create(WriteShellStdinErrorSchema, { error: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "fetchArgs") {
    const args = (execMsg as any).message.value;
    sendExecResult(execMsg, "fetchResult", create(FetchResultSchema, {
      result: { case: "error", value: create(FetchErrorSchema, { url: args.url ?? "", error: REJECT_REASON }) },
    }), sendFrame);
    return;
  }
  if (execCase === "diagnosticsArgs") {
    sendExecResult(execMsg, "diagnosticsResult", create(DiagnosticsResultSchema, {}), sendFrame);
    return;
  }

  // Unknown exec types
  const miscCaseMap: Record<string, string> = {
    listMcpResourcesExecArgs: "listMcpResourcesExecResult",
    readMcpResourceExecArgs: "readMcpResourceExecResult",
    recordScreenArgs: "recordScreenResult",
    computerUseArgs: "computerUseResult",
  };
  const resultCase = miscCaseMap[execCase as string];
  if (resultCase) {
    sendExecResult(execMsg, resultCase, create(McpResultSchema, {}), sendFrame);
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

export function derivePiSessionId(body: Pick<ChatCompletionRequest, "pi_session_id" | "user">): string | undefined {
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
  return createHash("sha256").update(`bridge:${firstUserText.slice(0, 200)}`).digest("hex").slice(0, 16);
}

export function deriveConversationKey(messages: OpenAIMessage[], sessionId?: string): string {
  if (sessionId) return deriveConversationKeyFromSessionId(sessionId);
  const firstUserMsg = messages.find((m) => m.role === "user");
  const firstUserText = firstUserMsg ? textContent(firstUserMsg.content) : "";
  return createHash("sha256").update(`conv:${firstUserText.slice(0, 200)}`).digest("hex").slice(0, 16);
}

export function cleanupSessionState(sessionId?: string): void {
  if (!sessionId) return;
  const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
  const convKey = deriveConversationKeyFromSessionId(sessionId);
  const active = activeBridges.get(bridgeKey);
  debugLog("session.cleanup", { sessionId, bridgeKey, convKey, hasActiveBridge: !!active, hadConversation: conversationStates.has(convKey) });
  if (active) cleanupBridge(active.bridge, active.heartbeatTimer, bridgeKey);
  conversationStates.delete(convKey);
}

export function deterministicConversationId(convKey: string): string {
  const hex = createHash("sha256").update(`cursor-conv-id:${convKey}`).digest("hex").slice(0, 32);
  return [
    hex.slice(0, 8), hex.slice(8, 12),
    `4${hex.slice(13, 16)}`,
    `${(0x8 | (parseInt(hex[16], 16) & 0x3)).toString(16)}${hex.slice(17, 20)}`,
    hex.slice(20, 32),
  ].join("-");
}

// ── Thinking tag filter ──

const THINKING_TAG_NAMES = ['think', 'thinking', 'reasoning', 'thought', 'think_intent'];
const MAX_THINKING_TAG_LEN = 16;

function createThinkingTagFilter() {
  let buffer = '';
  let inThinking = false;
  return {
    process(text: string) {
      const input = buffer + text;
      buffer = '';
      let content = '';
      let reasoning = '';
      let lastIdx = 0;
      const re = new RegExp(`<(/?)(?:${THINKING_TAG_NAMES.join('|')})\\s*>`, 'gi');
      let match: RegExpExecArray | null;
      while ((match = re.exec(input)) !== null) {
        const before = input.slice(lastIdx, match.index);
        if (inThinking) reasoning += before; else content += before;
        inThinking = match[1] !== '/';
        lastIdx = re.lastIndex;
      }
      const rest = input.slice(lastIdx);
      const ltPos = rest.lastIndexOf('<');
      if (ltPos >= 0 && rest.length - ltPos < MAX_THINKING_TAG_LEN && /^<\/?[a-z_]*$/i.test(rest.slice(ltPos))) {
        buffer = rest.slice(ltPos);
        const before = rest.slice(0, ltPos);
        if (inThinking) reasoning += before; else content += before;
      } else {
        if (inThinking) reasoning += rest; else content += rest;
      }
      return { content, reasoning };
    },
    flush() {
      const b = buffer;
      buffer = '';
      if (!b) return { content: '', reasoning: '' };
      return inThinking ? { content: '', reasoning: b } : { content: b, reasoning: '' };
    },
  };
}

// ── Connect frame parser ──

function createConnectFrameParser(
  onMessage: (bytes: Uint8Array) => void,
  onEndStream: (bytes: Uint8Array) => void,
): (incoming: Buffer) => void {
  let pending = Buffer.alloc(0);
  return (incoming: Buffer) => {
    pending = Buffer.concat([pending, incoming]);
    while (pending.length >= 5) {
      const flags = pending[0]!;
      const msgLen = pending.readUInt32BE(1);
      if (pending.length < 5 + msgLen) break;
      const messageBytes = pending.subarray(5, 5 + msgLen);
      pending = pending.subarray(5 + msgLen);
      if (flags & CONNECT_END_STREAM_FLAG) onEndStream(messageBytes);
      else onMessage(messageBytes);
    }
  };
}

function parseConnectEndStream(data: Uint8Array): Error | null {
  try {
    const payload = JSON.parse(new TextDecoder().decode(data));
    const error = payload?.error;
    if (error) return new Error(`Connect error ${error.code ?? "unknown"}: ${error.message ?? "Unknown error"}`);
    return null;
  } catch {
    return new Error("Failed to parse Connect end stream");
  }
}

function makeHeartbeatBytes(): Uint8Array {
  const heartbeat = create(AgentClientMessageSchema, {
    message: { case: "clientHeartbeat", value: create(ClientHeartbeatSchema, {}) },
  });
  return frameConnectMessage(toBinary(AgentClientMessageSchema, heartbeat));
}

function computeUsage(state: StreamState) {
  const completion_tokens = state.outputTokens;
  const total_tokens = state.totalTokens || completion_tokens;
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
      "Connection": "keep-alive",
    });
    for (const toolCall of toolCalls) {
      res.write(`data: ${JSON.stringify({
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model: modelId,
        choices: [{ index: 0, delta: { tool_calls: [toolCall] }, finish_reason: null }],
      })}\n\n`);
    }
    res.write(`data: ${JSON.stringify({
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model: modelId,
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    })}\n\n`);
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    id: completionId,
    object: "chat.completion",
    created,
    model: modelId,
    choices: [{
      index: 0,
      message: { role: "assistant", content: null, tool_calls: toolCalls },
      finish_reason: "tool_calls",
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  }));
}

// ── Streaming response ──

function startBridge(accessToken: string, requestBytes: Uint8Array) {
  const bridge = bridgeFactory({ accessToken, rpcPath: "/agent.v1.AgentService/Run" });
  debugLog("bridge.start_run", { requestBytes });
  bridge.write(frameConnectMessage(requestBytes));
  const heartbeatTimer = setInterval(() => bridge.write(makeHeartbeatBytes()), 5_000);
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
  req: IncomingMessage,
  res: ServerResponse,
  requestId: string,
): void {
  debugLog("stream.start", { requestId, bridgeKey, convKey, modelId });
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
    req,
    res,
    requestId,
  );
}

function sendCancelAction(
  bridge: BridgeHandle,
): void {
  debugLog("bridge.cancel_action", {});
  const action = create(ConversationActionSchema, {
    action: { case: "cancelAction", value: create(CancelActionSchema, {}) },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "conversationAction", value: action },
  });
  bridge.write(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
}

function cleanupBridge(
  bridge: BridgeHandle,
  heartbeatTimer: ReturnType<typeof setInterval>,
  bridgeKey: string,
): void {
  debugLog("bridge.cleanup", { bridgeKey, alive: bridge.alive });
  clearInterval(heartbeatTimer);
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
  req: IncomingMessage,
  res: ServerResponse,
  requestId?: string,
): void {
  debugLog("stream.writer_start", { requestId, bridgeKey, convKey, modelId, completedTurnCount: completedTurns.length, currentTurn });
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`;
  const created = Math.floor(Date.now() / 1000);

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
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
    id: completionId, object: "chat.completion.chunk", created, model: modelId,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });

  const makeUsageChunk = () => {
    const { prompt_tokens, completion_tokens, total_tokens } = computeUsage(state);
    return {
      id: completionId, object: "chat.completion.chunk", created, model: modelId,
      choices: [],
      usage: { prompt_tokens, completion_tokens, total_tokens },
    };
  };

  const state: StreamState = { toolCallIndex: 0, pendingExecs: [], outputTokens: 0, totalTokens: 0 };
  const tagFilter = createThinkingTagFilter();
  let mcpExecReceived = false;
  let cancelled = false;
  let latestCheckpoint: Uint8Array | null = null;

  // Detect client disconnect (e.g. user pressed Escape in pi)
  const onClientClose = () => {
    if (cancelled || closed) return;
    debugLog("stream.client_close", { requestId, bridgeKey, convKey });
    cancelled = true;
    cleanupBridge(bridge, heartbeatTimer, bridgeKey);
    closeResponse();
  };
  req.on("close", onClientClose);
  res.on("close", onClientClose);

  const processChunk = createConnectFrameParser(
    (messageBytes) => {
      try {
        const serverMessage = fromBinary(AgentServerMessageSchema, messageBytes);
        processServerMessage(
          serverMessage, blobStore, mcpTools,
          (data) => bridge.write(data),
          state,
          (text, isThinking) => {
            if (isThinking) {
              sendSSE(makeChunk({ reasoning_content: text }));
            } else {
              const { content, reasoning } = tagFilter.process(text);
              if (reasoning) sendSSE(makeChunk({ reasoning_content: reasoning }));
              if (content) {
                appendAssistantTextToTurn(currentTurn, content);
                sendSSE(makeChunk({ content }));
              }
            }
          },
          (exec) => {
            state.pendingExecs.push(exec);
            mcpExecReceived = true;

            const flushed = tagFilter.flush();
            if (flushed.reasoning) sendSSE(makeChunk({ reasoning_content: flushed.reasoning }));
            if (flushed.content) {
              appendAssistantTextToTurn(currentTurn, flushed.content);
              sendSSE(makeChunk({ content: flushed.content }));
            }

            currentTurn.steps.push({
              kind: "toolCall",
              toolCallId: exec.toolCallId,
              toolName: exec.toolName,
              arguments: parseToolCallArguments(exec.decodedArgs),
            });

            const toolCallIndex = state.toolCallIndex++;
            sendSSE(makeChunk({
              tool_calls: [{
                index: toolCallIndex, id: exec.toolCallId, type: "function",
                function: { name: exec.toolName, arguments: exec.decodedArgs },
              }],
            }));

            activeBridges.set(bridgeKey, {
              bridge, heartbeatTimer, blobStore, mcpTools, pendingExecs: state.pendingExecs, currentTurn,
            });
            debugLog("stream.tool_call_pause", { requestId, bridgeKey, exec, pendingExecs: state.pendingExecs, currentTurn });

            sendSSE(makeChunk({}, "tool_calls"));
            sendDone();
            closeResponse();
          },
          (checkpointBytes) => {
            latestCheckpoint = checkpointBytes;
            const stored = conversationStates.get(convKey);
            if (stored) {
              stored.checkpoint = checkpointBytes;
              for (const [k, v] of blobStore) stored.blobStore.set(k, v);

              stored.lastAccessMs = Date.now();
            }
            debugLog("stream.checkpoint_buffered", { requestId, convKey, checkpointBytes });
          },
        );
      } catch (err) {
        console.error("[cursor-provider] Stream message processing error:", err instanceof Error ? err.message : err);
      }
    },
    (endStreamBytes) => {
      const endError = parseConnectEndStream(endStreamBytes);
      if (endError) {
        console.error(`[cursor-provider] Cursor stream error (${modelId}):`, endError.message);
        conversationStates.delete(convKey);
        sendSSE(makeChunk({ content: endError.message }, "error"));
        sendSSE(makeUsageChunk());
        sendDone();
        closeResponse();
      }
    },
  );

  bridge.onData(processChunk);

  bridge.onClose((code) => {
    debugLog("stream.bridge_close", { requestId, bridgeKey, convKey, code, cancelled, mcpExecReceived, currentTurn, latestCheckpoint });
    clearInterval(heartbeatTimer);
    req.removeListener("close", onClientClose);
    res.removeListener("close", onClientClose);
    const stored = conversationStates.get(convKey);
    if (stored) {
      for (const [k, v] of blobStore) stored.blobStore.set(k, v);
      stored.lastAccessMs = Date.now();
      if (!cancelled && latestCheckpoint) {
        stored.checkpoint = latestCheckpoint;
        debugLog("stream.checkpoint_committed", { requestId, convKey, stored });
      }
    }
    if (cancelled) return;
    if (!mcpExecReceived) {
      const flushed = tagFilter.flush();
      if (flushed.reasoning) sendSSE(makeChunk({ reasoning_content: flushed.reasoning }));
      if (flushed.content) {
        appendAssistantTextToTurn(currentTurn, flushed.content);
        sendSSE(makeChunk({ content: flushed.content }));
      }
      sendSSE(makeChunk({}, "stop"));
      sendSSE(makeUsageChunk());
      sendDone();
      closeResponse();
    } else if (code !== 0) {
      sendSSE(makeChunk({ content: "Bridge connection lost" }, "error"));
      sendSSE(makeUsageChunk());
      sendDone();
      closeResponse();
      activeBridges.delete(bridgeKey);
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
  req: IncomingMessage,
  res: ServerResponse,
  stream: boolean,
  requestId?: string,
): void {
  const { bridge, heartbeatTimer, blobStore, mcpTools, pendingExecs, currentTurn } = active;
  debugLog("tool_resume.start", { requestId, bridgeKey, convKey, toolResults, pendingExecs, currentTurn });

  for (const result of toolResults) {
    const turnToolStep = currentTurn.steps.find((step) => step.kind === "toolCall" && step.toolCallId === result.toolCallId);
    if (turnToolStep) {
      turnToolStep.result = { content: result.content, isError: false };
    }
  }

  const turnResults = getTurnToolCallResults(currentTurn);
  const unresolvedExecs = pendingExecs.filter((exec) => !turnResults.has(exec.toolCallId));
  if (unresolvedExecs.length > 0) {
    activeBridges.set(bridgeKey, {
      bridge,
      heartbeatTimer,
      blobStore,
      mcpTools,
      pendingExecs,
      currentTurn,
    });
    debugLog("tool_resume.partial_wait", { requestId, bridgeKey, unresolvedExecs, currentTurn });
    respondWithPendingToolCalls(modelId, unresolvedExecs, stream, res);
    return;
  }

  for (const exec of pendingExecs) {
    const result = turnResults.get(exec.toolCallId);
    if (!result) continue;
    const mcpResult = create(McpResultSchema, {
      result: {
        case: "success",
        value: create(McpSuccessSchema, {
          content: [
            create(McpToolResultContentItemSchema, {
              content: { case: "text", value: create(McpTextContentSchema, { text: result.content }) },
            }),
          ],
          isError: false,
        }),
      },
    });

    const execClientMessage = create(ExecClientMessageSchema, {
      id: exec.execMsgId,
      execId: exec.execId,
      message: { case: "mcpResult" as any, value: mcpResult as any },
    });
    const clientMessage = create(AgentClientMessageSchema, {
      message: { case: "execClientMessage", value: execClientMessage },
    });
    bridge.write(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
    debugLog("tool_resume.sent_result", { requestId, exec, result });
  }

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
  convKey: string,
  completedTurns: ParsedTurn[],
  currentTurn: ParsedTurn,
  req: IncomingMessage,
  res: ServerResponse,
  requestId?: string,
): Promise<void> {
  debugLog("nonstream.start", { requestId, convKey, modelId, currentTurn, completedTurnCount: completedTurns.length });
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`;
  const created = Math.floor(Date.now() / 1000);

  const { bridge, heartbeatTimer } = startBridge(accessToken, payload.requestBytes);
  let cancelled = false;

  const onClientClose = () => {
    if (cancelled) return;
    debugLog("nonstream.client_close", { requestId, convKey });
    cancelled = true;
    clearInterval(heartbeatTimer);
    if (bridge.alive) {
      sendCancelAction(bridge);
      bridge.end();
    }
  };
  req.on("close", onClientClose);
  res.on("close", onClientClose);
  const state: StreamState = { toolCallIndex: 0, pendingExecs: [], outputTokens: 0, totalTokens: 0 };
  const tagFilter = createThinkingTagFilter();
  let fullText = "";
  let nonStreamError: Error | null = null;
  let latestCheckpoint: Uint8Array | null = null;

  return new Promise((resolve) => {
    bridge.onData(createConnectFrameParser(
      (messageBytes) => {
        try {
          const serverMessage = fromBinary(AgentServerMessageSchema, messageBytes);
          processServerMessage(
            serverMessage, payload.blobStore, payload.mcpTools,
            (data) => bridge.write(data),
            state,
            (text, isThinking) => {
              if (isThinking) return;
              const { content } = tagFilter.process(text);
              fullText += content;
              appendAssistantTextToTurn(currentTurn, content);
            },
            () => {},
            (checkpointBytes) => {
              latestCheckpoint = checkpointBytes;
              const stored = conversationStates.get(convKey);
              if (stored) {
                stored.checkpoint = checkpointBytes;
                for (const [k, v] of payload.blobStore) stored.blobStore.set(k, v);
  
                stored.lastAccessMs = Date.now();
              }
              debugLog("nonstream.checkpoint_buffered", { requestId, convKey, checkpointBytes });
            },
          );
        } catch (err) {
          console.error("[cursor-provider] Non-stream message processing error:", err instanceof Error ? err.message : err);
        }
      },
      (endStreamBytes) => {
        const endError = parseConnectEndStream(endStreamBytes);
        if (endError) {
          console.error(`[cursor-provider] Cursor non-stream error (${modelId}):`, endError.message);
          conversationStates.delete(convKey);
          nonStreamError = endError;
        }
      },
    ));

    bridge.onClose(() => {
      debugLog("nonstream.bridge_close", { requestId, convKey, cancelled, nonStreamError: nonStreamError?.message, currentTurn, latestCheckpoint });
      clearInterval(heartbeatTimer);
      req.removeListener("close", onClientClose);
      res.removeListener("close", onClientClose);
      const stored = conversationStates.get(convKey);
      if (stored) {
        for (const [k, v] of payload.blobStore) stored.blobStore.set(k, v);
        stored.lastAccessMs = Date.now();
        if (!cancelled && !nonStreamError && latestCheckpoint) {
          stored.checkpoint = latestCheckpoint;
          debugLog("nonstream.checkpoint_committed", { requestId, convKey, stored });
        }
      }

      if (cancelled) {
        if (!res.headersSent) {
          res.writeHead(499, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: "Client closed request", type: "aborted", code: "client_closed" } }));
        }
        resolve();
        return;
      }

      if (nonStreamError) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          error: { message: nonStreamError.message, type: "upstream_error", code: "cursor_error" },
        }));
        resolve();
        return;
      }

      const flushed = tagFilter.flush();
      fullText += flushed.content;
      appendAssistantTextToTurn(currentTurn, flushed.content);
      const usage = computeUsage(state);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        id: completionId, object: "chat.completion", created, model: modelId,
        choices: [{ index: 0, message: { role: "assistant", content: fullText }, finish_reason: "stop" }],
        usage,
      }));
      resolve();
    });
  });
}
