/**
 * Cursor Provider Extension for pi
 *
 * Provides access to Cursor models (Claude, GPT, Gemini, etc.) via:
 * 1. Browser-based PKCE OAuth login to Cursor
 * 2. Native Pi streamSimple provider translating Pi context → Cursor gRPC protocol
 *
 * Usage:
 *   /login cursor    — authenticate via browser
 *   /model           — select any Cursor model
 *
 * Based on https://github.com/ephraimduncan/opencode-cursor by Ephraim Duncan.
 */

import { AuthStorage, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join as pathJoin } from "node:path";
import {
  generateCursorAuthParams,
  getCursorAccessTokenFromEnv,
  getTokenExpiry,
  pollCursorAuth,
  refreshCursorToken,
} from "./auth.js";
import {
  isCursorProviderDebugEnabled,
  summarizeBase64ImageData,
  truncateDebugString,
} from "./debug.js";
import type { CursorModelRoutingByEffort, CursorToolResultImagePayload } from "./cursor-routing.js";
export type { CursorModelRouting, CursorToolResultImagePayload } from "./cursor-routing.js";
import {
  augmentCursorModels,
  buildNoReasoningEffortLookup,
  buildRawModelLookup,
  FALLBACK_MODELS,
  modelConfig,
  processModels,
  type ProcessedModel,
} from "./cursor-models.js";
export {
  applyNoReasoningEffort,
  applyRawCursorModelId,
  augmentCursorModels,
  buildEffortMap,
  buildNoReasoningEffortLookup,
  buildRawModelLookup,
  FALLBACK_MODELS,
  modelsFromParameterizedMetadata,
  parseModelId,
  processModels,
  supportsReasoningModelId,
} from "./cursor-models.js";
export type { ProcessedModel } from "./cursor-models.js";
import {
  cleanupSessionActiveBridge,
  cleanupSessionState,
  createCursorNativeStream,
  getCursorModels,
  getCursorParameterizedModels,
  touchActiveBridgeForSession,
  type CursorModel,
  type CursorParameterizedModel,
} from "./proxy.js";

// ── Extension debug and request context helpers ──

let extensionDebugLogFilePath: string | undefined;

function isExtensionDebugEnabled(): boolean {
  return isCursorProviderDebugEnabled();
}

function getExtensionDebugLogFilePath(): string {
  if (extensionDebugLogFilePath) return extensionDebugLogFilePath;
  const configured = process.env.PI_CURSOR_PROVIDER_EXTENSION_DEBUG_FILE?.trim();
  if (configured) {
    extensionDebugLogFilePath = configured;
    return extensionDebugLogFilePath;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  extensionDebugLogFilePath = pathJoin(
    tmpdir(),
    `pi-cursor-provider-extension-debug-${stamp}-${process.pid}.log`,
  );
  return extensionDebugLogFilePath;
}

function truncateDebugValue(value: string, max = 240): string {
  return truncateDebugString(value, max);
}

function summarizeImageBlock(type: unknown, mimeType: unknown, data: unknown): unknown {
  return {
    type,
    mimeType,
    ...(typeof data === "string"
      ? summarizeBase64ImageData(data)
      : { data: `<redacted base64 ${String(data ?? "").length} chars>` }),
  };
}

function summarizeDataImageUrl(url: string): unknown {
  const match = url.trim().match(/^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/is);
  if (!match)
    return {
      url: url.startsWith("data:image/")
        ? `<redacted data image ${url.length} chars>`
        : truncateDebugValue(url),
    };
  return {
    mimeType: match[1]?.toLowerCase(),
    ...summarizeBase64ImageData(match[2]!),
  };
}

function summarizeContent(content: unknown): unknown {
  if (typeof content === "string") return truncateDebugValue(content);
  if (!Array.isArray(content)) return content;
  return content.map((block) => {
    if (!block || typeof block !== "object") return block;
    const typed = block as Record<string, unknown>;
    switch (typed.type) {
      case "text":
        return { type: "text", text: truncateDebugValue(String(typed.text ?? "")) };
      case "thinking":
        return { type: "thinking", thinking: truncateDebugValue(String(typed.thinking ?? "")) };
      case "toolCall":
        return {
          type: "toolCall",
          id: typed.id,
          name: typed.name,
          arguments: typed.arguments,
        };
      case "image":
        return summarizeImageBlock("image", typed.mimeType, typed.data);
      case "image_url": {
        const url = (typed.image_url as Record<string, unknown> | undefined)?.url;
        const text = typeof url === "string" ? url : "";
        return { type: "image_url", image_url: summarizeDataImageUrl(text) };
      }
      default:
        return typed;
    }
  });
}

function summarizeMessage(message: unknown): unknown {
  if (!message || typeof message !== "object") return message;
  const typed = message as Record<string, unknown>;
  return {
    role: typed.role,
    stopReason: typed.stopReason,
    toolCallId: typed.toolCallId,
    toolName: typed.toolName,
    isError: typed.isError,
    errorMessage: typed.errorMessage,
    content: summarizeContent(typed.content),
  };
}

function summarizeBranchTail(
  ctx: {
    sessionManager?: {
      getBranch?: () => unknown[];
      getLeafId?: () => string;
      getSessionId?: () => string;
    };
  },
  limit = 6,
): unknown {
  try {
    const branch = ctx.sessionManager?.getBranch?.();
    if (!Array.isArray(branch)) return undefined;
    return {
      sessionId: ctx.sessionManager?.getSessionId?.(),
      leafId: ctx.sessionManager?.getLeafId?.(),
      size: branch.length,
      tail: branch.slice(-limit).map((entry) => {
        if (!entry || typeof entry !== "object") return entry;
        const typed = entry as Record<string, unknown>;
        return {
          type: typed.type,
          id: typed.id,
          parentId: typed.parentId,
          customType: typed.customType,
          message: summarizeMessage(typed.message),
        };
      }),
    };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

function payloadToolCallIds(payload: Record<string, unknown>): Set<string> {
  const ids = new Set<string>();
  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const typed = message as Record<string, unknown>;
    if (typed.role === "tool" && typeof typed.tool_call_id === "string" && typed.tool_call_id)
      ids.add(typed.tool_call_id);
  }
  return ids;
}

export function extractToolResultImagePayloads(
  ctx: { sessionManager?: { getBranch?: () => unknown[] } },
  payload: Record<string, unknown>,
): CursorToolResultImagePayload[] {
  const idsInPayload = payloadToolCallIds(payload);
  if (idsInPayload.size === 0) return [];
  const branch = ctx.sessionManager?.getBranch?.();
  if (!Array.isArray(branch)) return [];

  const byToolCallId = new Map<string, CursorToolResultImagePayload>();
  for (const entry of branch) {
    if (!entry || typeof entry !== "object") continue;
    const message = (entry as Record<string, unknown>).message;
    if (!message || typeof message !== "object") continue;
    const typed = message as Record<string, unknown>;
    const toolCallId = typeof typed.toolCallId === "string" ? typed.toolCallId : "";
    if (typed.role !== "toolResult" || !toolCallId || !idsInPayload.has(toolCallId)) continue;
    const content = Array.isArray(typed.content) ? typed.content : [];
    const images = content.flatMap((block) => {
      if (!block || typeof block !== "object") return [];
      const image = block as Record<string, unknown>;
      if (
        image.type !== "image" ||
        typeof image.data !== "string" ||
        typeof image.mimeType !== "string"
      )
        return [];
      return [{ data: image.data, mimeType: image.mimeType }];
    });
    if (images.length === 0) continue;
    const existing = byToolCallId.get(toolCallId);
    if (existing) existing.images.push(...images);
    else byToolCallId.set(toolCallId, { toolCallId, images });
  }
  return [...byToolCallId.values()];
}

function debugExtensionLog(event: string, data?: Record<string, unknown>): void {
  if (!isExtensionDebugEnabled()) return;
  const payload = JSON.stringify({
    ts: new Date().toISOString(),
    pid: process.pid,
    scope: "extension",
    event,
    ...data,
  });
  appendFileSync(getExtensionDebugLogFilePath(), `${payload}\n`, "utf8");
}

// ── Extension ──

const CURSOR_PROVIDER_ID = "cursor";

type StartupTokenSource = "env" | "pi_oauth" | "pi_oauth_refresh";

async function getStoredCursorOAuthAccessToken(): Promise<
  { accessToken: string; source: StartupTokenSource } | undefined
> {
  const authStorage = AuthStorage.create();
  const credential = authStorage.get(CURSOR_PROVIDER_ID);
  if (credential?.type !== "oauth") return undefined;

  if (Date.now() < credential.expires && credential.access) {
    return { accessToken: credential.access, source: "pi_oauth" };
  }

  const refreshed = await refreshCursorToken(credential.refresh);
  authStorage.set(CURSOR_PROVIDER_ID, { type: "oauth", ...refreshed });
  return { accessToken: refreshed.access, source: "pi_oauth_refresh" };
}

async function getStartupCursorAccessToken(): Promise<
  { accessToken: string; source: StartupTokenSource } | undefined
> {
  const envToken = getCursorAccessTokenFromEnv();
  if (envToken) return { accessToken: envToken, source: "env" };
  return getStoredCursorOAuthAccessToken();
}

export function registerSessionLifecycleCleanup(pi: ExtensionAPI) {
  const cleanupFullCurrentSession = (
    event: { type?: string; reason?: string },
    ctx: { sessionManager: { getSessionId(): string; getLeafId?: () => string } },
  ) => {
    debugExtensionLog("session.cleanup_hook", {
      cleanupKind: "full",
      eventType: event?.type,
      reason: event?.reason,
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
    });
    cleanupSessionState(ctx.sessionManager.getSessionId(), event?.reason ?? "session_shutdown", {
      eventType: event?.type ?? "session_shutdown",
    });
  };

  const cleanupActiveBridgeForTreeNavigation = (
    event: { type?: string },
    ctx: { sessionManager: { getSessionId(): string; getLeafId?: () => string } },
  ) => {
    debugExtensionLog("session.cleanup_hook", {
      cleanupKind: "activeBridge",
      eventType: event?.type,
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
    });
    cleanupSessionActiveBridge(ctx.sessionManager.getSessionId(), "session_tree", {
      eventType: event?.type ?? "session_tree",
    });
  };

  pi.on("session_shutdown", cleanupFullCurrentSession);
  pi.on("session_tree", cleanupActiveBridgeForTreeNavigation);
}

export function registerCursorModelSwitchCleanup(pi: ExtensionAPI) {
  pi.on(
    "model_select",
    async (
      event: { model?: { provider?: string }; previousModel?: { provider?: string } },
      ctx: { sessionManager: { getSessionId(): string; getLeafId?: () => string } },
    ) => {
      const nextProvider = event.model?.provider;
      const previousProvider = event.previousModel?.provider;
      if (nextProvider === previousProvider) return;
      if (nextProvider !== "cursor" && previousProvider !== "cursor") return;

      debugExtensionLog("model_select.cursor_boundary_cleanup", {
        cleanupKind: "activeBridge",
        sessionId: ctx.sessionManager.getSessionId(),
        leafId: ctx.sessionManager.getLeafId?.(),
        previousProvider,
        nextProvider,
      });
      cleanupSessionActiveBridge(
        ctx.sessionManager.getSessionId(),
        "model_select_cursor_boundary",
        { eventType: "model_select", previousProvider, nextProvider },
      );
    },
  );
}

export function registerCursorToolExecutionTtlRefresh(pi: ExtensionAPI) {
  const refresh = (
    event: { type?: string; toolCallId?: string; toolName?: string },
    ctx: {
      model?: { provider?: string };
      sessionManager: { getSessionId(): string; getLeafId?: () => string };
    },
  ) => {
    if (ctx.model?.provider !== "cursor") return;
    const refreshed = touchActiveBridgeForSession(
      ctx.sessionManager.getSessionId(),
      event?.type ?? "tool_execution",
    );
    if (!refreshed) return;
    debugExtensionLog("tool_execution.cursor_active_bridge_ttl_refreshed", {
      eventType: event?.type,
      toolCallId: event?.toolCallId,
      toolName: event?.toolName,
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
    });
  };

  pi.on("tool_execution_start", refresh);
  pi.on("tool_execution_update", refresh);
  pi.on("tool_execution_end", refresh);
}

function registerCursorPayloadContextHook(pi: ExtensionAPI) {
  pi.on("before_provider_request", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    const payload = (event as { payload?: unknown }).payload;
    if (!payload || typeof payload !== "object") return;
    const typedPayload = payload as Record<string, unknown>;
    if (!Array.isArray(typedPayload.messages)) return;

    const nextPayload: Record<string, unknown> = { ...typedPayload };
    if (typeof ctx.cwd === "string" && ctx.cwd.trim()) {
      nextPayload.cursor_workspace_path = ctx.cwd;
    }

    const imagePayloads = extractToolResultImagePayloads(ctx, nextPayload);
    if (imagePayloads.length > 0) nextPayload.cursor_tool_result_images = imagePayloads;
    return nextPayload;
  });
}

function registerExtensionDebugHooks(pi: ExtensionAPI) {
  if (!isExtensionDebugEnabled()) return;

  pi.on("message_start", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    debugExtensionLog("message.start", {
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
      model: ctx.model?.id,
      message: summarizeMessage((event as { message?: unknown }).message),
    });
  });

  pi.on("message_update", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    const typedEvent = event as {
      message?: unknown;
      assistantMessageEvent?: Record<string, unknown>;
    };
    debugExtensionLog("message.update", {
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
      model: ctx.model?.id,
      assistantMessageEvent: typedEvent.assistantMessageEvent
        ? {
            type: typedEvent.assistantMessageEvent.type,
            delta: truncateDebugValue(
              String(
                (typedEvent.assistantMessageEvent as Record<string, unknown>).delta ??
                  (typedEvent.assistantMessageEvent as Record<string, unknown>).content ??
                  "",
              ),
            ),
          }
        : undefined,
      message: summarizeMessage(typedEvent.message),
    });
  });

  pi.on("message_end", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    debugExtensionLog("message.end", {
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
      model: ctx.model?.id,
      message: summarizeMessage((event as { message?: unknown }).message),
      branch: summarizeBranchTail(ctx),
    });
  });

  pi.on("context", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    const typedEvent = event as { messages?: unknown[] };
    debugExtensionLog("context", {
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
      model: ctx.model?.id,
      messageCount: Array.isArray(typedEvent.messages) ? typedEvent.messages.length : undefined,
      messages: Array.isArray(typedEvent.messages)
        ? typedEvent.messages.slice(-8).map((message) => summarizeMessage(message))
        : undefined,
      branch: summarizeBranchTail(ctx),
    });
  });

  pi.on("turn_end", async (event, ctx) => {
    if (ctx.model?.provider !== "cursor") return;
    const typedEvent = event as { turnIndex?: number; message?: unknown; toolResults?: unknown[] };
    debugExtensionLog("turn.end", {
      sessionId: ctx.sessionManager.getSessionId(),
      leafId: ctx.sessionManager.getLeafId?.(),
      model: ctx.model?.id,
      turnIndex: typedEvent.turnIndex,
      message: summarizeMessage(typedEvent.message),
      toolResults: Array.isArray(typedEvent.toolResults)
        ? typedEvent.toolResults.map((message) => summarizeMessage(message))
        : undefined,
      branch: summarizeBranchTail(ctx),
    });
  });

  debugExtensionLog("extension.debug_hooks_registered", {
    logFile: getExtensionDebugLogFilePath(),
  });
}

export default async function (pi: ExtensionAPI) {
  // Current access token, updated by login/refresh/getApiKey
  let currentToken = "";
  let noReasoningEffortByModelId = new Map<string, string>();
  let rawModelByEffortByModelId = new Map<string, CursorModelRoutingByEffort>();

  const getAccessToken = async () => {
    if (!currentToken) throw new Error("Not logged in to Cursor. Run /login cursor");
    return currentToken;
  };

  const skipDedup = !!process.env.PI_CURSOR_RAW_MODELS;

  registerSessionLifecycleCleanup(pi);
  registerCursorModelSwitchCleanup(pi);
  registerCursorToolExecutionTtlRefresh(pi);
  registerCursorPayloadContextHook(pi);
  registerExtensionDebugHooks(pi);
  debugExtensionLog("extension.start", {
    mode: "native-streamSimple",
    debugLogFile: isExtensionDebugEnabled() ? getExtensionDebugLogFilePath() : undefined,
  });

  const startupModels = await discoverStartupModels();
  register(pi, startupModels.rawModels, startupModels.parameterizedModels);

  async function discoverStartupModels(): Promise<{
    rawModels: CursorModel[];
    parameterizedModels: CursorParameterizedModel[];
  }> {
    if (process.env.PI_OFFLINE) return { rawModels: FALLBACK_MODELS, parameterizedModels: [] };

    let startupToken: { accessToken: string; source: StartupTokenSource } | undefined;
    try {
      startupToken = await getStartupCursorAccessToken();
    } catch (err) {
      debugExtensionLog("model_discovery.startup.token_failed", {
        message: err instanceof Error ? err.message : String(err),
      });
    }

    if (!startupToken) {
      debugExtensionLog("model_discovery.startup.skipped", { reason: "no_cursor_oauth_token" });
      return { rawModels: FALLBACK_MODELS, parameterizedModels: [] };
    }

    try {
      currentToken = startupToken.accessToken;
      const [discovered, parameterized] = await Promise.all([
        getCursorModels(startupToken.accessToken),
        getCursorParameterizedModels(startupToken.accessToken),
      ]);
      debugExtensionLog("model_discovery.startup", {
        tokenSource: startupToken.source,
        discoveredCount: discovered.length,
        parameterizedCount: parameterized.length,
      });
      if (discovered.length > 0 || parameterized.length > 0) {
        return {
          rawModels: discovered.length > 0 ? discovered : FALLBACK_MODELS,
          parameterizedModels: parameterized,
        };
      }
    } catch (err) {
      debugExtensionLog("model_discovery.startup.failed", {
        tokenSource: startupToken.source,
        message: err instanceof Error ? err.message : String(err),
      });
    }

    return { rawModels: FALLBACK_MODELS, parameterizedModels: [] };
  }

  async function rediscoverAndRegisterModels(accessToken: string): Promise<void> {
    const [discovered, parameterized] = await Promise.all([
      getCursorModels(accessToken),
      getCursorParameterizedModels(accessToken),
    ]);
    if (discovered.length > 0 || parameterized.length > 0) {
      register(pi, discovered.length > 0 ? discovered : FALLBACK_MODELS, parameterized);
    }
  }

  function register(
    pi: ExtensionAPI,
    rawModels: CursorModel[],
    parameterizedModels: CursorParameterizedModel[] = [],
  ) {
    const augmentedModels = augmentCursorModels(rawModels, parameterizedModels);
    const processed = skipDedup
      ? augmentedModels.map((m) => ({ ...m, supportsEffort: false }) as ProcessedModel)
      : processModels(augmentedModels);
    noReasoningEffortByModelId = buildNoReasoningEffortLookup(processed);
    rawModelByEffortByModelId = buildRawModelLookup(processed);

    pi.registerProvider("cursor", {
      name: "Cursor",
      baseUrl: "https://api2.cursor.sh",
      api: "cursor-native",
      streamSimple: createCursorNativeStream({
        getAccessToken,
        getNoReasoningEffortByModelId: () => noReasoningEffortByModelId,
        getRawModelRoutingByModelId: () => rawModelByEffortByModelId,
      }),
      models: processed.map(modelConfig),
      oauth: {
        name: "Cursor",

        async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
          const { verifier, uuid, loginUrl } = await generateCursorAuthParams();
          callbacks.onAuth({ url: loginUrl });
          const { accessToken, refreshToken } = await pollCursorAuth(uuid, verifier);
          currentToken = accessToken;

          await rediscoverAndRegisterModels(accessToken);

          return {
            refresh: refreshToken,
            access: accessToken,
            expires: getTokenExpiry(accessToken),
          };
        },

        async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
          const refreshed = await refreshCursorToken(credentials.refresh);
          currentToken = refreshed.access;

          await rediscoverAndRegisterModels(refreshed.access);

          return refreshed as OAuthCredentials;
        },

        getApiKey(credentials: OAuthCredentials): string {
          currentToken = credentials.access;
          return "cursor-native";
        },
      },
    });
  }
}
