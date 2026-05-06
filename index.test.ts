import rawModels from "./cursor-models-raw.json";
import gpt55ParameterizedFixture from "./fixtures/cursor-gpt55-parameterized.json";
import { afterEach, describe, expect, test } from "vitest";
import { EventEmitter } from "node:events";
import { request as httpRequest } from "node:http";
import { spawnBridge } from "./bridge.ts";
import {
  applyNoReasoningEffort,
  buildEffortMap,
  buildNoReasoningEffortLookup,
  buildRawModelLookup,
  applyRawCursorModelId,
  augmentCursorModels,
  extractToolResultImagePayloads,
  FALLBACK_MODELS,
  modelsFromParameterizedMetadata,
  parseModelId,
  processModels,
  registerCursorModelSwitchCleanup,
  registerSessionLifecycleCleanup,
  supportsReasoningModelId,
} from "./index.ts";
import {
  resolveModelId,
  resolveRequestedModelId,
  inferCursorContextWindow,
  getCursorParameterizedModels,
  __testInternals,
  cleanupAllSessionState,
  cleanupSessionState,
  evictStaleConversations,
  deriveBridgeKey,
  deriveBridgeKeyFromSessionId,
  deriveConversationKey,
  deriveConversationKeyFromSessionId,
  derivePiSessionId,
  buildCursorRequest,
  createCursorNativeStream,
  parseMessages,
  setBridgeFactoryForTests,
  startProxy,
  stopProxy,
  writeSSEStreamForTests,
} from "./proxy.ts";
import type { CursorModel, ParsedTurn } from "./proxy.ts";
import { create, fromBinary, fromJson, toBinary, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import {
  AgentClientMessageSchema,
  AgentRunRequestSchema,
  AgentServerMessageSchema,
  ConversationStateStructureSchema,
  ConversationTurnStructureSchema,
  ConversationStepSchema,
  ExecServerMessageSchema,
  InteractionUpdateSchema,
  KvServerMessageSchema,
  McpArgsSchema,
  McpToolDefinitionSchema,
  RequestContextArgsSchema,
  SetBlobArgsSchema,
  TextDeltaUpdateSchema,
  UserMessageSchema,
} from "./proto/agent_pb.ts";

afterEach(() => {
  stopProxy();
  setBridgeFactoryForTests();
  cleanupAllSessionState();
});

// ── Helper ──

function m(id: string, name?: string): CursorModel {
  return { id, name: name ?? id, reasoning: true, contextWindow: 200_000, maxTokens: 64_000 };
}

// ── bridge process ──

describe("spawnBridge", () => {
  test("ignores writes after stdin is ended", async () => {
    const bridge = spawnBridge({
      accessToken: "test-token",
      rpcPath: "/agent.v1.AgentService/Run",
      url: "http://127.0.0.1:1",
    });
    bridge.end();
    expect(() => bridge.write(new Uint8Array([1, 2, 3]))).not.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(() => bridge.end()).not.toThrow();
    const closed = new Promise((resolve) => bridge.onClose(resolve));
    bridge.proc.kill();
    await Promise.race([closed, new Promise((resolve) => setTimeout(resolve, 250))]);
  });
});

// ── model metadata ──

describe("inferCursorContextWindow", () => {
  test("detects 1M and 272K context labels from Cursor names and IDs", () => {
    expect(inferCursorContextWindow("gpt-5.5", "GPT-5.5 272K Medium")).toBe(272_000);
    expect(inferCursorContextWindow("gpt-5.5-1m-medium", "GPT-5.5 1M")).toBe(1_000_000);
    expect(inferCursorContextWindow("gpt-5.3-codex", "Codex 5.3")).toBe(200_000);
  });
});

// ── parseModelId ──

describe("parseModelId", () => {
  test("plain model — no effort, no variant", () => {
    expect(parseModelId("composer-2")).toEqual({
      base: "composer-2",
      effort: "",
      fast: false,
      thinking: false,
    });
  });

  test("plain model with -fast suffix", () => {
    expect(parseModelId("composer-2-fast")).toEqual({
      base: "composer-2",
      effort: "",
      fast: true,
      thinking: false,
    });
  });

  test("model with effort suffix", () => {
    expect(parseModelId("gpt-5.4-medium")).toEqual({
      base: "gpt-5.4",
      effort: "medium",
      fast: false,
      thinking: false,
    });
  });

  test("model with effort + fast", () => {
    expect(parseModelId("gpt-5.4-high-fast")).toEqual({
      base: "gpt-5.4",
      effort: "high",
      fast: true,
      thinking: false,
    });
  });

  test("model with effort + thinking", () => {
    expect(parseModelId("claude-4.6-opus-high-thinking")).toEqual({
      base: "claude-4.6-opus",
      effort: "high",
      fast: false,
      thinking: true,
    });
  });

  test("max effort level", () => {
    expect(parseModelId("claude-4.6-opus-max")).toEqual({
      base: "claude-4.6-opus",
      effort: "max",
      fast: false,
      thinking: false,
    });
  });

  test("max effort + thinking", () => {
    expect(parseModelId("claude-4.6-opus-max-thinking")).toEqual({
      base: "claude-4.6-opus",
      effort: "max",
      fast: false,
      thinking: true,
    });
  });

  test("thinking + max effort order used by newer Cursor models", () => {
    expect(parseModelId("claude-opus-4-7-thinking-max")).toEqual({
      base: "claude-opus-4-7",
      effort: "max",
      fast: false,
      thinking: true,
    });
  });

  test("thinking + high effort + fast order used by newer Cursor models", () => {
    expect(parseModelId("claude-opus-4-7-thinking-high-fast")).toEqual({
      base: "claude-opus-4-7",
      effort: "high",
      fast: true,
      thinking: true,
    });
  });

  test("extra-high Cursor suffix normalizes to xhigh effort", () => {
    expect(parseModelId("gpt-5.5-extra-high")).toEqual({
      base: "gpt-5.5",
      effort: "xhigh",
      fast: false,
      thinking: false,
    });
  });

  test("extra-high Cursor suffix + fast normalizes to xhigh effort", () => {
    expect(parseModelId("gpt-5.5-extra-high-fast")).toEqual({
      base: "gpt-5.5",
      effort: "xhigh",
      fast: true,
      thinking: false,
    });
  });

  test("none effort level", () => {
    expect(parseModelId("gpt-5.4-mini-none")).toEqual({
      base: "gpt-5.4-mini",
      effort: "none",
      fast: false,
      thinking: false,
    });
  });

  test("xhigh effort", () => {
    expect(parseModelId("gpt-5.2-xhigh")).toEqual({
      base: "gpt-5.2",
      effort: "xhigh",
      fast: false,
      thinking: false,
    });
  });

  test("xhigh effort + fast", () => {
    expect(parseModelId("gpt-5.2-xhigh-fast")).toEqual({
      base: "gpt-5.2",
      effort: "xhigh",
      fast: true,
      thinking: false,
    });
  });

  test("codex-max model — max is part of base, not effort", () => {
    expect(parseModelId("gpt-5.1-codex-max-high")).toEqual({
      base: "gpt-5.1-codex-max",
      effort: "high",
      fast: false,
      thinking: false,
    });
  });

  test("codex-max + fast", () => {
    expect(parseModelId("gpt-5.1-codex-max-medium-fast")).toEqual({
      base: "gpt-5.1-codex-max",
      effort: "medium",
      fast: true,
      thinking: false,
    });
  });

  test("codex-mini model", () => {
    expect(parseModelId("gpt-5.1-codex-mini-high")).toEqual({
      base: "gpt-5.1-codex-mini",
      effort: "high",
      fast: false,
      thinking: false,
    });
  });

  test("spark-preview model", () => {
    expect(parseModelId("gpt-5.3-codex-spark-preview-high")).toEqual({
      base: "gpt-5.3-codex-spark-preview",
      effort: "high",
      fast: false,
      thinking: false,
    });
  });

  test("plain thinking model — no effort", () => {
    expect(parseModelId("grok-4-20-thinking")).toEqual({
      base: "grok-4-20",
      effort: "",
      fast: false,
      thinking: true,
    });
  });

  test("model without any suffix", () => {
    expect(parseModelId("kimi-k2.5")).toEqual({
      base: "kimi-k2.5",
      effort: "",
      fast: false,
      thinking: false,
    });
  });

  test("default model", () => {
    expect(parseModelId("default")).toEqual({
      base: "default",
      effort: "",
      fast: false,
      thinking: false,
    });
  });

  test("claude-4.6-sonnet-medium — effort is medium", () => {
    expect(parseModelId("claude-4.6-sonnet-medium")).toEqual({
      base: "claude-4.6-sonnet",
      effort: "medium",
      fast: false,
      thinking: false,
    });
  });

  test("claude-4.6-sonnet-medium-thinking", () => {
    expect(parseModelId("claude-4.6-sonnet-medium-thinking")).toEqual({
      base: "claude-4.6-sonnet",
      effort: "medium",
      fast: false,
      thinking: true,
    });
  });
});

// ── buildEffortMap ──

describe("buildEffortMap", () => {
  test("full range: none/low/medium/high/xhigh", () => {
    const map = buildEffortMap(new Set(["none", "low", "medium", "high", "xhigh"]));
    expect(map).toEqual({
      minimal: "none",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "xhigh",
    });
  });

  test("with default (empty) and medium", () => {
    const map = buildEffortMap(new Set(["", "low", "medium", "high"]));
    expect(map).toEqual({
      minimal: "low",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "high",
    });
  });

  test("default without medium — medium maps to empty", () => {
    const map = buildEffortMap(new Set(["", "low", "high", "xhigh"]));
    expect(map.medium).toBe("");
  });

  test("high+max only — all lower levels clamp to high", () => {
    const map = buildEffortMap(new Set(["high", "max"]));
    expect(map).toEqual({
      minimal: "high",
      low: "high",
      medium: "high",
      high: "high",
      xhigh: "max",
    });
  });

  test("none+low+medium+high+max", () => {
    const map = buildEffortMap(new Set(["none", "low", "medium", "high", "max"]));
    expect(map).toEqual({
      minimal: "none",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "max",
    });
  });

  test("low+high — medium falls back to low", () => {
    const map = buildEffortMap(new Set(["low", "high"]));
    expect(map).toEqual({ minimal: "low", low: "low", medium: "low", high: "high", xhigh: "high" });
  });
});

// ── processModels ──

describe("reasoning support", () => {
  test("derives reasoning from model ids", () => {
    expect(supportsReasoningModelId("gpt-5.4")).toBe(true);
    expect(supportsReasoningModelId("gpt-5.4-fast")).toBe(true);
    expect(supportsReasoningModelId("gpt-5.5")).toBe(true);
    expect(supportsReasoningModelId("composer-2")).toBe(true);
    expect(supportsReasoningModelId("default")).toBe(true);
    expect(supportsReasoningModelId("auto")).toBe(true);
    expect(supportsReasoningModelId("totally-unknown-model")).toBe(false);
  });

  test("fallback models keep derived reasoning enabled", () => {
    expect(FALLBACK_MODELS.length).toBeGreaterThan(0);
    expect(FALLBACK_MODELS.find((model) => model.id === "gpt-5.4-medium")?.reasoning).toBe(true);
    expect(FALLBACK_MODELS.find((model) => model.id === "gpt-5.5-medium")?.reasoning).toBe(true);
    expect(FALLBACK_MODELS.find((model) => model.id === "composer-2")?.reasoning).toBe(true);
  });
});

describe("processModels", () => {
  test("composer-2 — no effort variants, kept as-is", () => {
    const result = processModels([m("composer-2"), m("composer-2-fast")]);
    const c2 = result.find((r) => r.id === "composer-2");
    const c2f = result.find((r) => r.id === "composer-2-fast");
    expect(c2).toBeDefined();
    expect(c2!.supportsEffort).toBe(false);
    expect(c2f).toBeDefined();
    expect(c2f!.supportsEffort).toBe(false);
  });

  test("gpt-5.4 — deduped from low/medium/high/xhigh", () => {
    const result = processModels([
      m("gpt-5.4-low"),
      m("gpt-5.4-medium"),
      m("gpt-5.4-high"),
      m("gpt-5.4-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe("medium");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("gpt-5.4-fast — deduped from effort+fast variants", () => {
    const result = processModels([
      m("gpt-5.4-high-fast"),
      m("gpt-5.4-medium-fast"),
      m("gpt-5.4-xhigh-fast"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4-fast");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("augmentCursorModels adds GPT-5.5 context parameter choices to discovered models", () => {
    const augmented = augmentCursorModels([m("gpt-5.5-medium", "GPT-5.5 1M")]);
    const result = processModels(augmented);
    expect(result.find((r) => r.id === "gpt-5.5")?.contextWindow).toBe(272_000);
    expect(result.find((r) => r.id === "gpt-5.5-1m")?.contextWindow).toBe(1_000_000);
  });

  test("gpt-5.5 — deduped from 272K effort variants", () => {
    const result = processModels([
      m("gpt-5.5-none"),
      m("gpt-5.5-low"),
      m("gpt-5.5-medium", "GPT-5.5 272K"),
      m("gpt-5.5-high"),
      m("gpt-5.5-extra-high"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.5");
    expect(result[0].name).toBe("GPT-5.5 272K");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.minimal).toBe("none");
    expect(result[0].effortMap!.medium).toBe("medium");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("gpt-5.5-fast — deduped from effort+fast variants including low", () => {
    const result = processModels([
      m("gpt-5.5-low-fast"),
      m("gpt-5.5-medium-fast"),
      m("gpt-5.5-high-fast"),
      m("gpt-5.5-extra-high-fast"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.5-fast");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.low).toBe("low");
    expect(result[0].effortMap!.medium).toBe("medium");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("gpt-5.5-1m — deduped as a separate 1M context model", () => {
    const result = processModels([
      m("gpt-5.5-1m-low", "GPT-5.5 1M Low"),
      m("gpt-5.5-1m-medium", "GPT-5.5 1M"),
      m("gpt-5.5-1m-high", "GPT-5.5 1M High"),
      m("gpt-5.5-1m-extra-high", "GPT-5.5 1M Extra High"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.5-1m");
    expect(result[0].name).toBe("GPT-5.5 1M");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe("medium");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("default Cursor model is exposed as auto/Auto but routes to default", () => {
    const augmented = augmentCursorModels([m("default", "Auto")]);
    expect(augmented).toHaveLength(1);
    expect(augmented[0].id).toBe("auto");
    expect(augmented[0].name).toBe("Auto");
    expect(augmented[0].requestedModelId).toBe("default");

    const payload: Record<string, unknown> = { model: "auto" };
    applyRawCursorModelId(payload, buildRawModelLookup(processModels(augmented)));
    expect(payload.cursor_model_id).toBe("default");
  });

  test("raw -max-mode and -max-max IDs are normalized to -max while routing to the original ID", () => {
    const augmented = augmentCursorModels([
      m("composer-2-max-mode-fast", "Composer 2 Max Mode Fast"),
      m("composer-3-max-max-fast", "Composer 3 Max Max Fast"),
    ]);
    const processed = processModels(augmented);
    expect(processed.map((model) => model.id)).toEqual([
      "composer-2-max-fast",
      "composer-3-max-fast",
    ]);

    const payload: Record<string, unknown> = { model: "composer-2-max-fast" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("composer-2-max-mode-fast");

    const duplicateMaxPayload: Record<string, unknown> = { model: "composer-3-max-fast" };
    applyRawCursorModelId(duplicateMaxPayload, buildRawModelLookup(processed));
    expect(duplicateMaxPayload.cursor_model_id).toBe("composer-3-max-max-fast");
  });

  test("GPT-5.5 augmentation exposes max-mode rows but not impossible 1M fast variants", () => {
    const augmented = augmentCursorModels([m("gpt-5.5", "GPT-5.5")]);
    const processed = processModels(augmented);
    expect(processed.some((model) => model.id === "gpt-5.5-fast")).toBe(true);
    expect(processed.some((model) => model.id === "gpt-5.5-max")).toBe(true);
    expect(processed.some((model) => model.id === "gpt-5.5-max-fast")).toBe(true);
    expect(processed.some((model) => model.id === "gpt-5.5-1m")).toBe(true);
    expect(processed.some((model) => model.id === "gpt-5.5-1m-fast")).toBe(false);
    expect(augmented.some((model) => /^gpt-5\.5-1m-.*-fast$/.test(model.id))).toBe(false);
  });

  test("metadata-driven GPT-5.5 generation matches Cursor parameterized fixture", () => {
    const generated = modelsFromParameterizedMetadata(gpt55ParameterizedFixture as any);
    const processed = processModels(generated);
    const ids = processed.map((model) => model.id).sort();
    expect(ids).toEqual([
      "gpt-5.5",
      "gpt-5.5-1m",
      "gpt-5.5-fast",
      "gpt-5.5-max",
      "gpt-5.5-max-fast",
    ]);
    expect(processed.find((model) => model.id === "gpt-5.5")!.contextWindow).toBe(272_000);
    expect(
      processed.find((model) => model.id === "gpt-5.5-fast")!.rawRoutingByEffort!.high!
        .requestedMaxMode,
    ).toBe(false);
    expect(
      processed.find((model) => model.id === "gpt-5.5-max-fast")!.rawRoutingByEffort!.high,
    ).toEqual({
      modelId: "gpt-5.5",
      requestedMaxMode: true,
      parameters: [
        { id: "context", value: "272k" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    });
    expect(processed.find((model) => model.id === "gpt-5.5-1m")!.contextWindow).toBe(1_000_000);
    expect(
      processed.find((model) => model.id === "gpt-5.5-1m")!.rawRoutingByEffort!.high!.parameters,
    ).toContainEqual({ id: "fast", value: "false" });
    expect(generated.some((model) => model.id.includes("1m") && model.id.endsWith("fast"))).toBe(
      false,
    );
    const reasoningValues = generated
      .flatMap((model) => model.parameters ?? [])
      .filter((parameter) => parameter.id === "reasoning")
      .map((parameter) => parameter.value);
    expect(reasoningValues).not.toContain("minimal");
    expect(reasoningValues).not.toContain("max");
  });

  test("metadata-driven rows retain Cursor image-support metadata", () => {
    const generated = modelsFromParameterizedMetadata([
      {
        name: "text-only-model",
        clientDisplayName: "Text Only",
        supportsMaxMode: false,
        supportsNonMaxMode: true,
        supportsImages: false,
        variants: [
          { isMaxMode: false, parameters: [{ id: "reasoning", value: "low" }] },
          { isMaxMode: false, parameters: [{ id: "reasoning", value: "high" }] },
        ],
      },
    ] as any);

    expect(generated).toHaveLength(2);
    expect(generated.every((model) => model.supportsImages === false)).toBe(true);

    const augmented = augmentCursorModels([m("plain-text-only", "Plain Text Only")], [
      {
        name: "plain-text-only",
        clientDisplayName: "Plain Text Only",
        supportsImages: false,
        variants: [],
      },
    ] as any);
    expect(augmented.find((model) => model.id === "plain-text-only")?.supportsImages).toBe(false);
  });

  test("metadata-driven generation covers non-GPT-5.5 parameterized models", () => {
    const generated = modelsFromParameterizedMetadata([
      {
        name: "composer-2",
        clientDisplayName: "Composer 2",
        supportsMaxMode: true,
        supportsNonMaxMode: true,
        variants: [
          { isMaxMode: false, parameters: [{ id: "fast", value: "false" }] },
          { isMaxMode: false, parameters: [{ id: "fast", value: "true" }] },
        ],
      },
      {
        name: "gpt-5.3-codex",
        clientDisplayName: "Codex 5.3",
        supportsMaxMode: true,
        supportsNonMaxMode: true,
        variants: [
          {
            isMaxMode: false,
            parameters: [
              { id: "reasoning", value: "low" },
              { id: "fast", value: "false" },
            ],
          },
          {
            isMaxMode: false,
            parameters: [
              { id: "reasoning", value: "high" },
              { id: "fast", value: "false" },
            ],
          },
          {
            isMaxMode: false,
            parameters: [
              { id: "reasoning", value: "low" },
              { id: "fast", value: "true" },
            ],
          },
          {
            isMaxMode: false,
            parameters: [
              { id: "reasoning", value: "high" },
              { id: "fast", value: "true" },
            ],
          },
        ],
      },
      {
        name: "gpt-5.1-codex-max",
        clientDisplayName: "Codex 5.1 Max",
        supportsMaxMode: true,
        supportsNonMaxMode: true,
        variants: [
          {
            isMaxMode: false,
            parameters: [
              { id: "reasoning", value: "high" },
              { id: "fast", value: "true" },
            ],
          },
        ],
      },
      {
        name: "claude-opus-4-7",
        clientDisplayName: "Opus 4.7",
        supportsMaxMode: true,
        supportsNonMaxMode: true,
        variants: [
          {
            isMaxMode: false,
            parameters: [
              { id: "thinking", value: "true" },
              { id: "context", value: "300k" },
              { id: "effort", value: "high" },
            ],
          },
          {
            isMaxMode: false,
            parameters: [
              { id: "thinking", value: "true" },
              { id: "context", value: "300k" },
              { id: "effort", value: "xhigh" },
            ],
          },
        ],
      },
    ] as any);
    const processed = processModels(generated);
    expect(generated.some((model) => model.id.includes("max-mode"))).toBe(false);
    expect(generated.some((model) => model.id.includes("max-max"))).toBe(false);
    expect(processed.some((model) => model.id === "composer-2-fast")).toBe(true);
    expect(processed.find((model) => model.id === "composer-2-max-fast")!.requestedMaxMode).toBe(
      true,
    );
    expect(
      processed.find((model) => model.id === "gpt-5.3-codex-fast")!.rawRoutingByEffort!.high,
    ).toEqual({
      modelId: "gpt-5.3-codex",
      requestedMaxMode: false,
      parameters: [
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    });
    expect(
      processed.find((model) => model.id === "gpt-5.3-codex-max-fast")!.rawRoutingByEffort!.high!
        .requestedMaxMode,
    ).toBe(true);
    expect(
      processed.find((model) => model.id === "gpt-5.1-codex-max-fast")!.rawRoutingByEffort!.high,
    ).toMatchObject({ modelId: "gpt-5.1-codex-max", requestedMaxMode: true });
    expect(
      processed.find((model) => model.id === "claude-opus-4-7-thinking")!.rawRoutingByEffort!.xhigh,
    ).toEqual({
      modelId: "claude-opus-4-7",
      parameters: [
        { id: "thinking", value: "true" },
        { id: "context", value: "300k" },
        { id: "effort", value: "xhigh" },
      ],
      requestedMaxMode: false,
    });
    expect(
      processed.find((model) => model.id === "claude-opus-4-7-max-thinking")!.rawRoutingByEffort!
        .xhigh,
    ).toEqual({
      modelId: "claude-opus-4-7",
      parameters: [
        { id: "thinking", value: "true" },
        { id: "context", value: "300k" },
        { id: "effort", value: "xhigh" },
      ],
      requestedMaxMode: true,
    });
  });

  test("gpt-5.2 — deduped from default + effort variants", () => {
    const result = processModels([
      m("gpt-5.2"),
      m("gpt-5.2-high"),
      m("gpt-5.2-low"),
      m("gpt-5.2-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.2");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe(""); // no-suffix = default
    expect(result[0].effortMap!.high).toBe("high");
  });

  test("gpt-5.4-mini — has none effort", () => {
    const result = processModels([
      m("gpt-5.4-mini-low"),
      m("gpt-5.4-mini-medium"),
      m("gpt-5.4-mini-high"),
      m("gpt-5.4-mini-xhigh"),
      m("gpt-5.4-mini-none"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4-mini");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.minimal).toBe("none");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("no-reasoning lookup tracks models with none effort", () => {
    const result = processModels([
      m("gpt-5.4-mini-low"),
      m("gpt-5.4-mini-medium"),
      m("gpt-5.4-mini-none"),
      m("gpt-5.4-low"),
      m("gpt-5.4-medium"),
      m("gpt-5.4-high"),
    ]);
    const lookup = buildNoReasoningEffortLookup(result);
    expect(lookup.get("gpt-5.4-mini")).toBe("none");
    expect(lookup.has("gpt-5.4")).toBe(false);
  });

  test("claude-4.6-opus — high+max deduped, effort clamped to lowest", () => {
    const result = processModels([m("claude-4.6-opus-high"), m("claude-4.6-opus-max")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-opus");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.minimal).toBe("high");
    expect(result[0].effortMap!.low).toBe("high");
    expect(result[0].effortMap!.medium).toBe("high");
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.xhigh).toBe("max");
  });

  test("claude-4.6-opus-thinking — high+max thinking deduped", () => {
    const result = processModels([
      m("claude-4.6-opus-high-thinking"),
      m("claude-4.6-opus-max-thinking"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-opus-thinking");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.xhigh).toBe("max");
  });

  test("claude-opus-4-7-thinking — newer thinking-effort order deduped with exact raw IDs", () => {
    const result = processModels([
      m("claude-opus-4-7-thinking-low"),
      m("claude-opus-4-7-thinking-high"),
      m("claude-opus-4-7-thinking-max"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-opus-4-7-thinking");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.xhigh).toBe("max");
    expect(result[0].rawModelByEffort!.max).toBe("claude-opus-4-7-thinking-max");
  });

  test("raw model lookup applies exact Cursor IDs for deduped effort variants", () => {
    const processed = processModels([
      m("claude-opus-4-7-thinking-high"),
      m("claude-opus-4-7-thinking-max"),
    ]);
    const payload: Record<string, unknown> = {
      model: "claude-opus-4-7-thinking",
      reasoning_effort: "max",
    };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("claude-opus-4-7-thinking-max");
  });

  test("raw model lookup tolerates unmapped Pi effort values by applying the effort map", () => {
    const processed = processModels([m("claude-4.6-opus-high"), m("claude-4.6-opus-max")]);
    const payload: Record<string, unknown> = { model: "claude-4.6-opus", reasoning_effort: "low" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("claude-4.6-opus-high");
  });

  test("raw model lookup routes suffix-style fast models through exact Cursor IDs", () => {
    const processed = processModels([m("gpt-5.4-medium-fast"), m("gpt-5.4-high-fast")]);
    const payload: Record<string, unknown> = { model: "gpt-5.4-fast", reasoning_effort: "high" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("gpt-5.4-high-fast");
    expect(payload.cursor_model_parameters).toBeUndefined();
  });

  test("raw model lookup routes GPT-5.5 1M through requestedModel parameters", () => {
    const processed = processModels([
      {
        ...m("gpt-5.5-1m-high", "GPT-5.5 1M High"),
        requestedModelId: "gpt-5.5",
        requiresMaxMode: true,
        requestedMaxMode: true,
        parameters: [
          { id: "context", value: "1m" },
          { id: "reasoning", value: "high" },
          { id: "fast", value: "false" },
        ],
      },
      {
        ...m("gpt-5.5-1m-medium", "GPT-5.5 1M"),
        requestedModelId: "gpt-5.5",
        requiresMaxMode: true,
        requestedMaxMode: true,
        parameters: [
          { id: "context", value: "1m" },
          { id: "reasoning", value: "medium" },
          { id: "fast", value: "false" },
        ],
      },
    ]);
    const payload: Record<string, unknown> = { model: "gpt-5.5-1m", reasoning_effort: "high" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("gpt-5.5");
    expect(payload.cursor_model_parameters).toEqual([
      { id: "context", value: "1m" },
      { id: "reasoning", value: "high" },
      { id: "fast", value: "false" },
    ]);
    expect(payload.cursor_requires_max_mode).toBe(true);
    expect(payload.cursor_model_max_mode).toBe(true);
  });

  test("raw model lookup routes GPT-5.5 272K max fast mode through maxMode=true and fast=true", () => {
    const processed = processModels([
      {
        ...m("gpt-5.5-max-medium-fast", "GPT-5.5 272K Max Fast"),
        requestedModelId: "gpt-5.5",
        requestedMaxMode: true,
        parameters: [
          { id: "context", value: "272k" },
          { id: "reasoning", value: "medium" },
          { id: "fast", value: "true" },
        ],
      },
      {
        ...m("gpt-5.5-max-high-fast", "GPT-5.5 272K Max High Fast"),
        requestedModelId: "gpt-5.5",
        requestedMaxMode: true,
        parameters: [
          { id: "context", value: "272k" },
          { id: "reasoning", value: "high" },
          { id: "fast", value: "true" },
        ],
      },
    ]);
    const payload: Record<string, unknown> = {
      model: "gpt-5.5-max-fast",
      reasoning_effort: "high",
    };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("gpt-5.5");
    expect(payload.cursor_model_parameters).toEqual([
      { id: "context", value: "272k" },
      { id: "reasoning", value: "high" },
      { id: "fast", value: "true" },
    ]);
    expect(payload.cursor_requires_max_mode).toBeUndefined();
    expect(payload.cursor_model_max_mode).toBe(true);
  });

  test("raw model lookup routes GPT-5.5 272K fast mode through fast=true requestedModel parameter", () => {
    const processed = processModels([
      {
        ...m("gpt-5.5-medium-fast", "GPT-5.5 Fast"),
        requestedModelId: "gpt-5.5",
        requestedMaxMode: false,
        parameters: [
          { id: "context", value: "272k" },
          { id: "reasoning", value: "medium" },
          { id: "fast", value: "true" },
        ],
      },
      {
        ...m("gpt-5.5-high-fast", "GPT-5.5 High Fast"),
        requestedModelId: "gpt-5.5",
        requestedMaxMode: false,
        parameters: [
          { id: "context", value: "272k" },
          { id: "reasoning", value: "high" },
          { id: "fast", value: "true" },
        ],
      },
    ]);
    const payload: Record<string, unknown> = { model: "gpt-5.5-fast", reasoning_effort: "high" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("gpt-5.5");
    expect(payload.cursor_model_parameters).toEqual([
      { id: "context", value: "272k" },
      { id: "reasoning", value: "high" },
      { id: "fast", value: "true" },
    ]);
    expect(payload.cursor_requires_max_mode).toBeUndefined();
    expect(payload.cursor_model_max_mode).toBe(false);
  });

  test("raw model lookup routes non-effort parameterized rows", () => {
    const processed = processModels([
      {
        ...m("composer-2-max-fast", "Composer 2 Max Fast"),
        requestedModelId: "composer-2",
        requestedMaxMode: true,
        parameters: [{ id: "fast", value: "true" }],
      },
    ]);
    const payload: Record<string, unknown> = { model: "composer-2-max-fast" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("composer-2");
    expect(payload.cursor_model_parameters).toEqual([{ id: "fast", value: "true" }]);
    expect(payload.cursor_model_max_mode).toBe(true);
  });

  test("raw model lookup has a safe default route when no reasoning_effort is present", () => {
    const processed = processModels([m("gpt-5.4-low"), m("gpt-5.4-medium"), m("gpt-5.4-high")]);
    const payload: Record<string, unknown> = { model: "gpt-5.4" };
    applyRawCursorModelId(payload, buildRawModelLookup(processed));
    expect(payload.cursor_model_id).toBe("gpt-5.4-medium");
  });

  test("claude-4.5-opus-high — single effort variant, deduped to base", () => {
    const result = processModels([m("claude-4.5-opus-high")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.5-opus");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.minimal).toBe("high");
  });

  test("claude-4.6-sonnet-medium — single effort variant, deduped to base", () => {
    const result = processModels([m("claude-4.6-sonnet-medium")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-sonnet");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe("medium");
  });

  test("composer-2 — single model without effort, NOT deduped", () => {
    const result = processModels([m("composer-2")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("composer-2");
    expect(result[0].supportsEffort).toBe(false);
  });

  test("gpt-5.1-codex-max — deduped, max stays in base name", () => {
    const result = processModels([
      m("gpt-5.1-codex-max-low"),
      m("gpt-5.1-codex-max-medium"),
      m("gpt-5.1-codex-max-high"),
      m("gpt-5.1-codex-max-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.1-codex-max");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("gpt-5.3-codex-spark-preview — deduped", () => {
    const result = processModels([
      m("gpt-5.3-codex-spark-preview"),
      m("gpt-5.3-codex-spark-preview-high"),
      m("gpt-5.3-codex-spark-preview-low"),
      m("gpt-5.3-codex-spark-preview-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.3-codex-spark-preview");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("standalone models pass through", () => {
    const result = processModels([
      m("default"),
      m("gemini-3-flash"),
      m("kimi-k2.5"),
      m("grok-4-20"),
      m("grok-4-20-thinking"),
    ]);
    expect(result).toHaveLength(5);
    expect(result.every((r) => r.supportsEffort === false)).toBe(true);
  });

  test("uses representative name from medium variant", () => {
    const result = processModels([
      m("gpt-5.4-low", "GPT-5.4 1M Low"),
      m("gpt-5.4-medium", "GPT-5.4 1M"),
      m("gpt-5.4-high", "GPT-5.4 1M High"),
    ]);
    expect(result[0].name).toBe("GPT-5.4 1M");
  });

  test("uses representative name from default (no-suffix) variant", () => {
    const result = processModels([
      m("gpt-5.2", "GPT-5.2"),
      m("gpt-5.2-high", "GPT-5.2 High"),
      m("gpt-5.2-low", "GPT-5.2 Low"),
    ]);
    expect(result[0].name).toBe("GPT-5.2");
  });

  test("full raw model list dedup count", () => {
    const result = processModels(rawModels as CursorModel[]);
    // Should be significantly fewer than 83
    expect(result.length).toBeLessThan(50);
    expect(result.length).toBeGreaterThan(20);

    // Spot checks
    const composer2 = result.find((r) => r.id === "composer-2");
    expect(composer2).toBeDefined();
    expect(composer2!.supportsEffort).toBe(false);

    const gpt54 = result.find((r) => r.id === "gpt-5.4");
    expect(gpt54).toBeDefined();
    expect(gpt54!.supportsEffort).toBe(true);
    expect(gpt54!.name).toBe("GPT-5.4 1M");
    expect(gpt54!.contextWindow).toBe(1_000_000);

    const gpt55 = result.find((r) => r.id === "gpt-5.5");
    expect(gpt55).toBeDefined();
    expect(gpt55!.supportsEffort).toBe(true);
    expect(gpt55!.name).toBe("GPT-5.5 272K");
    expect(gpt55!.contextWindow).toBe(272_000);
    expect(gpt55!.effortMap!.minimal).toBe("none");
    expect(gpt55!.effortMap!.xhigh).toBe("xhigh");
    expect(gpt55!.rawModelByEffort!.xhigh).toBe("gpt-5.5-extra-high");

    const gpt55OneMillion = result.find((r) => r.id === "gpt-5.5-1m");
    expect(gpt55OneMillion).toBeDefined();
    expect(gpt55OneMillion!.supportsEffort).toBe(true);
    expect(gpt55OneMillion!.name).toBe("GPT-5.5 1M");
    expect(gpt55OneMillion!.contextWindow).toBe(1_000_000);
    expect(gpt55OneMillion!.rawRoutingByEffort!.high).toEqual({
      modelId: "gpt-5.5",
      requiresMaxMode: true,
      requestedMaxMode: true,
      parameters: [
        { id: "context", value: "1m" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "false" },
      ],
    });

    const gpt55Fast = result.find((r) => r.id === "gpt-5.5-fast");
    expect(gpt55Fast).toBeDefined();
    expect(gpt55Fast!.contextWindow).toBe(272_000);
    expect(gpt55Fast!.rawRoutingByEffort!.high!.requestedMaxMode).toBe(false);

    const gpt55MaxFast = result.find((r) => r.id === "gpt-5.5-max-fast");
    expect(gpt55MaxFast).toBeDefined();
    expect(gpt55MaxFast!.contextWindow).toBe(272_000);
    expect(gpt55MaxFast!.rawRoutingByEffort!.high).toEqual({
      modelId: "gpt-5.5",
      requestedMaxMode: true,
      parameters: [
        { id: "context", value: "272k" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    });

    expect(result.find((r) => r.id === "gpt-5.5-1m-fast")).toBeUndefined();

    // Opus should be deduped too
    const opus46 = result.find((r) => r.id === "claude-4.6-opus");
    expect(opus46).toBeDefined();
    expect(opus46!.supportsEffort).toBe(true);
    expect(result.find((r) => r.id === "claude-4.6-opus-high")).toBeUndefined();
    expect(result.find((r) => r.id === "claude-4.6-opus-max")).toBeUndefined();

    // No raw effort IDs should leak through for deduped models
    expect(result.find((r) => r.id === "gpt-5.4-medium")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.4-high")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-medium")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-high")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-extra-high")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-1m-medium")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-1m-high")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.5-1m-extra-high")).toBeUndefined();
    expect(result.find((r) => r.id === "gpt-5.2-low")).toBeUndefined();
  });
});

// ── no reasoning effort ──

describe("applyNoReasoningEffort", () => {
  test("maps thinking off to Cursor none effort when available", () => {
    const payload: Record<string, unknown> = { model: "gpt-5.4-mini" };
    applyNoReasoningEffort(payload, "off", new Map([["gpt-5.4-mini", "none"]]));
    expect(payload.reasoning_effort).toBe("none");
  });

  test("does not override explicit reasoning effort", () => {
    const payload: Record<string, unknown> = { model: "gpt-5.4-mini", reasoning_effort: "high" };
    applyNoReasoningEffort(payload, "off", new Map([["gpt-5.4-mini", "none"]]));
    expect(payload.reasoning_effort).toBe("high");
  });

  test("leaves models without none effort unchanged", () => {
    const payload: Record<string, unknown> = { model: "gpt-5.4" };
    applyNoReasoningEffort(payload, "off", new Map([["gpt-5.4-mini", "none"]]));
    expect(payload.reasoning_effort).toBeUndefined();
  });
});

// ── resolveModelId ──

describe("resolveModelId", () => {
  test("no effort — returns model as-is", () => {
    expect(resolveModelId("composer-2")).toBe("composer-2");
    expect(resolveModelId("composer-2", undefined)).toBe("composer-2");
    expect(resolveModelId("composer-2", "")).toBe("composer-2");
  });

  test("plain model + effort", () => {
    expect(resolveModelId("gpt-5.4", "medium")).toBe("gpt-5.4-medium");
    expect(resolveModelId("gpt-5.4", "high")).toBe("gpt-5.4-high");
    expect(resolveModelId("gpt-5.4", "xhigh")).toBe("gpt-5.4-xhigh");
    expect(resolveModelId("gpt-5.5", "high")).toBe("gpt-5.5-high");
    expect(resolveModelId("gpt-5.5-1m", "high")).toBe("gpt-5.5-1m-high");
  });

  test("fast model + effort — inserts before -fast", () => {
    expect(resolveModelId("gpt-5.4-fast", "medium")).toBe("gpt-5.4-medium-fast");
    expect(resolveModelId("gpt-5.4-fast", "high")).toBe("gpt-5.4-high-fast");
    expect(resolveModelId("gpt-5.5-fast", "high")).toBe("gpt-5.5-high-fast");
  });

  test("thinking model + effort — inserts before -thinking", () => {
    expect(resolveModelId("claude-4.6-opus-thinking", "high")).toBe(
      "claude-4.6-opus-high-thinking",
    );
    expect(resolveModelId("claude-4.6-opus-thinking", "max")).toBe("claude-4.6-opus-max-thinking");
  });

  test("codex-max model + effort", () => {
    expect(resolveModelId("gpt-5.1-codex-max", "high")).toBe("gpt-5.1-codex-max-high");
    expect(resolveModelId("gpt-5.1-codex-max", "medium")).toBe("gpt-5.1-codex-max-medium");
  });

  test("codex-max-fast model + effort", () => {
    expect(resolveModelId("gpt-5.1-codex-max-fast", "high")).toBe("gpt-5.1-codex-max-high-fast");
  });

  test("spark-preview model + effort", () => {
    expect(resolveModelId("gpt-5.3-codex-spark-preview", "xhigh")).toBe(
      "gpt-5.3-codex-spark-preview-xhigh",
    );
  });

  test("exact cursor model id override wins over suffix insertion", () => {
    expect(
      resolveRequestedModelId("claude-opus-4-7-thinking", "max", "claude-opus-4-7-thinking-max"),
    ).toBe("claude-opus-4-7-thinking-max");
  });
});

// ── Session key derivation ──

const msg = (role: "user" | "assistant" | "system", content: string) => ({ role, content });
const assistantStep = (text: string) => ({ kind: "assistantText", text }) as const;
const toolStep = (
  toolCallId: string,
  toolName: string,
  args: Record<string, unknown>,
  result?: {
    content: string;
    isError: boolean;
    images?: Array<{ data: Uint8Array; mimeType: string }>;
  },
) =>
  ({
    kind: "toolCall",
    toolCallId,
    toolName,
    arguments: args,
    ...(result ? { result } : {}),
  }) as const;
const turn = (userText: string, steps: ParsedTurn["steps"] = []): ParsedTurn => ({
  userText,
  steps,
});
const pngBytes = () => new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4]);
const pngBase64 = () => Buffer.from(pngBytes()).toString("base64");
const jpegBytes = () => new Uint8Array([0xff, 0xd8, 0xff, 1, 2, 3, 4]);
const jpegBase64 = () => Buffer.from(jpegBytes()).toString("base64");

describe("extractToolResultImagePayloads", () => {
  test("extracts image blocks from matching tool-result messages in the Pi branch", () => {
    const ctx = {
      sessionManager: {
        getBranch: () => [
          {
            message: {
              role: "toolResult",
              toolCallId: "tc1",
              content: [
                { type: "text", text: "screenshot" },
                { type: "image", data: pngBase64(), mimeType: "image/png" },
              ],
            },
          },
          {
            message: {
              role: "toolResult",
              toolCallId: "tc2",
              content: [{ type: "image", data: jpegBase64(), mimeType: "image/jpeg" }],
            },
          },
        ],
      },
    };

    const extracted = extractToolResultImagePayloads(ctx, {
      messages: [{ role: "tool", tool_call_id: "tc1", content: "(see attached image)" }],
    });

    expect(extracted).toEqual([
      { toolCallId: "tc1", images: [{ data: pngBase64(), mimeType: "image/png" }] },
    ]);
  });

  test("ignores branch images whose toolCallId is not in the provider payload", () => {
    const extracted = extractToolResultImagePayloads(
      {
        sessionManager: {
          getBranch: () => [
            {
              message: {
                role: "toolResult",
                toolCallId: "tc-other",
                content: [{ type: "image", data: pngBase64(), mimeType: "image/png" }],
              },
            },
          ],
        },
      },
      {
        messages: [{ role: "tool", tool_call_id: "tc1", content: "result" }],
      },
    );

    expect(extracted).toEqual([]);
  });
});

describe("deriveBridgeKey", () => {
  test("uses sessionId when provided", () => {
    const msgs = [msg("user", "hello")];
    const a = deriveBridgeKey(msgs, "session-abc");
    const b = deriveBridgeKey(msgs, "session-abc");
    expect(a).toBe(b);
  });

  test("different sessionIds produce different keys", () => {
    const msgs = [msg("user", "hello")];
    const a = deriveBridgeKey(msgs, "session-1");
    const b = deriveBridgeKey(msgs, "session-2");
    expect(a).not.toBe(b);
  });

  test("same sessionId ignores later messages", () => {
    const a = deriveBridgeKey([msg("user", "hello")], "session-1");
    const b = deriveBridgeKey([msg("user", "goodbye")], "session-1");
    expect(a).toBe(b);
  });

  test("falls back to first user message hash without sessionId", () => {
    const msgs1 = [msg("user", "hello")];
    const msgs2 = [msg("user", "hello"), msg("assistant", "hi"), msg("user", "bye")];
    expect(deriveBridgeKey(msgs1)).toBe(deriveBridgeKey(msgs2));
  });

  test("fallback differs by first user message", () => {
    const a = deriveBridgeKey([msg("user", "hello")]);
    const b = deriveBridgeKey([msg("user", "goodbye")]);
    expect(a).not.toBe(b);
  });
});

describe("deriveConversationKey", () => {
  test("same sessionId → same key regardless of messages", () => {
    const a = deriveConversationKey([msg("user", "hello")], "session-x");
    const b = deriveConversationKey([msg("user", "totally different")], "session-x");
    expect(a).toBe(b);
  });

  test("different sessionIds → different keys", () => {
    const a = deriveConversationKey([msg("user", "hello")], "session-1");
    const b = deriveConversationKey([msg("user", "hello")], "session-2");
    expect(a).not.toBe(b);
  });

  test("falls back to first user message hash without sessionId", () => {
    const a = deriveConversationKey([msg("user", "hello")]);
    const b = deriveConversationKey([msg("user", "hello"), msg("assistant", "hi")]);
    expect(a).toBe(b);
  });
});

describe("session cleanup", () => {
  function seedSessionState(sessionId: string) {
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const writes: Uint8Array[] = [];
    let ended = 0;
    const heartbeatTimer = setInterval(() => {}, 60_000);
    __testInternals.activeBridges.set(bridgeKey, {
      bridge: {
        get alive() {
          return true;
        },
        write(data: Uint8Array) {
          writes.push(data);
        },
        end() {
          ended++;
        },
        onData() {},
        onClose() {},
        proc: {} as any,
      } as any,
      heartbeatTimer,
      blobStore: new Map(),
      mcpTools: [],
      pendingExecs: [],
      currentTurn: turn("current"),
    });
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });
    return {
      bridgeKey,
      convKey,
      writes,
      get ended() {
        return ended;
      },
    };
  }

  test("cleanupSessionState removes active bridge and conversation for the session", () => {
    const seeded = seedSessionState("session-a");
    cleanupSessionState("session-a");
    expect(__testInternals.activeBridges.has(seeded.bridgeKey)).toBe(false);
    expect(__testInternals.conversationStates.has(seeded.convKey)).toBe(false);
    expect(seeded.writes.length).toBe(1);
    expect(seeded.ended).toBe(1);
  });

  test("cleanupSessionState does not touch another session", () => {
    const a = seedSessionState("session-a");
    const b = seedSessionState("session-b");
    cleanupSessionState("session-a");
    expect(__testInternals.activeBridges.has(a.bridgeKey)).toBe(false);
    expect(__testInternals.conversationStates.has(a.convKey)).toBe(false);
    expect(__testInternals.activeBridges.has(b.bridgeKey)).toBe(true);
    expect(__testInternals.conversationStates.has(b.convKey)).toBe(true);
  });
});

describe("session cleanup hook wiring", () => {
  test("registerSessionLifecycleCleanup wires switch/fork/tree/shutdown to cleanup current session", async () => {
    const handlers = new Map<string, Function>();
    const pi = {
      on(event: string, handler: Function) {
        handlers.set(event, handler);
      },
    } as any;

    registerSessionLifecycleCleanup(pi);

    const sessionId = "session-hook";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const heartbeatTimer = setInterval(() => {}, 60_000);
    __testInternals.activeBridges.set(bridgeKey, {
      bridge: {
        get alive() {
          return false;
        },
        write() {},
        end() {},
        onData() {},
        onClose() {},
        proc: {} as any,
      } as any,
      heartbeatTimer,
      blobStore: new Map(),
      mcpTools: [],
      pendingExecs: [],
      currentTurn: turn("current"),
    });
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const ctx = { sessionManager: { getSessionId: () => sessionId } };
    for (const event of [
      "session_before_switch",
      "session_before_fork",
      "session_before_tree",
      "session_shutdown",
    ]) {
      __testInternals.activeBridges.set(bridgeKey, {
        bridge: {
          get alive() {
            return false;
          },
          write() {},
          end() {},
          onData() {},
          onClose() {},
          proc: {} as any,
        } as any,
        heartbeatTimer,
        blobStore: new Map(),
        mcpTools: [],
        pendingExecs: [],
        currentTurn: turn("current"),
      });
      __testInternals.conversationStates.set(convKey, {
        conversationId: "conv",
        checkpoint: null,

        sessionScoped: true,
        blobStore: new Map(),
        lastAccessMs: Date.now(),
      });
      await handlers.get(event)?.({}, ctx);
      expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);
      expect(__testInternals.conversationStates.has(convKey)).toBe(false);
    }
  });
});

describe("model switch cleanup hook", () => {
  test("cleans cursor session state when crossing the cursor provider boundary", async () => {
    const handlers = new Map<string, Function>();
    const pi = {
      on(event: string, handler: Function) {
        handlers.set(event, handler);
      },
    } as any;

    registerCursorModelSwitchCleanup(pi);

    const sessionId = "session-model-switch";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const seedConversation = () => {
      __testInternals.conversationStates.set(convKey, {
        conversationId: "conv-model-switch",
        checkpoint: null,
        sessionScoped: true,
        blobStore: new Map(),
        lastAccessMs: Date.now(),
      });
    };
    const ctx = { sessionManager: { getSessionId: () => sessionId } };

    seedConversation();
    await handlers.get("model_select")?.(
      { previousModel: { provider: "anthropic" }, model: { provider: "cursor" } },
      ctx,
    );
    expect(__testInternals.conversationStates.has(convKey)).toBe(false);

    seedConversation();
    await handlers.get("model_select")?.(
      { previousModel: { provider: "cursor" }, model: { provider: "openai" } },
      ctx,
    );
    expect(__testInternals.conversationStates.has(convKey)).toBe(false);

    seedConversation();
    await handlers.get("model_select")?.(
      { previousModel: { provider: "anthropic" }, model: { provider: "openai" } },
      ctx,
    );
    expect(__testInternals.conversationStates.has(convKey)).toBe(true);
  });
});

describe("session-scoped eviction policy", () => {
  test("evictStaleConversations keeps session-scoped state past TTL", () => {
    const convKey = deriveConversationKeyFromSessionId("session-ttl");
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-session",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: 0,
    });

    evictStaleConversations(31 * 60 * 1000);
    expect(__testInternals.conversationStates.has(convKey)).toBe(true);
  });

  test("evictStaleConversations removes anonymous state past TTL", () => {
    const convKey = "anon-key";
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-anon",
      checkpoint: null,

      sessionScoped: false,
      blobStore: new Map(),
      lastAccessMs: 0,
    });

    evictStaleConversations(31 * 60 * 1000);
    expect(__testInternals.conversationStates.has(convKey)).toBe(false);
  });

  test("cleanupSessionState still removes session-scoped state explicitly", () => {
    const sessionId = "session-explicit";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-explicit",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: 0,
    });

    cleanupSessionState(sessionId);
    expect(__testInternals.conversationStates.has(convKey)).toBe(false);
  });
});

describe("derivePiSessionId", () => {
  test("prefers pi_session_id over user", () => {
    expect(derivePiSessionId({ pi_session_id: "a", user: "b" })).toBe("a");
  });

  test("falls back to user", () => {
    expect(derivePiSessionId({ user: "legacy" })).toBe("legacy");
  });

  test("trims whitespace", () => {
    expect(derivePiSessionId({ pi_session_id: "  x  " })).toBe("x");
  });

  test("returns undefined when empty", () => {
    expect(derivePiSessionId({ pi_session_id: "   ", user: "" })).toBeUndefined();
  });
});

// ── Turn reconstruction ──

function decodeRunRequest(payload: ReturnType<typeof buildCursorRequest>) {
  const clientMsg = fromBinary(AgentClientMessageSchema, payload.requestBytes);
  expect(clientMsg.message.case).toBe("runRequest");
  return clientMsg.message.value as InstanceType<(typeof AgentRunRequestSchema)["$typeName"]> & any;
}

function requestedModelSummary(req: any) {
  return {
    modelId: req.requestedModel.modelId,
    maxMode: req.requestedModel.maxMode,
    parameters: req.requestedModel.parameters.map((parameter: any) => ({
      id: parameter.id,
      value: parameter.value,
    })),
  };
}

function routedRequestSummary(rawModels: CursorModel[], model: string, reasoningEffort?: string) {
  const processed = processModels(rawModels);
  const body: Record<string, unknown> = {
    model,
    ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
  };
  applyRawCursorModelId(body, buildRawModelLookup(processed));
  const modelId = resolveRequestedModelId(
    String(body.model),
    typeof body.reasoning_effort === "string" ? body.reasoning_effort : undefined,
    typeof body.cursor_model_id === "string" ? body.cursor_model_id : undefined,
  );
  const maxMode =
    typeof body.cursor_model_max_mode === "boolean"
      ? body.cursor_model_max_mode
      : body.cursor_requires_max_mode === true;
  return requestedModelSummary(
    decodeRunRequest(
      buildCursorRequest(
        modelId,
        "system",
        "hello",
        [],
        "conv-1",
        null,
        undefined,
        maxMode,
        (body.cursor_model_parameters as any) ?? [],
      ),
    ),
  );
}

function resolveBlob(data: Uint8Array, blobStore?: Map<string, Uint8Array>): Uint8Array {
  if (blobStore && data.length === 32) {
    const resolved = blobStore.get(Buffer.from(data).toString("hex"));
    if (resolved) return resolved;
  }
  return data;
}

function decodeTurns(state: any, blobStore?: Map<string, Uint8Array>) {
  return (state.turns as Uint8Array[]).map((turnRef: Uint8Array) => {
    const turnBytes = resolveBlob(turnRef, blobStore);
    const turnStruct = fromBinary(ConversationTurnStructureSchema, turnBytes);
    expect(turnStruct.turn.case).toBe("agentConversationTurn");
    const agentTurn = turnStruct.turn.value as any;
    const userMsg = fromBinary(UserMessageSchema, resolveBlob(agentTurn.userMessage, blobStore));
    const steps = (agentTurn.steps as Uint8Array[]).map((s: Uint8Array) =>
      fromBinary(ConversationStepSchema, resolveBlob(s, blobStore)),
    );
    return { userMsg, steps };
  });
}

describe("buildCursorRequest — turn reconstruction", () => {
  test("no checkpoint, no turns — empty turns array", () => {
    const payload = buildCursorRequest("gpt-5", "system", "hello", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.conversationState.turns).toHaveLength(0);
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("hello");
  });

  test("uses requestedModel instead of legacy modelDetails", () => {
    const payload = buildCursorRequest("gpt-5", "system", "hello", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.modelDetails).toBeUndefined();
    expect(req.requestedModel.modelId).toBe("gpt-5");
    expect(req.requestedModel.maxMode).toBe(false);
    expect(req.requestedModel.parameters).toEqual([]);
  });

  test("max mode flag is sent without changing the model ID", () => {
    const payload = buildCursorRequest(
      "gpt-5.3-codex",
      "system",
      "hello",
      [],
      "conv-1",
      null,
      undefined,
      true,
    );
    const req = decodeRunRequest(payload);
    expect(req.modelDetails).toBeUndefined();
    expect(req.requestedModel.modelId).toBe("gpt-5.3-codex");
    expect(req.requestedModel.maxMode).toBe(true);
    expect(req.requestedModel.parameters).toEqual([]);
  });

  test("required max mode flag is sent for 1M parameterized models", () => {
    const payload = buildCursorRequest(
      "gpt-5.5",
      "system",
      "hello",
      [],
      "conv-1",
      null,
      undefined,
      true,
      [
        { id: "context", value: "1m" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    );
    const req = decodeRunRequest(payload);
    expect(req.requestedModel.modelId).toBe("gpt-5.5");
    expect(req.requestedModel.maxMode).toBe(true);
  });

  test("requestedModel includes Cursor parameterized model settings", () => {
    const payload = buildCursorRequest(
      "gpt-5.5",
      "system",
      "hello",
      [],
      "conv-1",
      null,
      undefined,
      false,
      [
        { id: "context", value: "1m" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    );
    const req = decodeRunRequest(payload);
    expect(req.requestedModel.modelId).toBe("gpt-5.5");
    expect(
      req.requestedModel.parameters.map((parameter: any) => ({
        id: parameter.id,
        value: parameter.value,
      })),
    ).toEqual([
      { id: "context", value: "1m" },
      { id: "reasoning", value: "high" },
      { id: "fast", value: "true" },
    ]);
  });

  test("routed fast/max model selections produce the expected Cursor requestedModel", () => {
    expect(
      routedRequestSummary(
        [m("gpt-5.4-medium-fast"), m("gpt-5.4-high-fast")],
        "gpt-5.4-fast",
        "high",
      ),
    ).toEqual({
      modelId: "gpt-5.4-high-fast",
      maxMode: false,
      parameters: [],
    });

    expect(
      routedRequestSummary(
        [
          {
            ...m("gpt-5.5-medium-fast", "GPT-5.5 Fast"),
            requestedModelId: "gpt-5.5",
            requestedMaxMode: false,
            parameters: [
              { id: "context", value: "272k" },
              { id: "reasoning", value: "medium" },
              { id: "fast", value: "true" },
            ],
          },
          {
            ...m("gpt-5.5-high-fast", "GPT-5.5 High Fast"),
            requestedModelId: "gpt-5.5",
            requestedMaxMode: false,
            parameters: [
              { id: "context", value: "272k" },
              { id: "reasoning", value: "high" },
              { id: "fast", value: "true" },
            ],
          },
        ],
        "gpt-5.5-fast",
        "high",
      ),
    ).toEqual({
      modelId: "gpt-5.5",
      maxMode: false,
      parameters: [
        { id: "context", value: "272k" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    });

    expect(
      routedRequestSummary(
        [
          {
            ...m("gpt-5.5-max-high-fast", "GPT-5.5 Max High Fast"),
            requestedModelId: "gpt-5.5",
            requestedMaxMode: true,
            parameters: [
              { id: "context", value: "272k" },
              { id: "reasoning", value: "high" },
              { id: "fast", value: "true" },
            ],
          },
        ],
        "gpt-5.5-max-fast",
        "high",
      ),
    ).toEqual({
      modelId: "gpt-5.5",
      maxMode: true,
      parameters: [
        { id: "context", value: "272k" },
        { id: "reasoning", value: "high" },
        { id: "fast", value: "true" },
      ],
    });

    expect(
      routedRequestSummary(
        [
          {
            ...m("composer-2-max-fast", "Composer 2 Max Fast"),
            requestedModelId: "composer-2",
            requestedMaxMode: true,
            parameters: [{ id: "fast", value: "true" }],
          },
        ],
        "composer-2-max-fast",
      ),
    ).toEqual({
      modelId: "composer-2",
      maxMode: true,
      parameters: [{ id: "fast", value: "true" }],
    });
  });

  test("includes MCP tools in the initial AgentRunRequest like Cursor CLI", () => {
    const tool = create(McpToolDefinitionSchema, {
      name: "read_file",
      providerIdentifier: "pi",
      toolName: "read_file",
      description: "Read a file",
      inputSchema: new Uint8Array([1, 2, 3]),
    });
    const payload = buildCursorRequest(
      "gpt-5",
      "system",
      "hello",
      [],
      "conv-1",
      null,
      undefined,
      false,
      [],
      [tool],
    );
    const req = decodeRunRequest(payload);
    expect(payload.mcpTools).toHaveLength(1);
    expect(req.mcpTools.mcpTools).toHaveLength(1);
    expect(req.mcpTools.mcpTools[0].name).toBe("read_file");
    expect(req.mcpTools.mcpTools[0].providerIdentifier).toBe("pi");
  });

  test("adds inline images to the current user message selected context", () => {
    const image = { data: pngBytes(), mimeType: "image/png" };
    const payload = buildCursorRequest(
      "gpt-5",
      "system",
      "describe this",
      [],
      "conv-1",
      null,
      undefined,
      false,
      [],
      [],
      [image],
    );
    const req = decodeRunRequest(payload);
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("describe this");
    expect(userAction.userMessage.selectedContext.selectedImages).toHaveLength(1);
    const selectedImage = userAction.userMessage.selectedContext.selectedImages[0];
    expect(selectedImage.mimeType).toBe("image/png");
    expect(selectedImage.dataOrBlobId.case).toBe("data");
    expect(Array.from(selectedImage.dataOrBlobId.value)).toEqual(Array.from(pngBytes()));
    expect(selectedImage.uuid).toMatch(/^[0-9a-f-]{36}$/i);
  });

  test("preserves images when reconstructing prior user turns", () => {
    const image = { data: jpegBytes(), mimeType: "image/jpeg" };
    const turns = [{ ...turn("what is this?", [assistantStep("a photo")]), userImages: [image] }];
    const payload = buildCursorRequest("gpt-5", "system", "thanks", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    expect(decoded[0].userMsg.selectedContext.selectedImages).toHaveLength(1);
    const selectedImage = decoded[0].userMsg.selectedContext.selectedImages[0];
    expect(selectedImage.mimeType).toBe("image/jpeg");
    expect(selectedImage.dataOrBlobId.case).toBe("data");
    expect(Array.from(selectedImage.dataOrBlobId.value)).toEqual(Array.from(jpegBytes()));
  });

  test("no checkpoint, with assistant-text turns — reconstructs protobuf turns without inline fallback", () => {
    const turns = [
      turn("first question", [assistantStep("first answer")]),
      turn("second question", [assistantStep("second answer")]),
    ];
    const payload = buildCursorRequest("gpt-5", "system", "third question", turns, "conv-1", null);
    const req = decodeRunRequest(payload);

    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    expect(decoded).toHaveLength(2);

    expect(decoded[0].userMsg.text).toBe("first question");
    expect(decoded[0].steps).toHaveLength(1);
    expect(decoded[0].steps[0].message.case).toBe("assistantMessage");
    expect((decoded[0].steps[0].message.value as any).text).toBe("first answer");

    expect(decoded[1].userMsg.text).toBe("second question");
    expect(decoded[1].steps[0].message.case).toBe("assistantMessage");
    expect((decoded[1].steps[0].message.value as any).text).toBe("second answer");

    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("third question");
    expect(userAction.userMessage.text).not.toContain("<conversation_history>");
  });

  test("no checkpoint, reconstructs tool-call steps and final assistant text", () => {
    const turns = [
      turn("inspect file", [
        toolStep(
          "tc1",
          "read",
          { path: "src/index.ts" },
          { content: "file contents", isError: false },
        ),
        assistantStep("I found the issue."),
      ]),
    ];
    const payload = buildCursorRequest("gpt-5", "system", "fix it", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState, payload.blobStore);

    expect(decoded).toHaveLength(1);
    expect(decoded[0].userMsg.text).toBe("inspect file");
    expect(decoded[0].steps).toHaveLength(2);

    const toolCallStep = decoded[0].steps[0]!;
    expect(toolCallStep.message.case).toBe("toolCall");
    expect(toolCallStep.message.value.tool.case).toBe("mcpToolCall");
    expect(toolCallStep.message.value.tool.value.args?.toolCallId).toBe("tc1");
    expect(toolCallStep.message.value.tool.value.args?.toolName).toBe("read");
    expect(toolCallStep.message.value.tool.value.result?.result.case).toBe("success");
    expect(
      toolCallStep.message.value.tool.value.result?.result.value.content[0]?.content.case,
    ).toBe("text");
    expect(
      toolCallStep.message.value.tool.value.result?.result.value.content[0]?.content.value.text,
    ).toBe("file contents");

    const finalAssistantStep = decoded[0].steps[1]!;
    expect(finalAssistantStep.message.case).toBe("assistantMessage");
    expect((finalAssistantStep.message.value as any).text).toBe("I found the issue.");

    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("fix it");
  });

  test("no checkpoint, reconstructs tool-call image result content", () => {
    const turns = [
      turn("capture screenshot", [
        toolStep(
          "tc1",
          "screenshot",
          {},
          { content: "", isError: false, images: [{ data: pngBytes(), mimeType: "image/png" }] },
        ),
      ]),
    ];
    const payload = buildCursorRequest(
      "gpt-5",
      "system",
      "what do you see?",
      turns,
      "conv-1",
      null,
    );
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    const toolCallStep = decoded[0].steps[0]!;
    expect(toolCallStep.message.case).toBe("toolCall");
    const success = toolCallStep.message.value.tool.value.result.result.value;
    expect(success.content).toHaveLength(1);
    expect(success.content[0].content.case).toBe("image");
    expect(success.content[0].content.value.mimeType).toBe("image/png");
    expect(Array.from(success.content[0].content.value.data)).toEqual(Array.from(pngBytes()));
  });

  test("no checkpoint, turn with no steps — no reconstructed steps", () => {
    const turns = [turn("hello")];
    const payload = buildCursorRequest("gpt-5", "system", "follow up", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    expect(decoded).toHaveLength(1);
    expect(decoded[0].userMsg.text).toBe("hello");
    expect(decoded[0].steps).toHaveLength(0);
  });

  test("with checkpoint — uses checkpoint, ignores turns", () => {
    const priorPayload = buildCursorRequest("gpt-5", "system", "hello", [], "conv-1", null);
    const priorReq = decodeRunRequest(priorPayload);
    const checkpoint = toBinary(ConversationStateStructureSchema, priorReq.conversationState);

    const turns = [turn("SHOULD NOT APPEAR", [assistantStep("SHOULD NOT APPEAR")])];
    const payload = buildCursorRequest("gpt-5", "system", "next", turns, "conv-1", checkpoint);
    const req = decodeRunRequest(payload);

    expect(req.conversationState.turns).toHaveLength(0);
  });

  test("system prompt stored in blobStore", () => {
    const payload = buildCursorRequest("gpt-5", "You are helpful", "hi", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.conversationState.rootPromptMessagesJson).toHaveLength(1);
    const blobId = Buffer.from(req.conversationState.rootPromptMessagesJson[0]).toString("hex");
    expect(payload.blobStore.has(blobId)).toBe(true);
    const blobData = JSON.parse(new TextDecoder().decode(payload.blobStore.get(blobId)!));
    expect(blobData.role).toBe("system");
    expect(blobData.content).toBe("You are helpful");
  });

  test("each reconstructed turn has a unique messageId", () => {
    const turns = [turn("a", [assistantStep("b")]), turn("a", [assistantStep("b")])];
    const payload = buildCursorRequest("gpt-5", "system", "c", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    expect(decoded[0].userMsg.messageId).not.toBe(decoded[1].userMsg.messageId);
  });
});

// ── Fork via checkpoint discard + reconstruction ──

describe("fork discards checkpoint, reconstruction takes over", () => {
  test("fork scenario — checkpoint discarded, turns reconstructed from messages", () => {
    const turns = [turn("first", [assistantStep("response1")])];
    const payload = buildCursorRequest("gpt-5", "system", "forked question", turns, "conv-1", null);
    const req = decodeRunRequest(payload);

    const decoded = decodeTurns(req.conversationState, payload.blobStore);
    expect(decoded).toHaveLength(1);
    expect(decoded[0].userMsg.text).toBe("first");
    expect((decoded[0].steps[0].message.value as any).text).toBe("response1");

    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("forked question");
    expect(userAction.userMessage.text).not.toContain("<conversation_history>");
  });

  test("fork to beginning — no turns, no reconstruction", () => {
    const payload = buildCursorRequest("gpt-5", "system", "start over", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.conversationState.turns).toHaveLength(0);
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("start over");
  });
});

// ── Tool-aware parsing ──

describe("parseMessages — structured tool turns", () => {
  test("extracts OpenAI-style data URL images from user content", () => {
    const parsed = parseMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "describe this" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
        ],
      },
    ]);

    expect(parsed.userText).toBe("describe this");
    expect(parsed.userImages).toHaveLength(1);
    expect(parsed.userImages[0].mimeType).toBe("image/png");
    expect(Array.from(parsed.userImages[0].data)).toEqual(Array.from(pngBytes()));
  });

  test("accepts image-only user prompts", () => {
    const parsed = parseMessages([
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
        ],
      },
    ]);

    expect(parsed.userText).toBe("");
    expect(parsed.userImages).toHaveLength(1);
    expect(parsed.userImages[0].mimeType).toBe("image/png");
  });

  test("rejects remote OpenAI image_url values with an explicit error", () => {
    expect(() =>
      parseMessages([
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            { type: "image_url", image_url: { url: "https://example.com/image.png" } },
          ],
        },
      ]),
    ).toThrow(/Remote image URLs are not supported/);
  });

  test("rejects user images that do not match Cursor CLI supported magic bytes", () => {
    expect(() =>
      parseMessages([
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AQIDBA==" } },
          ],
        },
      ]),
    ).toThrow(/Unsupported image type/);
  });

  test("rejects oversized user images using Cursor CLI's processed payload cap", () => {
    const bytes = new Uint8Array(5_242_881);
    bytes.set([0xff, 0xd8, 0xff]);
    expect(() =>
      parseMessages([
        {
          role: "user",
          content: [
            { type: "image", data: Buffer.from(bytes).toString("base64"), mimeType: "image/jpeg" },
          ],
        },
      ]),
    ).toThrow(/exceeds Cursor CLI's 5242880 byte limit/);
  });

  test("preserves images on completed prior turns", () => {
    const parsed = parseMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpegBase64()}` } },
        ],
      },
      { role: "assistant", content: "done" },
      { role: "user", content: "next" },
    ]);

    expect(parsed.userText).toBe("next");
    expect(parsed.turns).toHaveLength(1);
    expect(parsed.turns[0].userImages).toHaveLength(1);
    expect(parsed.turns[0].userImages?.[0].mimeType).toBe("image/jpeg");
    expect(Array.from(parsed.turns[0].userImages![0].data)).toEqual(Array.from(jpegBytes()));
  });

  test("extracts tool result images from inline tool content", () => {
    const parsed = parseMessages([
      { role: "user" as const, content: "capture screenshot" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          {
            id: "tc1",
            type: "function" as const,
            function: { name: "screenshot", arguments: "{}" },
          },
        ],
      },
      {
        role: "tool" as const,
        tool_call_id: "tc1",
        content: [
          { type: "text", text: "screenshot attached" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
        ],
      },
    ]);

    expect(parsed.toolResults).toHaveLength(1);
    expect(parsed.toolResults[0].content).toBe("screenshot attached");
    expect(parsed.toolResults[0].images).toHaveLength(1);
    expect(parsed.toolResults[0].images?.[0].mimeType).toBe("image/png");
    expect(Array.from(parsed.toolResults[0].images![0].data)).toEqual(Array.from(pngBytes()));
  });

  test("reattaches pi-ai synthetic tool-result image messages to the tool result", () => {
    const parsed = parseMessages([
      { role: "user" as const, content: "capture screenshot" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          {
            id: "tc1",
            type: "function" as const,
            function: { name: "screenshot", arguments: "{}" },
          },
        ],
      },
      { role: "tool" as const, tool_call_id: "tc1", content: "(see attached image)" },
      {
        role: "user" as const,
        content: [
          { type: "text", text: "Attached image(s) from tool result:" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
        ],
      },
    ]);

    expect(parsed.userText).toBe("capture screenshot");
    expect(parsed.toolResults).toHaveLength(1);
    expect(parsed.toolResults[0].content).toBe("");
    expect(parsed.toolResults[0].images).toHaveLength(1);
    expect(Array.from(parsed.toolResults[0].images![0].data)).toEqual(Array.from(pngBytes()));
  });

  test("preserves tool call, tool result, and final assistant text in a completed turn", () => {
    const parsed = parseMessages([
      { role: "system", content: "system" },
      { role: "user", content: "read file X" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "tc1", type: "function", function: { name: "read", arguments: '{"path":"X"}' } },
        ],
      },
      { role: "tool", content: "file contents here", tool_call_id: "tc1" },
      { role: "assistant", content: "Here is file X..." },
      { role: "user", content: "now do Y" },
    ]);

    expect(parsed.userText).toBe("now do Y");
    expect(parsed.toolResults).toEqual([]);
    expect(parsed.turns).toHaveLength(1);
    expect(parsed.turns[0]).toEqual(
      turn("read file X", [
        toolStep("tc1", "read", { path: "X" }, { content: "file contents here", isError: false }),
        assistantStep("Here is file X..."),
      ]),
    );
  });

  test("tool result continuation does not inflate completed turn count", () => {
    const initialMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
    ];
    const initial = parseMessages(initialMsgs);
    expect(initial.turns).toHaveLength(0);
    expect(initial.userText).toBe("read file X");

    const toolResultMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          {
            id: "tc1",
            type: "function" as const,
            function: { name: "read", arguments: '{"path":"X"}' },
          },
        ],
      },
      { role: "tool" as const, content: "file contents here", tool_call_id: "tc1" },
    ];
    const toolResult = parseMessages(toolResultMsgs);

    expect(toolResult.turns).toHaveLength(0);
    expect(toolResult.userText).toBe("read file X");
    expect(toolResult.toolResults).toEqual([{ toolCallId: "tc1", content: "file contents here" }]);

    const nextMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
      { role: "assistant" as const, content: "Here is file X..." },
      { role: "user" as const, content: "now do Y" },
    ];
    const next = parseMessages(nextMsgs);
    expect(next.turns.length).toBe(1);
  });

  test("multi-turn tool continuation keeps completed-history count stable", () => {
    const initialMsgs = [
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: "u1" },
      { role: "assistant" as const, content: "a1" },
      { role: "user" as const, content: "u2" },
      { role: "assistant" as const, content: "a2" },
      { role: "user" as const, content: "u3" },
    ];
    const initial = parseMessages(initialMsgs);
    expect(initial.turns.length).toBe(2);

    const toolResultMsgs = [
      ...initialMsgs.slice(0, -1),
      { role: "user" as const, content: "u3" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          { id: "t1", type: "function" as const, function: { name: "bash", arguments: "{}" } },
        ],
      },
      { role: "tool" as const, content: "output", tool_call_id: "t1" },
    ];
    const toolResult = parseMessages(toolResultMsgs);
    expect(toolResult.turns.length).toBe(2);
    expect(toolResult.toolResults).toEqual([{ toolCallId: "t1", content: "output" }]);

    const nextMsgs = [
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: "u1" },
      { role: "assistant" as const, content: "a1" },
      { role: "user" as const, content: "u2" },
      { role: "assistant" as const, content: "a2" },
      { role: "user" as const, content: "u3" },
      { role: "assistant" as const, content: "a3 with tool results" },
      { role: "user" as const, content: "u4" },
    ];
    const next = parseMessages(nextMsgs);
    expect(next.turns.length).toBe(3);
  });

  test("mixed resolved and unresolved tool calls stay in the in-flight turn", () => {
    const parsed = parseMessages([
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: "review it" },
      {
        role: "assistant" as const,
        content: "starting review",
        tool_calls: [
          {
            id: "t1",
            type: "function" as const,
            function: { name: "read", arguments: '{"path":"package.json"}' },
          },
        ],
      },
      { role: "tool" as const, content: "pkg", tool_call_id: "t1" },
      {
        role: "assistant" as const,
        content: "continuing review",
        tool_calls: [
          {
            id: "t2",
            type: "function" as const,
            function: { name: "read", arguments: '{"path":"README.md"}' },
          },
        ],
      },
    ]);

    expect(parsed.turns).toHaveLength(0);
    expect(parsed.userText).toBe("review it");
    expect(parsed.toolResults).toEqual([{ toolCallId: "t1", content: "pkg" }]);
  });
});

function frameConnectMessageForTest(data: Uint8Array, flags = 0): Buffer {
  const frame = Buffer.alloc(5 + data.length);
  frame[0] = flags;
  frame.writeUInt32BE(data.length, 1);
  frame.set(data, 5);
  return frame;
}

function decodeConnectFramesForTest(data: Uint8Array): Uint8Array[] {
  const frames: Uint8Array[] = [];
  let pending = Buffer.from(data);
  while (pending.length >= 5) {
    const length = pending.readUInt32BE(1);
    if (pending.length < 5 + length) break;
    frames.push(pending.subarray(5, 5 + length));
    pending = pending.subarray(5 + length);
  }
  return frames;
}

class FakeBridge {
  readonly proc = {
    kill: () => {
      this.close(143);
      return true;
    },
  };

  private aliveState = true;
  private dataCb: ((chunk: Buffer) => void) | null = null;
  private closeCb: ((code: number) => void) | null = null;
  private pendingCloseCode: number | null = null;
  private pendingServerChunks: Buffer[] = [];
  readonly clientMessages: any[] = [];

  constructor(
    readonly options: { accessToken: string; rpcPath: string; url?: string; unary?: boolean },
    private readonly onClientMessage?: (message: any, bridge: FakeBridge) => void,
  ) {}

  get alive() {
    return this.aliveState;
  }

  write(data: Uint8Array) {
    for (const frame of decodeConnectFramesForTest(data)) {
      const clientMessage = fromBinary(AgentClientMessageSchema, frame);
      this.clientMessages.push(clientMessage);
      this.onClientMessage?.(clientMessage, this);
    }
  }

  end() {
    this.close(0);
  }

  onData(cb: (chunk: Buffer) => void) {
    this.dataCb = cb;
    for (const chunk of this.pendingServerChunks.splice(0)) cb(chunk);
  }

  onClose(cb: (code: number) => void) {
    if (this.pendingCloseCode !== null) {
      const code = this.pendingCloseCode;
      queueMicrotask(() => cb(code));
      return;
    }
    this.closeCb = cb;
  }

  emitServerMessage(message: any) {
    const payload = toBinary(AgentServerMessageSchema, message);
    this.emitChunk(frameConnectMessageForTest(payload));
  }

  emitEndStream(payload: Record<string, unknown> = {}) {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    this.emitChunk(frameConnectMessageForTest(bytes, 0b00000010));
  }

  close(code = 0) {
    if (!this.aliveState) return;
    this.aliveState = false;
    if (this.closeCb) {
      const cb = this.closeCb;
      queueMicrotask(() => cb(code));
    } else {
      this.pendingCloseCode = code;
    }
  }

  private emitChunk(chunk: Buffer) {
    if (this.dataCb) {
      this.dataCb(chunk);
    } else {
      this.pendingServerChunks.push(chunk);
    }
  }
}

function makeTextDeltaMessage(text: string) {
  return create(AgentServerMessageSchema, {
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: { case: "textDelta", value: create(TextDeltaUpdateSchema, { text }) },
      }),
    },
  });
}

function makeCheckpointMessage() {
  return create(AgentServerMessageSchema, {
    message: {
      case: "conversationCheckpointUpdate",
      value: create(ConversationStateStructureSchema, {}),
    },
  });
}

function makeSetBlobMessage(blobId: Uint8Array, blobData: Uint8Array) {
  return create(AgentServerMessageSchema, {
    message: {
      case: "kvServerMessage",
      value: create(KvServerMessageSchema, {
        id: 1,
        message: {
          case: "setBlobArgs",
          value: create(SetBlobArgsSchema, { blobId, blobData }),
        },
      }),
    },
  });
}

function makeMcpExecMessage(toolCallId: string, toolName: string, args: Record<string, string>) {
  return create(AgentServerMessageSchema, {
    message: {
      case: "execServerMessage",
      value: create(ExecServerMessageSchema, {
        id: 1,
        execId: "exec-1",
        message: {
          case: "mcpArgs",
          value: create(McpArgsSchema, {
            name: toolName,
            toolName,
            toolCallId,
            providerIdentifier: "pi",
            args: Object.fromEntries(
              Object.entries(args).map(([key, value]) => [
                key,
                toBinary(ValueSchema, fromJson(ValueSchema, value)),
              ]),
            ),
          }),
        },
      }),
    },
  });
}

function makeRequestContextMessage() {
  return create(AgentServerMessageSchema, {
    message: {
      case: "execServerMessage",
      value: create(ExecServerMessageSchema, {
        id: 1,
        execId: "exec-context",
        message: { case: "requestContextArgs", value: create(RequestContextArgsSchema, {}) },
      }),
    },
  });
}

async function postChatCompletion(port: number, body: Record<string, unknown>) {
  return new Promise<{ statusCode: number; body: string }>((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path: "/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
      },
      (res) => {
        let data = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => resolve({ statusCode: res.statusCode ?? 0, body: data }));
        res.on("error", reject);
      },
    );
    req.on("error", reject);
    req.end(JSON.stringify(body));
  });
}

function nativeModel(id = "gpt-5") {
  return {
    id,
    name: id,
    api: "cursor-native",
    provider: "cursor",
    baseUrl: "https://api2.cursor.sh",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200_000,
    maxTokens: 64_000,
  } as any;
}

async function collectEvents(stream: AsyncIterable<any>): Promise<any[]> {
  const events: any[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe("native streamSimple provider", () => {
  test("image-only user request forwards selected images without the local proxy", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("image received"));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }),
    );

    const streamSimple = createCursorNativeStream({ getAccessToken: async () => "test-token" });
    const events = await collectEvents(
      streamSimple(
        nativeModel(),
        {
          messages: [
            {
              role: "user",
              content: [{ type: "image", data: pngBase64(), mimeType: "image/png" }],
              timestamp: Date.now(),
            },
          ],
        } as any,
        { sessionId: "native-image-session" },
      ),
    );

    expect(
      events.some((event) => event.type === "text_delta" && event.delta === "image received"),
    ).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "done", reason: "stop" });
    expect(runRequests).toHaveLength(1);
    const userMessage = runRequests[0].action.action.value.userMessage;
    expect(userMessage.text).toBe("");
    expect(userMessage.selectedContext.selectedImages).toHaveLength(1);
    const selectedImage = userMessage.selectedContext.selectedImages[0];
    expect(selectedImage.mimeType).toBe("image/png");
    expect(selectedImage.dataOrBlobId.case).toBe("data");
    expect(Array.from(selectedImage.dataOrBlobId.value)).toEqual(Array.from(pngBytes()));
  });

  test("tool-call continuation reuses the live bridge and emits pi-native tool events", async () => {
    const runRequests: any[] = [];
    const execClientMessages: any[] = [];
    const sessionId = "native-tool-session";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const convKey = deriveConversationKeyFromSessionId(sessionId);

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.emitServerMessage(makeMcpExecMessage("tc1", "read", { path: "README.md" }));
            return;
          }
          if (clientMessage.message.case === "execClientMessage") {
            execClientMessages.push(clientMessage.message.value);
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("I found the issue."));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }),
    );

    const streamSimple = createCursorNativeStream({ getAccessToken: async () => "test-token" });
    const firstEvents = await collectEvents(
      streamSimple(
        nativeModel(),
        {
          messages: [{ role: "user", content: "inspect file", timestamp: Date.now() }],
          tools: [
            {
              name: "read",
              description: "Read a file",
              parameters: { type: "object", properties: { path: { type: "string" } } },
            },
          ],
        } as any,
        { sessionId },
      ),
    );

    expect(
      firstEvents.some((event) => event.type === "toolcall_end" && event.toolCall.id === "tc1"),
    ).toBe(true);
    expect(firstEvents.at(-1)).toMatchObject({ type: "done", reason: "toolUse" });
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);

    const zeroUsage = {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    };
    const secondEvents = await collectEvents(
      streamSimple(
        nativeModel(),
        {
          messages: [
            { role: "user", content: "inspect file", timestamp: Date.now() },
            {
              role: "assistant",
              content: [
                { type: "toolCall", id: "tc1", name: "read", arguments: { path: "README.md" } },
              ],
              api: "cursor-native",
              provider: "cursor",
              model: "gpt-5",
              usage: zeroUsage,
              stopReason: "toolUse",
              timestamp: Date.now(),
            },
            {
              role: "toolResult",
              toolCallId: "tc1",
              toolName: "read",
              content: [{ type: "text", text: "README contents" }],
              isError: false,
              timestamp: Date.now(),
            },
          ],
        } as any,
        { sessionId },
      ),
    );

    expect(
      secondEvents.some(
        (event) => event.type === "text_delta" && event.delta === "I found the issue.",
      ),
    ).toBe(true);
    expect(runRequests).toHaveLength(1);
    expect(execClientMessages).toHaveLength(1);
    expect(execClientMessages[0].execId).toBe("exec-1");
    expect(execClientMessages[0].message.case).toBe("mcpResult");
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);
    expect(__testInternals.conversationStates.get(convKey)?.checkpoint).toBeTruthy();
  });

  test("aborting after a native tool-call pause cancels the live Cursor bridge", async () => {
    const bridges: FakeBridge[] = [];
    const sessionId = "native-abort-after-tool-pause";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);

    setBridgeFactoryForTests((options) => {
      const bridge = new FakeBridge(options, (clientMessage, fake) => {
        if (clientMessage.message.case === "runRequest") {
          fake.emitServerMessage(makeMcpExecMessage("tc-abort", "bash", { command: "rg foo" }));
        }
      });
      bridges.push(bridge);
      return bridge;
    });

    const controller = new AbortController();
    const streamSimple = createCursorNativeStream({ getAccessToken: async () => "test-token" });
    const events = await collectEvents(
      streamSimple(
        nativeModel(),
        {
          messages: [{ role: "user", content: "search", timestamp: Date.now() }],
          tools: [{ name: "bash", description: "Run shell", parameters: {} }],
        } as any,
        { sessionId, signal: controller.signal },
      ),
    );

    expect(events.at(-1)).toMatchObject({ type: "done", reason: "toolUse" });
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);

    controller.abort();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);
    expect(
      bridges[0].clientMessages.some(
        (message) =>
          message.message.case === "conversationAction" &&
          message.message.value.action.case === "cancelAction",
      ),
    ).toBe(true);
  });

  test("already-aborted native tool continuation reports aborted instead of lost continuation", async () => {
    const controller = new AbortController();
    controller.abort();

    const streamSimple = createCursorNativeStream({ getAccessToken: async () => "test-token" });
    const events = await collectEvents(
      streamSimple(
        nativeModel(),
        {
          messages: [
            { role: "user", content: "search", timestamp: Date.now() },
            {
              role: "assistant",
              content: [
                {
                  type: "toolCall",
                  id: "tc-aborted",
                  name: "bash",
                  arguments: { command: "rg foo" },
                },
              ],
              timestamp: Date.now(),
            },
            {
              role: "toolResult",
              toolCallId: "tc-aborted",
              toolName: "bash",
              content: [{ type: "text", text: "Operation aborted" }],
              isError: true,
              timestamp: Date.now(),
            },
          ],
        } as any,
        { sessionId: "native-already-aborted", signal: controller.signal },
      ),
    );

    expect(events.at(-1)).toMatchObject({
      type: "error",
      reason: "aborted",
      error: { errorMessage: "Aborted" },
    });
  });

  test("native request context reports the injected Pi workspace path", async () => {
    const runRequests: any[] = [];
    const execClientMessages: any[] = [];
    const workspacePath = "/tmp/pi-cursor-provider-workspace";

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.emitServerMessage(makeRequestContextMessage());
            return;
          }
          if (clientMessage.message.case === "execClientMessage") {
            execClientMessages.push(clientMessage.message.value);
            fake.emitServerMessage(makeTextDeltaMessage("workspace ready"));
            fake.emitServerMessage(makeCheckpointMessage());
            fake.close(0);
          }
        }),
    );

    const streamSimple = createCursorNativeStream({ getAccessToken: async () => "test-token" });
    const events = await collectEvents(
      streamSimple(
        nativeModel(),
        { messages: [{ role: "user", content: "check context", timestamp: Date.now() }] } as any,
        {
          sessionId: "native-workspace-context",
          onPayload: (payload) => ({
            ...(payload as Record<string, unknown>),
            cursor_workspace_path: workspacePath,
          }),
        },
      ),
    );

    expect(
      events.some((event) => event.type === "text_delta" && event.delta === "workspace ready"),
    ).toBe(true);
    expect(runRequests[0].conversationState.previousWorkspaceUris).toEqual([
      "file:///tmp/pi-cursor-provider-workspace",
    ]);
    const requestContext = execClientMessages[0].message.value.result.value.requestContext;
    expect(requestContext.env.workspacePaths).toEqual([workspacePath]);
  });

  test("native routing applies reasoning/model parameter maps without before_provider_request", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("routed"));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }),
    );

    const streamSimple = createCursorNativeStream({
      getAccessToken: async () => "test-token",
      getRawModelRoutingByModelId: () =>
        new Map([
          [
            "gpt-5.5",
            {
              high: {
                modelId: "gpt-5.5",
                parameters: [
                  { id: "context", value: "272k" },
                  { id: "reasoning", value: "high" },
                  { id: "fast", value: "false" },
                ],
                requestedMaxMode: true,
              },
            },
          ],
        ]),
    });

    await collectEvents(
      streamSimple(
        { ...nativeModel("gpt-5.5"), thinkingLevelMap: { high: "high" } },
        { messages: [{ role: "user", content: "hello", timestamp: Date.now() }] } as any,
        { sessionId: "native-routing-session", reasoning: "high" },
      ),
    );

    expect(runRequests).toHaveLength(1);
    expect(runRequests[0].requestedModel.modelId).toBe("gpt-5.5");
    expect(runRequests[0].requestedModel.maxMode).toBe(true);
    expect(runRequests[0].requestedModel.parameters.map((p: any) => `${p.id}=${p.value}`)).toEqual([
      "context=272k",
      "reasoning=high",
      "fast=false",
    ]);
  });
});

describe("proxy integration — session handling", () => {
  test("image-only user request forwards selected images through the HTTP proxy", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("image received"));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: "session-user-image",
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
          ],
        },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toContain("image received");
    expect(runRequests).toHaveLength(1);
    const userMessage = runRequests[0].action.action.value.userMessage;
    expect(userMessage.text).toBe("");
    expect(userMessage.selectedContext.selectedImages).toHaveLength(1);
    const selectedImage = userMessage.selectedContext.selectedImages[0];
    expect(selectedImage.mimeType).toBe("image/png");
    expect(selectedImage.dataOrBlobId.case).toBe("data");
    expect(Array.from(selectedImage.dataOrBlobId.value)).toEqual(Array.from(pngBytes()));
  });

  test("tool-call continuation reuses the live bridge and commits a checkpoint when the turn completes", async () => {
    const runRequests: any[] = [];
    const execClientMessages: any[] = [];
    const bridges: FakeBridge[] = [];

    setBridgeFactoryForTests((options) => {
      const bridge = new FakeBridge(options, (clientMessage, fake) => {
        if (clientMessage.message.case === "runRequest") {
          runRequests.push(clientMessage.message.value);
          fake.emitServerMessage(makeMcpExecMessage("tc1", "read", { path: "README.md" }));
          return;
        }

        if (clientMessage.message.case === "execClientMessage") {
          execClientMessages.push(clientMessage.message.value);
          setTimeout(() => {
            fake.emitServerMessage(makeTextDeltaMessage("I found the issue."));
            fake.emitServerMessage(makeCheckpointMessage());
            fake.close(0);
          }, 0);
        }
      });
      bridges.push(bridge);
      return bridge;
    });

    const sessionId = "session-tool";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const port = await startProxy(async () => "test-token");

    const first = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [{ role: "user", content: "inspect file" }],
      tools: [{ type: "function", function: { name: "read" } }],
    });

    expect(first.statusCode).toBe(200);
    expect(first.body).toContain('"finish_reason":"tool_calls"');
    expect(first.body).toContain('"id":"tc1"');
    expect(bridges).toHaveLength(1);
    expect(runRequests).toHaveLength(1);
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);

    const second = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "user", content: "inspect file" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "tc1",
              type: "function",
              function: { name: "read", arguments: '{"path":"README.md"}' },
            },
          ],
        },
        { role: "tool", content: "README contents", tool_call_id: "tc1" },
      ],
    });

    expect(second.statusCode).toBe(200);
    expect(second.body).toContain("I found the issue.");
    expect(runRequests).toHaveLength(1);
    expect(execClientMessages).toHaveLength(1);
    expect(execClientMessages[0].execId).toBe("exec-1");
    expect(execClientMessages[0].message.case).toBe("mcpResult");
    expect(execClientMessages[0].message.value.result.case).toBe("success");
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);

    const stored = __testInternals.conversationStates.get(convKey);
    expect(stored?.checkpoint).toBeTruthy();
  });

  test("tool-call continuation forwards tool result images as MCP image content", async () => {
    const execClientMessages: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            fake.emitServerMessage(makeMcpExecMessage("tc1", "screenshot", {}));
            return;
          }
          if (clientMessage.message.case === "execClientMessage") {
            execClientMessages.push(clientMessage.message.value);
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("I can see it."));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const sessionId = "session-tool-image";
    const first = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [{ role: "user", content: "capture screenshot" }],
      tools: [{ type: "function", function: { name: "screenshot" } }],
    });
    expect(first.statusCode).toBe(200);
    expect(first.body).toContain('"finish_reason":"tool_calls"');

    const second = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      cursor_tool_result_images: [
        { toolCallId: "tc1", images: [{ data: pngBase64(), mimeType: "image/png" }] },
      ],
      messages: [
        { role: "user", content: "capture screenshot" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "tc1", type: "function", function: { name: "screenshot", arguments: "{}" } },
          ],
        },
        { role: "tool", content: "(see attached image)", tool_call_id: "tc1" },
        {
          role: "user",
          content: [
            { type: "text", text: "Attached image(s) from tool result:" },
            { type: "image_url", image_url: { url: `data:image/png;base64,${pngBase64()}` } },
          ],
        },
      ],
    });

    expect(second.statusCode).toBe(200);
    expect(second.body).toContain("I can see it.");
    expect(execClientMessages).toHaveLength(1);
    const result = execClientMessages[0].message.value.result.value;
    expect(result.content).toHaveLength(1);
    expect(result.content[0].content.case).toBe("image");
    expect(result.content[0].content.value.mimeType).toBe("image/png");
    expect(Array.from(result.content[0].content.value.data)).toEqual(Array.from(pngBytes()));
  });

  test("lost tool-call continuation returns an explicit conflict instead of silently starting a new turn", async () => {
    const bridges: FakeBridge[] = [];
    setBridgeFactoryForTests((options) => {
      const bridge = new FakeBridge(options, (clientMessage, fake) => {
        if (clientMessage.message.case === "runRequest") {
          fake.emitServerMessage(makeMcpExecMessage("tc-lost", "read", { path: "README.md" }));
        }
      });
      bridges.push(bridge);
      return bridge;
    });

    const sessionId = "session-lost-tool-continuation";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const port = await startProxy(async () => "test-token");

    const first = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [{ role: "user", content: "inspect file" }],
      tools: [{ type: "function", function: { name: "read" } }],
    });

    expect(first.statusCode).toBe(200);
    expect(first.body).toContain('"finish_reason":"tool_calls"');
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);

    bridges[0].close(1);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);

    const second = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "user", content: "inspect file" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "tc-lost",
              type: "function",
              function: { name: "read", arguments: '{"path":"README.md"}' },
            },
          ],
        },
        { role: "tool", content: "README contents", tool_call_id: "tc-lost" },
      ],
    });

    expect(second.statusCode).toBe(409);
    expect(second.body).toContain("tool_continuation_lost");
  });

  test("stream:false with tools is rejected explicitly", async () => {
    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      stream: false,
      messages: [{ role: "user", content: "inspect file" }],
      tools: [{ type: "function", function: { name: "read" } }],
    });

    expect(response.statusCode).toBe(400);
    expect(response.body).toContain("nonstream_tools_unsupported");
  });

  test("tool_choice none suppresses exposed MCP tools", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      messages: [{ role: "user", content: "inspect file" }],
      tools: [{ type: "function", function: { name: "read" } }],
      tool_choice: "none",
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(runRequests[0].mcpTools.mcpTools).toHaveLength(0);
  });

  test("MCP tool definitions encode input_schema as Cursor's google.protobuf.Value", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      messages: [{ role: "user", content: "inspect file" }],
      tools: [
        {
          type: "function",
          function: {
            name: "read",
            description: "Read a file",
            parameters: {
              type: "object",
              properties: { path: { type: "string", description: "File path" } },
              required: ["path"],
            },
          },
        },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    const tool = runRequests[0].mcpTools.mcpTools[0];
    const decodedValue = fromBinary(ValueSchema, tool.inputSchema);
    expect(toJson(ValueSchema, decodedValue)).toEqual({
      type: "object",
      properties: { path: { type: "string", description: "File path" } },
      required: ["path"],
    });
  });

  test("max_tokens compatibility field is accepted as a no-op", async () => {
    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 10,
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
  });

  test("unsupported OpenAI request parameters are rejected explicitly", async () => {
    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      messages: [{ role: "user", content: "hello" }],
      temperature: 0.2,
      max_tokens: 10,
    });

    expect(response.statusCode).toBe(400);
    expect(response.body).toContain("unsupported_parameter");
    expect(response.body).toContain("temperature");
    expect(response.body).not.toContain("max_tokens");
  });

  test("unsupported tool_choice values are rejected explicitly", async () => {
    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      messages: [{ role: "user", content: "hello" }],
      tools: [{ type: "function", function: { name: "read" } }],
      tool_choice: "required",
    });

    expect(response.statusCode).toBe(400);
    expect(response.body).toContain("unsupported_tool_choice");
  });

  test("upstream stream errors preserve the last committed conversation checkpoint", async () => {
    const sessionId = "session-stream-error-preserve";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const priorTurns = [turn("earlier", [assistantStep("done")])];
    const priorPayload = buildCursorRequest(
      "gpt-5",
      "system",
      "next",
      priorTurns,
      "conv-stream-error",
      null,
    );
    const priorCheckpoint = toBinary(
      ConversationStateStructureSchema,
      decodeRunRequest(priorPayload).conversationState,
    );
    const failedBlobId = new Uint8Array([9, 9, 9, 9]);
    const failedBlobKey = Buffer.from(failedBlobId).toString("hex");

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-stream-error",
      checkpoint: priorCheckpoint,
      checkpointTurnCount: priorTurns.length,
      checkpointHistoryFingerprint: __testInternals.fingerprintCompletedTurns(priorTurns),
      sessionScoped: true,
      blobStore: new Map(priorPayload.blobStore),
      lastAccessMs: Date.now(),
    });

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            fake.emitServerMessage(
              makeSetBlobMessage(failedBlobId, new TextEncoder().encode("failed blob")),
            );
            fake.emitEndStream({ error: { code: "internal", message: "boom" } });
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "earlier" },
        { role: "assistant", content: "done" },
        { role: "user", content: "next" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(response.body).toContain("boom");
    expect(__testInternals.conversationStates.get(convKey)?.checkpoint).toEqual(priorCheckpoint);
    expect(__testInternals.conversationStates.get(convKey)?.blobStore.has(failedBlobKey)).toBe(
      false,
    );
  });

  test("upstream non-stream errors preserve the last committed conversation checkpoint", async () => {
    const sessionId = "session-nonstream-error-preserve";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const priorTurns = [turn("earlier", [assistantStep("done")])];
    const priorPayload = buildCursorRequest(
      "gpt-5",
      "system",
      "next",
      priorTurns,
      "conv-nonstream-error",
      null,
    );
    const priorCheckpoint = toBinary(
      ConversationStateStructureSchema,
      decodeRunRequest(priorPayload).conversationState,
    );

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-nonstream-error",
      checkpoint: priorCheckpoint,
      checkpointTurnCount: priorTurns.length,
      checkpointHistoryFingerprint: __testInternals.fingerprintCompletedTurns(priorTurns),
      sessionScoped: true,
      blobStore: new Map(priorPayload.blobStore),
      lastAccessMs: Date.now(),
    });

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            fake.emitEndStream({ error: { code: "internal", message: "boom" } });
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      stream: false,
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "earlier" },
        { role: "assistant", content: "done" },
        { role: "user", content: "next" },
      ],
    });

    expect(response.statusCode).toBe(502);
    expect(response.body).toContain("boom");
    expect(__testInternals.conversationStates.get(convKey)?.checkpoint).toEqual(priorCheckpoint);
  });

  test("partial tool-result batches stay in-flight until all pending tool results arrive", async () => {
    const execClientMessages: any[] = [];
    const sessionId = "session-partial-tools";
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const convKey = deriveConversationKeyFromSessionId(sessionId);

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-partial-tools",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const bridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "execClientMessage") {
          execClientMessages.push(clientMessage.message.value);
          if (execClientMessages.length === 2) {
            setTimeout(() => {
              fake.emitServerMessage(makeTextDeltaMessage("final review"));
              fake.emitServerMessage(makeCheckpointMessage());
              fake.close(0);
            }, 0);
          }
        }
      },
    );

    __testInternals.activeBridges.set(bridgeKey, {
      bridge: bridge as any,
      heartbeatTimer: setInterval(() => {}, 60_000),
      blobStore: new Map(),
      mcpTools: [],
      pendingExecs: [
        {
          execId: "exec-1",
          execMsgId: 1,
          toolCallId: "tc1",
          toolName: "read",
          decodedArgs: '{"path":"package.json"}',
        },
        {
          execId: "exec-2",
          execMsgId: 2,
          toolCallId: "tc2",
          toolName: "read",
          decodedArgs: '{"path":"README.md"}',
        },
      ],
      currentTurn: turn("review it", [
        assistantStep("starting review"),
        toolStep("tc1", "read", { path: "package.json" }),
        assistantStep("continuing review"),
        toolStep("tc2", "read", { path: "README.md" }),
      ]),
    });

    const port = await startProxy(async () => "test-token");

    const partial = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "user", content: "review it" },
        {
          role: "assistant",
          content: "starting review",
          tool_calls: [
            {
              id: "tc1",
              type: "function",
              function: { name: "read", arguments: '{"path":"package.json"}' },
            },
          ],
        },
        { role: "tool", content: "pkg", tool_call_id: "tc1" },
        {
          role: "assistant",
          content: "continuing review",
          tool_calls: [
            {
              id: "tc2",
              type: "function",
              function: { name: "read", arguments: '{"path":"README.md"}' },
            },
          ],
        },
      ],
    });

    expect(partial.statusCode).toBe(200);
    expect(partial.body).toContain('"finish_reason":"tool_calls"');
    expect(partial.body).toContain('"id":"tc2"');
    expect(partial.body).not.toContain('"id":"tc1"');
    expect(execClientMessages).toHaveLength(0);
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);
    const partialBridge = __testInternals.activeBridges.get(bridgeKey);
    const partialT1 = partialBridge?.currentTurn.steps.find(
      (step) => step.kind === "toolCall" && step.toolCallId === "tc1",
    );
    expect(partialT1 && partialT1.kind === "toolCall" ? partialT1.result?.content : undefined).toBe(
      "pkg",
    );

    const complete = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "user", content: "review it" },
        {
          role: "assistant",
          content: "starting review",
          tool_calls: [
            {
              id: "tc1",
              type: "function",
              function: { name: "read", arguments: '{"path":"package.json"}' },
            },
          ],
        },
        { role: "tool", content: "pkg", tool_call_id: "tc1" },
        {
          role: "assistant",
          content: "continuing review",
          tool_calls: [
            {
              id: "tc2",
              type: "function",
              function: { name: "read", arguments: '{"path":"README.md"}' },
            },
          ],
        },
        { role: "tool", content: "readme", tool_call_id: "tc2" },
      ],
    });

    expect(complete.statusCode).toBe(200);
    expect(complete.body).toContain("final review");
    expect(execClientMessages).toHaveLength(2);
    expect(execClientMessages.map((m) => m.execId)).toEqual(["exec-1", "exec-2"]);
    expect(
      execClientMessages.every(
        (m) => m.message.case === "mcpResult" && m.message.value.result.case === "success",
      ),
    ).toBe(true);
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);

    const stored = __testInternals.conversationStates.get(convKey);
    expect(stored?.checkpoint).toBeTruthy();
  });

  test("tool-call pause closes the SSE without cancelling the live bridge", async () => {
    let cancelCount = 0;
    const sessionId = "session-tool-pause-close";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const currentTurn = turn("inspect file");

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-tool-pause-close",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const bridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "conversationAction") {
          expect(clientMessage.message.value.action.case).toBe("cancelAction");
          cancelCount += 1;
          fake.close(0);
        }
      },
    );

    const req = new EventEmitter() as any;
    const res = new EventEmitter() as any;
    res.headersSent = false;
    res.writeHead = () => {
      res.headersSent = true;
      return res;
    };
    res.write = () => true;
    res.end = () => {
      res.headersSent = true;
      queueMicrotask(() => res.emit("close"));
      return res;
    };

    const heartbeatTimer = setInterval(() => {}, 60_000);
    writeSSEStreamForTests({
      bridge: bridge as any,
      heartbeatTimer,
      modelId: "gpt-5",
      bridgeKey,
      convKey,
      completedTurns: [],
      currentTurn,
      req,
      res,
    });

    bridge.emitServerMessage(makeMcpExecMessage("tc1", "read", { path: "README.md" }));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(cancelCount).toBe(0);
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(true);
  });

  test("stream cancellation sends cancelAction without committing pending checkpoint or blob store", async () => {
    let cancelCount = 0;
    const sessionId = "session-cancel";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const currentTurn = turn("interrupt me");
    const blobId = new Uint8Array([1, 2, 3, 4]);
    const blobKey = Buffer.from(blobId).toString("hex");
    const blobData = new TextEncoder().encode("blob payload");

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-cancel",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const bridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "conversationAction") {
          expect(clientMessage.message.value.action.case).toBe("cancelAction");
          cancelCount += 1;
          fake.close(0);
        }
      },
    );

    const req = new EventEmitter() as any;
    const res = new EventEmitter() as any;
    res.headersSent = false;
    res.writeHead = () => {
      res.headersSent = true;
      return res;
    };
    res.write = () => {
      queueMicrotask(() => res.emit("close"));
      return true;
    };
    res.end = () => {
      res.headersSent = true;
      return res;
    };

    const heartbeatTimer = setInterval(() => {}, 60_000);
    writeSSEStreamForTests({
      bridge: bridge as any,
      heartbeatTimer,
      modelId: "gpt-5",
      bridgeKey,
      convKey,
      completedTurns: [],
      currentTurn,
      req,
      res,
    });

    bridge.emitServerMessage(makeTextDeltaMessage("partial output"));
    bridge.emitServerMessage(makeSetBlobMessage(blobId, blobData));
    bridge.emitServerMessage(makeCheckpointMessage());
    await new Promise((resolve) => setTimeout(resolve, 0));

    const stored = __testInternals.conversationStates.get(convKey);
    expect(cancelCount).toBe(1);
    expect(stored).toBeDefined();
    expect(stored?.checkpoint).toBeNull();
    expect(stored?.blobStore.has(blobKey)).toBe(false);
    expect(__testInternals.activeBridges.has(bridgeKey)).toBe(false);
  });

  test("interrupt after a checkpoint does not reuse the uncommitted checkpoint on the next request", async () => {
    const sessionId = "session-interrupt-after-checkpoint";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const currentTurn = turn("interrupt me");

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-interrupt-after-checkpoint",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const interruptedBridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "conversationAction") {
          fake.close(0);
        }
      },
    );

    const req = new EventEmitter() as any;
    const res = new EventEmitter() as any;
    res.headersSent = false;
    res.writeHead = () => {
      res.headersSent = true;
      return res;
    };
    res.write = () => {
      queueMicrotask(() => res.emit("close"));
      return true;
    };
    res.end = () => {
      res.headersSent = true;
      return res;
    };

    writeSSEStreamForTests({
      bridge: interruptedBridge as any,
      heartbeatTimer: setInterval(() => {}, 60_000),
      modelId: "gpt-5",
      bridgeKey,
      convKey,
      completedTurns: [],
      currentTurn,
      req,
      res,
    });

    interruptedBridge.emitServerMessage(makeTextDeltaMessage("partial output"));
    interruptedBridge.emitServerMessage(makeCheckpointMessage());
    await new Promise((resolve) => setTimeout(resolve, 0));

    const storedCheckpoint = __testInternals.conversationStates.get(convKey)?.checkpoint;
    expect(storedCheckpoint).toBeNull();

    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "interrupt me" },
        { role: "user", content: "continue" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(runRequests[0].conversationId).toBe("conv-interrupt-after-checkpoint");
    expect(runRequests[0].conversationState.turns).toHaveLength(1);
    expect(runRequests[0].action.action.value.userMessage.text).toBe("continue");
  });

  test("interrupt after checkpoint reconstructs resumed history with partial assistant text", async () => {
    const sessionId = "session-interrupt-partial-assistant";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const currentTurn = turn("ask something");

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-partial-assistant",
      checkpoint: null,

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const interruptedBridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "conversationAction") {
          fake.close(0);
        }
      },
    );

    const req = new EventEmitter() as any;
    const res = new EventEmitter() as any;
    res.headersSent = false;
    res.writeHead = () => {
      res.headersSent = true;
      return res;
    };
    res.write = () => {
      queueMicrotask(() => res.emit("close"));
      return true;
    };
    res.end = () => {
      res.headersSent = true;
      return res;
    };

    writeSSEStreamForTests({
      bridge: interruptedBridge as any,
      heartbeatTimer: setInterval(() => {}, 60_000),
      modelId: "gpt-5",
      bridgeKey,
      convKey,
      completedTurns: [],
      currentTurn,
      req,
      res,
    });

    interruptedBridge.emitServerMessage(makeTextDeltaMessage("partial response text"));
    interruptedBridge.emitServerMessage(makeCheckpointMessage());
    await new Promise((resolve) => setTimeout(resolve, 0));

    const storedCheckpoint = __testInternals.conversationStates.get(convKey)?.checkpoint;
    expect(storedCheckpoint).toBeNull();

    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    // Pi includes the partial assistant text in the resumed message history
    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "ask something" },
        { role: "assistant", content: "partial response text" },
        { role: "user", content: "continue" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(runRequests[0].conversationId).toBe("conv-partial-assistant");
    expect(runRequests[0].conversationState.turns).toHaveLength(1);
    expect(runRequests[0].action.action.value.userMessage.text).toBe("continue");
  });

  test("interrupt before any new checkpoint discards prior checkpoint when resumed history includes the interrupted turn", async () => {
    const sessionId = "session-interrupt-before-checkpoint";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const bridgeKey = deriveBridgeKeyFromSessionId(sessionId);
    const priorTurns = [turn("earlier", [assistantStep("done")])];
    const priorPayload = buildCursorRequest(
      "gpt-5",
      "system",
      "next",
      priorTurns,
      "conv-old",
      null,
    );
    const priorCheckpoint = toBinary(
      ConversationStateStructureSchema,
      decodeRunRequest(priorPayload).conversationState,
    );

    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-old",
      checkpoint: priorCheckpoint,
      checkpointTurnCount: priorTurns.length,
      checkpointHistoryFingerprint: __testInternals.fingerprintCompletedTurns(priorTurns),
      sessionScoped: true,
      blobStore: new Map(priorPayload.blobStore),
      lastAccessMs: Date.now(),
    });

    const interruptedBridge = new FakeBridge(
      { accessToken: "test-token", rpcPath: "/agent.v1.AgentService/Run" },
      (clientMessage, fake) => {
        if (clientMessage.message.case === "conversationAction") {
          fake.close(0);
        }
      },
    );

    const req = new EventEmitter() as any;
    const res = new EventEmitter() as any;
    res.headersSent = false;
    res.writeHead = () => {
      res.headersSent = true;
      return res;
    };
    res.write = () => {
      queueMicrotask(() => res.emit("close"));
      return true;
    };
    res.end = () => {
      res.headersSent = true;
      return res;
    };

    writeSSEStreamForTests({
      bridge: interruptedBridge as any,
      heartbeatTimer: setInterval(() => {}, 60_000),
      modelId: "gpt-5",
      bridgeKey,
      convKey,
      completedTurns: priorTurns,
      currentTurn: turn("interrupt me"),
      req,
      res,
    });

    interruptedBridge.emitServerMessage(makeTextDeltaMessage("partial output"));
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Prior completed checkpoint survives the interrupted in-flight turn.
    expect(__testInternals.conversationStates.get(convKey)?.checkpoint).toEqual(priorCheckpoint);

    const runRequests: any[] = [];
    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "earlier" },
        { role: "assistant", content: "done" },
        { role: "user", content: "interrupt me" },
        { role: "user", content: "continue" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(runRequests[0].conversationId).toBe("conv-old");
    expect(
      toBinary(ConversationStateStructureSchema, runRequests[0].conversationState),
    ).not.toEqual(priorCheckpoint);
    expect(runRequests[0].conversationState.turns).toHaveLength(2);
    expect(runRequests[0].action.action.value.userMessage.text).toBe("continue");
  });

  test("provider switch history growth discards prior cursor checkpoint and reconstructs transcript", async () => {
    const runRequests: any[] = [];

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const sessionId = "session-provider-switch";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const storedTurns = [turn("hi", [assistantStep("Hi from Cursor")])];
    const priorPayload = buildCursorRequest(
      "gpt-5",
      "system",
      "next",
      storedTurns,
      "conv-provider-switch",
      null,
    );
    const priorCheckpoint = toBinary(
      ConversationStateStructureSchema,
      decodeRunRequest(priorPayload).conversationState,
    );
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-provider-switch",
      checkpoint: priorCheckpoint,
      checkpointTurnCount: storedTurns.length,
      checkpointHistoryFingerprint: __testInternals.fingerprintCompletedTurns(storedTurns),

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "hi" },
        { role: "assistant", content: "Hi from Cursor" },
        { role: "user", content: "second turn on another provider" },
        { role: "assistant", content: "Reply from another provider" },
        { role: "user", content: "now continue on cursor" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(
      toBinary(ConversationStateStructureSchema, runRequests[0].conversationState),
    ).not.toEqual(priorCheckpoint);
    expect(runRequests[0].conversationState.turns).toHaveLength(2);
    expect(runRequests[0].action.action.value.userMessage.text).toBe("now continue on cursor");
  });

  test("same-depth branch with different assistant text discards stale checkpoint", async () => {
    // Lifecycle hooks clean up real Pi forks, but the proxy still validates
    // completed-history fingerprints so stale checkpoints are not reused.
    const runRequests: any[] = [];

    setBridgeFactoryForTests(
      (options) =>
        new FakeBridge(options, (clientMessage, fake) => {
          if (clientMessage.message.case === "runRequest") {
            runRequests.push(clientMessage.message.value);
            fake.close(0);
          }
        }),
    );

    const sessionId = "session-branch";
    const convKey = deriveConversationKeyFromSessionId(sessionId);
    const storedTurns = [turn("first", [assistantStep("branch-a")])];
    const priorPayload = buildCursorRequest(
      "gpt-5",
      "system",
      "next",
      storedTurns,
      "conv-branch",
      null,
    );
    const priorRequest = decodeRunRequest(priorPayload);
    __testInternals.conversationStates.set(convKey, {
      conversationId: "conv-branch",
      checkpoint: toBinary(ConversationStateStructureSchema, priorRequest.conversationState),
      checkpointTurnCount: storedTurns.length,
      checkpointHistoryFingerprint: __testInternals.fingerprintCompletedTurns(storedTurns),

      sessionScoped: true,
      blobStore: new Map(),
      lastAccessMs: Date.now(),
    });

    const storedCheckpoint = __testInternals.conversationStates.get(convKey)?.checkpoint;

    const port = await startProxy(async () => "test-token");
    const response = await postChatCompletion(port, {
      model: "gpt-5",
      pi_session_id: sessionId,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "first" },
        { role: "assistant", content: "branch-b" },
        { role: "user", content: "next" },
      ],
    });

    expect(response.statusCode).toBe(200);
    expect(runRequests).toHaveLength(1);
    expect(
      toBinary(ConversationStateStructureSchema, runRequests[0].conversationState),
    ).not.toEqual(storedCheckpoint);
    expect(runRequests[0].conversationState.turns).toHaveLength(1);
  });
});

const liveCursorMetadataTest =
  process.env.LIVE_CURSOR_METADATA && process.env.CURSOR_ACCESS_TOKEN ? test : test.skip;

function normalizedParameterKey(parameters: { id: string; value: string }[] = []): string {
  return parameters
    .map((parameter) => `${parameter.id}=${parameter.value}`)
    .sort()
    .join(";");
}

function expectedContextWindowForParameter(context: string | undefined): number | undefined {
  if (context === "272k") return 272_000;
  if (context === "200k") return 200_000;
  if (context === "300k") return 300_000;
  if (context === "1m") return 1_000_000;
  return undefined;
}

liveCursorMetadataTest(
  "live Cursor metadata validates every generated parameterized route",
  async () => {
    const metadata = await getCursorParameterizedModels(process.env.CURSOR_ACCESS_TOKEN!);
    const generated = modelsFromParameterizedMetadata(metadata);
    const processed = processModels(generated);
    const parameterizedModelIds = metadata
      .filter((model) => model.variants.some((variant) => variant.parameters.length > 0))
      .map((model) => model.name);
    const generatedRequestedIds = new Set(generated.map((model) => model.requestedModelId));
    for (const modelId of parameterizedModelIds) {
      expect(
        generatedRequestedIds.has(modelId),
        `no generated rows for parameterized model ${modelId}`,
      ).toBe(true);
    }
    expect(generated.length).toBeGreaterThan(25);
    expect(new Set(processed.map((model) => model.id)).size).toBe(processed.length);
    expect(processed.some((model) => model.id === "gpt-5.5-max-fast")).toBe(true);
    expect(processed.some((model) => model.id === "gpt-5.5-1m-fast")).toBe(false);
    expect(processed.some((model) => model.id === "claude-opus-4-7")).toBe(true);
    expect(processed.some((model) => model.id === "claude-opus-4-7-max")).toBe(true);
    expect(processed.some((model) => model.id === "claude-opus-4-7-max-thinking")).toBe(true);

    for (const source of metadata.filter(
      (model) =>
        model.supportsMaxMode && model.variants.some((variant) => variant.parameters.length > 0),
    )) {
      expect(
        generated.some(
          (route) => route.requestedModelId === source.name && route.requestedMaxMode === true,
        ),
        `no max-mode rows generated for ${source.name}`,
      ).toBe(true);
    }

    for (const route of generated) {
      const source = metadata.find((model) => model.name === route.requestedModelId);
      expect(source, `missing metadata source for ${route.id}`).toBeDefined();
      const key = normalizedParameterKey(route.parameters);
      const variant = source!.variants.find(
        (variant) => normalizedParameterKey(variant.parameters) === key,
      );
      expect(
        variant,
        `route ${route.id} uses parameters not advertised by Cursor: ${key}`,
      ).toBeDefined();
      if (route.requestedMaxMode === true && variant!.isMaxMode !== true) {
        expect(
          source!.supportsMaxMode,
          `route ${route.id} sets maxMode for a model that does not support it`,
        ).toBe(true);
      }
      expect(route.parameters?.find((parameter) => parameter.id === "reasoning")?.value).not.toBe(
        "minimal",
      );
      expect(route.parameters?.find((parameter) => parameter.id === "reasoning")?.value).not.toBe(
        "max",
      );
      const expectedWindow = expectedContextWindowForParameter(
        route.parameters?.find((parameter) => parameter.id === "context")?.value,
      );
      if (expectedWindow) expect(route.contextWindow).toBe(expectedWindow);
    }
  },
);
