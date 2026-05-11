import type { CursorModelParameter } from "./cursor-wire.js";

export interface CursorToolResultImagePayload {
  toolCallId: string;
  images: Array<{ data: string; mimeType: string }>;
}

export interface CursorModelRouting {
  modelId: string;
  parameters?: CursorModelParameter[];
  requiresMaxMode?: boolean;
  requestedMaxMode?: boolean;
}

export type CursorModelRoutingByEffort = Record<string, CursorModelRouting>;

export function resolveCursorModelRouting(
  payload: Record<string, unknown>,
  rawRoutingByEffortByModelId: Map<string, CursorModelRoutingByEffort>,
): CursorModelRouting | undefined {
  if (typeof payload.model !== "string") return undefined;
  const rawRoutingByEffort = rawRoutingByEffortByModelId.get(payload.model);
  const effort = typeof payload.reasoning_effort === "string" ? payload.reasoning_effort : "";
  return rawRoutingByEffort?.[effort];
}

export function applyCursorModelRouting(
  payload: Record<string, unknown>,
  routing: CursorModelRouting | undefined,
): void {
  if (!routing) return;
  payload.cursor_model_id = routing.modelId;
  if (routing.parameters?.length) payload.cursor_model_parameters = routing.parameters;
  if (routing.requiresMaxMode) payload.cursor_requires_max_mode = true;
  if (typeof routing.requestedMaxMode === "boolean")
    payload.cursor_model_max_mode = routing.requestedMaxMode;
}
