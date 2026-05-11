export const DEFAULT_CURSOR_CLIENT_VERSION = "cli-2026.05.01-eea359f";

export function getCursorClientVersion(): string {
  return process.env.PI_CURSOR_CLIENT_VERSION || DEFAULT_CURSOR_CLIENT_VERSION;
}
