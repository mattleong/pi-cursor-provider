import { createHash } from "node:crypto";

export function isCursorProviderDebugEnabled(): boolean {
  const raw = process.env.PI_CURSOR_PROVIDER_DEBUG?.trim().toLowerCase();
  return !!raw && raw !== "0" && raw !== "false" && raw !== "off";
}

export function truncateDebugString(value: string, max = 4000): string {
  return value.length > max
    ? `${value.slice(0, max)}…<truncated ${value.length - max} chars>`
    : value;
}

export function summarizeByteData(bytes: Uint8Array): { byteLength: number; sha256: string } {
  return {
    byteLength: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex").slice(0, 16),
  };
}

export function summarizeBase64ImageData(data: string): {
  base64Length: number;
  byteLength?: number;
  sha256?: string;
} {
  const summary: { base64Length: number; byteLength?: number; sha256?: string } = {
    base64Length: data.length,
  };
  try {
    const bytes = Buffer.from(data.replace(/\s/g, ""), "base64");
    if (bytes.length > 0) Object.assign(summary, summarizeByteData(new Uint8Array(bytes)));
  } catch {}
  return summary;
}
