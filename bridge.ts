import { spawn, type ChildProcess } from "node:child_process";
import { resolve as pathResolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const CURSOR_API_URL = "https://api2.cursor.sh";
const CONNECT_END_STREAM_FLAG = 0b00000010;
const BRIDGE_PATH = pathResolve(dirname(fileURLToPath(import.meta.url)), "h2-bridge.mjs");

export interface SpawnBridgeOptions {
  accessToken: string;
  rpcPath: string;
  url?: string;
  unary?: boolean;
}

export interface BridgeHandle {
  proc: Pick<ChildProcess, "kill">;
  readonly alive: boolean;
  write(data: Uint8Array): void;
  end(): void;
  onData(cb: (chunk: Buffer) => void): void;
  onClose(cb: (code: number) => void): void;
}

export type BridgeFactory = (options: SpawnBridgeOptions) => BridgeHandle;
export type BridgeDebugLog = (event: string, data?: Record<string, unknown>) => void;

function noopDebugLog(): void {}

export function lpEncode(data: Uint8Array): Buffer {
  const buf = Buffer.alloc(4 + data.length);
  buf.writeUInt32BE(data.length, 0);
  buf.set(data, 4);
  return buf;
}

export function frameConnectMessage(data: Uint8Array, flags = 0): Buffer {
  const frame = Buffer.alloc(5 + data.length);
  frame[0] = flags;
  frame.writeUInt32BE(data.length, 1);
  frame.set(data, 5);
  return frame;
}

export function spawnBridge(
  options: SpawnBridgeOptions,
  debugLog: BridgeDebugLog = noopDebugLog,
): BridgeHandle {
  debugLog("bridge.spawn", {
    rpcPath: options.rpcPath,
    url: options.url ?? CURSOR_API_URL,
    unary: options.unary ?? false,
    cursorClientVersion: process.env.PI_CURSOR_CLIENT_VERSION || "cli-2026.05.01-eea359f",
  });
  const proc = spawn(process.execPath, [BRIDGE_PATH], {
    stdio: ["pipe", "pipe", "ignore"],
  });

  const config = JSON.stringify({
    accessToken: options.accessToken,
    url: options.url ?? CURSOR_API_URL,
    path: options.rpcPath,
    unary: options.unary ?? false,
  });
  const stdin = proc.stdin!;
  let exited = false;
  let exitCode = 1;
  let stdinUnavailable = false;
  let stdinEnded = false;

  const markStdinUnavailable = (reason: string, error?: unknown) => {
    stdinUnavailable = true;
    debugLog("bridge.stdin_unavailable", {
      reason,
      message: error instanceof Error ? error.message : undefined,
      code:
        error && typeof error === "object" && "code" in error
          ? String((error as { code?: unknown }).code)
          : undefined,
    });
  };

  stdin.on("error", (error) => markStdinUnavailable("error", error));
  stdin.on("close", () => markStdinUnavailable("close"));

  const canWriteStdin = () =>
    !exited &&
    !stdinUnavailable &&
    !stdinEnded &&
    !stdin.destroyed &&
    !stdin.writableEnded &&
    !stdin.writableFinished;

  const writeToStdin = (data: Uint8Array, kind: string): boolean => {
    if (!canWriteStdin()) {
      debugLog("bridge.stdin_write_skipped", {
        kind,
        exited,
        stdinUnavailable,
        stdinEnded,
        destroyed: stdin.destroyed,
        writableEnded: stdin.writableEnded,
        writableFinished: stdin.writableFinished,
      });
      return false;
    }
    try {
      return stdin.write(lpEncode(data));
    } catch (error) {
      markStdinUnavailable("write_failed", error);
      return false;
    }
  };

  writeToStdin(new TextEncoder().encode(config), "config");

  const cbs = {
    data: null as ((chunk: Buffer) => void) | null,
    close: null as ((code: number) => void) | null,
  };

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
    get alive() {
      return !exited;
    },
    write(data: Uint8Array) {
      writeToStdin(data, "data");
    },
    end() {
      if (stdinEnded) return;
      if (canWriteStdin()) {
        try {
          stdin.write(lpEncode(new Uint8Array(0)));
        } catch (error) {
          markStdinUnavailable("end_write_failed", error);
        }
      }
      stdinEnded = true;
      if (!stdin.destroyed && !stdin.writableEnded && !stdin.writableFinished) {
        try {
          stdin.end();
        } catch (error) {
          markStdinUnavailable("end_failed", error);
        }
      }
    },
    onData(cb: (chunk: Buffer) => void) {
      cbs.data = cb;
    },
    onClose(cb: (code: number) => void) {
      if (exited) {
        queueMicrotask(() => cb(exitCode));
      } else {
        cbs.close = cb;
      }
    },
  };
}

export function createConnectFrameParser(
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

export function parseConnectEndStream(data: Uint8Array): Error | null {
  try {
    const payload = JSON.parse(new TextDecoder().decode(data));
    const error = payload?.error;
    if (error)
      return new Error(
        `Connect error ${error.code ?? "unknown"}: ${error.message ?? "Unknown error"}`,
      );
    return null;
  } catch {
    return new Error("Failed to parse Connect end stream");
  }
}
