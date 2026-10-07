/** Centralized logging with levels and context. */

type Level = "debug" | "info" | "warn" | "error";

interface LogEntry {
  level: Level;
  message: string;
  context?: Record<string, unknown>;
  timestamp: string;
}

const LOGS_KEY = "groundedai.logs";
const MAX_LOGS = 100;

function getLogs(): LogEntry[] {
  try {
    const raw = localStorage.getItem(LOGS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveLogs(logs: LogEntry[]) {
  try {
    localStorage.setItem(LOGS_KEY, JSON.stringify(logs.slice(-MAX_LOGS)));
  } catch {
    /* ignore quota */
  }
}

function log(level: Level, message: string, context?: Record<string, unknown>) {
  const entry: LogEntry = {
    level,
    message,
    context,
    timestamp: new Date().toISOString(),
  };
  const logs = getLogs();
  logs.push(entry);
  saveLogs(logs);

  const style = {
    debug: "color: #64748b",
    info: "color: #0ea5e9",
    warn: "color: #f59e0b",
    error: "color: #ef4444; font-weight: bold",
  }[level];

  console[level](`%c[${level.toUpperCase()}] ${message}`, style, context ?? "");
}

export const logger = {
  debug: (msg: string, ctx?: Record<string, unknown>) => log("debug", msg, ctx),
  info: (msg: string, ctx?: Record<string, unknown>) => log("info", msg, ctx),
  warn: (msg: string, ctx?: Record<string, unknown>) => log("warn", msg, ctx),
  error: (msg: string, ctx?: Record<string, unknown>) => log("error", msg, ctx),
  getLogs: () => getLogs(),
  clearLogs: () => localStorage.removeItem("groundedai.logs"),
};
