/**
 * Typed fetch client: Bearer header, 401 -> login, 429 -> server detail,
 * 60s timeout (parity with the old api_client._TIMEOUT).
 */

export const API_BASE =
  import.meta.env.VITE_API_URL ?? "http://localhost:8001";

const TIMEOUT_MS = 60_000;
const TOKEN_KEY = "groundedai.token";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export interface LoginResponse {
  token: string;
  user_id: number;
  email: string;
  tier?: string;
}

/** JSON request with auth + timeout. Parsed server `detail` becomes the message. */
export async function api<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const token = getToken();

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init.headers,
      },
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === "AbortError";
    if (!timedOut) {
      console.error(`Cannot reach API at ${API_BASE}`, error);
    }
    throw new ApiError(
      0,
      timedOut
        ? "The service is taking longer than usual — please try again."
        : "Can't reach the server right now — please try again in a moment.",
    );
  } finally {
    clearTimeout(timer);
  }

  if (response.status === 401) {
    clearToken();
    if (!window.location.pathname.startsWith("/login")) {
      window.location.assign("/login");
    }
    throw new ApiError(401, "Session expired. Please sign in again.");
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      typeof body?.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(response.status, detail);
  }
  return body as T;
}


/** Multipart upload (PDFs): never set Content-Type, the browser bounds it. */
export async function apiUpload<T>(path: string, form: FormData): Promise<T> {
  const token = getToken();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      body: form,
      signal: controller.signal,
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === "AbortError";
    if (!timedOut) {
      console.error(`Cannot reach API at ${API_BASE}`, error);
    }
    throw new ApiError(
      0,
      timedOut
        ? "The upload is taking longer than usual — please try again."
        : "Can't reach the server right now — please try again in a moment.",
    );
  } finally {
    clearTimeout(timer);
  }
  if (response.status === 401) {
    clearToken();
    window.location.assign("/login");
    throw new ApiError(401, "Session expired. Please sign in again.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(
      response.status,
      typeof body?.detail === "string" ? body.detail : response.statusText,
    );
  }
  return body as T;
}
