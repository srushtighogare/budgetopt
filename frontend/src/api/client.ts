/**
 * Thin fetch wrapper for the BudgetOpt API.
 *
 * Tokens live in localStorage so a page reload keeps the session. On a 401 the
 * client tries the refresh token once, then signs the user out.
 */

const ACCESS = "budgetopt.access";
const REFRESH = "budgetopt.refresh";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown,
  ) {
    super(message);
  }
}

type Listener = () => void;
const logoutListeners = new Set<Listener>();
export const onLogout = (fn: Listener) => {
  logoutListeners.add(fn);
  return () => logoutListeners.delete(fn);
};

export const tokens = {
  get access() {
    return localStorage.getItem(ACCESS);
  },
  set(access: string, refresh: string) {
    localStorage.setItem(ACCESS, access);
    localStorage.setItem(REFRESH, refresh);
  },
  clear() {
    localStorage.removeItem(ACCESS);
    localStorage.removeItem(REFRESH);
    logoutListeners.forEach((fn) => fn());
  },
};

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as { detail?: unknown; error?: { message?: string } };
    if (typeof b.detail === "string") return b.detail;
    if (Array.isArray(b.detail) && b.detail[0]?.msg) return String(b.detail[0].msg);
    if (b.error?.message) return b.error.message;
  }
  return fallback;
}

let refreshing: Promise<boolean> | null = null;

async function refresh(): Promise<boolean> {
  const rt = localStorage.getItem(REFRESH);
  if (!rt) return false;
  refreshing ??= fetch("/api/v1/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: rt }),
  })
    .then(async (r) => {
      if (!r.ok) return false;
      const t = await r.json();
      tokens.set(t.access_token, t.refresh_token);
      return true;
    })
    .catch(() => false)
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {},
                             retried = false): Promise<T> {
  const headers = new Headers(init.headers);
  if (tokens.access) headers.set("Authorization", `Bearer ${tokens.access}`);
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }
  const res = await fetch(`/api/v1${path}`, { ...init, headers, body });

  if (res.status === 401 && !retried && (await refresh())) {
    return api<T>(path, init, true);
  }
  if (res.status === 401) {
    tokens.clear();
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => undefined);
  if (!res.ok) throw new ApiError(res.status, errorMessage(data, res.statusText), data);
  return data as T;
}

export async function login(email: string, password: string): Promise<void> {
  const res = await fetch("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  const data = await res.json().catch(() => undefined);
  if (!res.ok) throw new ApiError(res.status, errorMessage(data, "Login failed"), data);
  tokens.set(data.access_token, data.refresh_token);
}
