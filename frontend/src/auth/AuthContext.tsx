import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  API_BASE,
  api,
  clearToken,
  getToken,
  setToken,
  type LoginResponse,
} from "../api/client";

interface AuthUser {
  id: number;
}

interface AuthContextValue {
  user: AuthUser | null;
  login(response: LoginResponse): void;
  googleLogin(): void;
  guestLogin(): Promise<void>;
  demoLogin(): Promise<void>;
  /** `/auth` route: pull `#token=` out of the OAuth fragment. */
  consumeFragmentToken(): boolean;
  logout(): void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function readUser(token: string | null): AuthUser | null {
  if (!token) return null;
  try {
    const payload = JSON.parse(
      atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    if (typeof payload.exp === "number" && payload.exp * 1000 < Date.now()) {
      clearToken();
      return null;
    }
    return { id: Number(payload.sub) };
  } catch {
    clearToken();
    return null;
  }
}

function deviceId(): string {
  let id = localStorage.getItem("groundedai.device_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("groundedai.device_id", id);
  }
  return id;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() =>
    readUser(getToken()),
  );

  const login = useCallback((response: LoginResponse) => {
    setToken(response.token);
    setUser({ id: response.user_id });
  }, []);

  const googleLogin = useCallback(() => {
    window.location.assign(`${API_BASE}/auth/google`);
  }, []);

  const guestLogin = useCallback(async () => {
    login(
      await api<LoginResponse>("/auth/anonymous", {
        method: "POST",
        body: JSON.stringify({ device_id: deviceId() }),
      }),
    );
  }, [login]);

  const demoLogin = useCallback(async () => {
    login(await api<LoginResponse>("/auth/demo", { method: "POST" }));
  }, [login]);

  const consumeFragmentToken = useCallback(() => {
    const fragment = window.location.hash.replace(/^#/, "");
    const token = new URLSearchParams(fragment).get("token");
    if (!token) return false;
    setToken(token);
    setUser(readUser(token));
    window.history.replaceState(null, "", window.location.pathname);
    return true;
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      login,
      googleLogin,
      guestLogin,
      demoLogin,
      consumeFragmentToken,
      logout,
    }),
    [user, login, googleLogin, guestLogin, demoLogin, consumeFragmentToken, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
