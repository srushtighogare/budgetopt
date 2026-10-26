import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { api, login as apiLogin, onLogout, tokens } from "../api/client";
import type { User } from "../api/types";

interface AuthState {
  user: User | null;
  loading: boolean;
  isAdmin: boolean;
  isManager: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [hasToken, setHasToken] = useState(() => Boolean(tokens.access));

  useEffect(() => {
    const off = onLogout(() => {
      setHasToken(false);
      qc.clear();
    });
    return () => { off(); };
  }, [qc]);

  const me = useQuery({
    queryKey: ["me"],
    queryFn: () => api<User>("/auth/me"),
    enabled: hasToken,
    retry: false,
    staleTime: 5 * 60_000,
  });

  const value = useMemo<AuthState>(() => {
    const user = hasToken ? me.data ?? null : null;
    return {
      user,
      loading: hasToken && me.isLoading,
      isAdmin: user?.role === "org_admin",
      isManager: user?.role === "org_admin" || user?.role === "team_manager",
      login: async (email, password) => {
        await apiLogin(email, password);
        setHasToken(true);
        await qc.invalidateQueries({ queryKey: ["me"] });
      },
      logout: () => tokens.clear(),
    };
  }, [hasToken, me.data, me.isLoading, qc]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
