import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { useTeams } from "../api/queries";
import type { Team } from "../api/types";
import { useAuth } from "../auth/AuthContext";

/** The team the dashboard is looking at. Admins can switch; others see their own team. */
interface TeamState {
  team: Team | null;
  teams: Team[];
  setTeamId: (id: number) => void;
}

const Ctx = createContext<TeamState | null>(null);
const KEY = "budgetopt.team";

export function TeamProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const { data: teams = [] } = useTeams();
  const [teamId, setTeamId] = useState<number | null>(() => {
    const v = localStorage.getItem(KEY);
    return v ? Number(v) : null;
  });

  useEffect(() => {
    if (!teams.length) return;
    const valid = teams.some((t) => t.id === teamId);
    if (!valid) setTeamId(user?.team_id && teams.some((t) => t.id === user.team_id)
      ? user.team_id : teams[0].id);
  }, [teams, teamId, user]);

  const choose = (id: number) => {
    localStorage.setItem(KEY, String(id));
    setTeamId(id);
  };

  return (
    <Ctx.Provider value={{ team: teams.find((t) => t.id === teamId) ?? null, teams,
                           setTeamId: choose }}>
      {children}
    </Ctx.Provider>
  );
}

export function useTeam(): TeamState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTeam outside TeamProvider");
  return ctx;
}
