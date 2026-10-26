import { useQuery } from "@tanstack/react-query";

import { api } from "./client";
import type { LLMModel, Team, TeamSpend } from "./types";

export const useTeams = () =>
  useQuery({ queryKey: ["teams"], queryFn: () => api<Team[]>("/teams") });

export const useTeamSpend = (teamId: number | null | undefined) =>
  useQuery({
    queryKey: ["spend", teamId],
    queryFn: () => api<TeamSpend>(`/teams/${teamId}/spend`),
    enabled: Boolean(teamId),
  });

export const useModels = () =>
  useQuery({ queryKey: ["models"], queryFn: () => api<LLMModel[]>("/models") });
