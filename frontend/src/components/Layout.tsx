import {
  Bell,
  Bot,
  ChartLine,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  ListTree,
  LogOut,
  Sparkles,
  Users,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { titleCase } from "../lib/format";
import { useTeam } from "./TeamContext";
import { cn, Select } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/playground", label: "Playground", icon: Bot },
  { to: "/requests", label: "Request log", icon: ListTree },
  { to: "/forecast", label: "Forecast", icon: ChartLine },
  { to: "/alerts", label: "Alerts", icon: Bell },
  { to: "/optimization", label: "Optimization", icon: Sparkles, manager: true },
  { to: "/whatif", label: "What-if", icon: FlaskConical, manager: true },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/teams", label: "Teams & budgets", icon: Users },
];

export function Layout() {
  const { user, isManager, isAdmin, logout } = useAuth();
  const { team, teams, setTeamId } = useTeam();

  return (
    <div className="flex min-h-full">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-slate-200 bg-white md:flex">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="grid size-8 place-items-center rounded-lg bg-brand-600 text-white">
            <ChartLine className="size-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">BudgetOpt</p>
            <p className="text-[11px] text-slate-500">LLM cost optimization</p>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-3" aria-label="Main">
          {NAV.filter((n) => !n.manager || isManager).map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
                     className={({ isActive }) => cn(
                       "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm",
                       isActive ? "bg-brand-50 font-medium text-brand-700"
                         : "text-slate-600 hover:bg-slate-50 hover:text-slate-900")}>
              <n.icon className="size-4" /> {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-slate-200 p-4">
          <p className="truncate text-sm font-medium text-slate-800">{user?.full_name}</p>
          <p className="truncate text-xs text-slate-500">
            {user?.email} · {titleCase(user?.role ?? "")}
          </p>
          <button onClick={logout}
                  className="mt-3 flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800">
            <LogOut className="size-3.5" /> Sign out
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-slate-200 bg-white/90 px-6 py-3 backdrop-blur">
          <nav className="flex gap-3 overflow-x-auto text-sm md:hidden" aria-label="Main (mobile)">
            {NAV.filter((n) => !n.manager || isManager).map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end}
                       className={({ isActive }) => cn("whitespace-nowrap",
                         isActive ? "font-medium text-brand-700" : "text-slate-600")}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="hidden text-sm text-slate-500 md:block">
            {team ? <>Viewing <span className="font-medium text-slate-800">{team.name}</span></>
              : "No team"}
          </div>
          {isAdmin && teams.length > 1 && (
            <label className="flex items-center gap-2 text-xs text-slate-500">
              Team
              <Select value={team?.id ?? ""} onChange={(e) => setTeamId(Number(e.target.value))}
                      className="w-56 py-1.5" aria-label="Select team">
                {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </Select>
            </label>
          )}
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
