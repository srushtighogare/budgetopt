import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CircleDollarSign, PiggyBank, Repeat2, TrendingUp } from "lucide-react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import { useTeamSpend } from "../api/queries";
import type { Alert, Forecast, SavingsReport } from "../api/types";
import { BarList, SpendChart } from "../components/charts";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Card,
  Empty,
  ErrorBox,
  Meter,
  PageHeader,
  Spinner,
  Stat,
  statusTone,
} from "../components/ui";
import { dateTime, money, num, pct, ratioPct, titleCase } from "../lib/format";

export default function OverviewPage() {
  const { team } = useTeam();
  const id = team?.id;
  const spend = useTeamSpend(id);
  const report = useQuery({
    queryKey: ["savings", id],
    queryFn: () => api<SavingsReport>(`/reports/savings?team_id=${id}&days=30`),
    enabled: Boolean(id),
  });
  const forecast = useQuery({
    queryKey: ["forecast", id],
    queryFn: () => api<Forecast>(`/teams/${id}/forecast`),
    enabled: Boolean(id),
  });
  const alerts = useQuery({
    queryKey: ["alerts", id, "open"],
    queryFn: () => api<Alert[]>(`/alerts?team_id=${id}&acknowledged=false&limit=5`),
    enabled: Boolean(id),
  });

  if (!team) return <Spinner />;
  const s = spend.data;
  const r = report.data;
  const f = forecast.data;

  return (
    <>
      <PageHeader title="Overview"
                  description={`${team.name}: this month's budget, the last 30 days of usage and what routing saved.`} />
      <ErrorBox error={spend.error || report.error} />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Spent this month" icon={<CircleDollarSign className="size-4" />}
              value={s ? money(s.spend) : "…"}
              tone={s?.budget_state === "hard_block" ? "bad"
                : s?.budget_state === "soft_alert" ? "warn" : "default"}
              hint={s && <>
                <div className="mb-1.5"><Meter ratio={s.usage_ratio}
                                               threshold={team.alert_threshold} /></div>
                {ratioPct(s.usage_ratio)} of {money(s.monthly_budget)} budget
              </>} />
        <Stat label="Saved (30 days)" icon={<PiggyBank className="size-4" />} tone="good"
              value={r ? money(r.savings) : "…"}
              hint={r && (r.savings_pct !== null
                ? `${pct(r.savings_pct)} less than sending everything to the priciest model`
                : "No baseline yet")} />
        <Stat label="Month-end forecast" icon={<TrendingUp className="size-4" />}
              value={f ? money(f.projected_spend) : "…"}
              tone={f && (f.overrun_probability ?? 0) >= 0.5 ? "warn" : "default"}
              hint={f && <>
                {money(f.lower)} – {money(f.upper)} (90%)
                {(f.overrun_probability ?? 0) > 0 &&
                  ` · ${ratioPct(f.overrun_probability)} chance of overrun`}
              </>} />
        <Stat label="Requests (30 days)" icon={<Repeat2 className="size-4" />}
              value={r ? num(r.requests) : "…"}
              hint={r && `${num(r.cache_hits)} from cache · ${num(r.fallbacks)} fallbacks · ${num(r.blocked_by_budget)} blocked`} />
      </div>

      <div className="mt-6 grid items-start gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Daily spend vs baseline"
              subtitle="Baseline = the same requests sent to the most expensive model">
          {report.isLoading ? <Spinner /> : r?.daily.length
            ? <SpendChart data={r.daily} />
            : <Empty title="No requests in the last 30 days">
                Send one from the <Link className="text-brand-600 underline" to="/playground">Playground</Link>.
              </Empty>}
        </Card>

        <Card title="Open alerts" action={<Link to="/alerts" className="text-xs text-brand-600">View all</Link>}>
          {alerts.isLoading ? <Spinner /> : alerts.data?.length ? (
            <ul className="space-y-3">
              {alerts.data.map((a) => (
                <li key={a.id} className="text-sm">
                  <div className="mb-0.5 flex items-center gap-2">
                    <AlertTriangle className="size-3.5 text-amber-500" />
                    <Badge tone={statusTone(a.kind)}>{titleCase(a.kind)}</Badge>
                    <span className="text-xs text-slate-400">{dateTime(a.created_at)}</span>
                  </div>
                  <p className="text-slate-600">{a.message}</p>
                </li>
              ))}
            </ul>
          ) : <Empty title="No open alerts" />}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Spend by model (30 days)">
          {r?.by_model.length
            ? <BarList data={r.by_model.slice(0, 8).map((m) => ({
                name: m.name, value: m.spend, hint: `${num(m.requests)} requests` }))} />
            : <Empty title="No data yet" />}
        </Card>
        <Card title="Spend by task type (30 days)">
          {r?.by_task.length
            ? <BarList data={r.by_task.map((t) => ({
                name: titleCase(t.name), value: t.spend, hint: `${num(t.requests)} requests` }))} />
            : <Empty title="No data yet" />}
        </Card>
      </div>
    </>
  );
}
