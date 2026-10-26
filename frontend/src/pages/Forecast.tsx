import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";

import { api } from "../api/client";
import type { Forecast } from "../api/types";
import { ForecastChart, type ForecastPoint } from "../components/charts";
import { useTeam } from "../components/TeamContext";
import { Badge, Button, Card, ErrorBox, PageHeader, Spinner, Stat } from "../components/ui";
import { money, pct, ratioPct, shortDate, titleCase } from "../lib/format";

function toSeries(f: Forecast): ForecastPoint[] {
  let cum = 0;
  return f.daily.map((d) => {
    const isForecast = d.mean !== undefined;
    if (!isForecast && d.actual !== null) cum += d.actual;
    const first = isForecast && d.actual !== null;  // today: actual so far + projection
    if (first) cum += d.actual ?? 0;
    return {
      date: d.date,
      actual: !isForecast || first ? Math.round(cum * 1e6) / 1e6 : null,
      projected: isForecast ? (d as { cumulative_mean?: number }).cumulative_mean ?? null : null,
      band: isForecast ? [(d as { cumulative_lower?: number }).cumulative_lower ?? 0,
                          (d as { cumulative_upper?: number }).cumulative_upper ?? 0] : null,
    };
  });
}

export default function ForecastPage() {
  const { team } = useTeam();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["forecast", team?.id],
    queryFn: () => api<Forecast>(`/teams/${team?.id}/forecast`),
    enabled: Boolean(team),
  });
  const refresh = async () => {
    await api<Forecast>(`/teams/${team?.id}/forecast?refresh=true`);
    qc.invalidateQueries({ queryKey: ["forecast"] });
    qc.invalidateQueries({ queryKey: ["alerts"] });
  };

  if (!team) return <Spinner />;
  const f = q.data;
  const budget = f ? Number(f.monthly_budget) : 0;
  const bt = f?.backtest;

  return (
    <>
      <PageHeader title="Spend forecast"
                  description="Month-end spend projected from daily history with exponential smoothing (weekly pattern and trend chosen automatically). The shaded band is the 90% interval from 500 simulated futures."
                  action={<Button variant="secondary" onClick={refresh}><RefreshCw className="size-4" /> Recompute</Button>} />
      <ErrorBox error={q.error} />
      {q.isLoading || !f ? <Spinner /> : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Projected month total" value={money(f.projected_spend)}
                  tone={Number(f.projected_spend) > budget ? "warn" : "default"}
                  hint={`90%: ${money(f.lower)} – ${money(f.upper)}`} />
            <Stat label="Budget" value={money(budget)}
                  hint={`${shortDate(f.period_start)} – ${shortDate(f.period_end)}`} />
            <Stat label="Chance of overrun" value={ratioPct(f.overrun_probability)}
                  tone={(f.overrun_probability ?? 0) >= 0.5 ? "bad" : "good"}
                  hint={f.overrun_date ? `Likely around ${shortDate(f.overrun_date)}` : "Not expected this month"} />
            <Stat label="Model" value={<span className="text-base">{titleCase(f.method)}</span>}
                  hint={`Computed ${shortDate(f.created_at)}`} />
          </div>
          <Card className="mt-6" title="Cumulative spend this month">
            <ForecastChart data={toSeries(f)} budget={budget} />
          </Card>
          {bt && (
            <Card className="mt-6" title="How accurate is it? (last 7 days re-predicted)"
                  subtitle="The model is refitted without the last week and its prediction compared with what actually happened">
              <div className="grid gap-4 text-sm sm:grid-cols-4">
                <div><p className="text-xs text-slate-500">Actual 7-day spend</p>
                  <p className="num font-medium">{money(Number(bt.actual_total))}</p></div>
                <div><p className="text-xs text-slate-500">Forecast error</p>
                  <p className="num font-medium">{pct(Number(bt.forecast_total_error_pct), 1)}</p></div>
                <div><p className="text-xs text-slate-500">Naive average error</p>
                  <p className="num font-medium">{pct(Number(bt.naive_total_error_pct), 1)}</p></div>
                <div><p className="text-xs text-slate-500">Actual inside 90% interval</p>
                  <Badge tone={bt.actual_within_90pct_interval ? "green" : "yellow"}>
                    {bt.actual_within_90pct_interval ? "Yes" : "No"}</Badge></div>
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}
