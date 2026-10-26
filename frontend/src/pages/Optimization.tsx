import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { OptimizationRun, OptimizationRunSummary, PolicyRule } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ParetoChart, type ParetoPoint } from "../components/charts";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorBox,
  Field,
  Input,
  PageHeader,
  Select,
  Spinner,
  Stat,
  Table,
  Td,
} from "../components/ui";
import { dateTime, money, ms, titleCase } from "../lib/format";

const PRESET_LABEL: Record<string, string> = {
  cost_saver: "Cost saver", balanced: "Balanced", quality_first: "Quality first",
  optimal: "ILP optimum",
};

function RunForm({ teamId, onDone }: { teamId: number; onDone: (id: number) => void }) {
  const [form, setForm] = useState({ method: "nsga2", monthly_requests: "", budget: "",
                                     q_low: "6", q_medium: "7", q_high: "8", latency: "",
                                     generations: "150", ilp_objective: "max_quality",
                                     quality_floor: "8" });
  const run = useMutation({
    mutationFn: () => api<OptimizationRunSummary>("/optimization/runs", { method: "POST", json: {
      team_id: teamId, method: form.method,
      monthly_requests: form.monthly_requests ? Number(form.monthly_requests) : null,
      budget: form.budget ? Number(form.budget) : null,
      q_min: { low: Number(form.q_low), medium: Number(form.q_medium), high: Number(form.q_high) },
      latency_max_ms: form.latency ? Number(form.latency) : null,
      generations: Number(form.generations),
      ilp_objective: form.ilp_objective,
      quality_floor: form.method === "ilp" && form.ilp_objective === "min_cost"
        ? Number(form.quality_floor) : null,
    } }),
    onSuccess: (r) => onDone(r.id),
  });
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) =>
    setForm({ ...form, [k]: e.target.value });

  return (
    <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
      <Field label="Method">
        <Select value={form.method} onChange={set("method")}>
          <option value="nsga2">NSGA-II: cost × quality × latency trade-offs</option>
          <option value="ilp">ILP: exact optimum for one objective</option>
        </Select>
      </Field>
      {form.method === "ilp" && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Objective">
            <Select value={form.ilp_objective} onChange={set("ilp_objective")}>
              <option value="max_quality">Max quality within budget</option>
              <option value="min_cost">Min cost above quality floor</option>
            </Select>
          </Field>
          {form.ilp_objective === "min_cost" && (
            <Field label="Mean quality floor"><Input type="number" step="0.1" min="0" max="10"
                   value={form.quality_floor} onChange={set("quality_floor")} /></Field>
          )}
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <Field label="Requests per month" hint="Blank: from recent traffic">
          <Input type="number" min="1" value={form.monthly_requests} onChange={set("monthly_requests")} />
        </Field>
        <Field label="Monthly budget (USD)" hint="Blank: the team's budget">
          <Input type="number" min="0.01" step="0.01" value={form.budget} onChange={set("budget")} />
        </Field>
      </div>
      <fieldset>
        <legend className="mb-1 text-xs font-medium text-slate-600">Minimum quality per complexity band (0–10)</legend>
        <div className="grid grid-cols-3 gap-3">
          {(["q_low", "q_medium", "q_high"] as const).map((k) => (
            <Input key={k} type="number" step="0.5" min="0" max="10" value={form[k]}
                   onChange={set(k)} aria-label={k.replace("q_", "") + " band"} />
          ))}
        </div>
        <p className="mt-1 text-xs text-slate-400">Low · medium · high complexity. A model below the floor never serves that band.</p>
      </fieldset>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Max p95 latency (ms)" hint="Optional">
          <Input type="number" min="100" value={form.latency} onChange={set("latency")} />
        </Field>
        {form.method === "nsga2" && (
          <Field label="Generations"><Input type="number" min="10" max="1000"
                 value={form.generations} onChange={set("generations")} /></Field>
        )}
      </div>
      <ErrorBox error={run.error} />
      <Button type="submit" className="w-full" loading={run.isPending}>
        <Play className="size-4" /> Run optimization
      </Button>
    </form>
  );
}

function RulesTable({ runId, index }: { runId: number; index: number }) {
  const q = useQuery({
    queryKey: ["point", runId, index],
    queryFn: () => api<{ rules: Record<string, PolicyRule[]> }>(
      `/optimization/runs/${runId}/points/${index}`),
  });
  if (q.isLoading) return <Spinner />;
  const band = (k: string) => ["low", "medium", "high"].indexOf(k.split(":")[1]);
  const rules = Object.entries(q.data?.rules ?? {}).sort(
    ([a], [b]) => a.split(":")[0].localeCompare(b.split(":")[0]) || band(a) - band(b));
  return (
    <Table head={["Task type", "Complexity", "Traffic split"]}>
      {rules.map(([key, split]) => {
        const [task, band] = key.split(":");
        return (
          <tr key={key}>
            <Td>{titleCase(task)}</Td>
            <Td><Badge tone={band === "high" ? "red" : band === "medium" ? "yellow" : "green"}>{band}</Badge></Td>
            <Td>
              <div className="flex h-5 w-full max-w-md overflow-hidden rounded" title={split.map((s) => `${Math.round(s.share * 100)}% ${s.model}`).join(", ")}>
                {split.map((s, i) => (
                  <div key={s.model_id} className="flex items-center justify-center text-[10px] font-medium text-white"
                       style={{ width: `${s.share * 100}%`, background: ["#2a78d6", "#eb6834", "#1baf7a"][i] ?? "#94a3b8",
                                borderRight: "2px solid #fff" }}>
                    {s.share >= 0.15 ? `${Math.round(s.share * 100)}%` : ""}
                  </div>
                ))}
              </div>
              <p className="mt-0.5 text-xs text-slate-500">
                {split.map((s) => `${Math.round(s.share * 100)}% ${s.model}`).join(" · ")}</p>
            </Td>
          </tr>
        );
      })}
    </Table>
  );
}

function RunView({ runId }: { runId: number }) {
  const { team } = useTeam();
  const qc = useQueryClient();
  const run = useQuery({ queryKey: ["run", runId],
                         queryFn: () => api<OptimizationRun>(`/optimization/runs/${runId}`) });
  const [selected, setSelected] = useState<number | null>(null);
  const r = run.data;
  useEffect(() => {
    if (r) setSelected(r.presets.balanced?.index ?? r.presets.optimal?.index ?? 0);
  }, [r]);
  const apply = useMutation({
    mutationFn: (slot: "primary" | "savings") => api(`/optimization/runs/${runId}/apply`, {
      method: "POST", json: { team_id: team?.id, index: selected, slot } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["teams"] }),
  });

  if (run.isLoading || !r) return <Spinner />;
  const presetByIndex = Object.fromEntries(Object.entries(r.presets).map(([k, v]) => [v.index, k]));
  const front: ParetoPoint[] = r.front.map((p) => ({
    ...p, label: presetByIndex[p.index] ? PRESET_LABEL[presetByIndex[p.index]] : undefined }));
  const baselines: ParetoPoint[] = r.baselines.map((b, i) => ({
    index: -1 - i, monthly_cost: b.monthly_cost, quality: b.quality, latency_ms: b.latency_ms,
    label: b.name.replace("all traffic to ", "All → ") }));
  const sel = r.front.find((p) => p.index === selected);
  const lp = (r.metrics.lp_validation as { gap_pct: number }[] | undefined) ?? [];
  const priciest = r.baselines.reduce((a, b) => (b.monthly_cost > a.monthly_cost ? b : a),
                                      r.baselines[0]);

  return (
    <div className="space-y-6">
      {r.notes.map((n) => (
        <div key={n} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">{n}</div>
      ))}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Object.entries(r.presets).map(([k, p]) => (
          <button key={k} onClick={() => setSelected(p.index)} className="text-left">
            <Stat label={PRESET_LABEL[k] ?? k} value={money(p.monthly_cost)}
                  tone={p.feasible === false ? "bad" : "default"}
                  hint={`Quality ${p.quality.toFixed(2)} · p95 ${ms(p.latency_ms)}${p.feasible === false ? " · over budget" : ""}`} />
          </button>
        ))}
        {priciest && (
          <Stat label="Without optimization" value={money(priciest.monthly_cost)}
                hint={`${priciest.name} · quality ${priciest.quality.toFixed(2)}`} />
        )}
      </div>

      {r.method === "nsga2" && (
        <Card title="Pareto front: monthly cost vs quality"
              subtitle={`${r.points} non-dominated allocations · click a point to inspect it · ${r.traffic_source}`}>
          <ParetoChart front={front} baselines={baselines} selected={selected} onSelect={setSelected} />
          <div className="mt-3 -mx-5 border-t border-slate-100 pt-3">
            <Table head={["Without optimization", "Monthly cost", "Quality", "p95 latency"]}>
              {r.baselines.map((b) => (
                <tr key={b.name}>
                  <Td>{b.name}</Td>
                  <Td className="num">{money(b.monthly_cost)}</Td>
                  <Td className="num">{b.quality.toFixed(2)}</Td>
                  <Td className="num">{ms(b.latency_ms)}</Td>
                </tr>
              ))}
            </Table>
          </div>
          {lp.length > 0 && (
            <p className="mt-2 text-xs text-slate-500">
              Checked against the exact LP optimum at {lp.length} budget levels: largest gap{" "}
              {Math.max(...lp.map((v) => v.gap_pct)).toFixed(2)}%.
              Hypervolume {String(r.metrics.hypervolume)}.
            </p>
          )}
        </Card>
      )}

      {selected !== null && sel && (
        <Card title={`Allocation ${presetByIndex[selected] ? PRESET_LABEL[presetByIndex[selected]] : `#${selected}`}: ${money(sel.monthly_cost)}/month, quality ${sel.quality.toFixed(2)}`}
              subtitle="How each kind of request would be split across models"
              action={
                <div className="flex gap-2">
                  <Button variant="secondary" loading={apply.isPending && apply.variables === "savings"}
                          onClick={() => apply.mutate("savings")}>Use at budget alert</Button>
                  <Button loading={apply.isPending && apply.variables === "primary"}
                          onClick={() => apply.mutate("primary")}>Apply to {team?.name}</Button>
                </div>}>
          {apply.isSuccess && <p className="mb-3 text-sm text-emerald-700">Applied. New requests from {team?.name} now follow this policy.</p>}
          <ErrorBox error={apply.error} />
          <RulesTable runId={runId} index={selected} />
        </Card>
      )}
    </div>
  );
}

export default function OptimizationPage() {
  const { team } = useTeam();
  const { isAdmin } = useAuth();
  const runs = useQuery({
    queryKey: ["runs", team?.id],
    queryFn: () => api<OptimizationRunSummary[]>(`/optimization/runs?team_id=${team?.id}`),
    enabled: Boolean(team),
  });
  const qc = useQueryClient();
  const [runId, setRunId] = useState<number | null>(null);
  useEffect(() => { if (!runId && runs.data?.length) setRunId(runs.data[0].id); },
            [runs.data, runId]);

  if (!team) return <Spinner />;
  return (
    <>
      <PageHeader title="Optimization"
                  description="Find the best split of each kind of request across models, trading monthly cost against quality and latency. Models below the quality floor for a complexity band are never used for it." />
      <div className="grid gap-6 xl:grid-cols-4">
        <div className="grid gap-6 md:grid-cols-2 xl:col-span-1 xl:grid-cols-1 xl:content-start">
          {isAdmin && (
            <Card title="New run">
              <RunForm teamId={team.id} onDone={(id) => {
                qc.invalidateQueries({ queryKey: ["runs"] });
                setRunId(id);
              }} />
            </Card>
          )}
          <Card title="Previous runs">
            {runs.isLoading ? <Spinner /> : runs.data?.length ? (
              <ul className="-my-1 divide-y divide-slate-100">
                {runs.data.map((r) => (
                  <li key={r.id}>
                    <button onClick={() => setRunId(r.id)}
                            className={`w-full py-2 text-left text-sm ${r.id === runId ? "text-brand-700" : "text-slate-700"}`}>
                      <span className="font-medium">#{r.id} {r.method === "nsga2" ? "NSGA-II" : "ILP"}</span>
                      <span className="block text-xs text-slate-400">{dateTime(r.created_at)} · {r.points} points</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : <Empty title="No runs yet" />}
          </Card>
        </div>
        <div className="xl:col-span-3">
          {runId ? <RunView key={runId} runId={runId} />
            : <Card><Empty title="Run an optimization to see the trade-offs" /></Card>}
        </div>
      </div>
    </>
  );
}
