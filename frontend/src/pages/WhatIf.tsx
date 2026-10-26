import { useMutation } from "@tanstack/react-query";
import { FlaskConical, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Tooltip,
         XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import { useModels } from "../api/queries";
import type { WhatIfResult } from "../api/types";
import { C } from "../components/charts";
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
  Table,
  Td,
} from "../components/ui";
import { money, ms, pct } from "../lib/format";

interface ScenarioForm {
  name: string;
  allocation: string;
  model: string;
  policy_id: string;
  run_id: string;
  point_index: string;
  volume_growth_pct: string;
  price_model: string;
  price_factor: string;
}

const blank = (name: string, allocation = "actual"): ScenarioForm => ({
  name, allocation, model: "", policy_id: "", run_id: "", point_index: "0",
  volume_growth_pct: "0", price_model: "", price_factor: "1",
});

export default function WhatIfPage() {
  const { team } = useTeam();
  const models = useModels();
  const [days, setDays] = useState("30");
  const [scenarios, setScenarios] = useState<ScenarioForm[]>([
    blank("What actually happened"),
    blank("Cheapest model meeting the quality floor", "cheapest"),
    { ...blank("Same, with 50% more traffic", "cheapest"), volume_growth_pct: "50" },
  ]);

  const run = useMutation({
    mutationFn: () => api<WhatIfResult>("/whatif", { method: "POST", json: {
      team_id: team?.id, days: Number(days),
      scenarios: scenarios.map((s) => ({
        name: s.name, allocation: s.allocation,
        model: s.allocation === "model" ? s.model : null,
        policy_id: s.allocation === "policy" ? Number(s.policy_id) : null,
        run_id: s.allocation === "run_point" ? Number(s.run_id) : null,
        point_index: s.allocation === "run_point" ? Number(s.point_index) : null,
        volume_growth_pct: Number(s.volume_growth_pct || 0),
        price_multipliers: s.price_model && Number(s.price_factor) !== 1
          ? { [s.price_model]: Number(s.price_factor) } : {},
      })),
    } }),
  });

  const update = (i: number, patch: Partial<ScenarioForm>) =>
    setScenarios(scenarios.map((s, j) => (i === j ? { ...s, ...patch } : s)));
  const res = run.data;
  const chart = res?.scenarios.map((s) => ({ name: s.name, value: s.monthly_projection,
                                             fits: s.fits_budget }));

  return (
    <>
      <PageHeader title="What-if analysis"
                  description={`Replay ${team?.name ?? "the team"}'s real requests from recent days under different allocations, prices and volumes, without calling any model. Use it to check a change before applying it.`} />
      <Card title="Scenarios" action={
        <div className="flex items-end gap-2">
          <Field label="Replay last"><Select value={days} onChange={(e) => setDays(e.target.value)} className="w-28">
            {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d} days</option>)}
          </Select></Field>
          <Button variant="secondary" disabled={scenarios.length >= 10}
                  onClick={() => setScenarios([...scenarios, blank(`Scenario ${scenarios.length + 1}`, "model")])}>
            <Plus className="size-4" /> Add
          </Button>
          <Button loading={run.isPending} onClick={() => run.mutate()}>
            <FlaskConical className="size-4" /> Simulate
          </Button>
        </div>}>
        <div className="space-y-3">
          {scenarios.map((s, i) => (
            <div key={i} className="grid items-end gap-3 rounded-lg border border-slate-200 p-3 md:grid-cols-12">
              <div className="md:col-span-3"><Field label="Name"><Input value={s.name}
                   onChange={(e) => update(i, { name: e.target.value })} /></Field></div>
              <div className="md:col-span-3"><Field label="Allocation">
                <Select value={s.allocation} onChange={(e) => update(i, { allocation: e.target.value })}>
                  <option value="actual">What actually happened</option>
                  <option value="current_policy">Team's current policy</option>
                  <option value="cheapest">Cheapest meeting quality floor</option>
                  <option value="model">All traffic to one model</option>
                  <option value="policy">A saved policy</option>
                  <option value="run_point">An optimization result</option>
                </Select></Field></div>
              <div className="md:col-span-2">
                {s.allocation === "model" && <Field label="Model">
                  <Select value={s.model} onChange={(e) => update(i, { model: e.target.value })}>
                    <option value="">Choose…</option>
                    {models.data?.filter((m) => m.is_active).map((m) =>
                      <option key={m.id} value={m.name}>{m.display_name}</option>)}
                  </Select></Field>}
                {s.allocation === "policy" && <Field label="Policy id"><Input type="number"
                    value={s.policy_id} onChange={(e) => update(i, { policy_id: e.target.value })} /></Field>}
                {s.allocation === "run_point" && <div className="grid grid-cols-2 gap-2">
                  <Field label="Run"><Input type="number" value={s.run_id}
                         onChange={(e) => update(i, { run_id: e.target.value })} /></Field>
                  <Field label="Point"><Input type="number" min="0" value={s.point_index}
                         onChange={(e) => update(i, { point_index: e.target.value })} /></Field></div>}
              </div>
              <div className="md:col-span-1"><Field label="Volume %"><Input type="number"
                   value={s.volume_growth_pct} onChange={(e) => update(i, { volume_growth_pct: e.target.value })} /></Field></div>
              <div className="grid grid-cols-2 gap-2 md:col-span-2">
                <Field label="Price change"><Select value={s.price_model}
                       onChange={(e) => update(i, { price_model: e.target.value })}>
                  <option value="">None</option>
                  {models.data?.map((m) => <option key={m.id} value={m.name}>{m.display_name}</option>)}
                </Select></Field>
                <Field label="× price"><Input type="number" step="0.1" min="0.1" value={s.price_factor}
                       disabled={!s.price_model} onChange={(e) => update(i, { price_factor: e.target.value })} /></Field>
              </div>
              <div className="md:col-span-1">
                <Button variant="ghost" aria-label="Remove scenario" disabled={scenarios.length === 1}
                        onClick={() => setScenarios(scenarios.filter((_, j) => j !== i))}>
                  <Trash2 className="size-4" /></Button>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="mt-6"><ErrorBox error={run.error} /></div>
      {res && (res.requests_replayed === 0 ? (
        <Card><Empty title="Nothing to replay">{res.note}</Empty></Card>
      ) : (
        <div className="mt-2 space-y-6">
          <Card title="Projected monthly cost by scenario"
                subtitle={`${res.requests_replayed.toLocaleString()} real requests from the last ${res.days} days, replayed`}>
            <ResponsiveContainer width="100%" height={Math.max(160, (chart?.length ?? 0) * 44)}>
              <BarChart data={chart} layout="vertical" margin={{ top: 4, right: 80, left: 8, bottom: 4 }}>
                <CartesianGrid stroke={C.grid} horizontal={false} />
                <XAxis type="number" stroke={C.axis} tick={{ fill: C.muted, fontSize: 11 }}
                       tickFormatter={(v) => money(v, v < 10 ? 2 : 0)} />
                <YAxis type="category" dataKey="name" width={230} axisLine={false} tickLine={false}
                       tick={{ fill: C.ink2, fontSize: 12 }} />
                <Tooltip cursor={{ fill: "#f1f5f9" }} formatter={(v) => [money(Number(v)), "Monthly cost"]} />
                {res.monthly_budget !== undefined && (
                  <ReferenceLine x={res.monthly_budget} stroke={C.critical} strokeDasharray="4 4"
                                 label={{ value: `Budget ${money(res.monthly_budget)}`, fill: C.ink2,
                                          fontSize: 11, position: "top" }} />
                )}
                <Bar dataKey="value" fill={C.s1} radius={[0, 4, 4, 0]} maxBarSize={22}>
                  <LabelList dataKey="value" position="right" formatter={(v) => money(Number(v))}
                             style={{ fill: C.ink2, fontSize: 11 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Details">
            <Table head={["Scenario", "Cost (period)", "Per month", "Budget", "Saved vs premium",
                          "Quality", "p95 latency"]}>
              {res.scenarios.map((s) => (
                <tr key={s.name}>
                  <Td className="min-w-56 font-medium">{s.name}</Td>
                  <Td className="num">{money(s.cost)}</Td>
                  <Td className="num">{money(s.monthly_projection)}</Td>
                  <Td><Badge tone={s.fits_budget ? "green" : "red"}>{s.fits_budget ? "Fits" : "Over"}</Badge></Td>
                  <Td className="num text-emerald-700">{pct(s.savings_vs_baseline_pct, 1)}</Td>
                  <Td className="num">{s.mean_quality?.toFixed(2) ?? "—"}
                    {s.quality_coverage < 1 && <span className="ml-1 text-xs text-slate-400">({Math.round(s.quality_coverage * 100)}% profiled)</span>}</Td>
                  <Td className="num">{ms(s.mean_p95_latency_ms)}</Td>
                </tr>
              ))}
            </Table>
          </Card>
        </div>
      ))}
    </>
  );
}
