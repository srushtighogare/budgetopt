/**
 * Chart building blocks. Colours follow the validated reference palette: categorical
 * slots in fixed order (blue = actual / primary, orange = comparison), a single blue
 * ramp for magnitude, neutral ink for text, recessive grid and axes. Every chart has
 * a hover tooltip; every multi-series chart has a legend.
 */
import type { ReactNode } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { money, ms, shortDate } from "../lib/format";

export const C = {
  s1: "#2a78d6",
  s2: "#eb6834",
  s1Soft: "#b7d3f6",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  surface: "#fcfcfb",
  critical: "#d03b3b",
};

const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

/** Sequential blue step for a value in [lo, hi]. */
export function rampColor(v: number, lo: number, hi: number): string {
  const t = hi > lo ? (v - lo) / (hi - lo) : 1;
  return RAMP[Math.max(0, Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1))))];
}

const axisProps = {
  stroke: C.axis,
  tick: { fill: C.muted, fontSize: 11 },
  tickLine: false,
};

function TooltipBox({ title, rows }: { title: ReactNode;
                                        rows: { label: string; value: string; color?: string;
                                                dashed?: boolean }[] }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-medium text-slate-900">{title}</p>
      {rows.map((r) => (
        <p key={r.label} className="flex items-center gap-2 text-slate-600">
          {r.color && <span className="inline-block h-0.5 w-3"
                            style={{ background: r.color,
                                     borderTop: r.dashed ? `2px dashed ${r.color}` : undefined }} />}
          <span>{r.label}</span>
          <span className="num ml-auto pl-4 font-medium text-slate-900">{r.value}</span>
        </p>
      ))}
    </div>
  );
}

// --- Spend vs baseline over time -------------------------------------------------------

export function SpendChart({ data }: { data: { date: string; spend: number; baseline: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 4 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} {...axisProps} minTickGap={24} />
        <YAxis tickFormatter={(v) => money(v, v < 1 ? 2 : 0)} {...axisProps} width={56} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload, label }) =>
          active && payload?.length ? (
            <TooltipBox title={shortDate(String(label))} rows={[
              { label: "Actual spend", value: money(payload[0]?.payload.spend), color: C.s1 },
              { label: "Priciest-model baseline", value: money(payload[0]?.payload.baseline),
                color: C.s2, dashed: true },
            ]} />
          ) : null} />
        <Legend verticalAlign="top" align="right" height={28} iconType="plainline"
                wrapperStyle={{ fontSize: 12, color: C.ink2 }} />
        <Area type="monotone" dataKey="spend" name="Actual spend" stroke={C.s1} strokeWidth={2}
              fill={C.s1Soft} fillOpacity={0.5} dot={false} activeDot={{ r: 4 }} />
        <Line type="monotone" dataKey="baseline" name="Priciest-model baseline" stroke={C.s2}
              strokeWidth={2} strokeDasharray="5 4" dot={false} activeDot={{ r: 4 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// --- Forecast: cumulative actual + projection with interval ----------------------------

export interface ForecastPoint {
  date: string;
  actual: number | null;
  projected: number | null;
  band: [number, number] | null;
}

export function ForecastChart({ data, budget }: { data: ForecastPoint[]; budget: number }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 4 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} {...axisProps} minTickGap={24} />
        <YAxis tickFormatter={(v) => money(v, v < 10 ? 2 : 0)} {...axisProps} width={60}
               domain={[0, (max: number) => Math.max(max, budget) * 1.08]} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload, label }) => {
          if (!active || !payload?.length) return null;
          const p = payload[0].payload as ForecastPoint;
          const rows = [];
          if (p.actual !== null) rows.push({ label: "Spent (cumulative)", value: money(p.actual),
                                             color: C.s1 });
          if (p.projected !== null) rows.push({ label: "Forecast (cumulative)",
                                                value: money(p.projected), color: C.s1,
                                                dashed: true });
          if (p.band) rows.push({ label: "90% interval",
                                  value: `${money(p.band[0])} – ${money(p.band[1])}` });
          rows.push({ label: "Budget", value: money(budget), color: C.critical, dashed: true });
          return <TooltipBox title={shortDate(String(label))} rows={rows} />;
        }} />
        <Legend verticalAlign="top" align="right" height={28} iconType="plainline"
                wrapperStyle={{ fontSize: 12, color: C.ink2 }} />
        <Area dataKey="band" name="90% interval" stroke="none" fill={C.s1Soft} fillOpacity={0.6}
              activeDot={false} isAnimationActive={false} />
        <Line dataKey="actual" name="Spent" stroke={C.s1} strokeWidth={2} dot={false}
              activeDot={{ r: 4 }} connectNulls={false} />
        <Line dataKey="projected" name="Forecast" stroke={C.s1} strokeWidth={2}
              strokeDasharray="5 4" dot={false} activeDot={{ r: 4 }} />
        <ReferenceLine y={budget} stroke={C.critical} strokeDasharray="4 4"
                       label={{ value: `Budget ${money(budget)}`, position: "insideTopLeft",
                                fill: C.ink2, fontSize: 11 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// --- Horizontal bar list (magnitude, one series) -----------------------------------------

export function BarList({ data, valueLabel = "Spend", format = (v: number) => money(v) }: {
  data: { name: string; value: number; hint?: string }[];
  valueLabel?: string;
  format?: (v: number) => string;
}) {
  const height = Math.max(120, data.length * 34 + 20);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 72, bottom: 0, left: 0 }}
                barCategoryGap={8}>
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="name" width={150} {...axisProps} axisLine={false}
               tick={{ fill: C.ink2, fontSize: 12 }} />
        <Tooltip cursor={{ fill: "#f1f5f9" }} content={({ active, payload }) =>
          active && payload?.length ? (
            <TooltipBox title={payload[0].payload.name} rows={[
              { label: valueLabel, value: format(payload[0].payload.value), color: C.s1 },
              ...(payload[0].payload.hint ? [{ label: "", value: payload[0].payload.hint }] : []),
            ]} />
          ) : null} />
        <Bar dataKey="value" fill={C.s1} radius={[0, 4, 4, 0]} maxBarSize={20}>
          <LabelList dataKey="value" position="right" formatter={(v) => format(Number(v))}
                     style={{ fill: C.ink2, fontSize: 11 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// --- Pareto front (cost vs quality) -------------------------------------------------------

export interface ParetoPoint {
  index: number;
  monthly_cost: number;
  quality: number;
  latency_ms: number;
  label?: string;
}

export function ParetoChart({ front, baselines, selected, onSelect }: {
  front: ParetoPoint[];
  baselines: ParetoPoint[];
  selected?: number | null;
  onSelect?: (index: number) => void;
}) {
  const tooltip = (active?: boolean, payload?: readonly { payload?: unknown }[]) => {
    const p = payload?.[0]?.payload as ParetoPoint | undefined;
    if (!active || !p) return null;
    return <TooltipBox title={p.label ?? `Front point #${p.index}`} rows={[
      { label: "Monthly cost", value: money(p.monthly_cost) },
      { label: "Mean quality", value: `${p.quality.toFixed(2)} / 10` },
      { label: "Mean p95 latency", value: ms(p.latency_ms) },
    ]} />;
  };
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 16, right: 24, bottom: 16, left: 4 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis type="number" dataKey="monthly_cost" name="Monthly cost" {...axisProps}
               tickFormatter={(v) => money(v, v < 10 ? 2 : 0)} domain={["auto", "auto"]}
               label={{ value: "Monthly cost →", position: "insideBottomRight", offset: -8,
                        fill: C.muted, fontSize: 11 }} />
        <YAxis type="number" dataKey="quality" name="Quality" {...axisProps} width={44}
               domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(1)}
               label={{ value: "Quality ↑", angle: -90, position: "insideLeft", fill: C.muted,
                        fontSize: 11 }} />
        <ZAxis range={[70, 70]} />
        <Tooltip content={(props) => tooltip(props.active, props.payload)}
                 cursor={{ strokeDasharray: "3 3" }} />
        <Legend verticalAlign="top" align="right" height={28}
                wrapperStyle={{ fontSize: 12, color: C.ink2 }} />
        <Scatter name="Optimized allocations (Pareto front)" data={front} fill={C.s1}
                 stroke={C.surface} strokeWidth={2} cursor="pointer"
                 onClick={(p: { payload?: ParetoPoint }) =>
                   p.payload && onSelect?.(p.payload.index)}
                 shape={(props: { cx?: number; cy?: number; payload?: ParetoPoint }) => {
                   const isSel = props.payload?.index === selected;
                   return <circle cx={props.cx} cy={props.cy} r={isSel ? 8 : 5}
                                  fill={C.s1} stroke={isSel ? C.ink : C.surface}
                                  strokeWidth={2} />;
                 }}>
          <LabelList dataKey="label" position="top" style={{ fill: C.ink2, fontSize: 11 }} />
        </Scatter>
        <Scatter name="Without optimization (baselines)" data={baselines} fill={C.s2}
                 shape="diamond" stroke={C.surface} strokeWidth={2} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
