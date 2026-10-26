/** Display helpers. Per-request costs are fractions of a cent, so money adapts its precision. */

export function money(value: number | string | null | undefined, digits?: number): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "string" ? Number(value) : value;
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  const d = digits ?? (abs === 0 ? 2 : abs < 0.01 ? 6 : abs < 1 ? 4 : 2);
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

export function ratioPct(ratio: number | null | undefined, digits = 0): string {
  return ratio === null || ratio === undefined ? "—" : pct(ratio * 100, digits);
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", { maximumFractionDigits: digits,
                                         minimumFractionDigits: digits });
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

export function dateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-IN", { day: "2-digit", month: "short",
                                                 hour: "2-digit", minute: "2-digit" });
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
}

const ACRONYMS: Record<string, string> = { Qa: "QA", Api: "API", Ets: "ETS", Llm: "LLM",
                                          Ilp: "ILP" };

export const titleCase = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w+/g, (w) => {
    const t = w.charAt(0).toUpperCase() + w.slice(1);
    return ACRONYMS[t] ?? t;
  });
