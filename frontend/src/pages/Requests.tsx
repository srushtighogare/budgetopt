import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/client";
import type { RequestDetail, RequestLog } from "../api/types";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorBox,
  Modal,
  PageHeader,
  Select,
  Spinner,
  statusTone,
  Table,
  Td,
} from "../components/ui";
import { dateTime, money, ms, num, titleCase } from "../lib/format";

const STATUSES = ["success", "fallback_success", "cache_hit", "blocked_budget",
                  "blocked_constraint", "failed"];
const PAGE = 50;

export function RequestDetailView({ id }: { id: number }) {
  const q = useQuery({ queryKey: ["request", id],
                       queryFn: () => api<RequestDetail>(`/requests/${id}`) });
  if (q.isLoading) return <Spinner />;
  if (q.error) return <ErrorBox error={q.error} />;
  const r = q.data!;
  const w = r.candidates?.workload;
  return (
    <div className="space-y-5 text-sm">
      <div className="flex flex-wrap gap-2">
        <Badge tone={statusTone(r.status)}>{titleCase(r.status)}</Badge>
        {r.budget_state && <Badge tone={statusTone(r.budget_state)}>Budget: {titleCase(r.budget_state)}</Badge>}
        {r.fallback_used && <Badge tone="yellow">Fallback used</Badge>}
      </div>
      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Why this model</h3>
        <p className="leading-relaxed text-slate-700">{r.explanation}</p>
      </section>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[["Cost", money(r.cost)], ["Baseline", money(r.baseline_cost)],
          ["Saved", money(r.savings)], ["Latency", ms(r.latency_ms)],
          ["Tokens in / out", `${num(r.input_tokens)} / ${num(r.output_tokens)}`],
          ["Estimated", `${num(r.est_input_tokens)} / ${num(r.est_output_tokens)}`],
          ["Model", r.model ?? "—"], ["When", dateTime(r.created_at)]].map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-50 p-2.5">
            <p className="text-[11px] text-slate-500">{k}</p>
            <p className="num truncate font-medium text-slate-800">{v}</p>
          </div>
        ))}
      </div>
      {w && (
        <section>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Workload analysis</h3>
          <p className="text-slate-700">{titleCase(w.task_type)} (confidence {w.task_confidence.toFixed(2)};
            next: {w.task_alternatives.map(([t, p]) => `${titleCase(t)} ${p.toFixed(2)}`).join(", ") || "—"}),
            complexity {w.complexity_band} ({w.complexity.toFixed(2)}), analysed in {w.analysis_ms} ms.</p>
        </section>
      )}
      {r.candidates?.considered && (
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Candidates considered</h3>
          <Table head={["Model", "Est. cost", "Result"]}>
            {r.candidates.considered.map((c) => {
              const attempt = r.candidates?.attempts?.find((a) => a.model === c.model);
              return (
                <tr key={c.model}>
                  <Td>{c.model}</Td>
                  <Td className="num">{money(c.est_cost)}</Td>
                  <Td>{c.rejected_reason ? <span className="text-slate-500">Rejected: {c.rejected_reason}</span>
                    : attempt ? <Badge tone={attempt.outcome === "success" ? "green" : "red"}>
                        {attempt.outcome === "success" ? "Answered" : attempt.error_type ?? attempt.outcome}</Badge>
                      : <span className="text-slate-400">Eligible, not needed</span>}</Td>
                </tr>
              );
            })}
          </Table>
        </section>
      )}
      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Prompt</h3>
        {r.prompt_text
          ? <pre className="max-h-48 overflow-auto rounded-lg bg-slate-50 p-3 text-xs whitespace-pre-wrap">{r.prompt_text}</pre>
          : <p className="text-slate-500">Not stored: this team keeps only a SHA-256 hash of prompts.</p>}
      </section>
    </div>
  );
}

export default function RequestsPage() {
  const { team } = useTeam();
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<number | null>(null);

  const params = new URLSearchParams({ team_id: String(team?.id ?? ""), limit: String(PAGE),
                                       offset: String(page * PAGE) });
  if (status) params.set("status", status);
  const q = useQuery({ queryKey: ["requests", team?.id, status, page],
                       queryFn: () => api<RequestLog[]>(`/requests?${params}`),
                       enabled: Boolean(team) });

  return (
    <>
      <PageHeader title="Request log"
                  description="Every request with the model that served it, what it cost, what the baseline would have cost, and why. Click a row for the full routing decision." />
      <div className="mb-4 flex gap-3">
        <Select value={status} onChange={(e) => { setStatus(e.target.value); setPage(0); }}
                className="w-56" aria-label="Status filter">
          <option value="">All statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{titleCase(s)}</option>)}
        </Select>
      </div>
      <ErrorBox error={q.error} />
      <Card>
        {q.isLoading ? <Spinner /> : (
          <Table head={["When", "Status", "Model", "Tokens", "Cost", "Saved", "Latency", "Explanation"]}
                 empty={!q.data?.length && <Empty title="No requests yet" />}>
            {q.data?.map((r) => (
              <tr key={r.id} className="cursor-pointer hover:bg-slate-50" onClick={() => setOpen(r.id)}>
                <Td className="whitespace-nowrap text-slate-500">{dateTime(r.created_at)}</Td>
                <Td><Badge tone={statusTone(r.status)}>{titleCase(r.status)}</Badge></Td>
                <Td className="whitespace-nowrap">{r.model ?? "—"}</Td>
                <Td className="num whitespace-nowrap">{num(r.input_tokens)} / {num(r.output_tokens)}</Td>
                <Td className="num">{money(r.cost)}</Td>
                <Td className="num text-emerald-700">{money(r.savings)}</Td>
                <Td className="num">{ms(r.latency_ms)}</Td>
                <Td className="max-w-md truncate text-slate-500">{r.explanation}</Td>
              </tr>
            ))}
          </Table>
        )}
        <div className="mt-4 flex items-center justify-end gap-2 text-sm">
          <Button variant="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
          <span className="text-slate-500">Page {page + 1}</span>
          <Button variant="secondary" disabled={(q.data?.length ?? 0) < PAGE}
                  onClick={() => setPage(page + 1)}>Next</Button>
        </div>
      </Card>
      <Modal open={open !== null} onClose={() => setOpen(null)} title={`Request #${open}`} wide>
        {open !== null && <RequestDetailView id={open} />}
      </Modal>
    </>
  );
}
