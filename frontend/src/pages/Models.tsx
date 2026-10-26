import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PlugZap } from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { useModels } from "../api/queries";
import type { LLMModel, ProfilingSummary } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { rampColor } from "../components/charts";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorBox,
  Field,
  Input,
  Modal,
  PageHeader,
  Spinner,
  Table,
  Td,
} from "../components/ui";
import { money, ms, num, titleCase } from "../lib/format";

interface Provider { name: string; is_active: boolean; has_api_key: boolean;
                     adapter_available: boolean }

function QualityHeatmap({ data }: { data: ProfilingSummary }) {
  const [hover, setHover] = useState<string | null>(null);
  const rows = data.models.filter((m) => Object.values(m.tasks).some((t) => t.quality !== null));
  if (!rows.length) {
    return <Empty title="No quality profiles yet">
      Run <code className="text-xs">python -m app.services.profiling.cli quality</code> to profile models.
    </Empty>;
  }
  return (
    <div className="-mx-5 overflow-x-auto">
      <table className="min-w-full text-xs" aria-label="Quality per model and task">
        <thead>
          <tr className="text-slate-500">
            <th className="px-5 py-2 text-left font-medium">Model</th>
            {data.task_types.map((t) => <th key={t} className="px-2 py-2 font-medium">{titleCase(t)}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.model_id}>
              <td className="whitespace-nowrap px-5 py-1 font-medium text-slate-700">{m.display_name}</td>
              {data.task_types.map((t) => {
                const c = m.tasks[t];
                const q = c?.quality ?? null;
                const id = `${m.model_id}-${t}`;
                return (
                  <td key={t} className="relative p-0.5" onMouseEnter={() => setHover(id)}
                      onMouseLeave={() => setHover(null)}>
                    <div className="num grid h-9 place-items-center rounded-md text-[11px] font-medium"
                         style={{ background: q === null ? "#f1f5f9" : rampColor(q, 0, 10),
                                  color: q === null ? "#94a3b8" : q >= 6 ? "#fff" : "#0b0b0b" }}>
                      {q === null ? "–" : q.toFixed(1)}
                    </div>
                    {hover === id && c && (
                      <div className="absolute top-full left-1/2 z-20 mt-1 w-44 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-2 text-left shadow-md">
                        <p className="font-medium text-slate-900">{m.display_name}</p>
                        <p className="text-slate-500">{titleCase(t)}</p>
                        <p className="mt-1 text-slate-600">Quality <b className="num">{q?.toFixed(2) ?? "—"}</b> / 10
                          ({c.samples} graded)</p>
                        <p className="text-slate-600">Latency p50 {ms(c.latency_p50_ms)}, p95 {ms(c.latency_p95_ms)}</p>
                      </div>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex items-center gap-2 px-5 text-[11px] text-slate-500">
        <span>Quality 0</span>
        <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(to right, ${rampColor(0, 0, 10)}, ${rampColor(10, 0, 10)})` }} />
        <span>10</span>
        <span className="ml-3">Math is scored against gold answers; other tasks by a judge model that never grades itself.</span>
      </div>
    </div>
  );
}

function PriceModal({ model, onClose }: { model: LLMModel; onClose: () => void }) {
  const qc = useQueryClient();
  const [pin, setPin] = useState(String(Number(model.current_pricing?.input_per_1m ?? 0)));
  const [pout, setPout] = useState(String(Number(model.current_pricing?.output_per_1m ?? 0)));
  const save = useMutation({
    mutationFn: () => api(`/models/${model.id}/pricing`, { method: "POST",
      json: { input_per_1m: pin, output_per_1m: pout } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); onClose(); },
  });
  return (
    <Modal open onClose={onClose} title={`New price for ${model.display_name}`}>
      <p className="mb-4 text-sm text-slate-500">A new price version takes effect now. Older
        versions are kept, so past request costs stay correct.</p>
      <div className="grid grid-cols-2 gap-4">
        <Field label="Input, USD per 1M tokens"><Input type="number" step="0.0001" min="0"
               value={pin} onChange={(e) => setPin(e.target.value)} /></Field>
        <Field label="Output, USD per 1M tokens"><Input type="number" step="0.0001" min="0"
               value={pout} onChange={(e) => setPout(e.target.value)} /></Field>
      </div>
      <ErrorBox error={save.error} />
      <div className="mt-4 flex justify-end gap-2">
        <Button variant="secondary" onClick={onClose}>Cancel</Button>
        <Button loading={save.isPending} onClick={() => save.mutate()}>Save price</Button>
      </div>
    </Modal>
  );
}

function Providers() {
  const qc = useQueryClient();
  const providers = useQuery({ queryKey: ["providers"],
                               queryFn: () => api<Provider[]>("/providers") });
  const [editing, setEditing] = useState<string | null>(null);
  const [key, setKey] = useState("");
  const save = useMutation({
    mutationFn: () => api(`/providers/${editing}/key`, { method: "PUT", json: { api_key: key } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["providers"] }); setEditing(null); setKey(""); },
  });
  return (
    <Card title="Providers" subtitle="API keys are encrypted at rest and never shown again">
      <Table head={["Provider", "API key", "Status", ""]}>
        {providers.data?.map((p) => (
          <tr key={p.name}>
            <Td className="font-medium">{titleCase(p.name)}</Td>
            <Td>{p.has_api_key ? <Badge tone="green">Configured</Badge> : <Badge>Not set</Badge>}</Td>
            <Td>{p.is_active ? "Active" : "Disabled"}</Td>
            <Td className="text-right"><Button variant="ghost" onClick={() => setEditing(p.name)}>
              {p.has_api_key ? "Replace key" : "Set key"}</Button></Td>
          </tr>
        ))}
      </Table>
      <Modal open={editing !== null} onClose={() => setEditing(null)}
             title={`API key for ${titleCase(editing ?? "")}`}>
        <Field label="API key"><Input type="password" autoComplete="off" value={key}
               onChange={(e) => setKey(e.target.value)} /></Field>
        <ErrorBox error={save.error} />
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="secondary" onClick={() => setEditing(null)}>Cancel</Button>
          <Button loading={save.isPending} disabled={key.length < 8}
                  onClick={() => save.mutate()}>Save key</Button>
        </div>
      </Modal>
    </Card>
  );
}

export default function ModelsPage() {
  const { isAdmin } = useAuth();
  const models = useModels();
  const profiling = useQuery({ queryKey: ["profiling"],
                               queryFn: () => api<ProfilingSummary>("/profiling/summary") });
  const [pricing, setPricing] = useState<LLMModel | null>(null);
  const [testing, setTesting] = useState<number | null>(null);
  const [testResult, setTestResult] = useState<Record<number, string>>({});

  const test = async (m: LLMModel) => {
    setTesting(m.id);
    try {
      const r = await api<{ ok: boolean; text: string | null; latency_ms: number | null;
                            cost: string; attempts: { error_type?: string }[] }>(
        `/models/${m.id}/test`, { method: "POST" });
      setTestResult({ ...testResult, [m.id]: r.ok
        ? `OK in ${ms(r.latency_ms)}, ${money(r.cost)}`
        : `Failed: ${r.attempts.at(-1)?.error_type ?? "error"}` });
    } catch (e) {
      setTestResult({ ...testResult, [m.id]: (e as Error).message });
    } finally {
      setTesting(null);
    }
  };

  return (
    <>
      <PageHeader title="Models"
                  description="Every registered model with its current price, limits and measured quality per task. Prices are versioned; quality and latency come from profiling and live traffic." />
      <ErrorBox error={models.error} />
      <Card title="Model registry">
        {models.isLoading ? <Spinner /> : (
          <Table head={["Model", "Provider", "Input / 1M", "Output / 1M", "Context", "Privacy",
                        "Status", ...(isAdmin ? [""] : [])]}>
            {models.data?.map((m) => (
              <tr key={m.id}>
                <Td><p className="font-medium text-slate-900">{m.display_name}</p>
                  <p className="text-xs text-slate-400">{m.name}</p></Td>
                <Td>{titleCase(m.provider)}</Td>
                <Td className="num">{money(m.current_pricing?.input_per_1m, 3)}</Td>
                <Td className="num">{money(m.current_pricing?.output_per_1m, 3)}</Td>
                <Td className="num">{num(m.context_window / 1000)}k</Td>
                <Td>{titleCase(m.max_privacy_tier)}</Td>
                <Td><Badge tone={m.is_active ? "green" : "gray"}>{m.is_active ? "Active" : "Off"}</Badge></Td>
                {isAdmin && (
                  <Td className="whitespace-nowrap text-right">
                    {testResult[m.id] && <span className="mr-2 text-xs text-slate-500">{testResult[m.id]}</span>}
                    <Button variant="ghost" loading={testing === m.id} onClick={() => test(m)}
                            title="One tiny real call"><PlugZap className="size-4" /> Test</Button>
                    <Button variant="ghost" onClick={() => setPricing(m)}>Price</Button>
                  </Td>
                )}
              </tr>
            ))}
          </Table>
        )}
      </Card>

      <Card className="mt-6" title="Quality by task type (0–10)"
            subtitle="Profiled on a fixed set of 64 prompts, 8 per task type; the optimizer uses these numbers">
        {profiling.isLoading ? <Spinner /> : profiling.data && <QualityHeatmap data={profiling.data} />}
      </Card>

      {isAdmin && <div className="mt-6"><Providers /></div>}
      {pricing && <PriceModal model={pricing} onClose={() => setPricing(null)} />}
    </>
  );
}
