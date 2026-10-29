import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { useModels } from "../api/queries";
import type { RouteResult } from "../api/types";
import { Markdown } from "../components/Markdown";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Button,
  Card,
  ErrorBox,
  Field,
  Input,
  PageHeader,
  Select,
  Stat,
  statusTone,
  Textarea,
} from "../components/ui";
import { money, ms, num, titleCase } from "../lib/format";

const EXAMPLES = [
  "What is the capital of Australia?",
  "Translate 'Where is the railway station?' into Hindi.",
  "Write a Python function that checks whether a string is a palindrome.",
  "A shop sells pens at ₹12 each and gives 1 free for every 5 bought. How much do 17 pens cost?",
  "Compare microservices and a monolith for a 5-person startup and recommend one.",
];

export default function PlaygroundPage() {
  const { team } = useTeam();
  const models = useModels();
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [system, setSystem] = useState("");
  const [model, setModel] = useState("auto");
  const [maxTokens, setMaxTokens] = useState("512");
  const [useCache, setUseCache] = useState(true);

  const run = useMutation({
    mutationFn: () => api<RouteResult>("/route", { method: "POST", json: {
      team_id: team?.id, model, use_cache: useCache, max_tokens: Number(maxTokens),
      messages: [...(system ? [{ role: "system", content: system }] : []),
                 { role: "user", content: prompt }],
    } }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["spend"] });
      qc.invalidateQueries({ queryKey: ["requests"] });
      qc.invalidateQueries({ queryKey: ["savings"] });
    },
  });
  const r = run.data;

  return (
    <>
      <PageHeader title="Playground"
                  description={`Send a real request through BudgetOpt as ${team?.name ?? "your team"} and see how it is analysed, routed and priced. It counts against the team's budget.`} />
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2" title="Request">
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
            <Field label="Prompt">
              <Textarea rows={6} value={prompt} onChange={(e) => setPrompt(e.target.value)} required />
            </Field>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((ex) => (
                <button type="button" key={ex} onClick={() => setPrompt(ex)}
                        className="rounded-full border border-slate-200 px-2.5 py-1 text-xs text-slate-600 hover:border-brand-300 hover:text-brand-700">
                  {ex.length > 38 ? `${ex.slice(0, 38)}…` : ex}
                </button>
              ))}
            </div>
            <Field label="System prompt (optional)">
              <Input value={system} onChange={(e) => setSystem(e.target.value)}
                     placeholder="e.g. Answer in two sentences." />
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Model">
                <Select value={model} onChange={(e) => setModel(e.target.value)}>
                  <option value="auto">auto (let BudgetOpt choose)</option>
                  {models.data?.filter((m) => m.is_active).map((m) =>
                    <option key={m.id} value={m.name}>{m.display_name}</option>)}
                </Select>
              </Field>
              <Field label="Max output tokens">
                <Input type="number" min={16} max={8192} value={maxTokens}
                       onChange={(e) => setMaxTokens(e.target.value)} />
              </Field>
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
              Allow cached answers (exact and semantic)
            </label>
            <Button type="submit" loading={run.isPending} className="w-full">
              <Send className="size-4" /> Send
            </Button>
          </form>
        </Card>

        <div className="space-y-6 lg:col-span-3">
          <ErrorBox error={run.error} />
          {!r && !run.isPending && (
            <Card><p className="text-sm text-slate-500">The answer, the chosen model and the full
              routing explanation will appear here.</p></Card>
          )}
          {r && (
            <>
              <div className="grid grid-cols-2 gap-4 2xl:grid-cols-4">
                <Stat label="Model" value={<span className="text-base break-all">{r.model ?? "—"}</span>} />
                <Stat label="Cost" value={money(r.cost)} />
                <Stat label="Saved vs baseline" tone="good" value={money(r.savings)} />
                <Stat label="Latency" value={ms(r.latency_ms)} />
              </div>
              <Card title="Answer" action={<Badge tone={statusTone(r.status)}>{titleCase(r.status)}</Badge>}>
                {r.text ? <Markdown>{r.text}</Markdown>
                  : <p className="text-sm text-slate-500">No answer (see explanation).</p>}
                <p className="mt-3 text-xs text-slate-400">{num(r.input_tokens)} input · {num(r.output_tokens)} output tokens · request #{r.request_id}</p>
              </Card>
              <Card title="Why this model">
                <p className="text-sm leading-relaxed text-slate-700">{r.explanation}</p>
                {r.workload && (
                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                    {[["Task type", titleCase(r.workload.task_type)],
                      ["Confidence", r.workload.task_confidence.toFixed(2)],
                      ["Complexity", `${r.workload.complexity_band} (${r.workload.complexity.toFixed(2)})`],
                      ["Est. tokens", `${r.workload.est_input_tokens} / ${r.workload.est_output_tokens}`]]
                      .map(([k, v]) => (
                        <div key={k} className="rounded-lg bg-slate-50 p-2.5">
                          <p className="text-[11px] text-slate-500">{k}</p>
                          <p className="font-medium text-slate-800">{v}</p>
                        </div>
                      ))}
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
