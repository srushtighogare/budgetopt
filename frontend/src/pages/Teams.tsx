import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, KeyRound, Plus, Settings2, UserPlus } from "lucide-react";
import { useState, type FormEvent } from "react";

import { api } from "../api/client";
import { useTeamSpend } from "../api/queries";
import type { ApiKey, Role, Team, User } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorBox,
  Field,
  Input,
  Meter,
  Modal,
  PageHeader,
  Select,
  Spinner,
  statusTone,
  Table,
  Td,
} from "../components/ui";
import { dateTime, money, ratioPct, titleCase } from "../lib/format";

function SpendCell({ team }: { team: Team }) {
  const { data } = useTeamSpend(team.id);
  if (!data) return <span className="text-slate-400">…</span>;
  return (
    <div className="w-48">
      <div className="mb-1 flex justify-between text-xs">
        <span className="num">{money(data.spend)}</span>
        <span className="text-slate-500">{ratioPct(data.usage_ratio)}</span>
      </div>
      <Meter ratio={data.usage_ratio} threshold={team.alert_threshold} />
    </div>
  );
}

function TeamForm({ team, onDone }: { team?: Team; onDone: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    name: team?.name ?? "", monthly_budget: team ? String(Number(team.monthly_budget)) : "10",
    alert_threshold: String(team?.alert_threshold ?? 0.8),
    privacy_tier: team?.privacy_tier ?? "public", store_prompts: team?.store_prompts ?? false,
  });
  const save = useMutation({
    mutationFn: () => {
      const body = { ...form, alert_threshold: Number(form.alert_threshold),
                     monthly_budget: form.monthly_budget };
      return team ? api(`/teams/${team.id}`, { method: "PATCH", json: body })
        : api("/teams", { method: "POST", json: body });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      qc.invalidateQueries({ queryKey: ["spend"] });
      onDone();
    },
  });
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) =>
    setForm({ ...form, [k]: e.target.value });

  return (
    <form className="space-y-4" onSubmit={(e: FormEvent) => { e.preventDefault(); save.mutate(); }}>
      <Field label="Team name"><Input required value={form.name} onChange={set("name")} /></Field>
      <div className="grid grid-cols-2 gap-4">
        <Field label="Monthly budget (USD)">
          <Input type="number" min="0.01" step="0.01" required value={form.monthly_budget}
                 onChange={set("monthly_budget")} />
        </Field>
        <Field label="Alert threshold" hint="Routing switches to the cost-saver policy here">
          <Select value={form.alert_threshold} onChange={set("alert_threshold")}>
            {[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].map((v) =>
              <option key={v} value={v}>{Math.round(v * 100)}%</option>)}
          </Select>
        </Field>
      </div>
      <Field label="Privacy tier" hint="Only models cleared for this tier can serve the team">
        <Select value={form.privacy_tier} onChange={set("privacy_tier")}>
          <option value="public">Public</option>
          <option value="internal">Internal</option>
          <option value="confidential">Confidential</option>
        </Select>
      </Field>
      <label className="flex items-start gap-2 text-sm text-slate-700">
        <input type="checkbox" className="mt-0.5" checked={form.store_prompts}
               onChange={(e) => setForm({ ...form, store_prompts: e.target.checked })} />
        <span>Store prompt text in the request log (personal data is redacted). Off by default:
          only a SHA-256 hash is kept.</span>
      </label>
      <ErrorBox error={save.error} />
      <div className="flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onDone}>Cancel</Button>
        <Button type="submit" loading={save.isPending}>{team ? "Save" : "Create team"}</Button>
      </div>
    </form>
  );
}

function ApiKeys({ team }: { team: Team }) {
  const qc = useQueryClient();
  const keys = useQuery({ queryKey: ["keys", team.id],
                          queryFn: () => api<ApiKey[]>(`/teams/${team.id}/api-keys`) });
  const [name, setName] = useState("");
  const [created, setCreated] = useState<ApiKey | null>(null);
  const [copied, setCopied] = useState(false);
  const create = useMutation({
    mutationFn: () => api<ApiKey>(`/teams/${team.id}/api-keys`, { method: "POST",
                                                                  json: { name } }),
    onSuccess: (k) => { setCreated(k); setName(""); qc.invalidateQueries({ queryKey: ["keys"] }); },
  });
  const revoke = useMutation({
    mutationFn: (id: number) => api(`/teams/${team.id}/api-keys/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["keys"] }),
  });

  return (
    <Card title={<span className="flex items-center gap-2"><KeyRound className="size-4" /> API keys</span>}
          subtitle="Applications call /v1/chat/completions with these keys (OpenAI-compatible)">
      <form className="mb-4 flex gap-2"
            onSubmit={(e) => { e.preventDefault(); if (name) create.mutate(); }}>
        <Input placeholder="Key name, e.g. support-bot" value={name}
               onChange={(e) => setName(e.target.value)} className="max-w-xs" />
        <Button type="submit" loading={create.isPending}><Plus className="size-4" /> Create key</Button>
      </form>
      {created?.key && (
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm">
          <p className="mb-2 font-medium text-amber-900">Copy this key now: it won't be shown again.</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 overflow-x-auto rounded bg-white px-2 py-1 text-xs">{created.key}</code>
            <Button variant="secondary" onClick={() => {
              navigator.clipboard.writeText(created.key!);
              setCopied(true);
            }}>{copied ? <Check className="size-4" /> : <Copy className="size-4" />} Copy</Button>
          </div>
        </div>
      )}
      <ErrorBox error={keys.error || create.error} />
      {keys.isLoading ? <Spinner /> : (
        <Table head={["Name", "Key", "Status", "Created", "Last used", ""]}
               empty={!keys.data?.length && <Empty title="No API keys yet" />}>
          {keys.data?.map((k) => (
            <tr key={k.id}>
              <Td className="font-medium">{k.name}</Td>
              <Td><code className="text-xs text-slate-500">{k.prefix}…</code></Td>
              <Td><Badge tone={k.is_active ? "green" : "gray"}>{k.is_active ? "Active" : "Revoked"}</Badge></Td>
              <Td className="text-slate-500">{dateTime(k.created_at)}</Td>
              <Td className="text-slate-500">{k.last_used_at ? dateTime(k.last_used_at) : "Never"}</Td>
              <Td className="text-right">
                {k.is_active && <Button variant="ghost" onClick={() => revoke.mutate(k.id)}>Revoke</Button>}
              </Td>
            </tr>
          ))}
        </Table>
      )}
    </Card>
  );
}

function Members({ team }: { team: Team }) {
  const { isAdmin } = useAuth();
  const qc = useQueryClient();
  const users = useQuery({ queryKey: ["users"], queryFn: () => api<User[]>("/users") });
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ email: "", full_name: "", password: "",
                                     role: "member" as Role });
  const create = useMutation({
    mutationFn: () => api("/users", { method: "POST", json: { ...form, team_id: team.id } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["users"] }); setOpen(false); },
  });
  const members = users.data?.filter((u) => u.team_id === team.id) ?? [];

  return (
    <Card title="Members" action={
      <Button variant="secondary" onClick={() => setOpen(true)}><UserPlus className="size-4" /> Add</Button>}>
      {users.isLoading ? <Spinner /> : (
        <Table head={["Name", "Email", "Role"]} empty={!members.length && <Empty title="No members" />}>
          {members.map((u) => (
            <tr key={u.id}>
              <Td className="font-medium">{u.full_name}</Td>
              <Td className="text-slate-500">{u.email}</Td>
              <Td><Badge tone={u.role === "member" ? "gray" : "blue"}>{titleCase(u.role)}</Badge></Td>
            </tr>
          ))}
        </Table>
      )}
      <Modal open={open} onClose={() => setOpen(false)} title={`Add member to ${team.name}`}>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
          <Field label="Full name"><Input required value={form.full_name}
                 onChange={(e) => setForm({ ...form, full_name: e.target.value })} /></Field>
          <Field label="Email"><Input type="email" required value={form.email}
                 onChange={(e) => setForm({ ...form, email: e.target.value })} /></Field>
          <Field label="Initial password" hint="At least 10 characters, letters and digits">
            <Input type="password" required minLength={10} value={form.password}
                   onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
          {isAdmin && (
            <Field label="Role">
              <Select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
                <option value="member">Member</option>
                <option value="team_manager">Team manager</option>
              </Select>
            </Field>
          )}
          <ErrorBox error={create.error} />
          <div className="flex justify-end gap-2">
            <Button type="button" variant="secondary" onClick={() => setOpen(false)}>Cancel</Button>
            <Button type="submit" loading={create.isPending}>Add member</Button>
          </div>
        </form>
      </Modal>
    </Card>
  );
}

export default function TeamsPage() {
  const { isAdmin, isManager } = useAuth();
  const { teams, team: current, setTeamId } = useTeam();
  const [editing, setEditing] = useState<Team | "new" | null>(null);

  return (
    <>
      <PageHeader title="Teams & budgets"
                  description="Monthly budgets, alert thresholds and privacy settings. At the threshold, routing switches to the team's cost-saver policy; at 100% requests are blocked before reaching any model."
                  action={isAdmin && <Button onClick={() => setEditing("new")}><Plus className="size-4" /> New team</Button>} />
      <Card>
        <Table head={["Team", "Budget", "This month", "Alert at", "Privacy", "Policy", ""]}>
          {teams.map((t) => (
            <tr key={t.id} className={t.id === current?.id ? "bg-brand-50/40" : undefined}>
              <Td>
                <button className="font-medium text-slate-900 hover:text-brand-700"
                        onClick={() => setTeamId(t.id)}>{t.name}</button>
              </Td>
              <Td className="num">{money(t.monthly_budget)}</Td>
              <Td><SpendCell team={t} /></Td>
              <Td>{Math.round(t.alert_threshold * 100)}%</Td>
              <Td><Badge tone={statusTone(t.privacy_tier === "confidential" ? "anomaly" : "ok")}>
                {titleCase(t.privacy_tier)}</Badge></Td>
              <Td>{t.policy_id ? <Badge tone="blue">Policy #{t.policy_id}</Badge>
                : <span className="text-xs text-slate-400">Cheapest eligible</span>}</Td>
              <Td className="text-right">
                {isAdmin && <Button variant="ghost" onClick={() => setEditing(t)}
                                    aria-label={`Edit ${t.name}`}><Settings2 className="size-4" /></Button>}
              </Td>
            </tr>
          ))}
        </Table>
      </Card>

      {current && isManager && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <ApiKeys team={current} />
          <Members team={current} />
        </div>
      )}

      <Modal open={editing !== null} onClose={() => setEditing(null)}
             title={editing === "new" ? "New team" : `Edit ${editing?.name ?? ""}`}>
        {editing !== null && <TeamForm team={editing === "new" ? undefined : editing}
                                       onDone={() => setEditing(null)} />}
      </Modal>
    </>
  );
}
