import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/client";
import type { Alert } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useTeam } from "../components/TeamContext";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorBox,
  PageHeader,
  Select,
  Spinner,
  statusTone,
  Table,
  Td,
} from "../components/ui";
import { dateTime, titleCase } from "../lib/format";

const KINDS = ["budget_soft", "budget_hard", "forecast_overrun", "anomaly"];

export default function AlertsPage() {
  const { team } = useTeam();
  const { isManager, isAdmin } = useAuth();
  const qc = useQueryClient();
  const [kind, setKind] = useState("");
  const [open, setOpen] = useState("false");

  const params = new URLSearchParams({ team_id: String(team?.id ?? ""), limit: "200" });
  if (kind) params.set("kind", kind);
  if (open) params.set("acknowledged", open);
  const alerts = useQuery({
    queryKey: ["alerts", team?.id, kind, open],
    queryFn: () => api<Alert[]>(`/alerts?${params}`),
    enabled: Boolean(team),
  });
  const ack = useMutation({
    mutationFn: (id: number) => api(`/alerts/${id}/ack`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
  const scan = useMutation({
    mutationFn: async () => {
      await api("/monitoring/anomalies/scan", { method: "POST" });
      await api("/monitoring/forecasts/refresh", { method: "POST" });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["alerts"] });
      qc.invalidateQueries({ queryKey: ["forecast"] });
    },
  });

  return (
    <>
      <PageHeader title="Alerts"
                  description="Budget thresholds and blocks, likely overruns from the forecast, and unusual usage (each caller compared with its own usual activity at that time of day)."
                  action={isAdmin && <Button variant="secondary" loading={scan.isPending}
                                             onClick={() => scan.mutate()}>Run checks now</Button>} />
      <div className="mb-4 flex flex-wrap gap-3">
        <Select value={kind} onChange={(e) => setKind(e.target.value)} className="w-52"
                aria-label="Alert type">
          <option value="">All types</option>
          {KINDS.map((k) => <option key={k} value={k}>{titleCase(k)}</option>)}
        </Select>
        <Select value={open} onChange={(e) => setOpen(e.target.value)} className="w-44"
                aria-label="Status">
          <option value="false">Open</option>
          <option value="true">Acknowledged</option>
          <option value="">All</option>
        </Select>
      </div>
      <ErrorBox error={alerts.error || scan.error} />
      <Card>
        {alerts.isLoading ? <Spinner /> : (
          <Table head={["When", "Type", "Message", ""]}
                 empty={!alerts.data?.length && <Empty title="No alerts match these filters" />}>
            {alerts.data?.map((a) => (
              <tr key={a.id}>
                <Td className="whitespace-nowrap text-slate-500">{dateTime(a.created_at)}</Td>
                <Td><Badge tone={statusTone(a.kind)}>{titleCase(a.kind)}</Badge></Td>
                <Td className="max-w-2xl text-slate-700">{a.message}</Td>
                <Td className="text-right">
                  {!a.acknowledged && isManager &&
                    <Button variant="ghost" onClick={() => ack.mutate(a.id)}>Acknowledge</Button>}
                </Td>
              </tr>
            ))}
          </Table>
        )}
      </Card>
    </>
  );
}
