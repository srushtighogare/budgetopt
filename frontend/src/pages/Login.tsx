import { ChartLine } from "lucide-react";
import { useState, type FormEvent } from "react";

import { useAuth } from "../auth/AuthContext";
import { Button, ErrorBox, Field, Input } from "../components/ui";

export default function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid min-h-full place-items-center bg-gradient-to-br from-brand-50 via-white to-slate-50 px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-brand-600 text-white">
            <ChartLine className="size-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-900">BudgetOpt</h1>
            <p className="text-xs text-slate-500">Cost-optimal LLM selection and budget control</p>
          </div>
        </div>
        <form onSubmit={submit}
              className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <Field label="Email">
            <Input type="email" autoComplete="username" required value={email}
                   onChange={(e) => setEmail(e.target.value)} />
          </Field>
          <Field label="Password">
            <Input type="password" autoComplete="current-password" required value={password}
                   onChange={(e) => setPassword(e.target.value)} />
          </Field>
          <ErrorBox error={error} />
          <Button type="submit" className="w-full" loading={busy}>Sign in</Button>
        </form>
      </div>
    </div>
  );
}
