import { clsx } from "clsx";
import { AlertCircle, Loader2, X } from "lucide-react";
import { useEffect } from "react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

export function cn(...args: Parameters<typeof clsx>) {
  return clsx(...args);
}

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({ variant = "primary", loading, className, children, disabled,
                         ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant; loading?: boolean;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600",
        {
          primary: "bg-brand-600 text-white hover:bg-brand-700",
          secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
          ghost: "text-slate-600 hover:bg-slate-100",
          danger: "bg-rose-600 text-white hover:bg-rose-700",
        }[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="size-4 animate-spin" />}
      {children}
    </button>
  );
}

export function Card({ title, subtitle, action, className, children }: {
  title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; className?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("rounded-xl border border-slate-200 bg-white shadow-sm", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, tone = "default", icon }: {
  label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <p className={cn("num mt-2 text-2xl font-semibold", {
        default: "text-slate-900", good: "text-emerald-600", warn: "text-amber-600",
        bad: "text-rose-600",
      }[tone])}>{value}</p>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

const BADGE = {
  gray: "bg-slate-100 text-slate-700",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  yellow: "bg-amber-50 text-amber-800 ring-amber-600/20",
  red: "bg-rose-50 text-rose-700 ring-rose-600/20",
  blue: "bg-brand-50 text-brand-700 ring-brand-600/20",
};

export function Badge({ tone = "gray", children }: { tone?: keyof typeof BADGE;
                                                    children: ReactNode }) {
  return (
    <span className={cn("inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium whitespace-nowrap ring-1 ring-inset ring-slate-500/10",
                        BADGE[tone])}>
      {children}
    </span>
  );
}

export function statusTone(status: string): keyof typeof BADGE {
  if (["success", "ok", "cache_hit"].includes(status)) return "green";
  if (["fallback_success", "soft_alert", "budget_soft", "forecast_overrun"].includes(status))
    return "yellow";
  if (["failed", "blocked_budget", "blocked_constraint", "hard_block", "budget_hard",
       "anomaly"].includes(status)) return "red";
  return "gray";
}

export function Field({ label, hint, children }: { label: string; hint?: string;
                                                   children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-400">{hint}</span>}
    </label>
  );
}

const control = "rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-xs " +
  "focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20";

/** Full width unless the caller sets its own width (w-*) class. */
const sized = (className?: string) => cn(control, !/(^|\s)w-/.test(className ?? "") && "w-full",
                                         className);

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={sized(props.className)} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cn(sized(props.className), "pr-8")} />;
}

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn(sized(props.className), "font-[inherit]")} />;
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-slate-500" role="status">
      <Loader2 className="size-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"
         role="alert">
      <AlertCircle className="mt-0.5 size-4 shrink-0" />
      <span>{error instanceof Error ? error.message : String(error)}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 px-6 py-10 text-center">
      <p className="text-sm font-medium text-slate-700">{title}</p>
      {children && <div className="mt-1 text-sm text-slate-500">{children}</div>}
    </div>
  );
}

export function PageHeader({ title, description, action }: {
  title: string; description?: ReactNode; action?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-500">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function Table({ head, children, empty }: { head: ReactNode[]; children: ReactNode;
                                                  empty?: ReactNode }) {
  return (
    <div className="-mx-5 overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
            {head.map((h, i) => <th key={i} className="px-5 py-2.5 whitespace-nowrap">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">{children}</tbody>
      </table>
      {empty}
    </div>
  );
}

export function Td({ children, className, ...rest }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn("px-5 py-2.5 align-top", className)} {...rest}>{children}</td>;
}

export function Modal({ open, onClose, title, children, wide }: {
  open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 pt-16"
         onClick={onClose} role="dialog" aria-modal="true" aria-label={title}>
      <div className={cn("w-full rounded-xl bg-white shadow-xl", wide ? "max-w-3xl" : "max-w-lg")}
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100"
                  aria-label="Close"><X className="size-4" /></button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export function Meter({ ratio, threshold }: { ratio: number; threshold?: number }) {
  const pctValue = Math.min(100, Math.max(0, ratio * 100));
  const tone = ratio >= 1 ? "bg-rose-500" : threshold && ratio >= threshold ? "bg-amber-500"
    : "bg-brand-600";
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-100"
         role="meter" aria-valuenow={Math.round(pctValue)} aria-valuemin={0} aria-valuemax={100}>
      <div className={cn("h-full rounded-full", tone)} style={{ width: `${pctValue}%` }} />
      {threshold !== undefined && (
        <div className="absolute inset-y-0 w-px bg-slate-400" style={{ left: `${threshold * 100}%` }} />
      )}
    </div>
  );
}
