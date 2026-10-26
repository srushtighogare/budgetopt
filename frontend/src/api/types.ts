// Response shapes of the BudgetOpt API (only the fields the dashboard uses).

export type Role = "org_admin" | "team_manager" | "member";

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: Role;
  team_id: number | null;
  is_active: boolean;
}

export interface Team {
  id: number;
  name: string;
  monthly_budget: string;
  alert_threshold: number;
  privacy_tier: "public" | "internal" | "confidential";
  store_prompts: boolean;
  policy_id: number | null;
  created_at: string;
}

export interface TeamSpend {
  team_id: number;
  period_start: string;
  monthly_budget: string;
  spend: string;
  usage_ratio: number;
  budget_state: "ok" | "soft_alert" | "hard_block";
}

export interface ApiKey {
  id: number;
  team_id: number;
  name: string;
  prefix: string;
  is_active: boolean;
  created_at: string;
  last_used_at: string | null;
  key?: string;
}

export interface Pricing {
  id: number;
  input_per_1m: string;
  output_per_1m: string;
  effective_from: string;
}

export interface Performance {
  task_type: string;
  quality_score: number | null;
  quality_samples: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
}

export interface LLMModel {
  id: number;
  provider: string;
  name: string;
  display_name: string;
  context_window: number;
  max_output_tokens: number;
  max_privacy_tier: string;
  is_active: boolean;
  current_pricing: Pricing | null;
  performance: Performance[];
}

export interface RequestLog {
  id: number;
  team_id: number;
  user_id: number | null;
  api_key_id: number | null;
  status: string;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost: string;
  latency_ms: number | null;
  created_at: string;
  explanation: string | null;
  savings: string | null;
}

export interface RequestDetail extends RequestLog {
  prompt_text: string | null;
  est_input_tokens: number | null;
  est_output_tokens: number | null;
  budget_state: string | null;
  baseline_cost: string | null;
  fallback_used: boolean | null;
  candidates: {
    strategy?: string;
    considered?: { model: string; provider: string; est_cost: string | null;
                   rejected_reason: string | null }[];
    attempts?: { model: string; outcome: string; error_type?: string; latency_ms?: number }[];
    workload?: Workload;
  } | null;
}

export interface Workload {
  task_type: string;
  task_confidence: number;
  task_alternatives: [string, number][];
  complexity: number;
  complexity_band: string;
  est_input_tokens: number;
  est_output_tokens: number;
  analysis_ms: number;
  sources: Record<string, string>;
}

export interface RouteResult {
  request_id: number;
  status: string;
  text: string | null;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  cost: string;
  baseline_cost: string | null;
  savings: string | null;
  latency_ms: number | null;
  budget_state: string;
  explanation: string;
  attempts: { model: string; outcome: string; error_type?: string }[];
  workload: Workload | null;
}

export interface Alert {
  id: number;
  team_id: number;
  user_id: number | null;
  kind: "budget_soft" | "budget_hard" | "forecast_overrun" | "anomaly";
  message: string;
  details: Record<string, unknown> | null;
  acknowledged: boolean;
  created_at: string;
}

export interface ForecastDay {
  date: string;
  actual: number | null;
  mean?: number;
  lower?: number;
  upper?: number;
  cumulative_mean?: number;
}

export interface Forecast {
  id: number;
  team_id: number;
  method: string;
  created_at: string;
  period_start: string;
  period_end: string;
  projected_spend: string;
  lower: string;
  upper: string;
  overrun_date: string | null;
  overrun_probability: number | null;
  daily: ForecastDay[];
  backtest: Record<string, number | string | boolean | null> | null;
  monthly_budget: string;
}

export interface Breakdown {
  name: string;
  requests: number;
  spend: number;
  savings: number;
}

export interface SavingsReport {
  period: { from: string; to: string; days: number };
  team_id: number | null;
  requests: number;
  cache_hits: number;
  blocked_by_budget: number;
  failed: number;
  fallbacks: number;
  spend: number;
  baseline_spend: number;
  savings: number;
  routing_savings: number;
  cache_savings: number;
  savings_pct: number | null;
  monthly_projection: { spend: number; savings: number };
  median_latency_ms: number | null;
  by_model: Breakdown[];
  by_task: Breakdown[];
  by_team: Breakdown[] | null;
  daily: { date: string; spend: number; baseline: number }[];
  economic_feasibility: {
    platform_cost_per_month: number;
    llm_savings_per_month: number;
    net_benefit_per_month: number;
    roi_pct: number | null;
    economically_feasible: boolean;
  } | null;
}

export interface FrontPoint {
  index: number;
  monthly_cost: number;
  quality: number;
  latency_ms: number;
}

export interface OptimizationRunSummary {
  id: number;
  team_id: number | null;
  method: "nsga2" | "ilp";
  created_at: string;
  points: number;
  selected_index: number | null;
  presets: Record<string, FrontPoint & { feasible: boolean }>;
  baselines: { name: string; monthly_cost: number; quality: number; latency_ms: number }[];
  metrics: Record<string, unknown>;
  notes: string[];
  traffic_source: string;
}

export interface OptimizationRun extends OptimizationRunSummary {
  front: FrontPoint[];
  inputs: Record<string, unknown>;
}

export interface PolicyRule {
  model_id: number;
  model: string;
  share: number;
}

export interface ProfilingSummary {
  task_types: string[];
  models: { model_id: number; model: string; display_name: string;
            tasks: Record<string, { quality: number | null; samples: number;
                                    latency_p50_ms: number | null;
                                    latency_p95_ms: number | null }> }[];
  quality_run_active: boolean;
}

export interface WhatIfScenarioResult {
  name: string;
  allocation: string;
  volume_growth_pct: number;
  price_multipliers: Record<string, number>;
  cost: number;
  monthly_projection: number;
  fits_budget: boolean;
  mean_quality: number | null;
  quality_coverage: number;
  mean_p95_latency_ms: number;
  cost_by_model: Record<string, number>;
  cost_by_task: Record<string, number>;
  savings_vs_actual?: number;
  savings_vs_baseline: number;
  savings_vs_baseline_pct: number | null;
}

export interface WhatIfResult {
  team_id: number;
  days: number;
  requests_replayed: number;
  cache_hits: number;
  monthly_budget?: number;
  scenarios: WhatIfScenarioResult[];
  note?: string;
}
