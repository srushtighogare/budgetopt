# Security and privacy

How each requirement in the project report's *Safety and Security*, *Cybersecurity* and
*Ethical Considerations* sections is implemented, and which automated test covers it.
Test names refer to `backend/tests/`.

## Authentication

| Requirement | Implementation | Evidence |
|---|---|---|
| Passwords stored safely | argon2id hashes (`app/core/security.py`); minimum 10 characters, letters and digits | `test_password_hash_roundtrip`, `test_weak_password_rejected` |
| Session tokens | Short-lived JWT access tokens (30 min) and refresh tokens (7 days), HS256 with the algorithm pinned on decode; a refresh token is rejected as an access token | `test_token_type_is_enforced`, `test_tampered_token_rejected`, `test_refresh_token_flow` |
| No account enumeration | Unknown email and wrong password return the same error, and both run an argon2 verification so response time does not differ | `test_login_wrong_password_and_unknown_email_look_identical` |
| Brute-force protection | Failed logins are counted per account (5) and per IP (30) per 15 minutes in Redis; beyond that the endpoint returns 429 with `Retry-After` | `test_login_throttled_after_repeated_failures`, `test_successful_login_resets_failure_count` |
| Deactivated users | `is_active` is checked on every request and on token refresh | `get_current_user` and `/auth/refresh` |
| Application access | Applications call the gateway with team API keys: random 256-bit keys shown once, stored only as SHA-256, revocable | `test_api_key_shown_once`, `test_revoked_key_rejected` |

Every API operation except login, token refresh and `/health` requires a user token or an
API key; this was verified by enumerating the OpenAPI schema (44 of 47 operations require
authentication).

## Authorization (role-based access control)

| Role | Can see and do |
|---|---|
| Member | Own team's budget, forecast and alerts; only their own requests; the playground for their team |
| Team manager | Everything belonging to their team, team API keys, adding members, what-if analysis |
| Org admin | All teams in the organization, budgets, models, prices, provider keys, optimization and policies |

Checks are enforced in FastAPI dependencies (`app/core/deps.py`). Other organizations' teams
return 404 rather than 403, so their IDs cannot be probed. **Denied access attempts are
recorded in `audit_events`**, as are all changes to budgets, prices, keys and policies.

The report's own example, an employee trying to view another team's AI spend, is a test:
`test_member_cannot_view_other_teams_spend` (403 and an audit event), and again for the
forecast in `test_forecast_and_alert_endpoints`. Further cases:
`test_manager_can_only_add_members_to_own_team`, `test_other_organization_team_is_hidden`,
`test_playground_and_request_log_scoping`, `test_ilp_run_and_permissions`,
`test_whatif_validation_and_permissions`, `test_savings_report`.

## Secrets

- **Provider API keys** are encrypted at rest with Fernet (AES-128-CBC + HMAC); the API can set
  or remove them but never returns them (`test_provider_key_is_write_only`).
- **Configuration secrets** come from environment variables; `.env` is git-ignored and only
  `.env.example` is committed. The git history was scanned for provider key patterns: none.
- **Production guard**: outside `development`/`test` the API refuses to start with the
  development JWT secret, a missing encryption key, the default admin password or localhost
  CORS origins (`test_production_refuses_development_settings`).

## Input validation and abuse

- All request bodies are validated by Pydantic models (types, ranges, lengths); the gateway
  caps prompt size (`MAX_PROMPT_CHARS`) and message count, and rejects unknown models.
- SQL is built only through SQLAlchemy with bound parameters.
- Gateway rate limiting per API key / user (Redis, fixed window): `test_rate_limit_per_api_key`.
- Budget enforcement blocks over-budget teams before any LLM call:
  `test_over_budget_team_blocked_without_llm_call`.
- **Monitoring for abnormal usage**: each caller's last hour is compared with its own usage at
  the same time of day (robust z-score); spikes raise alerts:
  `test_spike_is_detected_but_daily_pattern_is_not`.

## Privacy of prompts and usage data

- By default only a **SHA-256 hash** of each prompt is stored. Teams can opt in to storing prompt
  text, which is then **PII-redacted** first (emails, phone numbers, card numbers, Aadhaar,
  PAN, API keys): `test_opted_in_prompts_are_stored_redacted`.
- The semantic cache keeps prompts **encrypted**, only until the cache entry expires (24 h), and
  is disabled for teams with the *confidential* privacy tier.
- Models are only used for teams whose privacy tier they are cleared for:
  `test_privacy_tier_excludes_all_public_models`.
- Caches are per team, so one team never receives another team's cached answer:
  `test_cache_bypass_rules`.

## Responsible model selection

The report requires that a model is never chosen only because it is cheaper. Every model has a
measured quality per task type, and a model below the quality floor for a request's complexity
band (6 / 7 / 8 out of 10) is never given that traffic by the optimizer or the router
(`test_quality_floor_per_band_controls_eligibility`, `test_cheapest_respects_quality_floor`).
Every routing decision is logged with a plain-language explanation of what was chosen, what was
rejected and why, and what it saved.

## Transport and deployment

- The dashboard is served by nginx with `Content-Security-Policy` (scripts and connections only
  from its own origin), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff` and
  `Referrer-Policy: same-origin`.
- Containers run as non-root users. PostgreSQL and Redis are bound to `127.0.0.1` only.
- HTTPS is expected to be terminated by the hosting platform or a reverse proxy in front of
  nginx; the API and dashboard need no changes for it.

## Known limitations

- Refresh tokens are stateless: revoking one before it expires (7 days) needs a denylist that
  is not implemented. Deactivating the user does block refresh.
- The dashboard keeps tokens in `localStorage`; the CSP and the absence of any raw-HTML
  rendering in the frontend are the mitigation against script injection.
- The exact-match cache stores answers in Redis unencrypted for 24 hours; Redis is not exposed
  outside the host.
- The development `docker-compose.yml` uses a fixed database password; change it for any shared
  deployment.
