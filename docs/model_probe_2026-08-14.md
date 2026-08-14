# Model availability probe — 2026-08-14

Empirical check of which LLM models this repo's credentials can actually call,
run via `scripts/probe_models.py` (tiny 16-token test generations, temperature 0,
prompt "Say OK"; 26 total test calls). Repro:

```bash
source ~/general/bin/activate && python scripts/probe_models.py
# targeted: python scripts/probe_models.py --openai-models gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna
```

## How auth works on this machine

- **Vertex AI (Gemini):** `genai.Client(vertexai=True, project=$GOOGLE_PROJECT_ID, location="global")`
  authenticates via **gcloud user ADC** at `~/.config/gcloud/application_default_credentials.json`
  (`type=authorized_user`, `quota_project=gemini-project-3-492422`).
  `GOOGLE_APPLICATION_CREDENTIALS` is unset; the vendored `./google-cloud-sdk` is not
  involved in API calls (the google-genai SDK reads ADC directly via google-auth).
  `GOOGLE_PROJECT_ID` comes from `.env` at the repo root (python-dotenv).
- **OpenAI:** `OPENAI_API_KEY` from `.env`.
- **Anthropic:** `ANTHROPIC_API_KEY` is **not configured** (not in `.env` or environment).

## Vertex AI (Gemini) — project `gemini-project-3-492422`, location `global`

`client.models.list(config={"query_base": True})` works and returned 23 models.

| Model | Status | Latency | Note |
|---|---|---|---|
| `gemini-3.1-pro-preview` | **OK** | 1.92s | current pipeline default (path_config.yaml); still the newest Pro on Vertex |
| `gemini-3-flash-preview` | **OK** | 1.02s | gemini_helper.py fallback default |
| `gemini-3.5-flash` | **OK** | 0.75s | |
| `gemini-3.5-flash-lite` | **OK** | 0.58s | high-volume / subagent tier |
| `gemini-3.6-flash` | **OK** | 1.22s | released 2026-07-21; coding/agentic focus |
| `gemini-3.7-flash` | **OK** | 0.87s | works despite being undocumented in official release notes (SDK leak); treat as preview-grade |
| `gemini-3-pro-image` | **OK** | 3.70s | image-gen line, not relevant here |
| `gemini-3.1-flash-image` | **OK** | 0.99s | image-gen line, not relevant here |
| `gemini-3.1-pro` | FAIL | — | ClientError 404 NOT_FOUND (no GA alias; only `-preview` exists) |
| `gemini-3-pro` | FAIL | — | 404 NOT_FOUND |
| `gemini-3.2-pro-preview` | FAIL | — | 404 NOT_FOUND (does not exist) |
| `gemini-3.5-pro-preview` | FAIL | — | 404 NOT_FOUND (does not exist) |
| `gemini-3.1-flash-preview` | FAIL | — | 404 NOT_FOUND |
| `gemini-3.2-flash-preview` | FAIL | — | 404 NOT_FOUND |

Also in the listing (not probed): `gemini-2.5-pro`, `gemini-2.5-flash`,
`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, older 2.0/1.5 models,
TTS/live/embedding variants.

Key fact: **the Pro line has not moved past `gemini-3.1-pro-preview`**; all the
2026 progress on Vertex is in the Flash line (3.5 → 3.6 → 3.7).

## OpenAI

`client.models.list()` works: 126 models, 49 matching gpt-5*/o*/codex. Newest
families: `gpt-5.6-{sol,terra,luna}` (2026-06-23), `gpt-5.5[-pro]` (2026-04),
`gpt-5.4[-mini|-nano|-pro]` (2026-03), `gpt-5.3-codex` (2026-02).

| Model | Status | Latency | Note |
|---|---|---|---|
| `gpt-5.6-sol` | **OK** | 2.95s | newest flagship; **rejects `temperature`** |
| `gpt-5.6-terra` | **OK** | 0.93s | balanced tier; **rejects `temperature`** |
| `gpt-5.6-luna` | **OK** | 0.94s | cheap tier; **rejects `temperature`** |
| `gpt-5.5` | **OK** | 2.17s | **rejects `temperature`** |
| `gpt-5.4-2026-03-05` | **OK** | 1.27s | referenced in repo; accepts `temperature=0` |
| `gpt-5.2` | **OK** | 3.09s | referenced in repo; accepts `temperature=0` |
| `gpt-5.3-codex` | **OK** | 0.71s | newest codex line |
| `gpt-5.4-mini` | **OK** | 0.63s | cheap/fast |

**Pipeline gotcha:** `src/llmsat/utils/chatgpt_helper.py` always passes
`temperature` to `responses.create`. Models ≥ gpt-5.5 (incl. all gpt-5.6
variants) return 400 for it, so adopting them requires dropping/making
conditional the `temperature` argument. gpt-5.2/5.3/5.4 still accept it.

## Anthropic

`ANTHROPIC_API_KEY` not set — **not configured**, not probed.

## Recommendations by role

| Role | Primary (Vertex, matches pipeline default path) | OpenAI alternative | Rationale |
|---|---|---|---|
| (a) Algorithm/mutation generation | `gemini-3.1-pro-preview` (keep) | `gpt-5.6-sol` | No newer Pro exists on Vertex; 3.1 Pro is still Google's top reasoning model. Sol is the newest flagship overall if cross-provider spend is acceptable. |
| (b) C code generation | `gemini-3.6-flash` | `gpt-5.3-codex` | 3.6 Flash is Google's 2026 coding/agentic-focused model at ~2.7x lower input cost than 3.1 Pro; 5.3-codex is OpenAI's coding-tuned line (fastest probe, 0.71s). |
| (c) Cheap analysis (memory-bank causal analysis) | `gemini-3.5-flash-lite` | `gpt-5.6-luna` | Cheapest verified tiers on each side; flash-lite was the lowest-latency Vertex probe (0.58s) and is positioned for high-volume subagent work. |

`gemini-3.7-flash` responded fine and may supersede 3.6-flash for role (b), but
it is absent from official release notes/pricing pages as of today — treat as
experimental until documented.

## Pricing notes (web, 2026-08-14 — third-party/blog sourced, verify before budgeting)

Per 1M tokens (input / output):

- `gemini-3.1-pro`: $2 / $12
- `gemini-3.6-flash`: $0.75 / $3.75 intro price through 2026-12-31, then $1.50 / $7.50
- `gemini-3.5-flash-lite`: low-cost tier; exact price not confirmed in probe sources
- `gpt-5.6-sol`: $5 / $30 (cached input −90%; >272K-token requests carry 2x input / 1.5x output surcharge)
- `gpt-5.6-terra`: $2 / $12 (after 2026-07-30 −20% cut)
- `gpt-5.6-luna`: $0.20 / $1.20 (after 2026-07-30 −80% cut)

Sources: OpenAI/Google announcement coverage and pricing aggregators
(openai.com/index/gpt-5-6, blog.google gemini-3-1-pro, finout.io, eesel.ai,
chatbase.co, layer3labs.io); Vertex availability verified empirically above.
