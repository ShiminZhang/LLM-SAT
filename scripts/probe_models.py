#!/usr/bin/env python3
"""Probe which LLM models are actually available to this project's credentials.

Tests, with tiny (~10 token) generations:
  - Vertex AI Gemini models via the same client setup as
    src/llmsat/utils/gemini_helper.py (genai.Client(vertexai=True, ...))
  - OpenAI models via the Responses API (same call shape as
    src/llmsat/utils/chatgpt_helper.py, including the temperature param the
    pipeline always passes -- so we learn if a model would reject it)
  - Anthropic: presence check only (no key -> "not configured")

Usage:
    source ~/general/bin/activate && python scripts/probe_models.py

Total API spend is trivial: <25 calls, each capped at 16 output tokens.
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MAX_TOTAL_TEST_CALLS = 24
_calls_made = {"n": 0}

# Curated Vertex candidates.
# "known-good": names already referenced in this repo (path_config.yaml,
# gemini_helper.py). "guess": plausible newer names. "web": names surfaced by a
# 2026-08-14 web check of Gemini API / Vertex release notes coverage.
VERTEX_CANDIDATES = [
    ("gemini-3.1-pro-preview", "known-good (path_config.yaml default)"),
    ("gemini-3-flash-preview", "known-good (gemini_helper.py fallback)"),
    ("gemini-3.1-pro", "guess: GA alias"),
    ("gemini-3-pro", "guess"),
    ("gemini-3.2-pro-preview", "guess"),
    ("gemini-3.5-pro-preview", "guess"),
    ("gemini-3.1-flash-preview", "guess"),
    ("gemini-3.2-flash-preview", "guess"),
    ("gemini-3.5-flash", "web: implied by 3.6-flash pricing note"),
    ("gemini-3.5-flash-lite", "web: released 2026, high-volume subagent tier"),
    ("gemini-3.6-flash", "web: released 2026-07-21, coding/agentic focus"),
    ("gemini-3.7-flash", "web: leaked in SDK, unconfirmed"),
]

# OpenAI models referenced in this repo's code/config.
OPENAI_REFERENCED = ["gpt-5.2", "gpt-5.4-2026-03-05"]

RESULTS = []  # rows: dict(provider, model, status, latency_s, note)


def record(provider, model, status, latency=None, note=""):
    RESULTS.append(
        {
            "provider": provider,
            "model": model,
            "status": status,
            "latency_s": latency,
            "note": note,
        }
    )


def _budget_ok():
    return _calls_made["n"] < MAX_TOTAL_TEST_CALLS


def _spend_call():
    _calls_made["n"] += 1


def _truncate(msg, n=160):
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= n else msg[: n - 3] + "..."


def load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def print_auth_diagnostics():
    print("=== Auth diagnostics ===")
    print(f"GOOGLE_PROJECT_ID set: {bool(os.environ.get('GOOGLE_PROJECT_ID'))}")
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {gac or '(unset)'}")
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    if adc.exists():
        try:
            import json

            d = json.loads(adc.read_text())
            print(
                f"gcloud ADC: {adc} (type={d.get('type')}, "
                f"quota_project={d.get('quota_project_id')})"
            )
        except Exception as e:
            print(f"gcloud ADC: {adc} (unreadable: {e})")
    else:
        print("gcloud ADC: not found")
    print(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    print()


# ---------------------------------------------------------------- Vertex ----


def vertex_list_models(client):
    """Try to enumerate models. Returns list of short names (may be empty)."""
    names = []
    for cfg in ({"query_base": True}, None):
        try:
            pager = client.models.list(config=cfg) if cfg else client.models.list()
            for i, m in enumerate(pager):
                if i >= 500:
                    break
                name = getattr(m, "name", "") or ""
                names.append(name.rsplit("/", 1)[-1])
            if names:
                print(f"[vertex] models.list(config={cfg}) returned {len(names)} models")
                break
        except Exception as e:
            print(f"[vertex] models.list(config={cfg}) failed: "
                  f"{type(e).__name__}: {_truncate(e)}")
    return sorted(set(n for n in names if n))


def vertex_test(client, model):
    if not _budget_ok():
        record("vertex", model, "SKIP", None, "call budget exhausted")
        return
    _spend_call()
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Say OK",
            config={"temperature": 0, "max_output_tokens": 16},
        )
        dt = time.perf_counter() - t0
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            note = f"reply={text!r}"
        else:
            note = "ok, empty text (16-token cap consumed by thinking)"
        record("vertex", model, "OK", dt, note)
    except Exception as e:
        dt = time.perf_counter() - t0
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        status = getattr(e, "status", None)
        record(
            "vertex",
            model,
            "FAIL",
            dt,
            f"{type(e).__name__} code={code} status={status} :: {_truncate(e)}",
        )


def probe_vertex():
    print("=== Vertex AI (Gemini) ===")
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        record("vertex", "(client)", "FAIL", None, "GOOGLE_PROJECT_ID not set")
        return
    try:
        from google import genai
    except ImportError as e:
        record("vertex", "(client)", "FAIL", None, f"google-genai not installed: {e}")
        return
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id  # mirror gemini_helper.py
    try:
        client = genai.Client(vertexai=True, project=project_id, location="global")
    except Exception as e:
        record("vertex", "(client)", "FAIL", None,
               f"client init failed: {type(e).__name__}: {_truncate(e)}")
        return

    listed = vertex_list_models(client)
    gemini_listed = [n for n in listed if "gemini" in n]
    if gemini_listed:
        print(f"[vertex] gemini models in listing: {gemini_listed}")

    probe_set = [name for name, _src in VERTEX_CANDIDATES]
    # Add up to 2 gemini-3* names discovered by the listing but not curated.
    extras = [n for n in gemini_listed if n.startswith("gemini-3") and n not in probe_set]
    for n in extras[:2]:
        probe_set.append(n)
        print(f"[vertex] adding listed model to probe set: {n}")

    for model in probe_set:
        vertex_test(client, model)
        print(f"[vertex] tested {model}: {RESULTS[-1]['status']}")
    print()


# ---------------------------------------------------------------- OpenAI ----


def openai_test(client, model):
    """Test with the pipeline's call shape (Responses API + temperature).

    If the model rejects the temperature param (gpt-5-era reasoning models do),
    retry without it so we still learn availability -- and flag that
    chatgpt_helper.py's unconditional temperature would 400 on this model.
    """
    if not _budget_ok():
        record("openai", model, "SKIP", None, "call budget exhausted")
        return
    _spend_call()
    t0 = time.perf_counter()
    temp_rejected = False
    try:
        try:
            resp = client.responses.create(
                model=model, input="Say OK", temperature=0, max_output_tokens=16
            )
        except Exception as e:
            if "temperature" in str(e).lower() and _budget_ok():
                temp_rejected = True
                _spend_call()
                resp = client.responses.create(
                    model=model, input="Say OK", max_output_tokens=16
                )
            else:
                raise
        dt = time.perf_counter() - t0
        text = (getattr(resp, "output_text", "") or "").strip()
        status = getattr(resp, "status", "")
        bits = []
        if text:
            bits.append(f"reply={text!r}")
        else:
            bits.append(f"resp.status={status} (16-token cap eaten by reasoning)")
        if temp_rejected:
            bits.append("REJECTS temperature param (chatgpt_helper.py would 400)")
        record("openai", model, "OK", dt, "; ".join(bits))
    except Exception as e:
        dt = time.perf_counter() - t0
        code = getattr(e, "status_code", None)
        record("openai", model, "FAIL", dt,
               f"{type(e).__name__} http={code} :: {_truncate(e)}")


def probe_openai(explicit_models=None):
    """explicit_models: if given, test exactly these ids (skip auto-selection)."""
    print("=== OpenAI ===")
    if not os.environ.get("OPENAI_API_KEY"):
        record("openai", "(client)", "FAIL", None, "OPENAI_API_KEY not set")
        return
    try:
        from openai import OpenAI
    except ImportError as e:
        record("openai", "(client)", "FAIL", None, f"openai not installed: {e}")
        return
    client = OpenAI()

    if explicit_models:
        print(f"[openai] testing explicit models: {explicit_models}")
        for model in explicit_models:
            openai_test(client, model)
            print(f"[openai] tested {model}: {RESULTS[-1]['status']}")
        print()
        return

    try:
        models = list(client.models.list())
    except Exception as e:
        record("openai", "(models.list)", "FAIL", None,
               f"{type(e).__name__}: {_truncate(e)}")
        return

    def interesting(mid):
        return mid.startswith(("gpt-5", "o3", "o4", "o5")) or "codex" in mid

    inter = sorted(
        (m for m in models if interesting(m.id)),
        key=lambda m: getattr(m, "created", 0),
        reverse=True,
    )
    print(f"[openai] models.list(): {len(models)} total; "
          f"{len(inter)} gpt-5*/o*/codex models (newest first):")
    for m in inter:
        created = getattr(m, "created", 0)
        day = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
        print(f"    {m.id}  (created {day})")

    ids = [m.id for m in inter]

    def newest(pred):
        for mid in ids:  # already newest-first
            if pred(mid):
                return mid
        return None

    import re

    # Flagship families may use bare ids (gpt-5.5) or named variants
    # (gpt-5.6-sol / -terra / -luna). Prefer sol > terra > luna > bare.
    variant_rank = {"sol": 0, "terra": 1, "luna": 2, None: 3}

    def flagship_key(mid):
        m = re.fullmatch(r"gpt-5(\.\d+)?(?:-(sol|terra|luna))?", mid)
        if not m:
            return None
        return variant_rank[m.group(2)]

    flagships = sorted(
        (i for i in ids if flagship_key(i) is not None),
        key=lambda i: (-next(m.created for m in inter if m.id == i), flagship_key(i)),
    )
    flagship = flagships[0] if flagships else None
    codex = newest(lambda i: "codex" in i)
    mini = newest(lambda i: re.fullmatch(r"gpt-5(\.\d+)?-mini", i))

    candidates = []
    for mid in OPENAI_REFERENCED + [flagship, codex, mini]:
        if mid and mid not in candidates:
            candidates.append(mid)
    candidates = candidates[:6]
    print(f"[openai] test candidates: {candidates}")

    for model in candidates:
        openai_test(client, model)
        print(f"[openai] tested {model}: {RESULTS[-1]['status']}")
    print()


# ------------------------------------------------------------- Anthropic ----


def probe_anthropic():
    print("=== Anthropic ===")
    if os.environ.get("ANTHROPIC_API_KEY"):
        record("anthropic", "(key)", "OK", None,
               "ANTHROPIC_API_KEY is set (no test call made)")
    else:
        record("anthropic", "(key)", "NOT CONFIGURED", None,
               "ANTHROPIC_API_KEY not set in .env or environment")
    print()


# --------------------------------------------------------------- Summary ----


def print_table():
    print("=== Summary ===")
    headers = ("PROVIDER", "MODEL", "STATUS", "LATENCY", "NOTE")
    rows = []
    for r in RESULTS:
        lat = f"{r['latency_s']:.2f}s" if r["latency_s"] is not None else "-"
        rows.append((r["provider"], r["model"], r["status"], lat, r["note"]))
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(4)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths) + "  {}"
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths), "----"))
    for row in rows:
        print(fmt.format(*row))
    print(f"\nTotal test generation calls made: {_calls_made['n']} "
          f"(budget {MAX_TOTAL_TEST_CALLS})")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--openai-models",
        help="Comma-separated OpenAI model ids to test exclusively "
        "(skips the Vertex probe and OpenAI auto-selection).",
    )
    args = ap.parse_args()

    load_env()
    print_auth_diagnostics()
    if args.openai_models:
        probe_openai([m.strip() for m in args.openai_models.split(",") if m.strip()])
    else:
        probe_vertex()
        probe_openai()
    probe_anthropic()
    print_table()
    ok = any(r["status"] == "OK" for r in RESULTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
