"""Shared helpers for the benchmark scripts (stdlib only)."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 180):
    """Return (status_code, parsed_json_or_text). Never raises on HTTP errors."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, str(e)


def route(base_url: str, payload: dict, max_attempts: int, backoff_s: float):
    """POST /route, waiting and retrying only when the failure was a rate limit.

    Free tiers return 429s under load; retrying them keeps rate limits from
    being counted as quality or routing failures. The attempt count is
    recorded so it stays visible in the results.
    """
    attempt = 0
    while True:
        attempt += 1
        start = time.perf_counter()
        status, body = http_json("POST", f"{base_url}/route", payload)
        client_ms = round((time.perf_counter() - start) * 1000, 2)
        detail = body.get("detail") if isinstance(body, dict) else None
        rate_limited = isinstance(detail, dict) and detail.get("rate_limited")
        if status == 503 and rate_limited and attempt < max_attempts:
            wait = backoff_s * attempt
            print(f"    rate limited, waiting {wait:.0f}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)
            continue
        return status, body, attempt, client_ms


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 2)


def mean(values: list[float]) -> float | None:
    # No rounding here: per-query costs are ~1e-5 USD and would collapse to 0.
    return sum(values) / len(values) if values else None


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return round((new - old) / old * 100, 1)


def fmt(v, suffix: str = "", digits: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{suffix}"
    return f"{v}{suffix}"
