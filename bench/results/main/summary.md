# Benchmark: main

Costs are estimates at paid-tier list prices (app/pricing.py); actual free-tier spend is $0.

## Per mode

| Mode | OK / N | Judge pass | Answer p50 / p95 (ms) | E2E p50 (ms) | Serving tokens/q | Est. serving $/1k q | Est. total $/1k q | Fallbacks (saved) | Rate-limited calls |
|---|---|---|---|---|---|---|---|---|---|
| always_gemini | 22/60 | 94.1% | 1,525 / 12,129 | 1,868 | 207 | 0.4857 | 0.5364 | 0 (0) | 5 |
| always_groq | 49/60 | 92.6% | 1,275 / 2,604 | 1,475 | 1,156 | 0.3267 | 0.3770 | 0 (0) | 28 |
| router | 58/60 | 100.0% | 4,290 / 32,111 | 4,673 | 1,489 | 0.6992 | 0.7537 | 24 (22) | 41 |

**router:** success rate 96.7% with fallback vs 60.0% without it (fallback reasons: {'unavailable': 21, None: 2, 'rate_limit': 1}).

## Router vs baselines (paired on queries both answered)

| Comparison | Pairs | Serving cost | Serving tokens | Answer latency p50 | p95 | Judge pass (router vs base) |
|---|---|---|---|---|---|---|
| router_vs_always_groq | 49 | 105.8% | 15.9% | 154.7% | 1,164.5% | 100.0% vs 92.6% (7.4 pp) |
| router_vs_always_gemini | 22 | -22.2% | 201.0% | -12.9% | -29.6% | 100.0% vs 94.1% (5.9 pp) |

Negative % = the router is cheaper / faster than that baseline.

## Classifier accuracy (router mode, vs hand labels)

- task_type: 82.2% | complexity: 75.6% | both: 62.2% (n=45)

Task-type confusion (rows = label, columns = predicted):

| label \ pred | code | creative | factual | reasoning |
|---|---|---|---|---|
| code | 10 | 0 | 2 | 0 |
| creative | 0 | 11 | 0 | 0 |
| factual | 0 | 0 | 10 | 1 |
| reasoning | 0 | 0 | 5 | 6 |
