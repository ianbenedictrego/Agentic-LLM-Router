# Benchmark: main-v2

Costs are estimates at paid-tier list prices (app/pricing.py); actual free-tier spend is $0.

## Per mode

| Mode | OK / N | Judge pass | Answer p50 / p95 (ms) | E2E p50 (ms) | Serving tokens/q | Est. serving $/1k q | Est. total $/1k q | Fallbacks (saved) | Rate-limited calls |
|---|---|---|---|---|---|---|---|---|---|
| always_gemini | 11/30 | 100.0% | 3,234 / 34,950 | 3,479 | 156 | 0.3601 | 0.4140 | 0 (0) | 1 |
| always_groq | 28/30 | 94.1% | 1,305 / 2,553 | 1,653 | 1,166 | 0.3293 | 0.3899 | 0 (0) | 11 |
| router | 30/30 | 100.0% | 3,308 / 18,827 | 3,805 | 1,781 | 0.7843 | 0.8790 | 17 (17) | 7 |

**router:** success rate 100.0% with fallback vs 43.3% without it (fallback reasons: {'unavailable': 17}).

## Router vs baselines (paired on queries both answered)

| Comparison | Pairs | Serving cost | Serving tokens | Answer latency p50 | p95 | Judge pass (router vs base) |
|---|---|---|---|---|---|---|
| router_vs_always_groq | 28 | 81.5% | 46.6% | 137.1% | 608.7% | 100.0% vs 94.1% (5.9 pp) |
| router_vs_always_gemini | 11 | -38.9% | 399.2% | -64.5% | -67.5% | 100.0% vs 100.0% (0.0 pp) |

Negative % = the router is cheaper / faster than that baseline.

## Classifier accuracy (router mode, vs hand labels)

- task_type: 83.3% | complexity: 86.7% | both: 73.3% (n=30)

Task-type confusion (rows = label, columns = predicted):

| label \ pred | code | conversational | creative | factual | reasoning |
|---|---|---|---|---|---|
| code | 6 | 0 | 0 | 0 | 0 |
| conversational | 0 | 3 | 0 | 0 | 3 |
| creative | 0 | 0 | 6 | 0 | 0 |
| factual | 0 | 0 | 0 | 6 | 0 |
| reasoning | 0 | 0 | 0 | 2 | 4 |
