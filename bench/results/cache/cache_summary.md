# Cache benchmark: cache

- Workload: 80 requests over 14 unique queries (Zipf popularity, some re-cased/re-spaced repeats)
- **Hit rate: 77.2%** (max possible 82.5%; 5 repeats missed because the earlier answer did not pass evaluation (judge failed or errored) and so was not cached)
- Latency p50: hit 5 ms vs miss 3,259 ms (653x faster); p95: hit 8 ms vs miss 21,013 ms
- **LLM calls avoided: 75.3%** (66 made vs ~268 without cache; each hit skips classifier + generation + judge)
- Tokens avoided: ~138,006 | est. cost avoided at list prices: $0.0434
