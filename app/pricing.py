"""List prices used to ESTIMATE what free-tier usage would cost on a paid tier.

The project runs on free tiers, so real spend is $0. These numbers turn token
counts into a comparable "cost-equivalent" figure. Update them when the
providers change their prices.

USD per 1M tokens (input, output). Checked 2026-11-24:
- Groq openai/gpt-oss-20b:   https://groq.com/pricing
- Gemini 3.5 Flash-Lite:     https://aistudio.google.com/docs/pricing
  (paid tier; output price includes thinking tokens)
"""

PRICES_PER_MTOK = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
