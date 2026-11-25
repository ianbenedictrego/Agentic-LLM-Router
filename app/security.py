"""Input screening rules.

Kept free of LLM / Redis imports so the security benchmark
(bench/run_security_benchmark.py) can test it offline without API keys.
"""

import re
from typing import Optional

MAX_QUERY_LENGTH = 1000

SUSPICIOUS_PATTERNS = [
    r"(ignore|forget|delete)\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?system\s+prompt",
    r"reveal\s+(your\s+)?instructions",
    r"show\s+(me\s+)?your\s+instructions",
    r"bypass\s+(your\s+)?security",
    r"bypass\s+(the\s+)?safety",
    r"give\s+me\s+(the\s+)?api\s+key",
    r"reveal\s+(the\s+)?api\s+key",
    r"show\s+(me\s+)?(the\s+)?api\s+key",
]


def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def screen_query(query: str) -> Optional[tuple[str, bool]]:
    """Return (reason, is_risk) if the query must be blocked, else None."""
    normalized = normalize_query(query)
    if not normalized:
        return "Query cannot be empty", False
    if len(normalized) > MAX_QUERY_LENGTH:
        return "Query exceeds maximum allowed length", False
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, normalized):
            return "Query contains a suspicious pattern", True
    return None
