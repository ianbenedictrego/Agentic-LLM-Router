"""Offline test of the input filter (app/security.py). No API keys or LLM calls.

    python bench/run_security_benchmark.py

Reports detection rate on attack prompts and false-positive rate on benign
look-alikes, and lists every miss so the patterns can be improved honestly.
"""

import sys
from pathlib import Path

from common import DATA_DIR, RESULTS_DIR, load_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from security import screen_query  # noqa: E402


def main():
    cases = load_jsonl(DATA_DIR / "security_cases.jsonl")
    tp = fn = fp = tn = 0
    missed, false_alarms = [], []
    for c in cases:
        blocked = screen_query(c["query"]) is not None
        if c["label"] == "attack":
            if blocked:
                tp += 1
            else:
                fn += 1
                missed.append(c)
        else:
            if blocked:
                fp += 1
                false_alarms.append(c)
            else:
                tn += 1

    attacks, benign = tp + fn, fp + tn
    lines = [
        "# Security filter benchmark", "",
        f"- Attack prompts blocked (recall): **{tp}/{attacks} = {tp / attacks:.0%}**",
        f"- Benign prompts wrongly blocked (false-positive rate): **{fp}/{benign} = {fp / benign:.0%}**",
        f"- Precision: {tp / (tp + fp):.0%}" if tp + fp else "- Precision: n/a",
        "", "## Missed attacks", "",
        *[f"- `{c['id']}` {c['query']}" for c in missed], "",
        "## False alarms", "",
        *([f"- `{c['id']}` {c['query']}" for c in false_alarms] or ["- none"]),
        "",
    ]
    md = "\n".join(lines)
    out = RESULTS_DIR / "security_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
