"""Check how far the LLM-as-a-Judge can be trusted, against your own labels.

1. Export a sample of judged answers to a CSV:
       python bench/judge_agreement.py export bench/results/<run-name> --n 50
2. Open bench/results/<run-name>/judge_review.csv and fill human_label with
   pass or fail for each row (judge the answer yourself; try not to look at
   the judge column first).
3. Score it:
       python bench/judge_agreement.py score bench/results/<run-name>

Reports raw agreement and Cohen's kappa (agreement corrected for chance).
"""

import argparse
import csv
import random
from pathlib import Path

from common import load_jsonl


def export(run_dir: Path, n: int, seed: int):
    rows = []
    for f in sorted(run_dir.glob("*.jsonl")):
        for r in load_jsonl(f):
            ev = str(r.get("evaluation") or "")
            if r.get("ok") and ev.startswith(("pass", "fail")) and r.get("response"):
                rows.append(r)
    if not rows:
        raise SystemExit("No judged answers with responses found in this run.")
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    out = run_dir / "judge_review.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "mode", "query", "response", "human_label", "judge_evaluation"])
        for r in sample:
            w.writerow([r["id"], r["mode"], r["query"], r["response"], "", r["evaluation"]])
    print(f"Wrote {len(sample)} rows to {out}. Fill human_label with pass/fail, then run 'score'.")


def score(run_dir: Path):
    path = run_dir / "judge_review.csv"
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip().lower() in ("pass", "fail")]
    if not rows:
        raise SystemExit(f"No rows in {path} have human_label set to pass/fail.")
    pairs = [(r["human_label"].strip().lower(), r["judge_evaluation"].split(":")[0]) for r in rows]
    n = len(pairs)
    agree = sum(h == j for h, j in pairs)
    po = agree / n
    ph = sum(h == "pass" for h, _ in pairs) / n
    pj = sum(j == "pass" for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    cm = {(h, j): sum(1 for p in pairs if p == (h, j)) for h in ("pass", "fail") for j in ("pass", "fail")}
    print(f"Judge vs human on {n} answers: agreement {po:.0%}, Cohen's kappa {kappa:.2f}")
    print(f"  human pass / judge pass: {cm[('pass', 'pass')]}   human pass / judge fail: {cm[('pass', 'fail')]}")
    print(f"  human fail / judge pass: {cm[('fail', 'pass')]}   human fail / judge fail: {cm[('fail', 'fail')]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("run_dir", type=Path)
    args = p.parse_args()
    if args.cmd == "export":
        export(args.run_dir, args.n, args.seed)
    else:
        score(args.run_dir)


if __name__ == "__main__":
    main()
