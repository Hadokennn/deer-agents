"""Eval runner CLI.

Usage:
    python scripts/run_eval.py tool                        # Layer 1 mock (regression)
    python scripts/run_eval.py tool --live                 # Layer 1 live (real MCP)
    python scripts/run_eval.py tool --case happy_path      # Single case
    python scripts/run_eval.py tool --tag cold-start       # Filter by tag
    python scripts/run_eval.py e2e                         # Layer 3 (full pipeline)
    python scripts/run_eval.py all                         # All layers
    python scripts/run_eval.py tool --save                 # Save JSON report
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.framework.report import print_report, save_report
from evals.framework.runner import run_eval


def main():
    parser = argparse.ArgumentParser(
        prog="run_eval", description="Run deer-agents evaluations"
    )
    parser.add_argument(
        "layer",
        choices=["tool", "e2e", "all"],
        help="Which evaluation layer to run",
    )
    parser.add_argument(
        "--agent", default="oncall", help="Agent to evaluate (default: oncall)"
    )
    parser.add_argument("--case", dest="case_id", help="Run specific case by id")
    parser.add_argument("--tag", help="Filter cases by tag")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real MCP/agent instead of mocks (Layer 1: real API; Layer 3: always live)",
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="JSON output only (no console)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to .deer-flow/eval-reports/",
    )
    args = parser.parse_args()

    layers = ["tool", "e2e"] if args.layer == "all" else [args.layer]
    case_ids = [args.case_id] if args.case_id else None
    tags = [args.tag] if args.tag else None

    for layer in layers:
        try:
            report = run_eval(
                layer,
                agent=args.agent,
                case_ids=case_ids,
                tags=tags,
                live=args.live,
            )
        except FileNotFoundError as e:
            if args.layer == "all":
                continue
            print(f"  Error: {e}")
            sys.exit(1)

        if not args.json_only:
            print_report(report)

        if args.save or args.json_only:
            path = save_report(report)
            print(f"  Report saved: {path}")


if __name__ == "__main__":
    main()
