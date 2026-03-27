# cli/__main__.py
"""Entry point for `deer` CLI or `python -m cli`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="deer",
        description="CLI agent system built on DeerFlow harness",
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=None,
        help="Agent to start with (default: from config.yaml)",
    )
    args = parser.parse_args()

    from cli.shell import DeerShell
    shell = DeerShell(agent_name=args.agent)
    shell.run()


if __name__ == "__main__":
    main()
