#!/usr/bin/env python3
"""CLI for the permanent mutation gate.

Usage:
    python tests/mutation/run.py [--only NAME] [--baseline]

Prints per-mutation KILLED/SURVIVED and a summary.  Exits non-zero when any
mutation survives, unless --baseline is given (report only).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import catalog  # noqa: E402
import harness  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mutation-testing gate")
    parser.add_argument("--only", metavar="NAME",
                        help="run a single catalog mutation by name")
    parser.add_argument("--baseline", action="store_true",
                        help="report only; never fail on survivors")
    args = parser.parse_args(argv)

    if args.only:
        entry = catalog.by_name(args.only)
        if entry is None:
            print("Unknown mutation: %s" % args.only, file=sys.stderr)
            return 2
        selected = [entry]
    else:
        selected = list(catalog.MUTATIONS)

    # Guard: the un-mutated suite must be green before any result is trusted.
    base = harness.run_suite(label="__sanity__")
    if not base["all_pass"]:
        print("Unmutated suite is NOT green -- refusing to run mutations.",
              file=sys.stderr)
        for line in base["failures"]:
            print("  ", line, file=sys.stderr)
        return 2
    print("Baseline suite: %d passed / 0 failed / %d skipped across %d files\n" %
          (base["passed"], base["skipped"], len(base["files"])))

    killed, survived = [], []
    started = time.time()
    for entry in selected:
        result = harness.run_suite(entry, label=entry["name"])
        if result["killed"]:
            killed.append(entry["name"])
            print("%-32s KILLED   (%d failing: %s)" %
                  (entry["name"], result["failed"], ", ".join(result["failures"][:2])))
        else:
            survived.append(entry["name"])
            print("%-32s SURVIVED" % entry["name"])
        sys.stdout.flush()

    print("\nSummary: %d killed / %d survived / %d total  (%.1fs)" %
          (len(killed), len(survived), len(selected), time.time() - started))
    print("Survived: %s" % (", ".join(survived) if survived else "(none)"))
    return 0 if args.baseline or not survived else 1


if __name__ == "__main__":
    sys.exit(main())
