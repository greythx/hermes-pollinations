"""Isolated mutation harness for the stdlib test suite.

The real ``__init__.py`` and the live caches are NEVER touched.  For every run
the harness creates a fresh directory under ``tests/_scratch_mutation/``
(gitignored), writes a COPY of the plugin source there -- optionally with one
catalog mutation applied -- copies the full suite beside it, and runs each test
as ``python tests/test_NAME.py``.  The scratch directory is removed afterwards.

A mutation is KILLED when at least one test FAILs/ERRORs, SURVIVED when every
test passes (SKIP is not a kill).  Baseline runs use no mutation.  A file whose
output has no parseable test status is a harness ABORT (recorded in ``aborts``,
never in ``failures``): it must fail the gate instead of scoring as a kill.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO: Path = Path(__file__).resolve().parents[2]
TESTS_DIR: Path = REPO / "tests"
SOURCE: Path = REPO / "__init__.py"
SCRATCH: Path = TESTS_DIR / "_scratch_mutation"

_STATUS = re.compile(r"^(PASS|FAIL|ERROR|SKIP)\s+(test_\w+)")
_KILL = {"FAIL", "ERROR"}


def test_files() -> List[Path]:
    """The full suite, in deterministic order."""
    return sorted(TESTS_DIR.glob("test_*.py"))


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def apply_mutation(src: str, mutation: Dict[str, Any]) -> str:
    """Apply one catalog mutation to SOURCE TEXT only (never to the real file)."""
    old = mutation["old_string"]
    count = src.count(old)
    if count != 1:
        raise ValueError("%s: anchor occurs %d times (expected 1)" %
                         (mutation.get("name", "?"), count))
    return src.replace(old, mutation["new_string"], 1)


def parse_output(text: str) -> Dict[str, str]:
    """Map test name -> PASS/FAIL/ERROR/SKIP from a stdlib runner's stdout."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        match = _STATUS.match(line)
        if match:
            out[match.group(2)] = match.group(1)
    return out


def clean() -> None:
    """Remove all mutation scratch state (safe: path is under tests/_scratch_*)."""
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)


def run_suite(mutation: Optional[Dict[str, Any]] = None, *, label: str = "run",
              keep: bool = False, timeout: int = 180,
              files: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Run the FULL suite against a mutated COPY of __init__.py.

    Returns a dict with per-test statuses, pass/fail counts, kill list and an
    ``all_pass`` flag.  ``keep=True`` leaves the scratch dir for debugging.
    """
    work = SCRATCH / label
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "tests").mkdir(parents=True)

    src = source_text()
    if mutation is not None:
        src = apply_mutation(src, mutation)
    (work / "__init__.py").write_text(src, encoding="utf-8")

    tests = list(files) if files is not None else test_files()
    for test in tests:
        shutil.copy2(test, work / "tests" / test.name)

    env = dict(os.environ)
    per_file: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    aborts: List[str] = []
    passed = failed = skipped = 0

    for test in tests:
        try:
            cp = subprocess.run(
                [sys.executable, str(work / "tests" / test.name)],
                cwd=str(work), env=env, capture_output=True, text=True,
                timeout=timeout)
            output = cp.stdout + ("\n" + cp.stderr if cp.stderr else "")
            returncode = cp.returncode
        except subprocess.TimeoutExpired as exc:
            output = str(exc)
            returncode = 124

        statuses = parse_output(output)
        if not statuses:
            # No parseable test status: the runner itself failed before any test
            # reported. This is a harness abort, NOT a mutation kill -- scoring
            # it as a kill would let a SyntaxError inflate the gate.
            aborts.append("%s::runner-crash [returncode=%d]" %
                          (test.name, returncode))
            per_file[test.name] = {
                "returncode": returncode,
                "statuses": {},
                "abort": True,
            }
            continue
        for name, status in statuses.items():
            if status in _KILL:
                failed += 1
                failures.append("%s::%s [%s]" % (test.name, name, status))
            elif status == "SKIP":
                skipped += 1
            else:
                passed += 1
        if returncode != 0 and not any(s in _KILL for s in statuses.values()):
            # A nonzero exit with no parsed kill is still a runner failure;
            # recording it keeps killed/all_pass honest without double-counting
            # a kill that was already parsed from this file's output.
            statuses["runner-error"] = "ERROR"
            failed += 1
            failures.append("%s::runner-error [ERROR]" % test.name)
        per_file[test.name] = {
            "returncode": returncode,
            "statuses": statuses,
        }

    result = {
        "label": label,
        "mutation": (mutation or {}).get("name", "baseline"),
        "files": [t.name for t in tests],
        "per_file": per_file,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
        "aborted": bool(aborts),
        "aborts": aborts,
        "killed": bool(failures),
        "all_pass": not failures and not aborts,
    }
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    result = run_suite(label="harness_selftest")
    print("%s: %d passed, %d failed, %d skipped" %
          (result["mutation"], result["passed"], result["failed"],
           result["skipped"]))
    for line in result["failures"]:
        print("  KILLED:", line)
