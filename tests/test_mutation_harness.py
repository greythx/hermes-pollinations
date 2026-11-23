"""Stdlib-only regression tests for the mutation harness runner-error path.

Run directly with ``python tests/test_mutation_harness.py``.  The harness is
loaded from its real location even when this file is copied into a mutation
scratch directory by ``tests/mutation/harness.py``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _load_harness():
    """Load harness.py from the repo (works from the repo and from scratch)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "mutation" / "harness.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "mutation_harness_under_test", candidate)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise unittest.SkipTest("tests/mutation/harness.py not found")


HARNESS = _load_harness()


class _FakeTestsDir:
    """Provide one tiny test file for ``run_suite(files=...)``."""

    def __init__(self):
        self._tmp = None
        self.files = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mutation-harness-unit-")
        fake = Path(self._tmp.name) / "test_fake_runner.py"
        fake.write_text('print("PASS test_fake_runner")\n', encoding="utf-8")
        self.files = [fake]
        return self.files

    def __exit__(self, *exc):
        if self._tmp is not None:
            self._tmp.cleanup()
        return False


def _run_with(stdout, returncode, label):
    """Call the real run_suite with a fixed CompletedProcess from subprocess."""
    with _FakeTestsDir() as files:
        fake = subprocess.CompletedProcess(
            args=["python", "test_fake_runner.py"],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )
        original = HARNESS.subprocess.run
        HARNESS.subprocess.run = lambda *args, **kwargs: fake
        try:
            return HARNESS.run_suite(files=files, label=label)
        finally:
            HARNESS.subprocess.run = original


def test_nonzero_exit_with_only_pass_status_is_not_all_pass():
    """RED: a crashed runner printing only PASS must not count as all_pass."""
    result = _run_with(
        "PASS test_fake_runner\n1 passed, 0 failed in 0.01s\n", 1,
        "unit_nonzero_pass_only")
    assert not result["all_pass"], result
    assert result["killed"] is True, result
    assert result["failed"] == 1, result
    assert any("runner-error" in f for f in result["failures"]), result


def test_zero_exit_with_pass_status_stays_all_pass_with_unchanged_counts():
    """GREEN: a zero-exit PASS-only run keeps its exact counts."""
    result = _run_with(
        "PASS test_fake_runner\n1 passed, 0 failed in 0.01s\n", 0,
        "unit_zero_pass")
    assert result["all_pass"] is True, result
    assert result["killed"] is False, result
    assert (result["passed"], result["failed"], result["skipped"]) == (1, 0, 0)
    assert result["failures"] == [], result
    statuses = result["per_file"]["test_fake_runner.py"]["statuses"]
    assert statuses == {"test_fake_runner": "PASS"}, statuses


def test_nonzero_exit_with_existing_kill_preserves_counts_without_double_count():
    """GREEN: a parsed kill stays the only failure on a nonzero exit."""
    result = _run_with(
        "FAIL test_fake_runner [FAIL]\n1 failed in 0.01s\n", 1,
        "unit_nonzero_kill")
    assert result["all_pass"] is False, result
    assert result["passed"] == 0, result
    assert result["failed"] == 1, result
    assert len(result["failures"]) == 1, result
    statuses = result["per_file"]["test_fake_runner.py"]["statuses"]
    assert statuses == {"test_fake_runner": "FAIL"}, statuses


def test_nonzero_exit_without_parsed_statuses_keeps_runner_crash_error():
    """GREEN: the existing <runner-crash> ERROR path stays intact."""
    result = _run_with("runner exploded before tests\n", 1,
                       "unit_nonzero_nostatus")
    assert result["all_pass"] is False, result
    assert result["failed"] == 1, result
    assert result["failures"] == ["test_fake_runner.py::<runner-crash> [ERROR]"], result
    statuses = result["per_file"]["test_fake_runner.py"]["statuses"]
    assert statuses == {"<runner-crash>": "ERROR"}, statuses


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except unittest.SkipTest as exc:
            skipped += 1
            print("SKIP", name, exc)
        except AssertionError as exc:
            failed += 1
            print("FAIL", name, exc)
        except Exception as exc:
            failed += 1
            print("ERROR", name, type(exc).__name__, exc)
    print("\n%d passed, %d failed, %d skipped" %
          (len(tests) - failed - skipped, failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
