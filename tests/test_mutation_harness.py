"""Stdlib-only regression tests for the mutation harness runner-error path.

Run directly with ``python tests/test_mutation_harness.py``.  The harness is
loaded from its real location even when this file is copied into a mutation
scratch directory by ``tests/mutation/harness.py``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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


def _load_run():
    """Load run.py from the repo (works from the repo and from scratch)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "mutation" / "run.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "mutation_run_under_test", candidate)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise unittest.SkipTest("tests/mutation/run.py not found")


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


def test_unparseable_runner_output_is_harness_abort_not_kill():
    """RED: no parseable statuses means a harness abort, never a mutation kill."""
    result = _run_with("runner exploded before tests\n", 1,
                       "unit_nonzero_nostatus")
    assert result["aborted"] is True, result
    assert len(result["aborts"]) == 1, result
    assert "runner-crash" in result["aborts"][0], result
    assert result["failed"] == 0, result
    assert result["failures"] == [], result
    assert result["killed"] is False, result
    assert result["all_pass"] is False, result
    statuses = result["per_file"]["test_fake_runner.py"]["statuses"]
    assert statuses == {}, statuses


def test_run_main_aborts_when_catalog_validation_fails():
    """RED: main() must call catalog.validate and abort on a broken catalog."""
    mod = _load_run()
    seen = []

    def boom(source):
        seen.append(source)
        raise AssertionError("bad anchor")

    orig_validate, orig_run_suite = mod.catalog.validate, mod.harness.run_suite
    mod.catalog.validate = boom
    mod.harness.run_suite = lambda *a, **k: {
        "all_pass": True, "killed": False, "aborted": False,
        "passed": 1, "failed": 0, "skipped": 0, "failures": [], "files": []}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = mod.main([])
    finally:
        mod.catalog.validate = orig_validate
        mod.harness.run_suite = orig_run_suite
    assert seen, "catalog.validate was never called by run.main()"
    assert rc != 0, rc


def test_run_main_exits_nonzero_when_a_mutation_aborts():
    """RED: a harness abort must fail the gate, not count as a kill."""
    mod = _load_run()
    fake_entry = {"name": "fake_abort", "description": "test", "finding": "test",
                  "old_string": "x", "new_string": "y"}
    labels = []

    def fake_run_suite(entry=None, *, label="run", **kwargs):
        labels.append(label)
        if label == "__sanity__":
            return {"all_pass": True, "killed": False, "aborted": False,
                    "passed": 1, "failed": 0, "skipped": 0, "failures": [],
                    "files": []}
        return {"all_pass": False, "killed": False, "aborted": True,
                "passed": 0, "failed": 0, "skipped": 0, "failures": [],
                "files": [], "aborts": ["test_x.py::runner-crash"]}

    orig_validate, orig_mut, orig_run = (mod.catalog.validate,
                                         mod.catalog.MUTATIONS,
                                         mod.harness.run_suite)
    mod.catalog.validate = lambda source: None
    mod.catalog.MUTATIONS = [fake_entry]
    mod.harness.run_suite = fake_run_suite
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = mod.main([])
    finally:
        mod.catalog.validate = orig_validate
        mod.catalog.MUTATIONS = orig_mut
        mod.harness.run_suite = orig_run
    assert labels.count("fake_abort") == 1, labels
    assert "ABORT" in buf.getvalue(), buf.getvalue()
    assert rc != 0, rc


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
