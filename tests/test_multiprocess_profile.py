"""Regression tests for multiprocess and routed-profile correctness (v1.7.0 T10).

Covers MP-1 (cross-process write guard: a newer disk write is never clobbered),
MP-2 (classic disk-first freshness), MP-3 (monotonic failure/success windows),
MP-4 (per-home cache state), MP-6 (profile-scoped MODEL_ENV) and flock
concurrency safety. Stdlib-only: run ``python3 tests/test_multiprocess_profile.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_mp"


def _load():
    try:
        import agent.image_gen_provider  # noqa: F401
    except ImportError:
        exe = shutil.which("hermes")
        if exe:
            sys.path.insert(0, str(Path(exe).resolve().parent.parent))
        try:
            import agent.image_gen_provider  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    spec = importlib.util.spec_from_file_location("pollinations_multiprocess_profile", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


ENTRY = {"name": "test/model", "category": "image", "paid_only": False,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def _setup(mod, home=None):
    home = Path(home or SCRATCH)
    # Full isolation: a leftover fresh disk cache from an earlier test would short-circuit
    # the fetch path these tests exercise.
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    setattr(mod, "_hermes_home", lambda: home)
    setattr(mod, "_api_key", lambda: None)
    setattr(mod, "load_image_gen_config", lambda: {})
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})
    return home


def test_newer_disk_write_not_clobbered():
    mod = _require()
    home = _setup(mod)
    newer = {"version": 1, "fetched": time.time() + 100, "models": [{"name": "newer/model"}]}
    mod.atomic_json_write(home / "cache" / "pollinations_catalog.json", newer)
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: [dict(ENTRY, name="older/model")])
    models = mod._catalog()
    assert [m["name"] for m in models] == ["older/model"], models
    disk = json.loads((home / "cache" / "pollinations_catalog.json").read_text())
    assert disk["models"] == [{"name": "newer/model"}], disk
    # An OLDER disk cache is overwritten normally.
    mod.atomic_json_write(home / "cache" / "pollinations_catalog.json",
                          {"version": 1, "fetched": time.time() - 7200,
                           "models": [{"name": "ancient/model"}]})
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._catalog()
    disk = json.loads((home / "cache" / "pollinations_catalog.json").read_text())
    assert [m["name"] for m in disk["models"]] == ["older/model"], disk


def test_classic_disk_first_freshness():
    mod = _require()
    home = _setup(mod)
    mod._classic_state().update({"at": time.time() - 10, "ids": ["old"],
                                 "negative_until": 0.0, "last_success": 0.0})
    mod.atomic_json_write(home / "cache" / "pollinations_classic_models.json",
                          {"version": 1, "fetched": time.time(), "ids": ["new"]})
    assert mod._classic_models() == ["new"]


def test_failure_window_is_monotonic():
    mod = _require()
    _setup(mod)

    def down(url, timeout=30.0):
        raise OSError("down")

    setattr(mod, "_http_get_json", down)
    mod._catalog()
    neg = mod._catalog_state()["negative_until"]
    assert abs(neg - (time.monotonic() + mod.NEGATIVE_TTL)) < 5, neg
    assert neg < time.time(), "window must not be wall-clock (epoch-scale)"


def test_per_home_state_isolation():
    mod = _require()
    home_a = SCRATCH / "home-a"
    home_b = SCRATCH / "home-b"
    _setup(mod, home_a)
    calls = {"n": 0}

    def down(url, timeout=30.0):
        calls["n"] += 1
        raise OSError("down")

    setattr(mod, "_http_get_json", down)
    mod._catalog()  # home A fails -> negative window
    assert mod._catalog_state()["negative_until"] > time.monotonic()
    _setup(mod, home_b)  # switch profile: fresh state, fetch attempted again
    mod._catalog()
    assert calls["n"] == 2, calls
    setattr(mod, "_hermes_home", lambda: home_a)
    mod._catalog()  # home A still suppressed
    assert calls["n"] == 2, calls


def test_model_env_profile_scoped():
    mod = _require()
    _setup(mod)
    import agent.secret_scope as ss
    orig_get_secret = ss.get_secret
    orig_env = os.environ.get(mod.MODEL_ENV)
    os.environ[mod.MODEL_ENV] = "launch-model"

    def scoped(name, *a, **kw):
        if name == mod.MODEL_ENV:
            return "scoped-model"
        return orig_get_secret(name, *a, **kw)

    setattr(ss, "get_secret", scoped)
    try:
        catalog = [dict(ENTRY, name="scoped-model"), dict(ENTRY, name="launch-model")]
        assert mod._resolve_model(None, catalog) == "scoped-model"

        def unscoped(name, *a, **kw):
            if name == mod.MODEL_ENV:
                raise ss.UnscopedSecretError("no scope")
            return orig_get_secret(name, *a, **kw)

        setattr(ss, "get_secret", unscoped)
        # No scope bound: the launch env must NOT leak into the routed context.
        assert mod._model_env() == ""
    finally:
        setattr(ss, "get_secret", orig_get_secret)
        if orig_env is None:
            os.environ.pop(mod.MODEL_ENV, None)
        else:
            os.environ[mod.MODEL_ENV] = orig_env


def test_guarded_write_concurrency_safe():
    mod = _require()
    home = _setup(mod)
    path = home / "cache" / "pollinations_catalog.json"
    errors = []

    def writer(i):
        try:
            for j in range(20):
                mod._guarded_cache_write(
                    path, {"version": 1, "fetched": time.time(),
                           "models": [{"name": f"m{i}-{j}"}]}, time.time())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    disk = json.loads(path.read_text())
    assert disk["version"] == 1 and isinstance(disk["models"], list), disk


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())