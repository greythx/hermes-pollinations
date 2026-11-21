"""Regression tests for id/prompt sanitization (audit M3).

Catalog names are guarded, but classic-endpoint ids and unknown-id passthrough
values also become URL path segments: a lone surrogate raises UnicodeEncodeError
out of quote(), and control-char/oversize ids produce null-id picker rows. Every
id source must share the same guard, and generate() must degrade to an error
response instead of raising. Stdlib-only: run ``python3 tests/test_sanitization.py``.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.TemporaryDirectory(prefix="pollinations-sanitization-")
atexit.register(_TMP.cleanup)
SCRATCH = Path(_TMP.name)


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
    spec = importlib.util.spec_from_file_location("pollinations_sanitization", ROOT / "__init__.py")
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


def _isolate(mod, tmp):
    """Scratch cache paths + reset in-memory caches so tests never touch live state."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    setattr(mod, "_cache_path", lambda: SCRATCH / f"{tmp}_catalog.json")
    setattr(mod, "_classic_cache_path", lambda: SCRATCH / f"{tmp}_classic.json")
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    for p in SCRATCH.glob(f"{tmp}_*.json"):
        p.unlink()


def _net_sentinel(mod):
    calls = []

    def _get(url, *, headers=None, timeout=120.0):
        calls.append(url)
        raise AssertionError("network reached")

    def _get_image(url, *, headers=None, timeout=120.0):
        calls.append(url)
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _get)
    setattr(mod, "_http_get_image", _get_image)
    return calls


def test_valid_id_guard():
    mod = _require()
    assert mod._valid_id("sana") is True
    assert mod._valid_id("publisher/model") is True
    assert mod._valid_id(" flux ") is True
    for bad in ("\ud800x", "a\nb", "a\x00b", "x" * 5000, "", "   ", 42, None, True):
        assert mod._valid_id(bad) is False, repr(bad)


def test_classic_fetch_filters_hostile_ids():
    mod = _require()
    _isolate(mod, "fetch")
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: ["\ud800evil", "sa\nna", "x" * 100000, "sana"])
    assert mod._classic_models() == ["sana"]


def test_classic_cache_load_filters_hostile_ids():
    mod = _require()
    _isolate(mod, "cacheload")
    path = mod._classic_cache_path()
    path.write_text(json.dumps({"version": 1, "fetched": time.time(),
                                "ids": ["\ud800evil", "sa\nna", "sana"]}))
    assert mod._load_classic_cache() == ["sana"]


def test_keyless_generate_with_hostile_classic_survives():
    mod = _require()
    _isolate(mod, "e2e")
    setattr(mod, "_api_key", lambda: None)
    setattr(mod, "load_image_gen_config", lambda: {})

    def _get_json(url, timeout=30.0):
        if "image.pollinations.ai/models" in url:
            return ["\ud800evil", "sana"]
        return [{"name": "other/model", "category": "image"}]

    setattr(mod, "_http_get_json", _get_json)
    calls = _net_sentinel(mod)
    res = mod.PollinationsImageGenProvider().generate("a cat")
    assert len(calls) == 1, (res, calls)
    assert "model=sana" in calls[0], calls


def test_surrogate_prompt_is_invalid_input():
    mod = _require()
    _isolate(mod, "prompt")
    setattr(mod, "_api_key", lambda: None)
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: ["sana"])
    calls = _net_sentinel(mod)
    res = mod.PollinationsImageGenProvider().generate("a cat \ud800")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert calls == [], calls


def test_surrogate_model_passthrough_skipped():
    mod = _require()
    _isolate(mod, "passthrough")
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: [])
    calls = _net_sentinel(mod)
    res = mod.PollinationsImageGenProvider().generate("a cat", model="\ud800x")
    assert res["success"] is False, res
    assert res["error_type"] == "no_model_available", res
    assert calls == [], calls


def test_keyless_picker_skips_invalid_classic_rows():
    mod = _require()
    _isolate(mod, "picker")
    setattr(mod, "_api_key", lambda: None)
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: ["sa\nna", "x" * 100000, "sana"])
    rows = mod.PollinationsImageGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert ids == ["sana"], ids
    assert all(i is not None for i in ids), ids


def test_control_char_prompt_still_encoded():
    mod = _require()
    _isolate(mod, "control")
    setattr(mod, "_api_key", lambda: None)
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: ["sana"])
    calls = _net_sentinel(mod)
    mod.PollinationsImageGenProvider().generate("a\nb\tc")
    assert len(calls) == 1, calls
    assert "a%0Ab%09c" in calls[0], calls


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
