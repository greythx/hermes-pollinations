"""Regression tests for the classic free-model cache guards and actual-size reporting.

The classic list's failure handling must mirror the catalog's: a down endpoint
costs one timeout per negative window even when a stale cache exists, and a
non-empty answer with zero usable ids is a failed fetch, not an authoritative
empty. A JPEG whose size the API does not announce is measured from its SOF
marker so the classic route's server-side clamp is reported honestly.
Stdlib-only: run ``python3 tests/test_classic_cache_guards.py``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_classic_guards"


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
    spec = importlib.util.spec_from_file_location("pollinations_classic_guards", ROOT / "__init__.py")
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


KNOWN_LIMITS = {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647}
QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}

_ORIG: dict = {}


def _setup(mod, *, key=None, catalog=None, classic=None):
    for name in ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
                 "_include_paid_models", "_http_get_json", "_http_get_bytes",
                 "_http_get_image", "save_b64_image", "_spec_limits",
                 "_cache_path", "_classic_cache_path"):
        _ORIG.setdefault(name, getattr(mod, name, None))
        if _ORIG[name] is not None:
            setattr(mod, name, _ORIG[name])
    SCRATCH.mkdir(exist_ok=True)
    for f in SCRATCH.iterdir():
        f.unlink()
    setattr(mod, "_cache_path", lambda: SCRATCH / "catalog.json")
    setattr(mod, "_classic_cache_path", lambda: SCRATCH / "classic.json")
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: list(catalog or []))
    if classic is not None:
        setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits", lambda: dict(KNOWN_LIMITS))
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")

    def _no_get(url, *, headers=None, timeout=120.0):
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _no_get)
    setattr(mod, "_http_get_image", _no_get)
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})


def _jpeg_bytes(w: int, h: int) -> bytes:
    """A structurally valid JPEG skeleton: APP0 + SOF0 carrying w/h + SOS + EOI."""
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    sof = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08"
           + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x00" * 10)
    return b"\xff\xd8" + app0 + sof + b"\xff\xda" + b"\x00\x02" + b"\xff\xd9"


def test_failure_with_stale_cache_sets_negative_window():
    mod = _require()
    _setup(mod)
    mod._store_classic_cache(["sana"], time.time())
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    attempts = {"n": 0}

    def down(url, timeout=30.0):
        attempts["n"] += 1
        raise OSError("endpoint down")

    setattr(mod, "_http_get_json", down)
    for _ in range(5):
        assert mod._classic_models() == ["sana"]
    assert attempts["n"] == 1, attempts
    assert mod._classic_state()["negative_until"] > time.monotonic()


def test_zero_valid_ids_preserves_cache():
    mod = _require()
    _setup(mod)
    mod._store_classic_cache(["sana"], time.time())
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: [{"name": "sana"}])
    assert mod._classic_models() == ["sana"]
    disk = json.loads((SCRATCH / "classic.json").read_text())
    assert disk["ids"] == ["sana"], disk
    assert mod._classic_state()["negative_until"] > time.monotonic()


def test_decoy_nonmodel_list_preserves_cache():
    mod = _require()
    _setup(mod)
    mod._store_classic_cache(["sana"], time.time())
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: {"data": [{"v": 1}]})
    assert mod._classic_models() == ["sana"]
    disk = json.loads((SCRATCH / "classic.json").read_text())
    assert disk["ids"] == ["sana"], disk


def test_truly_empty_list_stays_authoritative():
    mod = _require()
    _setup(mod)
    mod._store_classic_cache(["sana"], time.time())
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: [])
    assert mod._classic_models() == []
    disk = json.loads((SCRATCH / "classic.json").read_text())
    assert disk["ids"] == [], disk


def test_jpeg_dimensions_from_sof():
    mod = _require()
    _setup(mod)
    assert mod._jpeg_dimensions(_jpeg_bytes(37, 23)) == (37, 23)
    assert mod._jpeg_dimensions(_jpeg_bytes(768, 512)) == (768, 512)
    assert mod._jpeg_dimensions(b"\xff\xd8\xff\xd9") is None
    assert mod._jpeg_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) is None
    assert mod._jpeg_dimensions(b"") is None


def test_free_route_reports_actual_jpeg_size():
    mod = _require()
    _setup(mod, key=None, classic=["sana"])
    captured = {}

    def _get_image(url, *, headers=None, timeout=120.0):
        captured["url"] = url
        return _jpeg_bytes(768, 512), None

    setattr(mod, "_http_get_image", _get_image)
    res = mod.PollinationsImageGenProvider().generate("a cat", aspect_ratio="landscape", model="sana")
    assert res["success"] is True, res
    assert res["mode"] == "free", res
    assert res["width"] == 768 and res["height"] == 512, res
    assert res["requested_width"] == 1280 and res["requested_height"] == 768, res


def test_enter_route_jpeg_without_header_reports_actual():
    mod = _require()
    _setup(mod, key="sk_dummy", catalog=[dict(QUEST)])

    def _get_image(url, *, headers=None, timeout=120.0):
        return _jpeg_bytes(800, 600), None

    setattr(mod, "_http_get_image", _get_image)
    res = mod.PollinationsImageGenProvider().generate(
        "a cat", aspect_ratio="square", model="test/quest-model")
    assert res["success"] is True, res
    assert res["width"] == 800 and res["height"] == 600, res
    assert res["requested_width"] == 1024 and res["requested_height"] == 1024, res


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