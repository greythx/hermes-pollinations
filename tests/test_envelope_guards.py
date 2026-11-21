"""Regression tests for envelope/cache guards and image completeness (v1.7.0 T4).

Covers L12 (positional envelope fallback must not accept name-shaped non-model
lists), F5 (classic memory cache future-"at" freshness), F9 (non-string
endpoint fallback), M3 (source images must pass the completeness pre-filter),
F14 (generated output restricted to raster formats on both routes).
Stdlib-only: run ``python3 tests/test_envelope_guards.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import struct
import sys
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_envelope_guards"


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
    spec = importlib.util.spec_from_file_location("pollinations_envelope_guards", ROOT / "__init__.py")
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


def _png_bytes() -> bytes:
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


SVG = (b"<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'>"
       b"<rect width='10' height='10' fill='red'/></svg>")


def _setup(mod, *, fetch=None, classic_fetch=None):
    SCRATCH.mkdir(exist_ok=True)
    for f in SCRATCH.iterdir():
        f.unlink()
    setattr(mod, "_cache_path", lambda: SCRATCH / "catalog.json")
    setattr(mod, "_classic_cache_path", lambda: SCRATCH / "classic.json")
    setattr(mod, "_spec_cache_path", lambda: SCRATCH / "spec.json")
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})

    def _get_json(url, timeout=30.0):
        if "image/models" in url:
            if fetch is None:
                raise OSError("down")
            return fetch
        if classic_fetch is None:
            raise OSError("down")
        return classic_fetch

    setattr(mod, "_http_get_json", _get_json)


def test_catalog_fallback_rejects_name_shaped_non_models():
    mod = _require()
    _setup(mod, fetch={"tags": [{"name": "nsfw"}], "meta": [{"name": "v1"}]})
    assert mod._catalog() == []
    assert not (SCRATCH / "catalog.json").exists()


def test_catalog_fallback_accepts_image_shaped_rename():
    mod = _require()
    entry = {"name": "test/renamed", "category": "image", "paid_only": False,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    _setup(mod, fetch={"publisher_models": [entry]})
    models = mod._catalog()
    assert [m["name"] for m in models] == ["test/renamed"], models


def test_classic_fallback_requires_models_key():
    mod = _require()
    _setup(mod, classic_fetch={"versions": ["v1", "v2"], "free_models": ["sana"]})
    assert mod._classic_models() == ["sana"]
    _setup(mod, classic_fetch={"versions": ["v1", "v2"]})
    assert mod._classic_models() == []


def test_classic_memory_future_at_not_fresh():
    mod = _require()
    import time
    _setup(mod, classic_fetch=["fresh"])
    mod._classic_state().update({"at": time.time() + 3600, "ids": ["stale-id"],
                               "negative_until": 0.0, "last_success": 0.0})
    assert mod._classic_models() == ["fresh"]


def test_non_string_endpoint_not_an_image_signal():
    mod = _require()
    assert mod._is_image_model({"supported_endpoints": [{"note": "image support planned"}]}) is False
    assert mod._is_image_model({"supported_endpoints": ["/image/{prompt}"]}) is True


def test_incomplete_source_image_rejected():
    mod = _require()
    truncated = _png_bytes()[:-12]  # strip the IEND chunk
    try:
        mod._encode_input_image(truncated)
        raise AssertionError("incomplete source accepted")
    except ValueError:
        pass
    assert mod._encode_input_image(_png_bytes()).startswith("data:image/png;base64,")


def test_svg_output_rejected_t2i_and_edit():
    mod = _require()
    _setup(mod)  # keyed state, so generation reaches the response checks standalone
    entry = {"name": "test/raster", "category": "image", "paid_only": False,
             "supported_endpoints": ["/v1/images/edits"],
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    setattr(mod, "_catalog", lambda: [entry])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    setattr(mod, "_http_get_image", lambda url, *, headers=None, timeout=120.0: (SVG, None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/raster")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_response", res
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64,AAAA")
    setattr(mod, "_http_post_json", lambda url, *, headers, payload, timeout=120.0: {
        "data": [{"b64_json": base64.b64encode(SVG).decode()}]})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/raster", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_response", res


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