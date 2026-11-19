"""Regression tests for text-to-image route hardening (audit m3, m4, F7).

A keyed user whose saved model is served only by the classic API keeps working
on the classic route instead of 400ing on the enter API; produced dimensions
from X-Image-Dimensions are reported when present; a caller seed is honored.
Stdlib-only: run ``python3 tests/test_route_hardening.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import shutil
import struct
import sys
import unittest
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]


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
    spec = importlib.util.spec_from_file_location("pollinations_route_hardening", ROOT / "__init__.py")
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


_ORIG: dict = {}


def _route_setup(mod, *, catalog, classic, key="sk_dummy"):
    for name in ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
                 "_include_paid_models", "_http_get_bytes", "_http_get_image", "save_b64_image"):
        _ORIG.setdefault(name, getattr(mod, name, None))
        if _ORIG[name] is not None:
            setattr(mod, name, _ORIG[name])
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: catalog)
    setattr(mod, "_classic_models", lambda: classic)
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    captured = {}

    def _no_get(url, *, headers=None, timeout=120.0):
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _no_get)

    def _get_image(url, *, headers=None, timeout=120.0):
        captured["url"] = url
        captured["headers"] = headers or {}
        return _png_bytes(), None

    setattr(mod, "_http_get_image", _get_image)
    return captured


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def test_parse_image_dimensions_header():
    mod = _require()
    assert mod._parse_image_dimensions("800x600") == (800, 600)
    assert mod._parse_image_dimensions(" 800 X 600 ") == (800, 600)
    for bad in (None, "", "garbage", "0x600", "-1x5", "800x", "x600", 42):
        assert mod._parse_image_dimensions(bad) is None, repr(bad)


def test_produced_dimensions_reported():
    mod = _require()
    captured = _route_setup(mod, catalog=[dict(QUEST)], classic=[])

    def _get_image(url, *, headers=None, timeout=120.0):
        captured["url"] = url
        return _png_bytes(), (800, 600)

    setattr(mod, "_http_get_image", _get_image)
    res = mod.PollinationsImageGenProvider().generate(
        "a cat", aspect_ratio="square", model="test/quest-model")
    assert res["success"] is True, res
    assert res["width"] == 800 and res["height"] == 600, res
    assert res["requested_width"] == 1024 and res["requested_height"] == 1024, res


def test_requested_dimensions_kept_without_header():
    mod = _require()
    _route_setup(mod, catalog=[dict(QUEST)], classic=[])
    res = mod.PollinationsImageGenProvider().generate(
        "a cat", aspect_ratio="square", model="test/quest-model")
    assert res["success"] is True, res
    assert res["width"] == 1024 and res["height"] == 1024, res
    assert "requested_width" not in res, res


def test_t2i_seed_kwarg_honored():
    mod = _require()
    captured = _route_setup(mod, catalog=[dict(QUEST)], classic=[])
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model", seed=42)
    assert res["success"] is True, res
    assert parse_qs(urlsplit(captured["url"]).query)["seed"] == ["42"], captured["url"]


def test_t2i_invalid_seed_falls_back_to_random():
    mod = _require()
    captured = _route_setup(mod, catalog=[dict(QUEST)], classic=[])
    for bad in (True, -1, 2**31, "42"):
        mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model", seed=bad)
        seed = int(parse_qs(urlsplit(captured["url"]).query)["seed"][0])
        assert 0 <= seed <= mod.SEED_MAX, (bad, seed)


def test_keyed_classic_only_model_uses_free_route():
    mod = _require()
    captured = _route_setup(mod, catalog=[], classic=["sana"])
    res = mod.PollinationsImageGenProvider().generate("a cat", model="sana")
    assert res["success"] is True, res
    assert captured["url"].startswith("https://image.pollinations.ai/prompt/"), captured["url"]
    assert "Authorization" not in captured["headers"], captured["headers"]
    assert res["mode"] == "free", res


def test_keyed_catalog_model_still_uses_enter():
    mod = _require()
    sana_entry = {"name": "sana", "category": "image", "paid_only": False, "flat_rate": True,
                  "pricing": {"currency": "pollen", "completionImageTokens": "0"}}
    captured = _route_setup(mod, catalog=[sana_entry], classic=["sana"])
    res = mod.PollinationsImageGenProvider().generate("a cat", model="sana")
    assert res["success"] is True, res
    assert captured["url"].startswith("https://gen.pollinations.ai/image/"), captured["url"]
    assert captured["headers"].get("Authorization") == "Bearer sk_dummy", captured["headers"]
    assert res["mode"] == "quest", res


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
