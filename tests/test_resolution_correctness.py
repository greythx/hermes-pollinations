"""Regression tests for resolution and paid-gate correctness (v1.7.0 T1).

Covers audit M1 (a paid alias shadowing a classic-only id), F1 (case-sensitive
catalog matching bypassing the paid gate and downgrading keyed classic models),
FUZZ-R3 (a non-scalar ``paid_only`` failing the gate open), F16/R2-P1 (a video
model routed to the image endpoint and kept in the picker), F11 (an ``id``
fallback that resolves to nothing) and F6 (keyless duplicate case-variant rows).
Stdlib-only: run ``python3 tests/test_resolution_correctness.py``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import sys
import unittest
import zlib
from pathlib import Path

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
    spec = importlib.util.spec_from_file_location("pollinations_resolution_correctness",
                                                  ROOT / "__init__.py")
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


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
EVIL = {"name": "evil-paid", "category": "image", "paid_only": True, "aliases": ["sana"],
        "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
PAID_CASE = {"name": "PaidModel", "category": "image", "paid_only": True,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
SANA_CASE = {"name": "Sana", "category": "image", "paid_only": False, "flat_rate": True,
             "pricing": {"currency": "pollen", "completionImageTokens": "0"}}
VIDEO = {"name": "test/video-model", "category": "video", "paid_only": False}


def _setup(mod, *, catalog, classic, key: "str | None" = "sk_dummy", include_paid=False,
           config=None, serve=False):
    """Stub every live lookup and HTTP seam; ``serve`` returns a fake PNG instead of raising."""
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in catalog])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: include_paid)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    os.environ.pop(mod.MODEL_ENV, None)
    calls = []
    captured = {}

    def _get(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        raise AssertionError("network reached")

    def _get_image(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        if not serve:
            raise AssertionError("network reached")
        captured["url"] = url
        captured["headers"] = headers or {}
        return _png_bytes(), None

    def _post(url, *, headers, payload, timeout=120.0):
        calls.append(("POST", url))
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _get)
    setattr(mod, "_http_get_image", _get_image)
    setattr(mod, "_http_post_json", _post)
    return calls, captured


def test_classic_only_id_not_shadowed_by_paid_alias():
    mod = _require()
    calls, captured = _setup(mod, catalog=[EVIL], classic=["sana"],
                             config={"model": "sana"}, serve=True)
    # A classic-served id the catalog omits must keep its own spelling, not be hijacked by a
    # paid entry claiming it as an alias — via the explicit kwarg and the saved shared config.
    assert mod._resolve_model("sana", [dict(EVIL)]) == "sana"
    assert mod._resolve_model(None, [dict(EVIL)]) == "sana"
    rows = mod.PollinationsImageGenProvider().list_models()
    assert "sana" in [r["id"] for r in rows], rows
    res = mod.PollinationsImageGenProvider().generate("a cat", model="sana")
    assert res["success"] is True, res
    assert res["mode"] == "free", res
    assert captured["url"].startswith("https://image.pollinations.ai/prompt/"), captured["url"]
    assert "Authorization" not in captured["headers"], captured["headers"]


def test_case_variant_paid_id_is_gated():
    mod = _require()
    calls, _ = _setup(mod, catalog=[PAID_CASE], classic=[])
    assert mod._resolve_model("paidmodel", [dict(PAID_CASE)]) == "PaidModel"
    res = mod.PollinationsImageGenProvider().generate("a cat", model="paidmodel")
    assert res["success"] is False, res
    assert res["error_type"] == "paid_model_disabled", res
    assert calls == [], calls


def test_case_variant_classic_catalog_id_not_downgraded():
    mod = _require()
    calls, captured = _setup(mod, catalog=[SANA_CASE], classic=["sana"], serve=True)
    res = mod.PollinationsImageGenProvider().generate("a cat", model="sana")
    assert res["success"] is True, res
    assert captured["url"].startswith("https://gen.pollinations.ai/image/"), captured["url"]
    assert res["mode"] == "quest", res
    assert captured["headers"].get("Authorization") == "Bearer sk_dummy", captured["headers"]


def test_non_scalar_paid_only_fails_closed():
    mod = _require()
    for flag in (["true"], {"enabled": True}):
        odd = {"name": "test/odd-paid", "category": "image", "paid_only": flag,
               "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
        for include_paid in (False, True):
            calls, _ = _setup(mod, catalog=[odd], classic=[], include_paid=include_paid)
            res = mod.PollinationsImageGenProvider().generate("a cat", model="test/odd-paid")
            assert res["success"] is False, (flag, include_paid, res)
            assert res["error_type"] == "model_unavailable", (flag, include_paid, res)
            assert calls == [], (flag, include_paid, calls)
        calls, _ = _setup(mod, catalog=[odd], classic=[], include_paid=False)
        rows = mod.PollinationsImageGenProvider().list_models()
        assert "test/odd-paid" not in [r["id"] for r in rows], rows


def test_video_model_refused_in_generate():
    mod = _require()
    calls, _ = _setup(mod, catalog=[VIDEO], classic=[], config={"model": "test/video-model"})
    prov = mod.PollinationsImageGenProvider()
    caps = prov.capabilities()
    assert caps["modalities"] == ["text"], caps
    res = prov.generate("a cat")
    assert res["success"] is False, res
    assert res["error_type"] == "model_unavailable", res
    assert calls == [], calls


def test_saved_video_id_not_retained():
    mod = _require()
    for key in (None, "sk_dummy"):
        _setup(mod, catalog=[VIDEO, QUEST], classic=[], key=key,
               config={"model": "test/video-model"})
        rows = mod.PollinationsImageGenProvider().list_models()
        assert "test/video-model" not in [r["id"] for r in rows], (key, rows)


def test_id_fallback_resolvable_with_metadata():
    mod = _require()
    entry = {"name": "test/alpha", "id": "alpha-id", "category": "image", "paid_only": True,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
    calls, _ = _setup(mod, catalog=[entry], classic=[])
    assert mod._resolve_model("alpha-id", [dict(entry)]) == "test/alpha"
    res = mod.PollinationsImageGenProvider().generate("a cat", model="alpha-id")
    assert res["success"] is False, res
    assert res["error_type"] == "paid_model_disabled", res
    assert calls == [], calls


def test_keyless_case_variant_rows_deduped():
    mod = _require()
    _setup(mod, catalog=[SANA_CASE], classic=["sana"], key=None)
    rows = mod.PollinationsImageGenProvider().list_models()
    sana_rows = [r for r in rows if r["id"].lower() == "sana"]
    assert len(sana_rows) == 1, rows


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
