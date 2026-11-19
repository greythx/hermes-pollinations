"""Regression tests for catalog/cache hardening and parsing edges.

Covers audit m7, m8, m9, m12, m14, HC-F8/F10, ROB-F6/F7/F9/F10/F11, SEC-F9:
decoy-envelope unwrapping, the catalog success race guard, strict cache schema
and freshness, sanitized billing units, craftable completeness markers, tiny
price rendering, absurd rpm, passthrough convention, canonical-name preference,
live currency, and rename visibility. Stdlib-only.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.TemporaryDirectory(prefix="pollinations-hardening-")
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
    spec = importlib.util.spec_from_file_location("pollinations_hardening", ROOT / "__init__.py")
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


def _isolate(mod, tag):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    setattr(mod, "_cache_path", lambda: SCRATCH / f"{tag}_catalog.json")
    setattr(mod, "_classic_cache_path", lambda: SCRATCH / f"{tag}_classic.json")
    mod._CATALOG_CACHE.update({"negative_until": 0.0, "last_success": 0.0})
    mod._CLASSIC_CACHE.update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    for p in SCRATCH.glob(f"{tag}_*.json"):
        p.unlink()


def _png_bytes() -> bytes:
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


def test_unwrap_skips_decoy_errors_list():
    mod = _require()
    real = [{"name": "real/model", "category": "image"}]
    payload = {"errors": [{"name": "evil-model", "category": "image"}], "catalog": real}
    assert mod._unwrap_list(payload, ("models", "data", "items", "results")) == real


def test_unwrap_scalar_envelope_still_works():
    mod = _require()
    assert mod._unwrap_list({"foo": ["sana"]}, ("models",)) == ["sana"]
    assert mod._unwrap_list({"data": {"models": ["sana"]}}, ("models", "data")) == ["sana"]


def test_slow_stale_success_does_not_overwrite_newer_cache():
    mod = _require()
    _isolate(mod, "race")
    entered = threading.Event()
    release = threading.Event()
    state = {"calls": 0}

    def _get_json(url, timeout=30.0):
        state["calls"] += 1
        if state["calls"] == 1:
            entered.set()
            release.wait(5)
            return [{"name": "old/model", "category": "image"}]
        return [{"name": "new/model", "category": "image"}]

    setattr(mod, "_http_get_json", _get_json)
    worker = threading.Thread(target=mod._catalog)
    worker.start()
    assert entered.wait(5), "slow fetch never started"
    fresh = mod._catalog()
    assert [m["name"] for m in fresh] == ["new/model"], fresh
    release.set()
    worker.join(5)
    cached = json.loads(mod._cache_path().read_text())["models"]
    assert [m["name"] for m in cached] == ["new/model"], cached


def test_later_started_fetch_wins_even_when_slower():
    mod = _require()
    _isolate(mod, "race2")
    entered_a, entered_b = threading.Event(), threading.Event()
    release_a, release_b = threading.Event(), threading.Event()
    state = {"calls": 0}

    def _get_json(url, timeout=30.0):
        state["calls"] += 1
        if state["calls"] == 1:
            entered_a.set()
            release_a.wait(5)
            return [{"name": "old/model", "category": "image"}]
        entered_b.set()
        release_b.wait(5)
        return [{"name": "new/model", "category": "image"}]

    setattr(mod, "_http_get_json", _get_json)
    worker_a = threading.Thread(target=mod._catalog)
    worker_a.start()
    assert entered_a.wait(5), "fetch A never started"
    worker_b = threading.Thread(target=mod._catalog)
    worker_b.start()
    assert entered_b.wait(5), "fetch B never started"
    release_a.set()
    worker_a.join(5)  # earlier-started A completes first
    release_b.set()
    worker_b.join(5)  # later-started B completes last and must win
    cached = json.loads(mod._cache_path().read_text())["models"]
    assert [m["name"] for m in cached] == ["new/model"], cached


def _cache_trust_probe(mod, tag, cache_doc):
    _isolate(mod, tag)
    if cache_doc is not None:
        mod._cache_path().write_text(json.dumps(cache_doc))
    fetches = []

    def _get_json(url, timeout=30.0):
        fetches.append(url)
        return [{"name": "fresh/model", "category": "image"}]

    setattr(mod, "_http_get_json", _get_json)
    mod._catalog()
    return len(fetches)


def test_cache_schema_rejects_bool_and_float_versions():
    mod = _require()
    models = [{"name": "stale/model", "category": "image"}]
    for ver in (True, 1.0, "1", 2):
        assert _cache_trust_probe(mod, f"ver{ver}", {"version": ver, "fetched": time.time(),
                                                     "models": models}) == 1, ver
    assert _cache_trust_probe(mod, "verok", {"version": 1, "fetched": time.time(),
                                             "models": models}) == 0


def test_cache_without_version_is_not_trusted():
    mod = _require()
    models = [{"name": "stale/model", "category": "image"}]
    assert _cache_trust_probe(mod, "nover", {"fetched": time.time(), "models": models}) == 1


def test_future_fetched_timestamp_is_stale():
    mod = _require()
    models = [{"name": "stale/model", "category": "image"}]
    assert _cache_trust_probe(mod, "future", {"version": 1, "fetched": time.time() + 1e6,
                                              "models": models}) == 1


def test_pricing_unit_sanitized():
    mod = _require()
    m = {"name": "x", "category": "image",
         "pricing": {"completionImageTokens": "0.002"},
         "pricing_units": {"completionImageTokens": {"unit": "x\nINJECT-LINE"}}}
    label = mod._price_label(m, False, True)
    assert "\n" not in label, label
    m["pricing_units"]["completionImageTokens"]["unit"] = "y" * 100000
    assert len(mod._price_label(m, False, True)) < 5000


def test_crafted_png_marker_rejected():
    mod = _require()
    crafted = _png_bytes()[:40] + b"IEND\xaeB`\x82"
    assert mod._image_is_complete(crafted, "image/png") is False
    assert mod._image_is_complete(_png_bytes(), "image/png") is True


def test_crafted_webp_rejected_and_valid_accepted():
    mod = _require()
    crafted = b"RIFF" + struct.pack("<I", 20) + b"WEBP" + b"\x00" * 20
    assert mod._image_is_complete(crafted, "image/webp") is False
    valid = b"RIFF" + struct.pack("<I", 16) + b"WEBP" + b"VP8L" + b"\x00" * 8
    assert mod._image_is_complete(valid, "image/webp") is True


def test_svg_completeness_checked():
    mod = _require()
    assert mod._image_is_complete(b"<svg><rect/></svg>", "image/svg+xml") is True
    assert mod._image_is_complete(b"<svg><rect/>", "image/svg+xml") is False


def test_tiny_price_keeps_significant_digit():
    mod = _require()
    label = mod._format_price(1e-300)
    assert "e-300" in label and "1.00" in label, label
    assert label != "0.0000000000000000000000… pollen", label


def test_absurd_rpm_not_rendered():
    mod = _require()
    m = {"name": "x", "category": "image", "per_user_rpm": 1e308}
    assert mod._specs(m, False, True)["speed"] == ""
    m["per_user_rpm"] = 15
    assert mod._specs(m, False, True)["speed"] == "≤15 req/min"


def test_env_equal_to_shared_passes_through():
    mod = _require()
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {"model": "brand-new"})
    setattr(mod, "_classic_models", lambda: [])
    os.environ[mod.MODEL_ENV] = "brand-new"
    try:
        assert mod._resolve_model(None, []) == "brand-new"
    finally:
        os.environ.pop(mod.MODEL_ENV, None)
    # With no env/scoped choice, the injected shared kwarg carrying an unknown id stays blocked.
    assert mod._resolve_model("brand-new", []) is None


def test_canonical_name_beats_alias_shadow():
    mod = _require()
    # Stub the lookups so this test never reaches the live classic endpoint or real config,
    # regardless of test order.
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_classic_models", lambda: [])
    catalog = [{"name": "evil-paid", "category": "image", "paid_only": True, "aliases": ["flux"]},
               {"name": "flux", "category": "image", "paid_only": False}]
    assert mod._resolve_model("flux", catalog) == "flux"


def test_currency_read_from_catalog():
    mod = _require()
    m = {"name": "x", "category": "image", "flat_rate": True,
         "pricing": {"currency": "credits", "completionImageTokens": "0.002"}}
    assert mod._price_label(m, False, True) == "0.002 credits · needs key"
    m["pricing"]["currency"] = "pollen"
    assert mod._price_label(m, False, True) == "0.002 pollen · needs key"


def test_contract_gaps_logged():
    mod = _require()
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture(level=logging.DEBUG)
    mod.logger.addHandler(handler)
    mod.logger.setLevel(logging.DEBUG)
    try:
        mod._note_contract_gaps([{"name": "x"}])
        assert len(records) == 3, records
        records.clear()
        mod._note_contract_gaps([{"name": "x", "pricing": {}, "paid_only": False,
                                  "supported_endpoints": []}])
        assert records == [], records
    finally:
        mod.logger.removeHandler(handler)


def test_catalog_without_usable_ids_is_not_a_success():
    mod = _require()
    _isolate(mod, "noids")
    good = [{"name": "good/model", "category": "image"}]
    # Stale (not fresh) so a fetch is attempted and the decoy answer must not overwrite it.
    mod.atomic_json_write(mod._cache_path(), {"version": mod.CACHE_SCHEMA,
                                              "fetched": time.time() - mod.CATALOG_TTL - 10,
                                              "models": good})
    # A renamed envelope whose only list carries no model-shaped entries must not overwrite
    # the good cache: zero usable ids has the same effect as an empty answer.
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: {"meta": [{"v": 1}]})
    got = mod._catalog()
    assert [m["name"] for m in got] == ["good/model"], got
    cached = json.loads(mod._cache_path().read_text())["models"]
    assert [m["name"] for m in cached] == ["good/model"], cached


def test_cache_entries_without_ids_are_not_a_fresh_hit():
    mod = _require()
    _isolate(mod, "noids-cache")
    mod.atomic_json_write(mod._cache_path(), {"version": mod.CACHE_SCHEMA,
                                              "fetched": time.time(),
                                              "models": [{"v": 1}]})
    fresh = [{"name": "fresh/model", "category": "image"}]
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: fresh)
    got = mod._catalog()
    assert [m["name"] for m in got] == ["fresh/model"], got


def test_iso_date_absurd_year_blank():
    mod = _require()
    for raw in (1e18, 2**63, 1e15, -5, 0):
        assert mod._iso_date({"added_date": raw}) == "", raw
    assert mod._iso_date({"added_date": 1759795200000}) == "2025-10-07"
    assert mod._iso_date({"added_date": 1759795200}) == "2025-10-07"


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
