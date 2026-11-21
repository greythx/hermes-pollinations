"""Regression tests for Round-4 verification fixes (v1.7.0 T13).

Covers the keyed classic-served SVG routing exemption, redaction backslash-flood
DoS bound, fixUrl Unicode-authority confusables, retention SVG markers, zero-rate
variant vs paid base labeling, flat-label quantity agreement, SVG modality
signal, and normalized classic retention matching.
Stdlib-only: run ``python3 tests/test_round4_fixes.py``.
"""

from __future__ import annotations

import importlib.util
import shutil
import struct
import sys
import time
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_round4"


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
    spec = importlib.util.spec_from_file_location("pollinations_round4_fixes", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

_SEAMS = ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
          "_include_paid_models", "_spec_limits", "save_b64_image", "_hermes_home",
          "_http_get_json", "_http_get_image", "_http_post_json", "_to_data_url")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


SVG_ENTRY = {"name": "vector-x", "category": "image", "paid_only": False, "flat_rate": True,
             "description": "Create editable SVG artwork",
             "pricing": {"currency": "pollen", "completionImageTokens": "0.01"}}
QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def _setup(mod, *, catalog=None, key: "str | None" = "sk_dummy", config=None, classic=()):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    setattr(mod, "_hermes_home", lambda: SCRATCH)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog if catalog is not None else [QUEST])])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})


def _png(w, h):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


def test_keyed_classic_served_svg_refused():
    mod = _require()
    _setup(mod, catalog=[SVG_ENTRY], classic=["vector-x"])
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="vector-x")
    assert res["success"] is False, res
    assert res["error_type"] == "unsupported", res
    assert calls == [], calls  # keyed catalog-listed routes to enter: must refuse pre-request


def test_keyless_classic_served_svg_runs():
    mod = _require()
    _setup(mod, catalog=[SVG_ENTRY], key=None, classic=["vector-x"])
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="vector-x")
    assert res["success"] is True, res
    assert calls and "image.pollinations.ai" in calls[0], calls  # classic rasterizes to JPEG


def test_redaction_backslash_flood_bounded():
    mod = _require()
    start = time.monotonic()
    out = mod._redact_urls("\\" * 65536)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"redaction took {elapsed:.1f}s on a 64KB backslash flood"
    assert isinstance(out, str)


def test_fixurl_unicode_authority_rejected():
    mod = _require()
    import json

    class _R:
        headers = {}

        def __init__(self, body):
            self._body = body

        def iter_bytes(self):
            yield self._body

    for host in ("evil.example\uff3c.good.example", "evil.example\ufe68.good.example",
                 "good.example\uff20evil.example"):
        body = json.dumps({"error": {"message": "x", "fixUrl": f"https://{host}/fix"}}).encode()
        _, _, fix = mod._error_fields(_R(body))
        assert fix is None, (host, fix)
    body = json.dumps({"error": {"message": "x", "fixUrl": "https://good.example/fix"}}).encode()
    _, _, fix = mod._error_fields(_R(body))
    assert fix == "https://good.example/fix", fix


def test_retention_svg_marker_supersedes():
    mod = _require()
    paid_svg = dict(SVG_ENTRY, name="test/paid-svg", paid_only=True)
    _setup(mod, catalog=[paid_svg, QUEST], config={"model": "test/paid-svg"})
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/paid-svg")
    assert "SVG output unsupported" in row["price"], row["price"]


def test_zero_rate_variant_keeps_paid_base_figure():
    mod = _require()
    m = dict(QUEST, name="test/free-variant",
             pricing={"currency": "pollen", "completionImageTokens": "0.5"},
             pricing_variants=[{"name": "free_tier", "completionImageTokens": "0"}])
    label = mod._price_label(m, False, True)
    assert "Free base" not in label, label
    assert "0.5 pollen" in label and "(free variant)" in label, label
    # Sort keeps the true achievable floor (a free variant exists).
    assert mod._price_value(m) == 0.0, mod._price_value(m)


def test_flat_label_includes_quantity():
    mod = _require()
    m = dict(QUEST, name="test/qty",
             pricing={"currency": "pollen", "completionImageTokens": "0.5"},
             pricing_units={"completionImageTokens": {"unit": "image", "quantity": 2}})
    label = mod._price_label(m, False, True)
    assert label.startswith("1 pollen"), label  # 0.5 x 2, matching the sort estimate
    assert mod._price_value(m) == 1.0


def test_svg_modality_signal_detected():
    mod = _require()
    m = {"name": "test/modality-svg", "category": "image", "paid_only": False,
         "description": "Clean description with no format mention",
         "output_modalities": ["svg"],
         "pricing": {"currency": "pollen", "completionImageTokens": "0.01"}}
    assert mod._is_svg_product(m) is True
    _setup(mod, catalog=[m])
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/modality-svg")
    assert res["success"] is False and res["error_type"] == "unsupported", res
    assert calls == []


def test_retention_classic_normalized_match():
    mod = _require()
    _setup(mod, catalog=[], key=None, config={"model": "\uff53\uff41\uff4e\uff41"}, classic=["sana"])
    rows = mod.PollinationsImageGenProvider().list_models()
    assert any(r["id"] == "sana" for r in rows), rows


def test_retention_svg_marker_keyless_classic_exempt():
    mod = _require()
    entry = {"name": "Vector-X-Pro", "aliases": ["vector-x"], "output_modalities": ["svg"],
             "pricing": {"currency": "pollen", "completionImageTokens": "0.01"}}
    _setup(mod, catalog=[entry], key=None, config={"model": "Vector-X-Pro"}, classic=["vector-x"])
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "Vector-X-Pro")
    assert "SVG output unsupported" not in row["price"], row["price"]
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="Vector-X-Pro")
    assert res["success"] is True, res
    assert calls and "image.pollinations.ai" in calls[0], calls


def test_free_variant_mixed_keeps_pricier_warning():
    mod = _require()
    m = dict(QUEST, name="test/mixed",
             pricing={"currency": "pollen", "completionImageTokens": "0.5"},
             pricing_variants=[{"name": "free", "completionImageTokens": "0"},
                               {"name": "2k", "completionImageTokens": "2"}])
    label = mod._price_label(m, False, True)
    assert "(free variant)" in label, label
    assert "(variants may cost more)" in label, label


def test_svg_mime_modality_detected():
    mod = _require()
    m = {"name": "test/mime-svg", "description": "clean", "output_modalities": ["image/svg+xml"]}
    assert mod._is_svg_product(m) is True


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