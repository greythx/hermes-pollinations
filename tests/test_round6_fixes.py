"""Round-6 residual fixes: live variant floors, paid classic spelling, secret tails.

Every test is stdlib-only and loads a fresh copy of the plugin source; all network seams are
refusing fakes except where a captured fake PNG is needed to observe the selected route.
No live generation/edit request is made. Run ``python tests/test_round6_fixes.py``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import sys
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
            raise RuntimeError("Hermes core not importable; plugin cannot load")
    spec = importlib.util.spec_from_file_location("pollinations_round6_fixes", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()

# Mirrors the live catalog shape: x-ai/grok-imagine-image-2.0 publishes base 0.06 with
# pricing_variants[].pricing.completionImageTokens = 0.04. No live variant publishes a
# top-level completionImageTokens, but legacy fixtures do, so both shapes must work.
LIVE_NESTED = {
    "name": "x-ai/grok-imagine-image-2.0",
    "category": "image",
    "paid_only": True,
    "flat_rate": True,
    "pricing": {"currency": "pollen", "completionImageTokens": "0.06"},
    "pricing_variants": [
        {"name": "low_1k", "pricing": {"completionImageTokens": "0.04"}},
    ],
}


def test_variant_floor_reads_nested_live_shape():
    # A live-shaped cheaper variant must become the advertised floor and the sort value.
    label = MOD._price_label(LIVE_NESTED, False, True)
    assert label.startswith("from 0.04 pollen"), label
    assert MOD._price_value(LIVE_NESTED) == 0.04, MOD._price_value(LIVE_NESTED)
    # Legacy top-level variants stay supported (older catalog snapshots and test fixtures).
    legacy = dict(LIVE_NESTED, pricing_variants=[
        {"name": "low_1k", "completionImageTokens": "0.04"}])
    assert MOD._price_value(legacy) == 0.04, MOD._price_value(legacy)
    # A costlier variant must not invent a cheaper floor.
    dearer = dict(LIVE_NESTED, pricing_variants=[
        {"name": "2k", "pricing": {"completionImageTokens": "0.09"}}])
    assert MOD._price_value(dearer) == 0.06, MOD._price_value(dearer)


PAID_CLASSIC = {"name": "Sana", "category": "image", "paid_only": True,
                "pricing": {"currency": "pollen", "completionImageTokens": "0"}}
FREE_CLASSIC = {"name": "Sana", "category": "image", "paid_only": False, "flat_rate": True,
                "pricing": {"currency": "pollen", "completionImageTokens": "0"}}


def _png_bytes() -> bytes:
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
            + chunk(b"IEND", b""))


def _setup_gate(mod, *, catalog, config=None, include_paid=False):
    mod._api_key = lambda: "sk_dummy"
    mod.load_image_gen_config = lambda: dict(config or {})
    mod._catalog = lambda: [dict(m) for m in catalog]
    mod._classic_models = lambda: ["sana"]
    mod._include_paid_models = lambda pid: include_paid
    mod._spec_limits = lambda: {"edit_prompt_max": 32000, "seed_min": -1,
                                "seed_max": 2147483647}
    mod.save_b64_image = lambda b64, *, prefix="image", extension="png": "/tmp/t.png"
    os.environ.pop(mod.MODEL_ENV, None)
    calls, captured = [], {}

    def _get(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        raise AssertionError("network reached")

    def _get_image(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        captured["url"], captured["headers"] = url, headers or {}
        return _png_bytes(), None

    def _post(url, *, headers, payload, timeout=120.0):
        calls.append(("POST", url))
        raise AssertionError("network reached")

    mod._http_get_bytes, mod._http_get_image, mod._http_post_json = _get, _get_image, _post
    return calls, captured


def test_classic_paid_canonical_gated_both_paths():
    # A canonical paid-only catalog name that is also classic-served must keep its canonical
    # spelling when addressed by the endpoint's case variant, or the paid gate cannot see it.
    for use_config in (False, True):
        calls, captured = _setup_gate(MOD, catalog=[PAID_CLASSIC],
                                      config={"model": "sana"} if use_config else {})
        assert MOD._resolve_model("sana", [dict(PAID_CLASSIC)]) == "Sana"
        provider = MOD.PollinationsImageGenProvider()
        res = provider.generate("a cat") if use_config else provider.generate("a cat", model="sana")
        assert res["success"] is False, (use_config, res)
        assert res["error_type"] == "paid_model_disabled", (use_config, res)
        assert calls == [], (use_config, calls)
    # A genuinely free classic id still resolves to its endpoint spelling and runs.
    calls, captured = _setup_gate(MOD, catalog=[FREE_CLASSIC])
    assert MOD._resolve_model("sana", [dict(FREE_CLASSIC)]) == "Sana"
    res = MOD.PollinationsImageGenProvider().generate("a cat", model="sana")
    assert res["success"] is True, res
    assert res["mode"] == "quest", res
    assert captured["url"].startswith("https://gen.pollinations.ai/image/"), captured
    assert captured["headers"].get("Authorization") == "Bearer sk_dummy", captured


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
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
