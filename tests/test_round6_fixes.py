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
