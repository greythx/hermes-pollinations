"""Regression tests for per-image cost ordering (audit M2).

The picker's cheapest-first promise must compare estimated per-image costs:
``pricing_units.quantity`` scales the published rate (megapixel billing uses
quantity 1e6), per-token models cannot be compared to per-image rates and sort
after them, and unpublished prices sort last instead of first. Stdlib-only:
run ``python3 tests/test_price_sort.py`` (or pytest, when installed).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
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
    spec = importlib.util.spec_from_file_location("pollinations_price_sort", ROOT / "__init__.py")
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


def _model(name, price, **extra):
    entry = {"name": name, "category": "image", "paid_only": False,
             "pricing": {"currency": "pollen", "completionImageTokens": price}}
    entry.update(extra)
    return entry


def test_megapixel_quantity_normalized_to_per_image():
    mod = _require()
    m = _model("mp", "0.00000002", flat_rate=True,
               pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}})
    value = mod._price_value(m)
    # 2e-8 x 1e6 = 0.02 per megapixel; the plugin's own sizes are ~1 MP per image.
    assert 0.019 < value < 0.022, value
    assert mod._price_class(m) == 0, mod._price_class(m)


def test_per_image_price_unchanged():
    mod = _require()
    m = _model("img", "0.002", flat_rate=True)
    assert mod._price_value(m) == 0.002
    assert mod._price_class(m) == 0


def test_per_token_rate_kept_but_classed_separately():
    mod = _require()
    m = _model("tok", "0.0000075", flat_rate=False)
    assert mod._price_value(m) == 0.0000075
    assert mod._price_class(m) == 1


def test_unpublished_price_classed_last():
    mod = _require()
    m = {"name": "unknown", "category": "image"}
    assert mod._price_class(m) == 2
    assert mod._price_value(m) == 0.0


def test_zero_price_is_classed_free():
    mod = _require()
    m = _model("free", "0", flat_rate=False)
    assert mod._price_class(m) == 0
    assert mod._price_value(m) == 0.0


def test_bool_quantity_is_not_a_multiplier():
    mod = _require()
    m = _model("mp", "0.00000002", flat_rate=True,
               pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": True}})
    assert mod._price_value(m) < 1e-6, mod._price_value(m)


def test_string_quantity_parsed():
    mod = _require()
    m = _model("mp", "0.00000002", flat_rate=True,
               pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": "1000000"}})
    assert 0.019 < mod._price_value(m) < 0.022, mod._price_value(m)


def test_picker_order_free_perimage_permegapixel_pertoken_unknown():
    mod = _require()
    catalog = [
        _model("a-unknown", None),
        _model("b-token", "0.0000075", flat_rate=False),
        _model("c-megapixel", "0.00000002", flat_rate=True,
               pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}}),
        _model("d-image", "0.002", flat_rate=True),
        _model("e-free", "0", flat_rate=True),
    ]
    catalog[0].pop("pricing")
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: catalog)
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    rows = mod.PollinationsImageGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert ids == ["e-free", "d-image", "c-megapixel", "b-token", "a-unknown"], ids


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
