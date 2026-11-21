"""Regression tests for catalog price parsing.

The Pollinations catalog publishes every ``pricing`` rate as a string — its
OpenAPI contract types the values ``string`` — while older snapshots sent
numbers. Both must read as the same amount, and a boolean must never read as
one. Stdlib-only: run ``python3 tests/test_catalog_pricing.py`` (or pytest,
when installed). Skips where Hermes core is not importable, since the plugin
module imports it at load time.
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
    spec = importlib.util.spec_from_file_location("pollinations_under_test", ROOT / "__init__.py")
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


def _model(price, **extra):
    """A catalog-shaped image model entry carrying one published price."""
    entry = {"name": "publisher/model", "category": "image",
             "pricing": {"currency": "pollen", "completionImageTokens": price}}
    entry.update(extra)
    return entry


def test_string_price_renders_as_amount():
    mod = _require()
    label = mod._price_label(_model("0.002", flat_rate=True), False, True)
    assert label == "0.002 pollen · needs key", label


def test_numeric_price_still_renders():
    mod = _require()
    label = mod._price_label(_model(0.002, flat_rate=True), False, True)
    assert label == "0.002 pollen · needs key", label


def test_string_price_orders_by_amount():
    mod = _require()
    assert mod._price_value(_model("0.04")) == 0.04
    assert mod._price_value(_model("0.0000075")) == 0.0000075


def test_zero_string_price_is_free():
    mod = _require()
    model = _model("0", flat_rate=True)
    assert mod._advertises_free(model) is True
    assert mod._price_label(model, False, True).startswith("Free"), \
        mod._price_label(model, False, True)


def test_boolean_false_is_not_a_free_price():
    mod = _require()
    assert mod._advertises_free(_model(False)) is False
    assert mod._format_price(False) == "No published price"


def test_boolean_true_is_not_one_pollen():
    mod = _require()
    assert mod._format_price(True) == "No published price"
    assert mod._price_value(_model(True)) == 0.0


def test_unparseable_prices_read_as_unpublished():
    mod = _require()
    for raw in ("abc", "", "   ", None, {}, [], "NaN", "Infinity",
                float("nan"), float("inf"), -1, "-0.5"):
        assert mod._format_price(raw) == "No published price", repr(raw)


def test_price_variants_mark_a_floor():
    mod = _require()
    model = _model("0.05", flat_rate=True, pricing_variants=[{"resolution": "2k"}])
    label = mod._price_label(model, False, True)
    assert label.startswith("from 0.05 pollen"), label


def test_per_unit_billing_labels_survive_string_prices():
    mod = _require()
    per_mp = _model("0.00000002", pricing_units={
        "completionImageTokens": {"unit": "megapixel", "quantity": 1000000}})
    assert "per megapixel · needs key" in mod._price_label(per_mp, False, True)
    per_token = _model("0.0000075", flat_rate=False)
    assert mod._price_label(per_token, False, True) == "charged per token · needs key"


def test_keyless_free_classic_label_unchanged():
    mod = _require()
    label = mod._price_label(_model("0.002"), True, False)
    assert label == "Free · watermark · reduced size · no key needed", label


def test_whitespace_padded_string_price_renders():
    mod = _require()
    label = mod._price_label(_model("  0.002  ", flat_rate=True), False, True)
    assert label == "0.002 pollen · needs key", label


def test_lenient_float_spellings_are_not_prices():
    mod = _require()
    # float() alone would read underscores and non-ASCII digits as amounts.
    for raw in ("1_000", "١٢", "0x10", "1,000", "1e", "--1"):
        assert mod._format_price(raw) == "No published price", repr(raw)


def test_huge_integer_price_is_unpublished():
    mod = _require()
    assert mod._format_price(10**400) == "No published price"


def test_tiny_positive_price_does_not_round_to_zero():
    mod = _require()
    label = mod._format_price(1e-13)
    assert label == "0.0000000000001 pollen", label


def test_negative_string_price_is_unpublished():
    mod = _require()
    label = mod._price_label(_model("-1", flat_rate=False), False, True)
    assert label == "No published price · needs key", label


def test_underflowed_string_price_is_not_free():
    mod = _require()
    # float("1e-400") underflows to 0.0 without raising; that is not a free price.
    for raw in ("1e-400", "-1e-400"):
        assert mod._format_price(raw) == "No published price", repr(raw)
        assert mod._advertises_free(_model(raw)) is False, repr(raw)
    label = mod._price_label(_model("1e-400", flat_rate=True), False, True)
    assert label == "No published price · needs key", label
    assert mod._format_price("0e5") == "Free"  # a real zero stays free


def test_large_prices_keep_integer_zeros():
    mod = _require()
    assert mod._format_price(1e20) == "1" + "0" * 20 + " pollen"
    assert mod._format_price("1e20") == "1" + "0" * 20 + " pollen"
    assert mod._format_price(1e30) == "1" + "0" * 23 + "… pollen"


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
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
