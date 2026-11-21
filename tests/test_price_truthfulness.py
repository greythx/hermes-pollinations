"""Regression tests for price truthfulness (v1.7.0 T5).

Covers L2 (pricing_adjustments fees in the estimate/label), L3/F8 (the
"from" floor marker is unit-independent), L4 (text-token-only models class
as per-token, not unpublished), R2-P6 (promptImageTokens input-billing hint)
and R2-P5 (_text renders newlines as spaces, not fused words).
Stdlib-only: run ``python3 tests/test_price_truthfulness.py``.
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
    spec = importlib.util.spec_from_file_location("pollinations_price_truthfulness", ROOT / "__init__.py")
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


def _model(price: "str | None" = "0.03", **extra):
    m = {"name": "test/priced", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": price}}
    m.update(extra)
    return m


def _adj(price, unit="generation", currency="pollen", kind="image", quantity=1):
    return {"name": "fee", "kind": kind, "price": price, "currency": currency,
            "label": "fee", "quantity": quantity, "unit": unit}


def test_flat_fee_included_in_value_and_label():
    mod = _require()
    m = _model("0.03", pricing_adjustments=[_adj("0.04")])
    assert abs(mod._price_value(m) - 0.07) < 1e-12, mod._price_value(m)
    label = mod._price_label(m, False, True)
    assert "0.07 pollen" in label, label
    assert "fees extra" not in label, label


def test_megapixel_fee_hint_and_estimate():
    mod = _require()
    m = _model("0.00000002", flat_rate=False,
               pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}},
               pricing_adjustments=[_adj("0.01")])
    expected = 0.00000002 * 1000000 * mod._TYPICAL_MP + 0.01
    assert abs(mod._price_value(m) - expected) < 1e-9, mod._price_value(m)
    label = mod._price_label(m, False, True)
    assert "charged per megapixel" in label, label
    assert "fees extra" in label, label


def test_unconvertible_fees_hint_only():
    mod = _require()
    for unit, currency in (("GPU second", "pollen"), ("USD", "pollen"), ("generation", "USD")):
        m = _model("0.03", pricing_adjustments=[_adj("1", unit=unit, currency=currency)])
        assert abs(mod._price_value(m) - 0.03) < 1e-12, (unit, currency, mod._price_value(m))
        label = mod._price_label(m, False, True)
        assert "fees extra" in label, (unit, currency, label)
    video_fee = _model("0.03", pricing_adjustments=[_adj("1", kind="video")])
    assert "fees extra" not in mod._price_label(video_fee, False, True)


def test_text_token_only_model_classes_per_token():
    mod = _require()
    m = {"name": "test/text-billed", "category": "image", "paid_only": True,
         "pricing": {"currency": "pollen", "promptTextTokens": "0.000004",
                     "completionTextTokens": "0.00002"}}
    assert mod._price_class(m) == 1, mod._price_class(m)
    assert mod._price_value(m) == 0.00002, mod._price_value(m)
    label = mod._price_label(m, False, True)
    assert "charged per token" in label, label
    assert "No published price" not in label, label


def test_from_marker_unit_independent():
    mod = _require()
    mp = _model("0.00000002", flat_rate=False,
                pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}},
                pricing_variants=[{"name": "edit"}])
    assert mod._price_label(mp, False, True).startswith("from charged per megapixel"), \
        mod._price_label(mp, False, True)
    flat = _model("0.002", pricing_variants=[{"name": "2k"}])
    assert mod._price_label(flat, False, True).startswith("from 0.002 pollen")
    zero = _model("0", pricing_variants=[{"name": "edit"}])
    assert "variants may cost more" in mod._price_label(zero, False, True)
    unpublished = _model(None, pricing_variants=[{"name": "edit"}])
    del unpublished["pricing"]["completionImageTokens"]
    assert mod._price_label(unpublished, False, True).startswith("No published price")


def test_input_billing_hint():
    mod = _require()
    m = _model("0.002")
    m["pricing"]["promptImageTokens"] = "0.008"
    assert "input billed" in mod._price_label(m, False, True)
    assert "input billed" not in mod._price_label(m, False, False)
    assert "input billed" not in mod._price_label(_model("0.002"), False, True)


def test_text_newlines_become_spaces():
    mod = _require()
    assert mod._text("FLUX.2 Pro\n\nQuest Accepted") == "FLUX.2 Pro Quest Accepted"
    assert mod._text("a\r\nb\tc") == "a b c"
    assert mod._text("  padded  ") == "padded"


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