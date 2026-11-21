"""Regression tests for picker/capabilities/default state consistency (v1.7.0 T2).

Covers F2 (capabilities and picker must not advertise what generate refuses),
R2-P2 (default_model returns the resolved configured model for the CLI marker),
R2-P3 (keyed passthrough saved model gets a minimal row), R2-P4 (no keyless
"Free pollen" name-fallback claim for non-classic rows), R2-P7 (retained paid
row carries a disabled hint while the toggle is off).
Stdlib-only: run ``python3 tests/test_state_consistency.py``.
"""

from __future__ import annotations

import base64
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
    spec = importlib.util.spec_from_file_location("pollinations_state_consistency", ROOT / "__init__.py")
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


PAID_EDIT = {"name": "test/paid-edit", "category": "image", "paid_only": True,
             "supported_endpoints": ["/v1/images/edits"], "resolutions": ["1k", "2k"],
             "max_reference_images": 4,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
FREE_NAME = {"name": "test/free-named", "category": "image", "paid_only": False,
             "title": "Totally FREE model",
             "pricing": {"currency": "pollen"}}


def _setup(mod, *, catalog, classic=None, key: "str | None" = "sk_dummy", include_paid=False,
           config=None):
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in catalog])
    setattr(mod, "_classic_models", lambda: list(classic or []))
    setattr(mod, "_include_paid_models", lambda pid: include_paid)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})


def test_capabilities_respect_paid_gate():
    mod = _require()
    _setup(mod, catalog=[PAID_EDIT], config={"model": "test/paid-edit"}, include_paid=False)
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps["modalities"] == ["text"], caps
    assert caps["max_reference_images"] == 0, caps
    assert not caps.get("creative_controls"), caps
    _setup(mod, catalog=[PAID_EDIT], config={"model": "test/paid-edit"}, include_paid=True)
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"], caps
    assert caps["max_reference_images"] == 4, caps
    assert caps.get("creative_controls") == ["resolution"], caps


def test_keyed_paid_classic_row_hidden_while_toggle_off():
    mod = _require()
    paid_classic = dict(PAID_EDIT, name="test/paid-classic")
    _setup(mod, catalog=[paid_classic, QUEST], classic=["test/paid-classic"], include_paid=False)
    rows = mod.PollinationsImageGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert "test/paid-classic" not in ids, ids
    assert "test/quest-model" in ids, ids
    # Keyless the same model IS runnable on the free classic route: keep it.
    _setup(mod, catalog=[paid_classic, QUEST], classic=["test/paid-classic"], key=None)
    rows = mod.PollinationsImageGenProvider().list_models()
    assert "test/paid-classic" in [r["id"] for r in rows], rows


def test_default_model_returns_resolved_configured():
    mod = _require()
    alias_entry = dict(QUEST, name="test/canonical", aliases=["short-alias"])
    other = dict(QUEST, name="test/other",
                 pricing={"currency": "pollen", "completionImageTokens": "0.001"})
    # The derived default would be test/other (first Quest-eligible); the configured alias
    # must win so the CLI marker lands on the row actually in use.
    _setup(mod, catalog=[other, alias_entry], config={"model": "short-alias"})
    assert mod.PollinationsImageGenProvider().default_model() == "test/canonical"
    _setup(mod, catalog=[alias_entry], classic=["sana"], key=None)
    assert mod.PollinationsImageGenProvider().default_model() == "sana"


def test_keyed_passthrough_saved_model_gets_row():
    mod = _require()
    _setup(mod, catalog=[QUEST], config={"pollinations": {"model": "brand-new-xyz"}})
    rows = mod.PollinationsImageGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert "brand-new-xyz" in ids, ids
    row = next(r for r in rows if r["id"] == "brand-new-xyz")
    assert "No published price" in row["price"], row
    # Keyless, an unknown non-classic id cannot run: no row.
    _setup(mod, catalog=[QUEST], key=None, config={"pollinations": {"model": "brand-new-xyz"}})
    rows = mod.PollinationsImageGenProvider().list_models()
    assert "brand-new-xyz" not in [r["id"] for r in rows], rows


def test_no_keyless_free_claim_from_name_alone():
    mod = _require()
    for key in (None, "sk_dummy"):
        _setup(mod, catalog=[FREE_NAME], key=key, config={"model": "test/free-named"})
        rows = mod.PollinationsImageGenProvider().list_models()
        row = next(r for r in rows if r["id"] == "test/free-named")
        assert "Free pollen" not in row["price"], (key, row["price"])
        assert "No published price" in row["price"], (key, row["price"])


def test_retained_paid_row_carries_disabled_hint():
    mod = _require()
    _setup(mod, catalog=[PAID_EDIT, QUEST], config={"model": "test/paid-edit"}, include_paid=False)
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/paid-edit")
    assert "disabled" in row["price"].lower(), row["price"]
    _setup(mod, catalog=[PAID_EDIT, QUEST], config={"model": "test/paid-edit"}, include_paid=True)
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/paid-edit")
    assert "disabled" not in row["price"].lower(), row["price"]


def test_future_aspect_ratio_degrades_to_square():
    mod = _require()
    _setup(mod, catalog=[QUEST])
    orig_aspect = mod.resolve_aspect_ratio
    orig_get_image = getattr(mod, "_http_get_image", None)
    setattr(mod, "resolve_aspect_ratio", lambda a: "wide")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
        "AAAABJRU5ErkJggg==")
    setattr(mod, "_http_get_image", lambda url, *, headers=None, timeout=120.0: (png, None))
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    try:
        res = mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model")
        assert res["success"] is True, res
        assert res["width"] == 1024 and res["height"] == 1024, res
    finally:
        setattr(mod, "resolve_aspect_ratio", orig_aspect)
        if orig_get_image is not None:
            setattr(mod, "_http_get_image", orig_get_image)


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