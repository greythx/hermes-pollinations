"""Regression tests for the paid-model invocation gate (audit M1).

The ``include_paid_models`` toggle must gate invocation, not just picker
visibility: with a key set and the toggle off, a paid-only catalog model is
refused before any request or source-image fetch. Keyless behavior, the
unknown-id passthrough and the picker gate are preserved. Stdlib-only: run
``python3 tests/test_paid_gate.py`` (or pytest, when installed).
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
    spec = importlib.util.spec_from_file_location("pollinations_paid_gate", ROOT / "__init__.py")
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


PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
           "AAAABJRU5ErkJggg==")

PAID = {"name": "test/paid-model", "category": "image", "paid_only": True,
        "pricing": {"currency": "pollen", "completionImageTokens": "0.5"},
        "supported_endpoints": ["/v1/images/edits"]}
QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False,
         "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def _setup(mod, *, key: "str | None" = "sk_dummy", include_paid=False):
    mod._api_key = lambda: key
    mod.load_image_gen_config = lambda: {}
    mod._catalog = lambda: [dict(PAID), dict(QUEST)]
    mod._classic_models = lambda: []
    mod._include_paid_models = lambda pid: include_paid
    calls = []

    def _get(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        raise AssertionError("network reached")

    def _get_image(url, *, headers=None, timeout=120.0):
        calls.append(("GET", url))
        raise AssertionError("network reached")

    def _post(url, *, headers, payload, timeout=120.0):
        calls.append(("POST", url))
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _get)
    setattr(mod, "_http_get_image", _get_image)
    setattr(mod, "_http_post_json", _post)
    return calls


def test_paid_model_refused_when_toggle_off():
    mod = _require()
    calls = _setup(mod, include_paid=False)
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/paid-model")
    assert res["success"] is False, res
    assert res["error_type"] == "paid_model_disabled", res
    assert "paid" in res["error"].lower(), res
    assert calls == [], calls


def test_paid_edit_refused_before_source_fetch():
    mod = _require()
    calls = _setup(mod, include_paid=False)
    res = mod.PollinationsImageGenProvider().generate(
        "make it green", model="test/paid-model",
        image_url=f"data:image/png;base64,{PNG_B64}")
    assert res["success"] is False, res
    assert res["error_type"] == "paid_model_disabled", res
    assert calls == [], calls


def test_paid_model_allowed_when_toggle_on():
    mod = _require()
    calls = _setup(mod, include_paid=True)
    mod.PollinationsImageGenProvider().generate("a cat", model="test/paid-model")
    assert len(calls) == 1 and calls[0][0] == "GET", calls
    assert "model=test%2Fpaid-model" in calls[0][1], calls


def test_quest_model_unaffected_by_gate():
    mod = _require()
    calls = _setup(mod, include_paid=False)
    mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model")
    assert len(calls) == 1 and calls[0][0] == "GET", calls
    assert "model=test%2Fquest-model" in calls[0][1], calls


def test_keyless_paid_model_reports_model_unavailable():
    mod = _require()
    calls = _setup(mod, key=None, include_paid=False)
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/paid-model")
    assert res["success"] is False, res
    assert res["error_type"] == "model_unavailable", res
    assert calls == [], calls


def test_unknown_id_passthrough_not_gated():
    mod = _require()
    calls = _setup(mod, include_paid=False)
    mod.PollinationsImageGenProvider().generate("a cat", model="brand-new-model")
    assert len(calls) == 1 and calls[0][0] == "GET", calls
    assert "model=brand-new-model" in calls[0][1], calls


def test_picker_still_hides_paid_when_off():
    mod = _require()
    _setup(mod, include_paid=False)
    rows = mod.PollinationsImageGenProvider().list_models()
    assert [r["id"] for r in rows] == ["test/quest-model"], rows


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
