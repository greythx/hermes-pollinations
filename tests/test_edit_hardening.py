"""Regression tests for edit-path hardening (audit m2, m5, m6, m11, m13, F10, F7).

Edit prompts obey the spec's 32000-char limit; duplicate sources are deduped;
unpublished reference caps default conservatively; the payload budget is
accounted BEFORE base64 blow-up and includes the prompt; unsafe response URLs
are refused; the usage block reaches the core's token accounting; a caller seed
is honored. Stdlib-only: run ``python3 tests/test_edit_hardening.py``.
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
    spec = importlib.util.spec_from_file_location("pollinations_edit_hardening", ROOT / "__init__.py")
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

_ORIG: dict = {}


def _edit_setup(mod, *, entry_extra=None, post_response=None):
    # Snapshot originals once so every test starts from pristine module state.
    for name in ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
                 "_include_paid_models", "_to_data_url", "_http_post_json", "_http_get_bytes",
                 "_http_get_image", "cache_url_best_effort", "record_token_usage",
                 "save_b64_image", "MAX_INPUT_IMAGE_BYTES", "MAX_EDIT_PAYLOAD_BYTES"):
        _ORIG.setdefault(name, getattr(mod, name, None))
        if _ORIG[name] is not None:
            setattr(mod, name, _ORIG[name])
    entry = {"name": "test/edit-model", "category": "image", "paid_only": False,
             "supported_endpoints": ["/v1/images/edits"], "flat_rate": True,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    entry.update(entry_extra or {})
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: [entry])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_to_data_url", lambda v: f"data:image/png;base64,{PNG_B64}")
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")

    def _no_get(url, *, headers=None, timeout=120.0):
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _no_get)
    captured = {}

    def _post(url, *, headers, payload, timeout=120.0):
        captured["url"] = url
        captured["payload"] = payload
        return post_response if post_response is not None else {"data": [{"b64_json": PNG_B64}]}

    setattr(mod, "_http_post_json", _post)
    return entry, captured


def _refs(n):
    return [f"https://example.com/img{i}.png" for i in range(n)]


def test_edit_prompt_over_spec_limit_rejected():
    mod = _require()
    _, captured = _edit_setup(mod)
    res = mod.PollinationsImageGenProvider().generate(
        "x" * 32001, model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert "payload" not in captured, captured


def test_duplicate_sources_deduped():
    mod = _require()
    _, captured = _edit_setup(mod)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/a.png", "https://example.com/b.png"])
    assert res["success"] is True, res
    assert len(captured["payload"]["image"]) == 2, captured["payload"]


def test_unpublished_reference_cap_defaults_conservative():
    mod = _require()
    _, captured = _edit_setup(mod)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=_refs(6))
    assert res["success"] is True, res
    assert len(captured["payload"]["image"]) == 4, captured["payload"]
    assert res["dropped_references"] == 3, res


def test_published_reference_cap_honored():
    mod = _require()
    _, captured = _edit_setup(mod, entry_extra={"max_reference_images": 2})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=_refs(6))
    assert res["success"] is True, res
    assert len(captured["payload"]["image"]) == 2, captured["payload"]
    assert res["dropped_references"] == 5, res


def test_payload_cap_rejects_before_encoding_next():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_INPUT_IMAGE_BYTES", 1000)
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 3000)
    calls = []

    def _encode(value):
        calls.append(value)
        return "data:image/png;base64," + "A" * 1380

    setattr(mod, "_to_data_url", _encode)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"])
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert len(calls) == 1, calls


def test_prompt_counted_against_payload_cap():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 2000)
    calls = []
    setattr(mod, "_to_data_url", lambda v: calls.append(v) or "data:image/png;base64,AAAA")
    res = mod.PollinationsImageGenProvider().generate(
        "x" * 1500, model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert calls == [], calls


def test_usage_recorded_after_edit():
    mod = _require()
    usage = {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
    _edit_setup(mod, post_response={"data": [{"b64_json": PNG_B64}], "usage": usage})
    recorded = []
    setattr(mod, "record_token_usage",
            lambda u, *, model, provider, base_url=None: recorded.append((u, model, provider)))
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is True, res
    assert recorded == [(usage, "test/edit-model", "pollinations")], recorded


def test_unsafe_edit_response_url_rejected():
    mod = _require()
    _edit_setup(mod, post_response={"data": [{"url": "http://169.254.169.254/latest/meta-data/"}]})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_response", res
    assert res["image"] is None, res


def test_safe_edit_response_url_cached():
    mod = _require()
    _edit_setup(mod, post_response={"data": [{"url": "https://example.com/img.png"}]})
    seen = {}

    def _cache(url, *, prefix, label, log=None, on_fail=None):
        seen["url"] = url
        seen["on_fail"] = on_fail
        return "/tmp/cached.png"

    setattr(mod, "cache_url_best_effort", _cache)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is True, res
    assert res["image"] == "/tmp/cached.png", res
    assert res["remote"] is True, res
    assert seen["url"] == "https://example.com/img.png", seen
    assert callable(seen["on_fail"]), seen


def test_edit_seed_kwarg_honored():
    mod = _require()
    _, captured = _edit_setup(mod)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png", seed=42)
    assert res["success"] is True, res
    assert captured["payload"]["seed"] == 42, captured["payload"]


def test_edit_invalid_seed_falls_back_to_random():
    mod = _require()
    _, captured = _edit_setup(mod)
    for bad in (True, -1, 2**31, "42", None):
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/edit-model", image_url="https://example.com/a.png", seed=bad)
        assert res["success"] is True, res
        seed = captured["payload"]["seed"]
        assert isinstance(seed, int) and 0 <= seed <= mod.SEED_MAX, (bad, seed)


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
