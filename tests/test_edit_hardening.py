"""Regression tests for edit-path hardening (audit m2, m5, m6, m11, m13, F10, F7).

Edit prompts obey the live spec's prompt limit; duplicate sources are deduped;
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
                 "save_b64_image", "MAX_INPUT_IMAGE_BYTES", "MAX_EDIT_PAYLOAD_BYTES",
                 "_spec_limits"):
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
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "_to_data_url", lambda v, limit=1000000: f"data:image/png;base64,{PNG_B64}")
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


def test_small_second_reference_fits_after_large_first():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_INPUT_IMAGE_BYTES", 10000)
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 3000)
    calls = []

    def _encode(value, limit=10000):
        calls.append(limit)
        if len(calls) == 1:
            return "data:image/png;base64," + "A" * 1380
        return "data:image/png;base64," + "A" * max(0, (limit * 3) // 4 - 20)

    setattr(mod, "_to_data_url", _encode)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"])
    assert res["success"] is True, res
    assert len(calls) == 2, calls
    assert len(res["image"]) > 0


def test_payload_rejected_when_budget_exhausted():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_INPUT_IMAGE_BYTES", 1000)
    # Sized against the SERIALIZED fixed-field overhead (~120B): the first source encodes at
    # the 1000B input cap, the second's remaining budget is below the minimum source size,
    # so it is refused BEFORE encoding.
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 1600)
    calls = []

    def _encode(value, limit=1000):
        calls.append(limit)
        return "data:image/png;base64," + "A" * 1380

    setattr(mod, "_to_data_url", _encode)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"])
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert len(calls) == 1, calls


def test_encoder_limit_bounded_by_remaining_budget():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_INPUT_IMAGE_BYTES", 10000)
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 3000)
    limits = []

    def _encode(value, limit=10000):
        limits.append(limit)
        return "data:image/png;base64," + "A" * max(0, (limit * 3) // 4 - 20)

    setattr(mod, "_to_data_url", _encode)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"])
    assert res["success"] is True, res
    assert len(limits) == 2, limits
    assert all(0 < lim <= 10000 for lim in limits), limits
    assert limits[1] < limits[0], limits


def test_prompt_counted_against_payload_cap():
    mod = _require()
    _edit_setup(mod)
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 2000)
    calls = []
    setattr(mod, "_to_data_url", lambda v, limit=1000000: calls.append(v) or "data:image/png;base64,AAAA")
    # 1900 prompt bytes + serialized framing exceed the 2000B cap before any source is read.
    res = mod.PollinationsImageGenProvider().generate(
        "x" * 1900, model="test/edit-model", image_url="https://example.com/a.png")
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
    for bad in (True, 2**31, "42", None):
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/edit-model", image_url="https://example.com/a.png", seed=bad)
        assert res["success"] is True, res
        seed = captured["payload"]["seed"]
        # type() is int, not isinstance: True is an int subclass and must not pass as a seed.
        assert type(seed) is int and 0 <= seed <= 2147483647, (bad, seed)


def test_usage_recorded_even_when_save_fails():
    mod = _require()
    usage = {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
    _edit_setup(mod, post_response={"data": [{"b64_json": PNG_B64}], "usage": usage})
    recorded = []
    setattr(mod, "record_token_usage",
            lambda u, *, model, provider, base_url=None: recorded.append(u))

    def _boom(b64, *, prefix="image", extension="png"):
        raise OSError("disk full")

    setattr(mod, "save_b64_image", _boom)
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "io_error", res
    # The upstream edit completed and billed tokens; a local save failure must not lose that.
    assert recorded == [usage], recorded


def test_edit_inband_error_url_redacted():
    mod = _require()
    _edit_setup(mod, post_response={"error": {"message":
        "fetch failed https://cdn.example/img?sig=SUPERSECRET"}})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert "SUPERSECRET" not in res["error"], res
    assert "[url]" in res["error"], res


def test_edit_inband_error_escaped_url_redacted():
    mod = _require()
    _edit_setup(mod, post_response={"error": {"message":
        "fetch failed https:\\/\\/cdn.example/img?sig=ESCAPED"}})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert "ESCAPED" not in res["error"], res
    assert "[url]" in res["error"], res


def test_edit_inband_error_string_redacted():
    mod = _require()
    _edit_setup(mod, post_response={"error": "failed https://x.example/i?sig=STR"})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert "STR" not in res["error"], res
    assert "[url]" in res["error"], res


def test_published_zero_cap_disables_edits():
    mod = _require()
    entry, _ = _edit_setup(mod, entry_extra={"max_reference_images": 0})
    assert mod._reference_cap(entry) == 0
    assert mod._supports_edits(entry) is False
    prov = mod.PollinationsImageGenProvider()
    caps = prov.capabilities()
    assert caps["modalities"] == ["text"], caps
    assert caps["max_reference_images"] == 0, caps
    res = prov.generate("edit", model="test/edit-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "unsupported", res


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
