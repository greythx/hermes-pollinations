"""Regression tests for live API limits read from the published OpenAPI spec.

The edit prompt maxLength and the seed range are spec facts, not plugin
constants: they are fetched live (TTL-cached) and enforced when known. When
the spec cannot be reached and nothing is cached, the plugin skips its
client-side pre-checks and lets the server enforce its own contract; caller
seeds pass through verbatim. Stdlib-only: run
``python3 tests/test_spec_limits.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
import time
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_spec_limits"


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
    spec = importlib.util.spec_from_file_location("pollinations_spec_limits", ROOT / "__init__.py")
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
ENTRY = {"name": "test/spec-model", "category": "image", "paid_only": False,
         "supported_endpoints": ["/v1/images/edits"], "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
KNOWN = {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647}
UNKNOWN = {"edit_prompt_max": None, "seed_min": None, "seed_max": None}


def _spec_doc(prompt_max: object = 32000, seed_min: object = -1,
              seed_max: object = 2147483647, use_ref=True):
    """A minimal OpenAPI-shaped document carrying the two limits the plugin reads."""
    body = {"type": "object", "required": ["prompt", "image"],
            "properties": {"prompt": {"type": "string", "maxLength": prompt_max}}}
    schema = {"$ref": "#/components/schemas/CreateImageEditRequest"} if use_ref else body
    doc = {"paths": {
        "/image/{prompt}": {"get": {"parameters": [
            {"name": "width", "schema": {"type": "integer"}},
            {"name": "seed", "schema": {"type": "integer",
                                         "minimum": seed_min, "maximum": seed_max}}]}},
        "/v1/images/edits": {"post": {"requestBody": {
            "content": {"application/json": {"schema": schema}}}}}}}
    if use_ref:
        doc["components"] = {"schemas": {"CreateImageEditRequest": body}}
    return doc


_ORIG: dict = {}


def _setup(mod, *, fetch=None):
    """Isolate the module: scratch caches, fake key, no network but the spec fetch."""
    for name in ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
                 "_include_paid_models", "_http_get_json", "_http_get_bytes",
                 "_http_get_image", "_http_post_json", "_to_data_url", "save_b64_image",
                 "_cache_path", "_classic_cache_path", "_spec_cache_path"):
        _ORIG.setdefault(name, getattr(mod, name, None))
        if _ORIG[name] is not None:
            setattr(mod, name, _ORIG[name])
    SCRATCH.mkdir(exist_ok=True)
    for f in SCRATCH.iterdir():
        f.unlink()
    setattr(mod, "_cache_path", lambda: SCRATCH / "catalog.json")
    setattr(mod, "_classic_cache_path", lambda: SCRATCH / "classic.json")
    setattr(mod, "_spec_cache_path", lambda: SCRATCH / "spec.json")
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: [dict(ENTRY)])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_to_data_url", lambda v, limit=1000000: f"data:image/png;base64,{PNG_B64}")
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")

    def _no_get(url, *, headers=None, timeout=120.0):
        raise AssertionError("network reached")

    setattr(mod, "_http_get_bytes", _no_get)
    setattr(mod, "_http_get_image", _no_get)
    captured = {}

    def _post(url, *, headers, payload, timeout=120.0):
        captured["payload"] = payload
        return {"data": [{"b64_json": PNG_B64}]}

    setattr(mod, "_http_post_json", _post)
    calls = {"n": 0}

    def _get_json(url, timeout=30.0):
        calls["n"] += 1
        if fetch is None:
            raise OSError("spec unreachable")
        return fetch

    setattr(mod, "_http_get_json", _get_json)
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})
    return captured, calls


def _t2i_capture(mod, captured):
    def _get_image(url, *, headers=None, timeout=120.0):
        captured["url"] = url
        return base64.b64decode(PNG_B64), None
    setattr(mod, "_http_get_image", _get_image)


def test_spec_limits_parsed_ref_and_inline():
    mod = _require()
    for use_ref in (True, False):
        _setup(mod, fetch=_spec_doc(use_ref=use_ref))
        assert mod._spec_limits() == KNOWN, (use_ref, mod._spec_limits())


def test_spec_limits_fresh_cache_skips_refetch():
    mod = _require()
    _, calls = _setup(mod, fetch=_spec_doc())
    assert mod._spec_limits() == KNOWN
    assert mod._spec_limits() == KNOWN
    assert calls["n"] == 1, calls


def test_spec_failure_serves_stale_and_remembers():
    mod = _require()
    _setup(mod, fetch=_spec_doc())
    mod._spec_limits()  # seed a good cache
    raw = json.loads((SCRATCH / "spec.json").read_text())
    raw["fetched"] = time.time() - mod.SPEC_TTL - 10
    mod.atomic_json_write(SCRATCH / "spec.json", raw)
    attempts = {"n": 0}

    def down(url, timeout=30.0):
        attempts["n"] += 1
        raise OSError("down")

    setattr(mod, "_http_get_json", down)
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})
    lim = UNKNOWN
    for _ in range(5):
        lim = mod._spec_limits()
    assert lim == KNOWN, lim
    assert attempts["n"] == 1, attempts
    assert mod._spec_state()["negative_until"] > time.monotonic()


def test_spec_unreachable_no_cache_degrades():
    mod = _require()
    _setup(mod, fetch=None)
    assert mod._spec_limits() == UNKNOWN


def test_spec_malformed_degrades():
    mod = _require()
    cases = [
        (_spec_doc(prompt_max=True), dict(KNOWN, edit_prompt_max=None)),
        (_spec_doc(prompt_max="32000"), dict(KNOWN, edit_prompt_max=None)),
        (_spec_doc(prompt_max=-5), dict(KNOWN, edit_prompt_max=None)),
        ({"paths": {}}, UNKNOWN),
        (_spec_doc(seed_min=100, seed_max=50), dict(KNOWN, seed_min=None, seed_max=None)),
        ({"paths": {"/image/{prompt}": {"get": {"parameters": [
            {"name": "seed", "schema": {"minimum": True, "maximum": "x"}}]}}}},
         dict(UNKNOWN, edit_prompt_max=None)),
    ]
    for doc, want in cases:
        _setup(mod, fetch=doc)
        got = mod._spec_limits()
        assert got == want, (doc, got, want)


def test_foreign_spec_cache_schema_ignored():
    mod = _require()
    _setup(mod, fetch=_spec_doc())
    mod.atomic_json_write(SCRATCH / "spec.json",
                          {"version": True, "fetched": time.time(),
                           "limits": {"edit_prompt_max": 999, "seed_min": 0, "seed_max": 9}})
    assert mod._spec_limits() == KNOWN


def test_edit_prompt_limit_read_from_spec():
    mod = _require()
    captured, _ = _setup(mod, fetch=_spec_doc(prompt_max=100))
    prov = mod.PollinationsImageGenProvider()
    res = prov.generate("x" * 101, model="test/spec-model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert "100" in res["error"], res
    assert "payload" not in captured, captured
    res = prov.generate("x" * 100, model="test/spec-model", image_url="https://example.com/a.png")
    assert res["success"] is True, res
    assert len(captured["payload"]["prompt"]) == 100


def test_edit_prompt_no_precheck_when_spec_unknown():
    mod = _require()
    captured, _ = _setup(mod, fetch=None)
    res = mod.PollinationsImageGenProvider().generate(
        "x" * 40000, model="test/spec-model", image_url="https://example.com/a.png")
    # No client-side limit is known: the server enforces its own contract.
    assert res["success"] is True, res
    assert len(captured["payload"]["prompt"]) == 40000


def test_seed_minus_one_forwarded_both_routes():
    mod = _require()
    captured, _ = _setup(mod, fetch=_spec_doc())
    prov = mod.PollinationsImageGenProvider()
    res = prov.generate("edit", model="test/spec-model",
                        image_url="https://example.com/a.png", seed=-1)
    assert res["success"] is True, res
    assert captured["payload"]["seed"] == -1, captured["payload"]
    _t2i_capture(mod, captured)
    res = prov.generate("a cat", model="test/spec-model", seed=-1)
    assert res["success"] is True, res
    assert parse_qs(urlsplit(captured["url"]).query)["seed"] == ["-1"], captured["url"]


def test_seed_out_of_live_range_randomized():
    mod = _require()
    captured, _ = _setup(mod, fetch=_spec_doc(seed_max=100))
    seen = {}
    orig_random = mod.random

    class _R:
        def randint(self, a, b):
            seen["range"] = (a, b)
            return 7

    setattr(mod, "random", _R())
    try:
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/spec-model", image_url="https://example.com/a.png", seed=101)
        assert res["success"] is True, res
        assert captured["payload"]["seed"] == 7, captured["payload"]
        assert seen["range"] == (0, 100), seen
    finally:
        setattr(mod, "random", orig_random)


def test_seed_unknown_spec_forwarded_verbatim():
    mod = _require()
    captured, _ = _setup(mod, fetch=None)
    prov = mod.PollinationsImageGenProvider()
    for s in (-1, 2**40, 0):
        res = prov.generate("edit", model="test/spec-model",
                            image_url="https://example.com/a.png", seed=s)
        assert res["success"] is True, res
        assert captured["payload"]["seed"] == s, (s, captured["payload"])
    res = prov.generate("edit", model="test/spec-model",
                        image_url="https://example.com/a.png", seed=True)
    assert res["success"] is True, res
    assert type(captured["payload"]["seed"]) is int, captured["payload"]


def test_spec_without_limits_is_failed_fetch():
    mod = _require()
    _, calls = _setup(mod, fetch={"openapi": "3.1.0", "paths": {}})
    assert mod._spec_limits() == UNKNOWN
    assert not (SCRATCH / "spec.json").exists()
    assert mod._spec_state()["negative_until"] > time.monotonic()
    assert mod._spec_limits() == UNKNOWN
    assert calls["n"] == 1, calls


def test_polluted_all_unknown_cache_ignored():
    mod = _require()
    _setup(mod, fetch=_spec_doc())
    mod.atomic_json_write(SCRATCH / "spec.json",
                          {"version": 1, "fetched": time.time(),
                           "limits": {"edit_prompt_max": None, "seed_min": None,
                                      "seed_max": None}})
    assert mod._spec_limits() == KNOWN


def test_random_seed_respects_positive_lower_bound():
    mod = _require()
    captured, _ = _setup(mod, fetch=_spec_doc(seed_min=100, seed_max=200))
    seen = {}
    orig_random = mod.random

    class _R:
        def randint(self, a, b):
            seen["range"] = (a, b)
            return 150

    setattr(mod, "random", _R())
    try:
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/spec-model", image_url="https://example.com/a.png")
        assert res["success"] is True, res
        assert captured["payload"]["seed"] == 150, captured["payload"]
        assert seen["range"] == (100, 200), seen
    finally:
        setattr(mod, "random", orig_random)


def test_random_seed_negative_only_range_never_raises():
    mod = _require()
    for lo, hi, want in ((-1, -1, -1), (-5, -2, -2)):
        captured, _ = _setup(mod, fetch=_spec_doc(seed_min=lo, seed_max=hi))
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/spec-model", image_url="https://example.com/a.png")
        assert res["success"] is True, res
        assert captured["payload"]["seed"] == want, (lo, hi, captured["payload"])


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