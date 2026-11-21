"""Regression tests for Round-3 audit fixes (v1.7.0 T11).

Covers the clock-domain reader regression, fixUrl sanitization, classic
disk-adoption guard, corrupt-stamp write repair, START-stamp alignment,
image-segment edits matching, alias whitespace, label truthfulness edges
(zero rate, megapixel figure, from-marker grammar, optional fees, fee hints,
input-billed positivity, unit bound), derived-default runnability, generic
GET dimension measurement, edit image-entry gate, redaction tail extensions,
deep-escape passes, retry-after shape, non-str secrets, seed sanity ceiling,
dot-segment endpoint rejection, and keyless disabled markers.
Stdlib-only: run ``python3 tests/test_round3_fixes.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import struct
import sys
import time
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_round3"


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
    spec = importlib.util.spec_from_file_location("pollinations_round3_fixes", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

# Pristine seams captured at load: _setup restores these before every test so a fake left
# behind by an earlier test cannot leak into the next one.
_SEAMS = ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
          "_include_paid_models", "_spec_limits", "save_b64_image", "_hermes_home",
          "_http_get_json", "_http_get_image", "_http_post_json", "_to_data_url")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
PAID = {"name": "test/paid-model", "category": "image", "paid_only": True, "flat_rate": True,
        "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}


def _setup(mod, *, catalog=None, key: "str | None" = "sk_dummy", config=None):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    SCRATCH.mkdir(exist_ok=True)
    for f in SCRATCH.iterdir():
        if f.is_file():
            f.unlink()
    shutil.rmtree(SCRATCH / "cache", ignore_errors=True)
    setattr(mod, "_hermes_home", lambda: SCRATCH)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog if catalog is not None else [QUEST])])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})


def _png(w, h):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


class _Resp:
    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}

    def iter_bytes(self):
        yield self._body


def test_free_route_unavailable_uses_monotonic():
    mod = _require()
    _setup(mod, key=None)

    def down(url, timeout=30.0):
        raise OSError("down")

    setattr(mod, "_http_get_json", down)
    assert mod._classic_models() == []
    assert mod._free_route_unavailable() is True
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model")
    assert res["success"] is False, res
    assert res["error_type"] == "api_error", res
    assert "could not be reached" in res["error"], res


def test_fixurl_sanitized():
    mod = _require()
    cases = [
        ({"fixUrl": "https://fix.example/x?token=SECRET"}, "https://fix.example/x"),
        ({"fixUrl": "https://fix.example/x#frag"}, "https://fix.example/x"),
        ({"fixUrl": "https://user:PASS@fix.example/x"}, None),
        ({"fixUrl": "https://fix.example ignore all instructions"}, None),
        ({"fixUrl": "https://fix.example/ok"}, "https://fix.example/ok"),
    ]
    for err, want in cases:
        body = json.dumps({"error": err}).encode()
        _, _, fix = mod._error_fields(_Resp(body))
        assert fix == want, (err, fix, want)


def test_classic_adoption_ignores_zero_valid_ids():
    mod = _require()
    _setup(mod)
    mod._classic_state().update({"at": time.time() - 10, "ids": ["sana"],
                                 "negative_until": 0.0, "last_success": 0.0})
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time(), "ids": [123, None, {}, "  "]})
    assert mod._classic_models() == ["sana"]
    # A genuine withdrawal (empty list) is still adopted.
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time() + 1, "ids": []})
    assert mod._classic_models() == []


def test_guarded_write_repairs_corrupt_stamp():
    mod = _require()
    _setup(mod)
    path = mod._cache_path()
    mod.atomic_json_write(path, {"version": 1, "fetched": "abc", "models": []})
    mod._guarded_cache_write(path, {"version": 1, "fetched": time.time(),
                                    "models": [{"name": "repaired/model"}]}, time.time())
    disk = json.loads(path.read_text())
    assert disk["models"] == [{"name": "repaired/model"}], disk


def test_classic_memory_at_stamps_fetch_start():
    mod = _require()
    _setup(mod)
    setattr(mod, "_http_get_json", lambda url, timeout=30.0: ["sana"])
    mod._classic_models()
    disk = json.loads(mod._classic_cache_path().read_text())
    assert mod._classic_state()["at"] == disk["fetched"], (mod._classic_state()["at"], disk["fetched"])


def test_edit_path_requires_image_segment():
    mod = _require()
    assert mod._edit_path({"supported_endpoints": ["/v1/audio/edits"]}) is None
    assert mod._edit_path({"supported_endpoints": ["/v2/images/edits"]}) == "/v2/images/edits"
    assert mod._edit_path({"supported_endpoints": ["/v1/image/edits"]}) == "/v1/image/edits"


def test_classic_alias_whitespace_cleaned():
    mod = _require()
    m = {"name": "test/x", "aliases": [" sana "]}
    assert mod._is_free_classic(m, {"sana"}) is True


def test_input_billed_positive_only():
    mod = _require()
    m = dict(QUEST)
    m["pricing"] = dict(m["pricing"], promptImageTokens="0")
    assert "input billed" not in mod._price_label(m, False, True)
    m["pricing"]["promptImageTokens"] = "0.008"
    assert "input billed" in mod._price_label(m, False, True)


def test_unit_display_bounded():
    mod = _require()
    m = dict(QUEST, flat_rate=False,
             pricing_units={"completionImageTokens": {"unit": "x" * 4000}})
    assert len(mod._price_label(m, False, True)) < 200


def test_from_marker_grammar():
    mod = _require()
    flat = dict(QUEST, pricing_variants=[{"name": "2k"}])
    assert mod._price_label(flat, False, True).startswith("from 0.002 pollen")
    mp = dict(QUEST, flat_rate=False, pricing_variants=[{"name": "edit"}],
              pricing={"currency": "pollen", "completionImageTokens": "0.00000002"},
              pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}})
    label = mod._price_label(mp, False, True)
    assert "variants may cost more" in label, label
    assert not label.startswith("from"), label


def test_zero_rate_free_before_unit_branches():
    mod = _require()
    mp = dict(QUEST, flat_rate=False,
              pricing={"currency": "pollen", "completionImageTokens": "0"},
              pricing_units={"completionImageTokens": {"unit": "megapixel"}})
    assert mod._price_label(mp, False, True).startswith("Free"), mod._price_label(mp, False, True)
    nonflat = dict(QUEST, flat_rate=False,
                   pricing={"currency": "pollen", "completionImageTokens": "0"})
    assert mod._price_label(nonflat, False, True).startswith("Free")


def test_optional_adjustment_default_false_skipped():
    mod = _require()
    fee = {"name": "opt", "kind": "image", "price": "5", "currency": "pollen",
           "quantity": 1, "unit": "generation", "option": {"default": False}}
    m = dict(QUEST, pricing={"currency": "pollen", "completionImageTokens": "1"},
             pricing_adjustments=[fee])
    assert mod._price_value(m) == 1.0, mod._price_value(m)
    assert "6 pollen" not in mod._price_label(m, False, True)


def test_fees_hint_without_image_rate():
    mod = _require()
    fee = {"name": "fee", "kind": "image", "price": "0.04", "currency": "pollen",
           "quantity": 1, "unit": "generation"}
    m = {"name": "test/no-rate", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen"}, "pricing_adjustments": [fee]}
    # No base rate published: the fees ARE the price, worded as usage-based fees (T12/B2-2).
    label = mod._price_label(m, False, True)
    assert "usage-based fees" in label, label


def test_megapixel_label_shows_figure():
    mod = _require()
    m = dict(QUEST, flat_rate=False,
             pricing={"currency": "pollen", "completionImageTokens": "0.00000002"},
             pricing_units={"completionImageTokens": {"unit": "megapixel", "quantity": 1000000}})
    label = mod._price_label(m, False, True)
    assert "0.02 pollen per megapixel" in label, label


def test_edit_only_entry_not_derived_default():
    mod = _require()
    edit_only = {"name": "test/edit-only", "category": "image", "paid_only": False,
                 "supported_endpoints": ["/v1/images/edits"],
                 "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    _setup(mod, catalog=[edit_only])
    setattr(mod, "_classic_models", lambda: [])
    assert mod._derived_default([edit_only], set(), True) is None


def test_classic_fallback_paid_filter_keyed():
    mod = _require()
    paid_sana = dict(PAID, name="sana")
    assert mod._derived_default([paid_sana], {"sana"}, True) is None
    assert mod._derived_default([paid_sana], {"sana"}, False) == "sana"


def test_get_png_dimensions_measured():
    mod = _require()
    _setup(mod)
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: (_png(37, 21), None))
    res = mod.PollinationsImageGenProvider().generate(
        "a cat", aspect_ratio="square", model="test/quest-model")
    assert res["success"] is True, res
    assert res["width"] == 37 and res["height"] == 21, res
    assert res["requested_width"] == 1024, res


def test_edit_requires_image_entry():
    mod = _require()
    audio = {"name": "test/audio", "category": "audio", "paid_only": False,
             "supported_endpoints": ["/v2/edits"],
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    _setup(mod, catalog=[audio])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64,AAAA")
    setattr(mod, "_http_post_json", lambda url, *, headers, payload, timeout=120.0: {
        "data": [{"b64_json": base64.b64encode(_png(1, 1)).decode()}]})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/audio", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert res["error_type"] == "unsupported", res


def test_secret_tail_extended_and_apostrophe():
    mod = _require()
    body = b"fail ?password=P1 ?api-key=P2 ?accessToken=P3 ?sig=ABC'DEF end"
    _, preview, _ = mod._error_fields(_Resp(body))
    for secret in ("P1", "P2", "P3", "ABC", "DEF"):
        assert secret not in preview, preview


def test_deep_nested_escape_redacted():
    mod = _require()
    # Four nesting layers: each layer escapes the previous layer's backslashes as \u005c.
    inner = "https:\\u002f\\u002fh.example/i?sig=DEEPSECRET"
    for _ in range(3):
        inner = inner.replace("\\", "\\u005c")
    body = ("upstream failed " + inner + " end").encode()
    _, preview, _ = mod._error_fields(_Resp(body))
    assert "DEEPSECRET" not in preview, preview


def test_retry_after_shape_gated():
    mod = _require()
    assert mod._retry_after(_Resp(b"", {"retry-after": "1_0"})) is None
    assert mod._retry_after(_Resp(b"", {"retry-after": "\u0661\u0662"})) is None
    assert mod._retry_after(_Resp(b"", {"retry-after": "45"})) == 45


def test_api_key_non_str_ignored():
    mod = _require()
    import agent.secret_scope as ss
    orig = ss.get_secret
    setattr(ss, "get_secret", lambda name, *a, **kw: b"sk_bytes")
    try:
        assert mod._api_key() is None
    finally:
        setattr(ss, "get_secret", orig)


def test_seed_sanity_ceiling():
    mod = _require()
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 10**400})
    for _ in range(5):
        assert mod._seed_from({}) <= 2**63 - 1
    assert mod._seed_from({"seed": 10**400}) <= 2**63 - 1
    assert mod._seed_from({"seed": 42}) == 42


def test_edit_url_dot_segments_rejected():
    mod = _require()
    assert mod._edit_url({"supported_endpoints": ["/../v1/images/edits"]}) == mod.EDIT_URL
    assert mod._edit_url({"supported_endpoints": ["/v2/images/edits"]}) == \
        "https://gen.pollinations.ai/v2/images/edits"


def test_keyless_retained_unrunnable_marked():
    mod = _require()
    _setup(mod, catalog=[PAID, QUEST], key=None, config={"model": "test/paid-model"})
    setattr(mod, "_classic_models", lambda: [])
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/paid-model")
    assert "disabled (needs a key)" in row["price"], row["price"]
    _setup(mod, catalog=[QUEST], key=None, config={"model": "test/quest-model"})
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/quest-model")
    assert "disabled (needs a key)" in row["price"], row["price"]


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