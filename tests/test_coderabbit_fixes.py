"""Regression tests for the CodeRabbit PR #4 review fixes (v1.7.0 T15).

Covers envelope ambiguity, free-route 403 wording, non-finite cache stamps,
backoff disk adoption, store-winner semantics, alternate-id classic routing,
text-billed zero-image-rate labels, free-variant fee wording, keyed SVG
derived-default skipping, normalized-canonical precedence, unreadable-billing
toggle-on hiding, and exact serialized payload budgeting.
Stdlib-only: run ``python3 tests/test_coderabbit_fixes.py``.
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
SCRATCH = Path(__file__).resolve().parent / "_scratch_cr"


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
    spec = importlib.util.spec_from_file_location("pollinations_cr_fixes", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

_SEAMS = ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
          "_include_paid_models", "_spec_limits", "save_b64_image", "_hermes_home",
          "_http_get_json", "_http_get_image", "_http_post_json", "_to_data_url",
          "MAX_EDIT_PAYLOAD_BYTES")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def _setup(mod, *, catalog=None, key: "str | None" = "sk_dummy", config=None, classic=(),
           include_paid=False):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    setattr(mod, "_hermes_home", lambda: SCRATCH)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog if catalog is not None else [QUEST])])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: include_paid)
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


def test_ambiguous_envelope_not_authoritative():
    mod = _require()
    payload = {"model_versions": ["v1"], "free_models": ["sana"]}
    out = mod._unwrap_list(payload, ("models", "data"),
                           fallback_key_ok=lambda k: "model" in k.lower(),
                           fallback_item_ok=lambda items: any(mod._valid_id(it) for it in items))
    assert out is None, out
    # A single qualifying list still resolves.
    out = mod._unwrap_list({"free_models": ["sana"]}, ("models",),
                           fallback_key_ok=lambda k: "model" in k.lower(),
                           fallback_item_ok=lambda items: any(mod._valid_id(it) for it in items))
    assert out == ["sana"], out
    # A model-shaped list disambiguates even among several candidates.
    out = mod._unwrap_list({"tags": ["x"], "entries": [{"name": "real/model"}]}, ("models",),
                           fallback_key_ok=lambda k: True,
                           fallback_item_ok=lambda items: any(
                               isinstance(i, dict) and i.get("name") for i in items))
    assert out == [{"name": "real/model"}], out


def test_free_route_403_wording():
    mod = _require()
    exc = mod._HTTPError("https://image.pollinations.ai/prompt/x", 403)
    fail = mod.error_factory(mod.PROVIDER_ID, "landscape", model="m", prompt="p")
    res = mod._request_failed(exc, action="generation", mode="free", fail=fail)
    assert res["success"] is False
    assert "key lacks permission" not in res["error"], res["error"]
    assert "enter.pollinations.ai" not in res["error"], res["error"]
    res2 = mod._request_failed(exc, action="generation", mode="quest", fail=fail)
    assert "key lacks permission" in res2["error"], res2["error"]


def test_non_finite_stamp_repaired():
    mod = _require()
    _setup(mod)
    path = mod._cache_path()
    mod.atomic_json_write(path, {"version": 1, "fetched": "Infinity", "models": []})
    mod._guarded_cache_write(path, {"version": 1, "fetched": time.time(),
                                    "models": [{"name": "ok/model"}]}, time.time())
    disk = json.loads(path.read_text())
    assert disk["models"] == [{"name": "ok/model"}], disk


def test_backoff_adopts_newer_withdrawal():
    mod = _require()
    _setup(mod)
    # The REAL lookup must run (the _setup stub would make this vacuous): restore the
    # pristine seam, expire memory past the TTL, keep the negative window active, and
    # fail loudly on any network fetch.
    setattr(mod, "_classic_models", _PRISTINE["_classic_models"])

    calls = []

    def _boom(url, timeout=30.0):
        calls.append(url)
        raise OSError("network fetch during backoff")

    setattr(mod, "_http_get_json", _boom)
    mod._classic_state().update({"at": time.time() - 7200, "ids": ["sana"],
                                 "negative_until": time.monotonic() + 60,
                                 "last_success": time.monotonic()})
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time(), "ids": []})
    assert mod._classic_models() == []
    # The fetch handler swallows exceptions, so assert on the recorded calls: a regressed
    # backoff branch must not reach the network at all.
    assert calls == [], calls


def test_malformed_ids_not_a_withdrawal():
    mod = _require()
    _setup(mod)
    setattr(mod, "_classic_models", _PRISTINE["_classic_models"])  # the REAL lookup
    mod._classic_state().update({"at": time.time() - 10, "ids": ["sana"],
                                 "negative_until": 0.0, "last_success": 0.0})
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time(), "ids": "invalid"})
    assert mod._classic_models() == ["sana"]  # fresh memory kept; malformed disk ignored
    # Store skip path: a newer malformed write must not clear the fetched list either.
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time() + 10, "ids": "invalid"})
    winner = mod._store_classic_cache(["fresh"], time.time())
    assert winner == ["fresh"], winner


def test_store_returns_newer_disk_winner():
    mod = _require()
    _setup(mod)
    mod.atomic_json_write(mod._classic_cache_path(),
                          {"version": 1, "fetched": time.time() + 5, "ids": ["new"]})
    winner = mod._store_classic_cache(["old"], time.time())
    assert winner == ["new"], winner
    assert mod._classic_state()["ids"] == ["new"]


def test_alternate_id_classic_routing():
    mod = _require()
    entry = {"name": "picture-v2", "id": "picture", "category": "image", "paid_only": False,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    _setup(mod, catalog=[entry], key=None, classic=["picture"])
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="picture")
    assert res["success"] is True, res
    assert calls and "image.pollinations.ai" in calls[0] and "model=picture" in calls[0], calls


def test_zero_image_rate_text_billed_not_free():
    mod = _require()
    m = dict(QUEST, name="test/text-billed",
             pricing={"currency": "pollen", "completionImageTokens": "0",
                      "completionTextTokens": "0.001"})
    label = mod._price_label(m, False, True)
    assert "text billed" in label, label
    assert not label.startswith("Free ·") and label != "Free", label


def test_free_variant_with_fees_shows_variant_cost():
    mod = _require()
    fee = {"name": "fee", "kind": "image", "price": "0.04", "currency": "pollen",
           "quantity": 1, "unit": "generation"}
    m = dict(QUEST, name="test/fee-variant",
             pricing={"currency": "pollen", "completionImageTokens": "0.5"},
             pricing_variants=[{"name": "free", "completionImageTokens": "0"}],
             pricing_adjustments=[fee])
    label = mod._price_label(m, False, True)
    assert "(free variant)" not in label, label
    assert "0.04 pollen" in label and "variant" in label, label


def test_keyed_svg_classic_not_derived_default():
    mod = _require()
    svg = {"name": "vector-x", "category": "image", "paid_only": False, "flat_rate": True,
           "description": "Create editable SVG artwork",
           "pricing": {"currency": "pollen", "completionImageTokens": "0.01"}}
    _setup(mod, catalog=[svg, QUEST], classic=["vector-x"])
    assert mod._derived_default([svg, QUEST], {"vector-x"}, True) == "test/quest-model"
    assert mod._derived_default([svg, QUEST], {"vector-x"}, False) == "vector-x"


def test_normalized_canonical_beats_exact_alias():
    mod = _require()
    canonical = dict(QUEST, name="Picture")
    hijack = {"name": "other/paid", "category": "image", "paid_only": True,
              "aliases": ["picture"],
              "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
    _setup(mod, catalog=[canonical, hijack])
    assert mod._resolve_model("picture", [canonical, hijack]) == "Picture"


def test_classic_branch_canonical_beats_alias():
    mod = _require()
    canonical = dict(QUEST, name="Picture")
    claimant = dict(QUEST, name="other/free", aliases=["picture"])
    _setup(mod, catalog=[canonical, claimant], classic=["picture"])
    # Classic serves 'picture': the canonical 'Picture' must win over the exact alias claim,
    # keyed included; the paid/unreadable hijack guards are covered in test_resolution_correctness.
    assert mod._resolve_model("picture", [canonical, claimant]) == "Picture"


def test_case_variant_alias_collision_exact_wins():
    mod = _require()
    first = dict(QUEST, name="first/free", aliases=["Picture"])
    second = dict(QUEST, name="second/free", aliases=["picture"])
    # Classic branch: the exact alias claimant wins over the case-folded alias.
    _setup(mod, catalog=[first, second], classic=["picture"])
    assert mod._resolve_model("picture", [first, second]) == "second/free"
    # Non-classic path ranks identically.
    _setup(mod, catalog=[first, second], classic=[])
    assert mod._resolve_model("picture", [first, second]) == "second/free"


def test_unreadable_billing_hidden_with_toggle_on():
    mod = _require()
    unreadable = {"name": "test/unreadable", "category": "image", "paid_only": [],
                  "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 4,
                  "pricing": {"currency": "pollen", "completionImageTokens": "0.01"}}
    _setup(mod, catalog=[unreadable, QUEST], include_paid=True,
           config={"model": "test/unreadable"})
    rows = mod.PollinationsImageGenProvider().list_models()
    assert not any(r["id"] == "test/unreadable" and "disabled" not in r["price"] for r in rows), rows
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps["modalities"] == ["text"], caps
    assert "test/unreadable" not in mod._visible_edit_models([unreadable], include_paid=True)


def test_exact_serialized_budget():
    mod = _require()
    entry = dict(QUEST, name="ok/model", supported_endpoints=["/v1/images/edits"],
                 max_reference_images=4)
    _setup(mod, catalog=[entry])
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 30000)
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64," + "A" * 400)
    posted = []
    setattr(mod, "_http_post_json",
            lambda url, *, headers, payload, timeout=120.0: posted.append(payload) or {
                "data": [{"b64_json": base64.b64encode(_png(1, 1)).decode()}]})
    # 28000 UTF-8 bytes but 42000 serialized bytes (each newline escapes to \n): over budget.
    res = mod.PollinationsImageGenProvider().generate(
        "a\n" * 14000, model="ok/model", image_url="https://example.com/a.png")
    assert res["success"] is False, res
    assert "too large" in res["error"], res
    assert posted == [], posted
    # Control: same raw byte count without escapes fits and posts.
    res2 = mod.PollinationsImageGenProvider().generate(
        "a" * 28000, model="ok/model", image_url="https://example.com/a.png")
    assert res2["success"] is True, res2
    assert len(posted) == 1


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