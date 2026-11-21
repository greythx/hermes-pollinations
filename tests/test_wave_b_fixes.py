"""Regression tests for Round-3 wave-B fixes (v1.7.0 T12).

Covers Unicode-normalized id gating (NFKC/casefold/dash confusables), fixUrl
backslash authority, extended credential tails + fragment tails + single-slash
URLs, edit-hint flag consistency, symlink-safe lockfile/cache writes, framing
reserve for long model ids, SVG-product fail-closed handling, variant floor
truthfulness, and usage-based-fee wording.
Stdlib-only: run ``python3 tests/test_wave_b_fixes.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import struct
import sys
import time
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_scratch_waveb"


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
    spec = importlib.util.spec_from_file_location("pollinations_wave_b_fixes", ROOT / "__init__.py")
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
          "_http_get_json", "_http_get_image", "_http_post_json", "_to_data_url")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


PAID_FLUX = {"name": "flux-pro", "category": "image", "paid_only": True, "flat_rate": True,
             "pricing": {"currency": "pollen", "completionImageTokens": "0.5"}}
QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def _setup(mod, *, catalog=None, key: "str | None" = "sk_dummy", config=None):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    setattr(mod, "_hermes_home", lambda: SCRATCH)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog if catalog is not None else [QUEST])])
    setattr(mod, "_classic_models", lambda: [])
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


def test_unicode_normalized_paid_ids_gated():
    mod = _require()
    _setup(mod, catalog=[PAID_FLUX, QUEST])
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    for variant in ("\ufb02ux-pro", "\uff46\uff4c\uff55\uff58\uff0d\uff50\uff52\uff4f",
                    "flux\u2011pro", "FLUX-PRO"):
        res = mod.PollinationsImageGenProvider().generate("a cat", model=variant)
        assert res["success"] is False, (variant, res)
        assert res["error_type"] == "paid_model_disabled", (variant, res)
    assert calls == [], calls  # nothing reached the network


def test_unicode_normalized_free_id_resolves():
    mod = _require()
    free = dict(QUEST, name="free-model")
    _setup(mod, catalog=[free])
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: (_png(1, 1), None))
    res = mod.PollinationsImageGenProvider().generate("a cat", model="\uff46\uff52\uff45\uff45\uff0d\uff4d\uff4f\uff44\uff45\uff4c")
    assert res["success"] is True, res
    assert res["model"] == "free-model", res


def test_fixurl_backslash_authority_rejected():
    mod = _require()
    body = json.dumps({"error": {"message": "x", "fixUrl": "https://evil.example\\.gen.pollinations.ai/fix"}}).encode()
    _, _, fix = mod._error_fields(_Resp(body))
    assert fix is None, fix


def test_extended_credential_tails_masked():
    mod = _require()
    body = (b"fail ?Policy=P1&Key-Pair-Id=K1 ?X-Goog-Signature=G1 ?code=C1 "
            b"#access_token=T1 ?jwt=J1 end")
    _, preview, _ = mod._error_fields(_Resp(body))
    for secret in ("P1", "K1", "G1", "C1", "T1", "J1"):
        assert secret not in preview, preview


def test_single_slash_url_redacted():
    mod = _require()
    body = b"upstream said https:/media.pollinations.ai/i/abc?id=SECRET end"
    _, preview, _ = mod._error_fields(_Resp(body))
    assert "SECRET" not in preview, preview
    assert "media.pollinations.ai" not in preview, preview


def test_edit_hint_uses_paid_flag_and_skips_svg():
    mod = _require()
    unreadable = {"name": "test/unreadable", "category": "image", "paid_only": {"x": 1},
                  "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 4}
    svg = {"name": "test/svg", "category": "image", "paid_only": False,
           "description": "Create editable SVG artwork",
           "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 4}
    names = mod._visible_edit_models([unreadable, svg, QUEST], include_paid=False)
    assert "test/unreadable" not in names, names
    assert "test/svg" not in names, names


def test_lockfile_symlink_not_followed():
    mod = _require()
    _setup(mod)
    cache_dir = SCRATCH / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    victim = SCRATCH / "victim.txt"
    lock = mod._cache_path().with_name(mod._cache_path().name + ".lock")
    lock.symlink_to(victim)
    mod._guarded_cache_write(mod._cache_path(),
                             {"version": 1, "fetched": time.time(), "models": [{"name": "ok/model"}]},
                             time.time())
    assert not victim.exists(), "lockfile symlink target was created"
    disk = json.loads(mod._cache_path().read_text())
    assert disk["models"] == [{"name": "ok/model"}], disk


def test_cache_symlink_not_followed():
    mod = _require()
    _setup(mod)
    cache_dir = SCRATCH / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    victim = SCRATCH / "victim.txt"
    victim.write_text("DO NOT OVERWRITE")
    mod._cache_path().symlink_to(victim)
    mod._guarded_cache_write(mod._cache_path(),
                             {"version": 1, "fetched": time.time(), "models": [{"name": "ok/model"}]},
                             time.time())
    assert victim.read_text() == "DO NOT OVERWRITE"
    assert not mod._cache_path().is_symlink()
    disk = json.loads(mod._cache_path().read_text())
    assert disk["models"] == [{"name": "ok/model"}], disk


def test_framing_reserve_counts_model_id():
    mod = _require()
    long_id = "x" * 4000
    entry = dict(QUEST, name=long_id, supported_endpoints=["/v1/images/edits"],
                 max_reference_images=4)
    short_entry = dict(QUEST, name="ok/model", supported_endpoints=["/v1/images/edits"],
                       max_reference_images=4)
    _setup(mod, catalog=[entry, short_entry])
    posted = []
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64," + "A" * 400)
    setattr(mod, "_http_post_json",
            lambda url, *, headers, payload, timeout=120.0: posted.append(payload) or {
                "data": [{"b64_json": base64.b64encode(_png(1, 1)).decode()}]})
    # Scoped budget + a prompt UNDER the 32000-char cap, so the model-id overhead is what
    # pushes the serialized body over the limit (with the real 40MiB budget the prompt-length
    # gate fired first and this test never reached the budget check).
    orig_budget = mod.MAX_EDIT_PAYLOAD_BYTES
    setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", 25000)
    try:
        prompt = "p" * 22000
        res = mod.PollinationsImageGenProvider().generate(
            prompt, model=long_id, image_url="https://example.com/a.png")
        assert res["success"] is False, res
        assert "too large" in res["error"], res
        assert posted == [], posted
        # Control: a short model id with the same prompt and image fits and posts.
        res2 = mod.PollinationsImageGenProvider().generate(
            prompt, model="ok/model", image_url="https://example.com/a.png")
        assert res2["success"] is True, res2
        assert len(posted) == 1
    finally:
        setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", orig_budget)


def test_svg_product_refused_before_spend():
    mod = _require()
    svg = {"name": "test/vector", "category": "image", "paid_only": True, "flat_rate": True,
           "description": "Create editable SVG artwork from text",
           "pricing": {"currency": "pollen", "completionImageTokens": "0.01"},
           "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 5}
    _setup(mod, catalog=[svg, QUEST], config={"model": "test/vector"})
    setattr(mod, "_include_paid_models", lambda pid: True)
    calls = []
    setattr(mod, "_http_get_image",
            lambda url, *, headers=None, timeout=120.0: calls.append(url) or (_png(1, 1), None))
    setattr(mod, "_http_post_json",
            lambda url, **kw: calls.append(url) or {"data": [{"b64_json": "AAAA"}]})
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/vector")
    assert res["success"] is False, res
    assert res["error_type"] == "unsupported", res
    assert "SVG" in res["error"], res
    res2 = mod.PollinationsImageGenProvider().generate(
        "edit", model="test/vector", image_url="https://example.com/a.png")
    assert res2["success"] is False, res2
    assert calls == [], calls
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps["modalities"] == ["text"], caps
    rows = mod.PollinationsImageGenProvider().list_models()
    row = next(r for r in rows if r["id"] == "test/vector")
    assert "SVG" in row["price"], row


def test_variant_floor_truthful():
    mod = _require()
    m = dict(QUEST, name="test/variants",
             pricing={"currency": "pollen", "completionImageTokens": "0.06"},
             pricing_variants=[{"name": "low_1k", "completionImageTokens": "0.04"}])
    label = mod._price_label(m, False, True)
    assert label.startswith("from 0.04 pollen"), label
    assert mod._price_value(m) == 0.04, mod._price_value(m)
    dearer = dict(m, pricing_variants=[{"name": "2k", "completionImageTokens": "0.09"}])
    assert mod._price_label(dearer, False, True).startswith("from 0.06 pollen")
    assert mod._price_value(dearer) == 0.06


def test_usage_based_fees_wording():
    mod = _require()
    fee = {"name": "gpu", "kind": "image", "price": "0.000975", "currency": "pollen",
           "quantity": 1, "unit": "gpu-second"}
    m = {"name": "test/fee-only", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen"}, "pricing_adjustments": [fee]}
    label = mod._price_label(m, False, True)
    assert "usage-based fees" in label, label
    assert "No published price" not in label, label


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