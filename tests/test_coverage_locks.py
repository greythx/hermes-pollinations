"""Coverage locks for guards not tied to T1-T8 behavior changes (v1.7.0 T9).

Locks: POST no-redirect policy, bounded-read ceiling/deadline, t2i completeness
variants, _as_bool string matrix, classic-route paid label, classic fresh
memory hit, output_modalities discriminator, keyless resolution withholding,
reference-cap edges, empty-catalog default. Stdlib-only: run
``python3 tests/test_coverage_locks.py``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
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
    spec = importlib.util.spec_from_file_location("pollinations_coverage_locks", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

# Pristine seam captured at load: other tests monkeypatch _classic_models itself.
_REAL_CLASSIC_MODELS = MOD._classic_models if MOD is not None else None


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}


def test_edit_post_does_not_follow_redirects():
    mod = _require()
    seen = {}

    class _Stream:
        status_code = 200
        headers = {}

        def iter_bytes(self):
            yield b'{"data": []}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Client:
        def stream(self, method, url, **kw):
            return _Stream()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _factory(timeout, follow_redirects=True):
        seen["follow_redirects"] = follow_redirects
        return _Client()

    orig = mod._ssrf_client
    setattr(mod, "_ssrf_client", _factory)
    try:
        mod._http_post_json("https://gen.pollinations.ai/v1/images/edits", headers={}, payload={})
        assert seen["follow_redirects"] is False, seen
    finally:
        setattr(mod, "_ssrf_client", orig)


def test_bounded_read_aborts_oversize_and_slow():
    mod = _require()

    class _Resp:
        def __init__(self, chunks, delay=0.0):
            self._chunks = chunks
            self._delay = delay

        def iter_bytes(self):
            for c in self._chunks:
                if self._delay:
                    time.sleep(self._delay)
                yield c

    try:
        mod._bounded_read(_Resp([b"x" * 100, b"y" * 100]), "https://x", limit=150)
        raise AssertionError("oversize body accepted")
    except ValueError:
        pass
    orig_timeout = mod.HTTP_TIMEOUT
    setattr(mod, "HTTP_TIMEOUT", 0.05)
    try:
        mod._bounded_read(_Resp([b"x", b"y"], delay=0.1), "https://x", limit=1000)
        raise AssertionError("slow body accepted")
    except ValueError:
        pass
    finally:
        setattr(mod, "HTTP_TIMEOUT", orig_timeout)


def test_t2i_incomplete_jpeg_gif_webp_rejected():
    mod = _require()
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: [dict(QUEST)])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    bodies = [
        b"\xff\xd8\xff\xe0" + b"\x00" * 40,                     # JPEG without EOI
        b"GIF89a" + b"\x00" * 20,                               # GIF without trailer
        b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"junk",  # WebP bad first chunk
    ]
    for body in bodies:
        setattr(mod, "_http_get_image",
                lambda url, *, headers=None, timeout=120.0, _b=body: (_b, None))
        res = mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model")
        assert res["success"] is False, (body[:8], res)
        assert res["error_type"] == "invalid_response", (body[:8], res)


def test_as_bool_string_matrix():
    mod = _require()
    for v in ("false", "FALSE", "0", "", "no", "off", None, [], {}, 0):
        assert mod._as_bool(v) is False, repr(v)
    for v in ("true", "TRUE", "1", "yes", "on", " True ", 1, 2.5, True):
        assert mod._as_bool(v) is True, repr(v)


def test_price_label_classic_route_paid_only():
    mod = _require()
    m = dict(QUEST, paid_only=True)
    label = mod._price_label(m, True, True, classic_route=True)
    assert label.startswith("Free on the classic route"), label
    assert "watermark" in label, label


def test_classic_fresh_memory_hit_skips_fetch():
    mod = _require()
    orig = mod._classic_state().copy()
    mod._classic_state().update({"at": time.time(), "ids": ["memory-id"],
                               "negative_until": 0.0, "last_success": 0.0})

    def _boom(url, timeout=30.0):
        raise AssertionError("fetch on fresh memory hit")

    orig_fetch = mod._http_get_json
    setattr(mod, "_http_get_json", _boom)
    try:
        assert _REAL_CLASSIC_MODELS is not None
        assert _REAL_CLASSIC_MODELS() == ["memory-id"]
    finally:
        setattr(mod, "_http_get_json", orig_fetch)
        mod._classic_state().update(orig)


def test_output_modalities_discriminator():
    mod = _require()
    assert mod._is_image_model({"output_modalities": ["image"]}) is True
    assert mod._is_image_model({"output_modalities": ["video"]}) is False
    assert mod._is_image_model({"category": "video", "output_modalities": ["image"]}) is False
    assert mod._is_image_model({"category": "image"}) is True


def test_capabilities_resolution_keyless_withheld():
    mod = _require()
    entry = dict(QUEST, resolutions=["1k", "2k"])
    setattr(mod, "_catalog", lambda: [entry])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "load_image_gen_config", lambda: {"model": "test/quest-model"})
    setattr(mod, "_api_key", lambda: None)
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert not caps.get("creative_controls"), caps
    setattr(mod, "_api_key", lambda: "sk_dummy")
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps.get("creative_controls") == ["resolution"], caps


def test_reference_cap_edges():
    mod = _require()
    cases = [({"max_reference_images": "3"}, 3), ({"max_reference_images": 100}, 16),
             ({"max_reference_images": True}, 4), ({"max_reference_images": "nan"}, 4),
             ({"max_reference_images": -2}, 0), ({"max_reference_images": 0}, 0),
             ({"max_reference_images": 3.9}, 3), ({}, 4)]
    for entry, want in cases:
        assert mod._reference_cap(entry) == want, (entry, mod._reference_cap(entry), want)


def test_derived_default_empty_and_paid_only():
    mod = _require()
    setattr(mod, "_classic_models", lambda: [])
    assert mod._derived_default([], set(), True) is None
    paid = dict(QUEST, paid_only=True)
    assert mod._derived_default([paid], set(), True) is None
    assert mod._derived_default([paid], {"test/quest-model"}, False) == "test/quest-model"


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