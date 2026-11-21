"""Regression tests for edit payload exactness and produced sizes (v1.7.0 T6).

Covers R2-E1 (an exactly-at-cap data URI must be accepted, like local/URL
sources), L14 (the payload budget counts the prompt's UTF-8 bytes, not code
points) and M4 (edit responses report the produced dimensions measured from
the decoded image, not the requested ones).
Stdlib-only: run ``python3 tests/test_edit_exactness.py``.
"""

from __future__ import annotations

import base64
import importlib.util
import shutil
import struct
import sys
import unittest
import zlib
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
    spec = importlib.util.spec_from_file_location("pollinations_edit_exactness", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

# Pristine seams captured at load: later tests monkeypatch these, and the cap test below
# must exercise the REAL _to_data_url, not a fake left behind by an earlier test.
_REAL_TO_DATA_URL = MOD._to_data_url if MOD is not None else None


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


def _png_exact(total: int, w: int = 1, h: int = 1) -> bytes:
    """A structurally complete PNG of exactly *total* bytes (IEND last)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    fixed = 8 + 25 + 12  # magic + IHDR chunk + IEND chunk
    idat_data_len = total - fixed - 12
    assert idat_data_len >= 0
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", b"\x00" * idat_data_len) + chunk(b"IEND", b""))


def _setup(mod, *, post_response=None, payload_cap=None):
    entry = {"name": "test/edit-model", "category": "image", "paid_only": False,
             "supported_endpoints": ["/v1/images/edits"],
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: [entry])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    setattr(mod, "record_token_usage", lambda u, *, model, provider, base_url=None: None)
    if payload_cap is not None:
        setattr(mod, "MAX_EDIT_PAYLOAD_BYTES", payload_cap)
    calls = []
    setattr(mod, "_to_data_url",
            lambda v, limit=1000000: calls.append(limit) or "data:image/png;base64,AAAA")
    captured = {}

    def _post(url, *, headers, payload, timeout=120.0):
        captured["payload"] = payload
        return post_response if post_response is not None else {
            "data": [{"b64_json": base64.b64encode(_png_exact(200)).decode()}]}

    setattr(mod, "_http_post_json", _post)
    return calls, captured


def test_exact_cap_data_uri_accepted():
    mod = _require()
    orig_cap = mod.MAX_INPUT_IMAGE_BYTES
    setattr(mod, "MAX_INPUT_IMAGE_BYTES", 1000)
    try:
        raw = _png_exact(1000)
        assert len(raw) == 1000
        real = _REAL_TO_DATA_URL
        assert real is not None
        uri = "data:image/png;base64," + base64.b64encode(raw).decode()
        assert real(uri, limit=1000).startswith("data:image/png;base64,")
        big = _png_exact(1001)
        try:
            real("data:image/png;base64," + base64.b64encode(big).decode(), limit=1000)
            raise AssertionError("over-cap data URI accepted")
        except ValueError:
            pass
    finally:
        setattr(mod, "MAX_INPUT_IMAGE_BYTES", orig_cap)


def test_budget_counts_prompt_utf8_bytes():
    mod = _require()
    calls, _ = _setup(mod, payload_cap=3000)
    res = mod.PollinationsImageGenProvider().generate(
        "\u00e9" * 1500, model="test/edit-model", image_url="https://example.com/a.png")
    # 1500 code points fit the old char-count budget (1500+1024<3000) but the UTF-8 encoding
    # is 3000 bytes (+1024 framing) and must be rejected before any source is read.
    assert res["success"] is False, res
    assert res["error_type"] == "invalid_input", res
    assert calls == [], calls


def test_edit_reports_produced_png_dimensions():
    mod = _require()
    png = _png_exact(200, w=37, h=23)
    _, captured = _setup(mod, post_response={
        "data": [{"b64_json": base64.b64encode(png).decode()}]})
    res = mod.PollinationsImageGenProvider().generate(
        "edit", aspect_ratio="square", model="test/edit-model",
        image_url="https://example.com/a.png")
    assert res["success"] is True, res
    assert res["width"] == 37 and res["height"] == 23, res
    assert res["requested_width"] == 1024 and res["requested_height"] == 1024, res


def test_structural_dimension_parsers():
    mod = _require()
    assert mod._image_dimensions(_png_exact(200, w=800, h=600), "image/png") == (800, 600)
    gif = b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00" * 20 + b";"
    assert mod._image_dimensions(gif, "image/gif") == (320, 240)
    # WebP VP8X extended canvas: 24-bit little-endian (width-1, height-1) at bytes 24..30.
    vp8x = (b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"VP8X"
            + (10).to_bytes(4, "little") + b"\x00" * 4
            + (639).to_bytes(3, "little") + (479).to_bytes(3, "little"))
    assert mod._image_dimensions(vp8x, "image/webp") == (640, 480)
    assert mod._image_dimensions(b"not an image", "image/png") is None


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