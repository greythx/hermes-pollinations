"""Regression tests for URL redaction and error transport (v1.7.0 T3).

Covers M2 (apostrophe/control-char/protocol-relative redaction bypasses), the
fuzz F-2 class (unicode-escaped scheme/colon spellings on BOTH error paths),
bare credential query tails, F10 (a 3xx POST response must not parse as a
completed edit), L13 (Retry-After display clamp), L5 (actionable 403), L6
(neutral QUEST_POLLEN_RESTRICTED wording) and L8 (live fixUrl surfacing).
Stdlib-only: run ``python3 tests/test_redaction_transport.py``.
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
    spec = importlib.util.spec_from_file_location("pollinations_redaction_transport", ROOT / "__init__.py")
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


class _Resp:
    def __init__(self, body: bytes, headers=None):
        self._body = body
        self.headers = headers or {}

    def iter_bytes(self):
        yield self._body


def _preview(mod, body: bytes) -> str:
    _, preview, _ = mod._error_fields(_Resp(body))
    return preview


def test_unicode_escaped_scheme_redacted():
    mod = _require()
    variants = [
        b"upstream failed \\u0068ttps://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed https\\u003a//cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed h\\u0074tps://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed http\\u0073://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed htt\\u0070://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed \\u0048ttps://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed \\u0048TTPS://cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed \\u0068ttps\\u003a//cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed \\u0068\\u0074\\u0074\\u0070\\u0073\\u003a//cdn.example/i?sig=ZZSECRETZZ end",
        b"upstream failed \\\\u0068ttps://cdn.example/i?sig=ZZSECRETZZ end",
    ]
    for body in variants:
        preview = _preview(mod, body)
        assert "ZZSECRETZZ" not in preview, (body, preview)
        assert "[url]" in preview, (body, preview)


def test_apostrophe_and_control_char_urls_redacted():
    mod = _require()
    bodies = [
        b"failed https://cdn.example/i?sig=SEC'RET end",
        b"failed https://cdn.example/i?sig=SEC\x00RET end",
        b"failed https://cdn.example/i?sig=SEC\x1bRET end",
        b"failed //cdn.example/i?sig=PROTOSECRET end",
    ]
    for body in bodies:
        preview = _preview(mod, body)
        for secret in ("SEC'RET", "SECRET", "PROTOSECRET"):
            assert secret not in preview, (body, preview)


def test_bare_credential_query_tails_masked():
    mod = _require()
    preview = _preview(mod, b"request failed ?sig=BARESECRET&x=1 and ?token=TOKSECRET end")
    assert "BARESECRET" not in preview, preview
    assert "TOKSECRET" not in preview, preview
    assert "[redacted]" in preview, preview


def test_json_extracted_message_unicode_escape_redacted():
    mod = _require()
    # Double-encoded: after json.loads the message still carries literal \u0068 escapes.
    body = b'{"error": {"message": "edit failed \\\\u0068ttps://cdn.example/i?sig=JSONSECRET"}}'
    preview = _preview(mod, body)
    assert "JSONSECRET" not in preview, preview
    assert "[url]" in preview, preview


def test_inband_edit_error_unicode_escape_redacted():
    mod = _require()
    entry = {"name": "test/edit-model", "category": "image", "paid_only": False,
             "supported_endpoints": ["/v1/images/edits"],
             "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
    names = ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
             "_include_paid_models", "_spec_limits", "_to_data_url", "save_b64_image",
             "_http_post_json")
    orig = {n: getattr(mod, n) for n in names}
    setattr(mod, "_api_key", lambda: "sk_dummy")
    setattr(mod, "load_image_gen_config", lambda: {})
    setattr(mod, "_catalog", lambda: [entry])
    setattr(mod, "_classic_models", lambda: [])
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64,AAAA")
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    setattr(mod, "_http_post_json", lambda url, *, headers, payload, timeout=120.0: {
        "error": {"message": "edit failed \\u0068ttps://cdn.example/i?sig=EDITSECRET"}})
    try:
        res = mod.PollinationsImageGenProvider().generate(
            "edit", model="test/edit-model", image_url="https://example.com/a.png")
        assert res["success"] is False, res
        assert "EDITSECRET" not in res["error"], res["error"]
        assert "[url]" in res["error"], res["error"]
    finally:
        for n, v in orig.items():
            setattr(mod, n, v)


class _FakeStream:
    def __init__(self, status, body):
        self.status_code = status
        self.headers = {}
        self._body = body

    def iter_bytes(self):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def stream(self, method, url, **kw):
        return self._resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_3xx_with_json_is_an_error_not_success():
    mod = _require()
    orig = mod._ssrf_client
    setattr(mod, "_ssrf_client", lambda timeout, follow_redirects=True: _FakeClient(
        _FakeStream(302, b'{"data": [{"b64_json": "AAAA"}]}')))
    try:
        try:
            mod._http_post_json("https://gen.pollinations.ai/v1/images/edits",
                                headers={}, payload={})
            raise AssertionError("302 JSON parsed as success")
        except mod._HTTPError as exc:
            assert exc.status == 302, exc.status
    finally:
        setattr(mod, "_ssrf_client", orig)


def test_retry_after_display_clamped():
    mod = _require()
    for raw in ("1e308", "999999999999", "1e10"):
        val = mod._retry_after(_Resp(b"", {"retry-after": raw}))
        assert val is not None and 0 <= val <= 86400, (raw, val)
    assert mod._retry_after(_Resp(b"", {"retry-after": "45"})) == 45
    assert mod._retry_after(_Resp(b"", {"retry-after": "garbage"})) is None


def test_403_actionable_guidance():
    mod = _require()
    exc = mod._HTTPError("https://gen.pollinations.ai/image/x", 403, "FORBIDDEN")
    res = mod._request_failed(exc, action="image generation", mode="quest",
                              fail=lambda msg, kind: {"error": msg, "error_type": kind})
    assert res["error_type"] == "auth_required", res
    assert "permission" in res["error"].lower(), res


def test_restricted_wording_neutral():
    mod = _require()
    exc = mod._HTTPError("https://gen.pollinations.ai/image/x", 402, "",
                         error_code="QUEST_POLLEN_RESTRICTED")
    res = mod._request_failed(exc, action="image generation", mode="quest",
                              fail=lambda msg, kind: {"error": msg, "error_type": kind})
    assert "restricted" in res["error"].lower(), res
    assert res["error_type"] == "payment_required", res


def test_fixurl_extracted_and_surfaced():
    mod = _require()
    body = b'{"error": {"code": "X", "message": "nope", "fixUrl": "https://enter.pollinations.ai/fix"}}'
    code, preview, fix = mod._error_fields(_Resp(body))
    assert fix == "https://enter.pollinations.ai/fix", fix
    exc = mod._HTTPError("https://gen.pollinations.ai/image/x", 402, preview,
                         error_code=code, fix_url=fix)
    res = mod._request_failed(exc, action="image generation", mode="quest",
                              fail=lambda msg, kind: {"error": msg, "error_type": kind})
    assert "https://enter.pollinations.ai/fix" in res["error"], res
    # A non-https or hostile fixUrl is ignored.
    for bad in ("http://evil.example/fix", "javascript:alert(1)", "https://x.example/" + "a" * 3000):
        _, _, f = mod._error_fields(_Resp(
            ('{"error": {"fixUrl": "' + bad + '"}}').encode()))
        assert f is None, bad


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