"""Regression tests for HTTP error semantics and safe error previews (audit m1, m10, F7).

402 is a balance/payment condition on the keyed routes (waiting cannot fix a
Paid-Pollen restriction), 429 is the rate limit, and the free route's throttle
message must be visible to the core's text-based rate-limit classifier. Error
previews must extract error.message from the FULL body before truncating, so a
long body cannot leak sibling fields such as signed URLs. Stdlib-only.
"""

from __future__ import annotations

import importlib.util
import json
import re
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
    spec = importlib.util.spec_from_file_location("pollinations_error_mapping", ROOT / "__init__.py")
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


class _FakeResponse:
    def __init__(self, body: bytes, headers=None, status_code: int = 200):
        self._body = body
        self.headers = headers or {}
        self.status_code = status_code

    def iter_bytes(self):
        yield self._body


def _fail(mod):
    return mod.error_factory("pollinations", "square", model="m", prompt="p")


def _http_error(mod, status, *, body="", code=None, retry_after=None):
    return mod._HTTPError("https://gen.pollinations.ai/image/x", status, body,
                          retry_after, error_code=code)


def test_keyed_402_quest_pollen_only_is_payment_required():
    mod = _require()
    exc = _http_error(mod, 402, body="This model requires Paid Pollen", code="QUEST_POLLEN_ONLY")
    res = mod._request_failed(exc, action="image generation", mode="quest", fail=_fail(mod))
    assert res["error_type"] == "payment_required", res
    assert "paid pollen" in res["error"].lower(), res
    assert "wait" not in res["error"].lower(), res


def test_keyed_402_generic_mentions_right_balance():
    mod = _require()
    exc = _http_error(mod, 402, body="Balance too low", code="INSUFFICIENT_BALANCE")
    quest = mod._request_failed(exc, action="image generation", mode="quest", fail=_fail(mod))
    paid = mod._request_failed(exc, action="image generation", mode="paid", fail=_fail(mod))
    assert quest["error_type"] == "payment_required", quest
    assert "quest pollen" in quest["error"].lower(), quest
    assert "paid pollen" in paid["error"].lower(), paid


def test_keyed_429_is_rate_limited_with_retry_after():
    mod = _require()
    exc = _http_error(mod, 429, retry_after=45)
    res = mod._request_failed(exc, action="image generation", mode="quest", fail=_fail(mod))
    assert res["error_type"] == "rate_limited", res
    assert "rate limit" in res["error"].lower(), res
    assert "45" in res["error"], res


def test_free_402_matches_core_classifier():
    mod = _require()
    from agent.tool_result_classification import is_rate_limited
    exc = _http_error(mod, 402, retry_after=45)
    res = mod._request_failed(exc, action="image generation", mode="free", fail=_fail(mod))
    assert res["error_type"] == "rate_limited", res
    assert is_rate_limited(res), res


def test_free_429_matches_core_classifier():
    mod = _require()
    from agent.tool_result_classification import is_rate_limited
    exc = _http_error(mod, 429, retry_after=45)
    res = mod._request_failed(exc, action="image generation", mode="free", fail=_fail(mod))
    assert res["error_type"] == "rate_limited", res
    assert is_rate_limited(res), res


def test_pk_401_explains_app_key_change():
    mod = _require()
    setattr(mod, "_api_key", lambda: "pk_dummy")
    exc = _http_error(mod, 401)
    res = mod._request_failed(exc, action="image generation", mode="quest", fail=_fail(mod))
    assert res["error_type"] == "auth_required", res
    assert "app" in res["error"].lower() and "sk_" in res["error"], res


def test_sk_401_keeps_key_guidance():
    mod = _require()
    setattr(mod, "_api_key", lambda: "sk_dummy")
    exc = _http_error(mod, 401)
    res = mod._request_failed(exc, action="image generation", mode="quest", fail=_fail(mod))
    assert res["error_type"] == "auth_required", res
    assert "POLLINATIONS_API_KEY" in res["error"], res


def test_error_fields_extracts_code_and_message():
    mod = _require()
    body = json.dumps({"error": {"code": "BAD_REQUEST", "message": "bad model"}}).encode()
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert code == "BAD_REQUEST", code
    assert preview == "bad model", preview


def test_body_preview_long_json_extracts_message_only():
    mod = _require()
    payload = {"url": "https://cdn.example/img?sig=SUPER_SECRET_SIGNATURE",
               "error": {"message": "m" * 600}}
    body = json.dumps(payload).encode()
    assert len(body) > 500
    _, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPER_SECRET" not in preview and "sig=" not in preview, preview
    assert preview.startswith("mmm"), preview
    assert len(preview) <= 200, len(preview)


def test_body_preview_non_json_redacts_urls():
    mod = _require()
    body = b"failed see https://cdn.example/x?sig=SECRET now"
    _, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SECRET" not in preview, preview
    assert "https://" not in preview, preview


def test_body_preview_strips_control_chars():
    mod = _require()
    body = json.dumps({"error": {"message": "bad\x00\x1bmodel\nhere"}}).encode()
    _, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "\x00" not in preview and "\x1b" not in preview and "\n" not in preview, preview


def test_binary_error_body_not_echoed():
    mod = _require()
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
    code, preview, _fix = mod._error_fields(_FakeResponse(jpeg))
    assert code is None, code
    assert preview == "(non-text error body)", preview


def test_oversize_error_body_preserves_status_mapping():
    mod = _require()
    # A 429 with a 100KiB image body (the classic route can answer errors with a full image):
    # the preview must not raise, and the _HTTPError status/Retry-After mapping must survive.
    big = b"\xff\xd8\xff\xe0" + b"\x00" * (100 * 1024)
    resp = _FakeResponse(big, headers={"retry-after": "45"}, status_code=429)

    class _Ctx:
        def __enter__(self):
            return resp

        def __exit__(self, *a):
            return False

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            return _Ctx()

    orig = mod._ssrf_client
    setattr(mod, "_ssrf_client", lambda timeout, **kw: _Client())
    try:
        try:
            mod._http_get_image("https://image.pollinations.ai/prompt/x")
            raise AssertionError("expected _HTTPError")
        except mod._HTTPError as exc:
            assert exc.status == 429, exc.status
            assert exc.retry_after == 45, exc.retry_after
            res = mod._request_failed(exc, action="image generation", mode="free",
                                      fail=_fail(mod))
            assert res["error_type"] == "rate_limited", res
    finally:
        setattr(mod, "_ssrf_client", orig)


def test_oversize_non_json_error_body_redacts_urls():
    mod = _require()
    # The URL must sit INSIDE the 64KiB prefix _error_fields reads, or the test passes
    # vacuously (truncation hides the secret even with redaction deleted).
    body = ("upstream failure https://cdn.example/img?sig=SUPERSECRET "
            + "x" * (100 * 1024)).encode()
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview
    assert "[url]" in preview, preview
    assert len(preview) <= 200, len(preview)


def test_escaped_signed_url_redacted_in_truncated_body():
    mod = _require()
    # A JSON body longer than the 64KiB prefix: truncation breaks json.loads, so the raw-text
    # fallback must still redact JSON-escaped slashes (https:\/\/). The URL sits INSIDE the
    # prefix so the fallback actually sees it.
    body = ('{"url": "https:\\/\\/cdn.example\\/img?sig=SUPERSECRET", '
            '"error": {"message": "' + "x" * 70000 + '"}}').encode()
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview
    assert "cdn.example" not in preview, preview


def test_escaped_signed_url_redacted_in_short_invalid_body():
    mod = _require()
    body = b"not json at all https:\\/\\/cdn.example\\/img?sig=SUPERSECRET tail"
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview
    assert "cdn.example" not in preview, preview


def test_escaped_url_redacted_in_valid_json_without_message():
    mod = _require()
    # Parses fine but carries no error.message: the fallback sees the RAW text, where the
    # slashes are still escaped (plain and \\u002f forms).
    body = b'{"url": "https:\\/\\/cdn.example\\/img?sig=SUPERSECRET", "code": 400}'
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview
    body = b'{"url": "https:\\u002f\\u002fcdn.example\\/img?sig=SUPERSECRET"}'
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview


def test_signed_url_inside_error_message_redacted():
    mod = _require()
    body = json.dumps({"error": {"message": "fetch failed https://cdn.example/img?sig=SUPERSECRET"}}).encode()
    code, preview, _fix = mod._error_fields(_FakeResponse(body))
    assert "SUPERSECRET" not in preview, preview
    assert "fetch failed" in preview, preview


class _RawBodyResponse:
    headers = {}

    def __init__(self, body: bytes):
        self._body = body

    def iter_bytes(self):
        yield self._body


def test_double_escaped_urls_redacted():
    mod = _require()
    dbl = b"\\\\"  # two literal backslashes
    bodies = [
        b"not json https:" + dbl + b"/" + dbl + b"/cdn.example/img?sig=DOUBLE tail",
        b"not json https:" + dbl + b"u002f" + dbl + b"u002fcdn.example/img?sig=UNI tail",
        b"not json https:" + dbl + b"/" + b"\\/" + b"cdn.example/img?sig=MIXED tail",
    ]
    for body in bodies:
        _, preview, _fix = mod._error_fields(_RawBodyResponse(body))
        assert "[url]" in preview, preview
        for secret in ("DOUBLE", "UNI", "MIXED"):
            assert secret not in preview, preview


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
