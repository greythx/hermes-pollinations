"""Coverage-gap regression tests (v1.7.x test hardening).

Locks down defensive branches that review probes exercised but the suite never
ran: secret-backend degradation, config hostility, setup/register wiring,
capabilities resolution control, local/remote source-ingestion guards, edit
error/result branches, cache-write fallbacks, route modes, WebP variants, and
edit-hint bounds. Stdlib-only: run ``python3 tests/test_coverage_gaps.py``.
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
SCRATCH = Path(__file__).resolve().parent / "_scratch_covgap"


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
    spec = importlib.util.spec_from_file_location("pollinations_covgap", ROOT / "__init__.py")
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
          "_ssrf_client", "cache_url_best_effort", "record_token_usage",
          "mkdir_under_hermes_home", "MAX_EDIT_PAYLOAD_BYTES", "MAX_INPUT_IMAGE_BYTES")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


QUEST = {"name": "test/quest-model", "category": "image", "paid_only": False, "flat_rate": True,
         "pricing": {"currency": "pollen", "completionImageTokens": "0.002"}}
EDIT_ENTRY = dict(QUEST, name="test/edit-model", supported_endpoints=["/v1/images/edits"],
                  max_reference_images=4)


def _setup(mod, *, catalog=None, key: "str | None" = "sk_dummy", config=None, classic=()):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    setattr(mod, "_hermes_home", lambda: SCRATCH)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog if catalog is not None else [QUEST])])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": f"/tmp/{prefix}_test.{extension}")
    mod._catalog_state().update({"negative_until": 0.0, "last_success": 0.0})
    mod._classic_state().update({"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})
    mod._spec_state().update({"negative_until": 0.0, "last_success": 0.0})


def _png(w=1, h=1):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


PNG_B64 = base64.b64encode(_png()).decode()


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception as other:
        raise AssertionError(f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, no exception")


# --- secret backend degradation ---------------------------------------------

def test_api_key_unscoped_warns_once():
    mod = _require()
    _setup(mod)
    setattr(mod, "_api_key", _PRISTINE["_api_key"])  # the REAL reader, not the setup stub
    import agent.secret_scope as ss
    orig = ss.get_secret
    mod._UNSCOPED_WARNED_HOMES.clear()

    def unscoped(name, *a, **kw):
        raise ss.UnscopedSecretError(name)

    ss.get_secret = unscoped
    try:
        assert mod._api_key() is None
        assert str(SCRATCH) in mod._UNSCOPED_WARNED_HOMES
        assert mod._api_key() is None  # second call: debug path, still keyless
    finally:
        ss.get_secret = orig
        mod._UNSCOPED_WARNED_HOMES.clear()


def test_api_key_backend_failure_and_shapes():
    mod = _require()
    _setup(mod)
    setattr(mod, "_api_key", _PRISTINE["_api_key"])
    import agent.secret_scope as ss
    orig = ss.get_secret
    try:
        def boom(name, *a, **kw):
            raise RuntimeError("backend down")
        ss.get_secret = boom
        assert mod._api_key() is None
        ss.get_secret = lambda name, *a, **kw: "   "
        assert mod._api_key() is None
        ss.get_secret = lambda name, *a, **kw: b"sk_bytes"
        assert mod._api_key() is None
        ss.get_secret = lambda name, *a, **kw: " sk_ok "
        assert mod._api_key() == "sk_ok"
    finally:
        ss.get_secret = orig


def test_model_env_degradation():
    mod = _require()
    _setup(mod)
    import agent.secret_scope as ss
    orig = ss.get_secret
    try:
        def unscoped(name, *a, **kw):
            raise ss.UnscopedSecretError(name)
        ss.get_secret = unscoped
        assert mod._model_env() == ""
        def boom(name, *a, **kw):
            raise RuntimeError("down")
        ss.get_secret = boom
        assert mod._model_env() == ""
        ss.get_secret = lambda name, *a, **kw: " some/model "
        assert mod._model_env() == "some/model"
    finally:
        ss.get_secret = orig


def test_include_paid_hostile_config():
    mod = _require()
    setattr(mod, "_include_paid_models", _PRISTINE["_include_paid_models"])
    import hermes_cli.config as hc
    orig = hc.load_config_readonly
    pid = "image_gen/pollinations"
    try:
        for cfg, want in [
            (None, False),
            ({"plugins": None}, False),
            ({"plugins": {"entries": {pid: None}}}, False),
            ({"plugins": {"entries": {pid: {"settings": None}}}}, False),
            ({"plugins": {"entries": {pid: {"settings": {"include_paid_models": "true"}}}}}, True),
            ({"plugins": {"entries": {pid: {"settings": {"include_paid_models": "false"}}}}}, False),
            ({"plugins": {"entries": {pid: {"settings": {"include_paid_models": []}}}}}, False),
            ({"plugins": {"entries": {pid: {"settings": {"include_paid_models": True}}}}}, True),
        ]:
            hc.load_config_readonly = lambda cfg=cfg: cfg
            assert mod._include_paid_models(pid) is want, (cfg, want)
        def boom():
            raise RuntimeError("config unreadable")
        hc.load_config_readonly = boom
        assert mod._include_paid_models(pid) is False
    finally:
        hc.load_config_readonly = orig


# --- setup / register / capabilities ----------------------------------------

def test_setup_schema_shape():
    mod = _require()
    schema = mod.PollinationsImageGenProvider().get_setup_schema()
    assert isinstance(schema, dict) and schema, schema
    text = json.dumps(schema)
    assert "POLLINATIONS_API_KEY" in text, text
    assert "enter.pollinations.ai/keys" in text, text


def test_register_ctx_fallback():
    mod = _require()

    class Ctx:
        def __init__(self, pid="__missing__"):
            if pid != "__missing__":
                self.plugin_id = pid
            self.registered = []

        def register_image_gen_provider(self, provider):
            self.registered.append(provider)

    for pid, want in [(None, mod.PLUGIN_ID), ("", mod.PLUGIN_ID),
                      ("__missing__", mod.PLUGIN_ID), ("image_gen/pollinations", "image_gen/pollinations")]:
        ctx = Ctx(pid)
        mod.register(ctx)
        assert len(ctx.registered) == 1
        assert ctx.registered[0]._plugin_id == want, (pid, want)


def test_capabilities_resolution_control():
    mod = _require()
    entry = dict(EDIT_ENTRY, resolutions=["1k", "2k"])
    _setup(mod, catalog=[entry], config={"model": "test/edit-model"})
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps.get("creative_controls") == ["resolution"], caps
    assert caps["modalities"] == ["text", "image"], caps
    _setup(mod, catalog=[entry], key=None, config={"model": "test/edit-model"})
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps.get("creative_controls") == [], caps
    assert caps["modalities"] == ["text"], caps


# --- local source ingestion ---------------------------------------------------

def test_local_ingestion_guards():
    mod = _require()
    _setup(mod)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    blocked = SCRATCH / ".env"
    blocked.write_text("SECRET=***")
    assert _raises(lambda: mod._local_image_to_data_url(str(blocked)))
    assert _raises(lambda: mod._local_image_to_data_url(str(SCRATCH / "missing.png")))
    assert _raises(lambda: mod._local_image_to_data_url(str(SCRATCH)))  # a directory
    empty = SCRATCH / "empty.png"
    empty.write_bytes(b"")
    assert _raises(lambda: mod._local_image_to_data_url(str(empty)))
    text = SCRATCH / "text.png"
    text.write_bytes(b"this is not an image at all")
    assert _raises(lambda: mod._local_image_to_data_url(str(text)))
    good = SCRATCH / "good.png"
    good.write_bytes(_png(3, 2))
    assert mod._local_image_to_data_url(str(good)).startswith("data:image/png;base64,")
    assert _raises(lambda: mod._local_image_to_data_url(str(good), limit=10))


# --- remote source ingestion --------------------------------------------------

class _FakeResp:
    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield self._body


class _FakeStream:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def stream(self, method, url):
        return _FakeStream(self._resp)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_remote_ssrf_blocked():
    mod = _require()
    _setup(mod)

    def no_network(timeout, **kw):
        raise AssertionError("SSRF guard let a private address reach the HTTP client")

    setattr(mod, "_ssrf_client", no_network)
    # Assert the GUARD rejects (its specific error), not just that some ValueError happens:
    # a regressed guard would reach no_network and fail loudly instead of passing vacuously.
    for url in ("http://169.254.169.254/latest/meta-data", "http://127.0.0.1:8080/x.png"):
        try:
            mod._remote_image_to_data_url(url)
            raise AssertionError(f"SSRF guard did not block {url}")
        except ValueError as exc:
            assert "SSRF" in str(exc), (url, exc)


def test_remote_content_length_and_body():
    mod = _require()
    _setup(mod)
    setattr(mod, "_ssrf_client",
            lambda timeout, **kw: _FakeClient(_FakeResp(_png(), {"content-length": "999999999"})))
    assert _raises(lambda: mod._remote_image_to_data_url("https://example.com/a.png", limit=1000))
    setattr(mod, "_ssrf_client", lambda timeout, **kw: _FakeClient(_FakeResp(_png(3, 2))))
    assert mod._remote_image_to_data_url("https://example.com/a.png").startswith("data:image/png;base64,")
    setattr(mod, "_ssrf_client",
            lambda timeout, **kw: _FakeClient(_FakeResp(b"not an image")))
    assert _raises(lambda: mod._remote_image_to_data_url("https://example.com/a.png"))


# --- edit endpoint pinning ----------------------------------------------------

def test_edit_url_hostile_matrix():
    mod = _require()
    hostile = [
        "https://evil.example.com/v1/images/edits",
        "http://gen.pollinations.ai/v1/images/edits",
        "https://gen.pollinations.ai:8443/v1/images/edits",
        "https://user:pass@gen.pollinations.ai/v1/images/edits",
        "https://gen.pollinations.ai.evil.com/v1/images/edits",
        "/../v1/images/edits",
        "/v1/audio/edits",
        "//evil.example.com/v1/images/edits",
    ]
    for endpoint in hostile:
        assert mod._edit_url({"supported_endpoints": [endpoint]}) == mod.EDIT_URL, endpoint
    assert mod._edit_url({"supported_endpoints": ["/v2/images/edits"]}) == \
        "https://gen.pollinations.ai/v2/images/edits"
    assert mod._edit_url({}) == mod.EDIT_URL
    assert mod._edit_url(None) == mod.EDIT_URL


# --- edit response branches ---------------------------------------------------

def _edit_call(mod, post=None, extra_seams=None):
    _setup(mod, catalog=[EDIT_ENTRY])
    setattr(mod, "_to_data_url", lambda v, limit=1000000: "data:image/png;base64," + PNG_B64)
    if post is not None:
        setattr(mod, "_http_post_json", post)
    for name, value in (extra_seams or {}).items():
        setattr(mod, name, value)  # AFTER setup, which restores pristine seams
    return mod.PollinationsImageGenProvider().generate(
        "edit", model="test/edit-model", image_url="https://example.com/a.png")


def test_edit_http_errors_map():
    mod = _require()
    def post_429(url, *, headers, payload, timeout=120.0):
        raise mod._HTTPError(url, 429, retry_after=30)
    res = _edit_call(mod, post=post_429)
    assert res["success"] is False and res["error_type"] == "rate_limited", res
    def post_401(url, *, headers, payload, timeout=120.0):
        raise mod._HTTPError(url, 401)
    res = _edit_call(mod, post=post_401)
    assert res["success"] is False and res["error_type"] == "auth_required", res


def test_edit_inband_error_shapes():
    mod = _require()
    res = _edit_call(mod, post=lambda url, **kw: {"error": {"message": "structured failure"}})
    assert res["success"] is False and "structured failure" in res["error"], res
    res = _edit_call(mod, post=lambda url, **kw: {"error": "plain failure"})
    assert res["success"] is False and "plain failure" in res["error"], res
    res = _edit_call(mod, post=lambda url, **kw: "garbage")
    assert res["success"] is False and res["error_type"] == "invalid_response", res
    res = _edit_call(mod, post=lambda url, **kw: {"data": [{"b64_json": "!!!not-base64!!!"}]})
    assert res["success"] is False and res["error_type"] == "invalid_response", res
    res = _edit_call(mod, post=lambda url, **kw: {"data": []})
    assert res["success"] is False, res


def test_edit_url_result_and_usage():
    mod = _require()
    cached = []
    recorded = []
    seams = {
        "cache_url_best_effort": lambda url, **kw: cached.append(url) or "/tmp/cached_edit.png",
        "record_token_usage": lambda u, *, model, provider, base_url=None: recorded.append((u, model, provider)),
    }
    res = _edit_call(mod, post=lambda url, **kw: {
        "data": [{"url": "https://media.pollinations.ai/x.png"}],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}, extra_seams=seams)
    assert res["success"] is True, res
    assert res.get("remote") is True, res
    assert cached == ["https://media.pollinations.ai/x.png"], cached
    assert recorded and recorded[0][1] == "test/edit-model", recorded
    res = _edit_call(mod, post=lambda url, **kw: {
        "data": [{"b64_json": PNG_B64}], "usage": {"total_tokens": 7}}, extra_seams=seams)
    assert res["success"] is True, res


def test_save_oserror_maps_io_error():
    mod = _require()
    _setup(mod, catalog=[QUEST])
    setattr(mod, "_http_get_image", lambda url, *, headers=None, timeout=120.0: (_png(), None))
    def bad_save(b64, *, prefix="image", extension="png"):
        raise OSError("disk full")
    setattr(mod, "save_b64_image", bad_save)
    res = mod.PollinationsImageGenProvider().generate("a cat", model="test/quest-model")
    assert res["success"] is False and res["error_type"] == "io_error", res
    res = _edit_call(mod, post=lambda url, **kw: {"data": [{"b64_json": PNG_B64}]},
                     extra_seams={"save_b64_image": bad_save})
    assert res["success"] is False and res["error_type"] == "io_error", res


# --- cache-write fallbacks ----------------------------------------------------

def test_guarded_write_without_fcntl():
    mod = _require()
    _setup(mod)
    saved = sys.modules.get("fcntl")
    sys.modules["fcntl"] = None  # type: ignore[assignment]  # forces ImportError on `import fcntl`
    try:
        ok = mod._guarded_cache_write(mod._cache_path(),
                                      {"version": 1, "fetched": time.time(), "models": []},
                                      time.time())
        assert ok is True
        assert mod._cache_path().exists()
    finally:
        if saved is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = saved


def test_guarded_write_lock_open_failure():
    mod = _require()
    _setup(mod)
    orig_open = os.open
    def bad_open(path, *a, **kw):
        if str(path).endswith(".lock"):
            raise OSError("cannot open lock")
        return orig_open(path, *a, **kw)  # the atomic writer still needs os.open
    os.open = bad_open
    try:
        ok = mod._guarded_cache_write(mod._cache_path(),
                                      {"version": 1, "fetched": time.time(), "models": []},
                                      time.time())
        assert ok is True
        assert mod._cache_path().exists()
    finally:
        os.open = orig_open


def test_guarded_write_mkdir_failure():
    mod = _require()
    _setup(mod)
    def bad_mkdir(path):
        raise RuntimeError("home tombstoned")
    setattr(mod, "mkdir_under_hermes_home", bad_mkdir)
    ok = mod._guarded_cache_write(mod._cache_path(),
                                  {"version": 1, "fetched": time.time(), "models": []},
                                  time.time())
    assert ok is False
    assert not mod._cache_path().exists()


# --- route modes --------------------------------------------------------------

def test_route_modes():
    mod = _require()
    _setup(mod, catalog=[QUEST])
    route = mod._text_to_image_route("p", "test/quest-model", QUEST, [QUEST], 1024, 1024, 42, "")
    assert route.error is None
    assert route.extra.get("mode") == "quest", route.extra
    assert route.url.startswith(mod.ENTER_IMAGE_URL), route.url
    paid = dict(QUEST, paid_only=True)
    route = mod._text_to_image_route("p", "test/quest-model", paid, [paid], 1024, 1024, 42, "")
    assert route.extra.get("mode") == "paid", route.extra
    _setup(mod, catalog=[QUEST], key=None, classic=["test/quest-model"])
    route = mod._text_to_image_route("p", "test/quest-model", QUEST, [QUEST], 1024, 1024, 42, "")
    assert route.extra.get("mode") == "free", route.extra
    assert route.url.startswith(mod.CLASSIC_IMAGE_URL), route.url
    assert "Authorization" not in route.headers


# --- WebP dimension variants --------------------------------------------------

def test_webp_variants():
    mod = _require()
    def riff(chunk, payload):
        return b"RIFF" + struct.pack("<I", len(payload)) + b"WEBP" + chunk + payload
    vp8x = riff(b"VP8X", b"\x0a\x00\x00\x00" + b"\x00\x00\x00\x00"
                + (37 - 1).to_bytes(3, "little") + (21 - 1).to_bytes(3, "little"))
    assert mod._webp_dimensions(vp8x) == (37, 21), mod._webp_dimensions(vp8x)
    bits = (40 - 1) | ((25 - 1) << 14)
    vp8l = riff(b"VP8L", b"\x0a\x00\x00\x00" + b"\x2f" + struct.pack("<I", bits))
    assert mod._webp_dimensions(vp8l) == (40, 25), mod._webp_dimensions(vp8l)
    vp8 = riff(b"VP8 ", b"\x0a\x00\x00\x00" + b"\x00\x00\x00" + b"\x9d\x01\x2a"
               + struct.pack("<HH", 50, 30))
    assert mod._webp_dimensions(vp8) == (50, 30), mod._webp_dimensions(vp8)
    assert mod._webp_dimensions(vp8x[:25]) is None
    assert mod._webp_dimensions(b"RIFF\x00\x00\x00\x00WEBPXXXX" + b"\x00" * 20) is None


# --- edit hint bounds ---------------------------------------------------------

def test_edit_hint_bounded():
    mod = _require()
    many = [dict(EDIT_ENTRY, name=f"test/edit-{i}") for i in range(mod.EDIT_HINT_MODELS + 5)]
    names = mod._visible_edit_models(many, include_paid=False)
    assert 0 < len(names) <= mod.EDIT_HINT_MODELS, names
    assert all(n.startswith("test/edit-") for n in names)


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