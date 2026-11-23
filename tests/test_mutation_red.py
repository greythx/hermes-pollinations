"""RED regression tests for the nine MUTATION-V2 baseline survivors.

Every test here asserts the CORRECT v1.7.0 behavior and therefore FAILS on the
mutant named in its docstring (verified with ``python tests/mutation/run.py
--only NAME``) and PASSES on the shipped source. Stdlib only: run
``python3 tests/test_mutation_red.py``.
"""

from __future__ import annotations

import importlib.util
import os
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
    spec = importlib.util.spec_from_file_location("pollinations_mutation_red",
                                                  ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    MOD = _load()
except unittest.SkipTest:
    MOD = None

_SEAMS = ("_api_key", "load_image_gen_config", "_catalog", "_classic_models",
          "_include_paid_models", "_spec_limits", "save_b64_image", "_hermes_home")
_PRISTINE = {n: getattr(MOD, n) for n in _SEAMS} if MOD is not None else {}


def _require():
    if MOD is None:
        raise unittest.SkipTest("Hermes core not importable; plugin cannot load")
    return MOD


def _setup(mod, *, catalog=None, classic=(), key="sk_dummy", config=None):
    for n, v in _PRISTINE.items():
        setattr(mod, n, v)
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in (catalog or [])])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})
    setattr(mod, "save_b64_image",
            lambda b64, *, prefix="image", extension="png": "/tmp/mutation_test.png")
    os.environ.pop(mod.MODEL_ENV, None)


def _free(name, **extra):
    return dict({"name": name, "category": "image", "paid_only": False, "flat_rate": True,
                 "pricing": {"currency": "pollen", "completionImageTokens": "0"}}, **extra)


# --- M1: an exact canonical id beats a case-folded first-wins alias claim -----
def test_M1_alias_shadow_red():
    """KILLS M1_alias_shadow: exact canonical name wins over folded first entry."""
    mod = _require()
    _setup(mod, classic=[])
    catalog = [
        {"name": "FLUX", "category": "image", "paid_only": True},
        {"name": "flux", "category": "image", "paid_only": False},
    ]
    # The direct canonical check must return the caller's exact spelling; the
    # mutant falls through to the case-folded index (first entry wins) -> "FLUX".
    assert mod._resolve_model("flux", catalog) == "flux"


# --- M2: apostrophe is a legal URL tail character ----------------------------
def test_M2_apostrophe_tail_red():
    """KILLS M2_apostrophe_tail: an apostrophe must not split a signed URL."""
    mod = _require()
    preview = mod._redact_urls("see https://example.com/o'brien/tok/SECRETVALUE here")
    assert preview == "see [url] here", preview
    assert "brien" not in preview and "SECRETVALUE" not in preview, preview


# --- M2: protocol-relative URLs are redacted too ------------------------------
def test_M2_protocol_relative_red():
    """KILLS M2_protocol_relative: //host URLs must match the redactor."""
    mod = _require()
    preview = mod._redact_urls("see //evil.example.com/signed/path?sig=ABC here")
    assert preview == "see [url] here", preview
    assert "evil.example.com" not in preview, preview


# --- F6: case-variant catalog ids collapse into one picker row ----------------
def test_F6_dup_rows_red():
    """KILLS F6_dup_rows: keyless picker dedupes catalog case variants."""
    mod = _require()
    _setup(mod, key=None, classic=["model"],
           catalog=[_free("Model"), _free("model")])
    rows = mod.PollinationsImageGenProvider().list_models()
    ids = [row["id"] for row in rows]
    assert ids == ["Model"], ids


# --- F16: a video configured model gets text-only capabilities ----------------
def test_F16_video_capabilities_red():
    """KILLS F16_video_capabilities: video entry must not advertise editing."""
    mod = _require()
    _setup(mod, key="sk_dummy", config={"model": "vid"},
           catalog=[{"name": "vid", "category": "video", "paid_only": False,
                     "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 2}])
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps == {"modalities": ["text"], "max_reference_images": 0}, caps


# --- R2-P4: a name containing "free" as a substring is not a free claim ------
def test_R2P4_free_name_lie_red():
    """KILLS R2P4_free_name_lie: word-boundary free match, not substring."""
    mod = _require()
    assert mod._advertises_free({"name": "freebie-pro", "pricing": {}}) is False
    assert mod._advertises_free({"name": "free pro", "pricing": {}}) is True


# --- L13: a past HTTP-date Retry-After clamps to zero ------------------------
def test_L13_retry_date_clamp_red():
    """KILLS L13_retry_date_clamp: HTTP-date delay is clamped to >=0."""
    mod = _require()

    class Resp:
        headers = {"retry-after": "Wed, 01 Jan 1970 00:00:00 GMT"}

    delay = mod._retry_after(Resp())
    assert delay == 0, delay
    assert 0 <= delay <= mod.RETRY_AFTER_MAX, delay


# --- MP-4: catalog cache state is scoped to the active Hermes home -----------
def test_MP4_home_keyed_catalog_red():
    """KILLS MP4_home_keyed_catalog: per-home failure windows are independent."""
    mod = _require()
    try:
        mod._hermes_home = lambda: Path("/tmp/mutation_home_a")
        a = mod._catalog_state()
        a["negative_until"] = 111.0
        mod._hermes_home = lambda: Path("/tmp/mutation_home_b")
        b = mod._catalog_state()
        assert a is not b, "catalog state shared across homes"
        assert b["negative_until"] == 0.0, b
    finally:
        mod._hermes_home = _PRISTINE["_hermes_home"]


# --- MP-4: classic cache state is scoped to the active Hermes home -----------
def test_MP4_home_keyed_classic_red():
    """KILLS MP4_home_keyed_classic: per-home classic timestamps are independent."""
    mod = _require()
    try:
        mod._hermes_home = lambda: Path("/tmp/mutation_home_a")
        a = mod._classic_state()
        a["at"] = 111.0
        mod._hermes_home = lambda: Path("/tmp/mutation_home_b")
        b = mod._classic_state()
        assert a is not b, "classic state shared across homes"
        assert b["at"] == 0.0, b
    finally:
        mod._hermes_home = _PRISTINE["_hermes_home"]


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
