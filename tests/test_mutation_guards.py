"""Minimal permanent guards for the nine MUTATION-V2 baseline survivors.

One focused assertion per mutation. Each passes on the shipped v1.7.0 source
and fails on its mutant; together they take the gate to 42/42 killed.
Stdlib only: run ``python3 tests/test_mutation_guards.py``.
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
    spec = importlib.util.spec_from_file_location("pollinations_mutation_guards",
                                                  ROOT / "__init__.py")
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


def _stub(mod, *, catalog=(), classic=(), key="sk", config=None):
    setattr(mod, "_api_key", lambda: key)
    setattr(mod, "load_image_gen_config", lambda: dict(config or {}))
    setattr(mod, "_catalog", lambda: [dict(m) for m in catalog])
    setattr(mod, "_classic_models", lambda: list(classic))
    setattr(mod, "_include_paid_models", lambda pid: False)
    setattr(mod, "_spec_limits",
            lambda: {"edit_prompt_max": 32000, "seed_min": -1, "seed_max": 2147483647})


def test_M1_alias_shadow_guard():
    mod = _require()
    _stub(mod, catalog=[{"name": "FLUX", "category": "image", "paid_only": True},
                        {"name": "flux", "category": "image", "paid_only": False}])
    assert mod._resolve_model("flux", mod._catalog()) == "flux"


def test_M2_apostrophe_tail_guard():
    mod = _require()
    assert "brien" not in mod._redact_urls("x https://e.com/o'brien/tok/S here")


def test_M2_protocol_relative_guard():
    mod = _require()
    assert mod._redact_urls("x //evil.example.com/s here") == "x [url] here"


def test_F6_dup_rows_guard():
    mod = _require()
    free = {"category": "image", "paid_only": False, "flat_rate": True,
            "pricing": {"currency": "pollen", "completionImageTokens": "0"}}
    _stub(mod, key=None, classic=["model"],
          catalog=[dict(free, name="Model"), dict(free, name="model")])
    assert len(mod.PollinationsImageGenProvider().list_models()) == 1


def test_F16_video_capabilities_guard():
    mod = _require()
    _stub(mod, config={"model": "vid"},
          catalog=[{"name": "vid", "category": "video", "paid_only": False,
                    "supported_endpoints": ["/v1/images/edits"], "max_reference_images": 2}])
    caps = mod.PollinationsImageGenProvider().capabilities()
    assert caps["modalities"] == ["text"] and caps["max_reference_images"] == 0, caps


def test_R2P4_free_name_lie_guard():
    mod = _require()
    assert mod._advertises_free({"name": "freebie-pro", "pricing": {}}) is False


def test_L13_retry_date_clamp_guard():
    mod = _require()

    class Resp:
        headers = {"retry-after": "Wed, 01 Jan 1970 00:00:00 GMT"}

    assert mod._retry_after(Resp()) == 0


def test_MP4_home_keyed_catalog_guard():
    mod = _require()
    setattr(mod, "_hermes_home", lambda: Path("/tmp/mguard_a"))
    a = mod._catalog_state()
    a["negative_until"] = 7.0
    setattr(mod, "_hermes_home", lambda: Path("/tmp/mguard_b"))
    assert mod._catalog_state()["negative_until"] == 0.0 and mod._catalog_state() is not a


def test_MP4_home_keyed_classic_guard():
    mod = _require()
    setattr(mod, "_hermes_home", lambda: Path("/tmp/mguard_a"))
    a = mod._classic_state()
    a["at"] = 7.0
    setattr(mod, "_hermes_home", lambda: Path("/tmp/mguard_b"))
    assert mod._classic_state()["at"] == 0.0 and mod._classic_state() is not a


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
