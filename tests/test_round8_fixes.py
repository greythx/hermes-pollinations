"""Round-8 residual redaction fixes: semicolon-delimited fields and URL tails.

Stdlib-only.  Every case loads a fresh copy of the plugin source so the file can
also run unchanged from a mutation-scratch directory.  No live generation and
no network access.  Run ``python tests/test_round8_fixes.py``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
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
            raise RuntimeError("Hermes core not importable; plugin cannot load")
    spec = importlib.util.spec_from_file_location("pollinations_round8_fixes", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()
SECRET = "LEAKME_9988"


def test_semicolon_code_prose_then_signature_secret_is_masked():
    """'; code=404' is prose, but the following '; sig=' field must still mask."""
    out = MOD._redact_urls(f"; code=404; sig={SECRET}")
    assert SECRET not in out, out
    assert "[redacted]" in out, out
    assert "; code=404" in out, out


def test_semicolon_key_prose_then_token_secret_is_masked():
    """'; key=abc' is prose, but the following '; token=' field must still mask."""
    out = MOD._redact_urls(f"; key=abc; token={SECRET}")
    assert SECRET not in out, out
    assert "[redacted]" in out, out
    assert "; key=abc" in out, out


def test_schemeless_host_path_code_then_signature_secret_is_masked():
    """Scheme-less authority/path forms need the same per-field scan."""
    out = MOD._redact_urls(f"host/x; code=404; sig={SECRET}")
    assert SECRET not in out, out
    assert "[redacted]" in out, out
    assert "host/x; code=404" in out, out


def test_url_attached_semicolon_secret_spaced_and_unspaced():
    """A URL's trailing ';' must not hide the credential field after it."""
    spaced = MOD._redact_urls(f"https://host/p; sig={SECRET}")
    assert SECRET not in spaced, spaced
    assert spaced == "[url]; sig=[redacted]", spaced

    unspaced = MOD._redact_urls(f"https://host/p;sig={SECRET}")
    assert SECRET not in unspaced, unspaced
    # An unspaced ';' is consumed by the contiguous URL, so the whole string is
    # one URL and is fully masked (no credential field survives to be scanned).
    assert unspaced in ("[url];sig=[redacted]", "[url]"), unspaced


def test_internal_semicolon_url_redacted_without_tail_leak():
    """RED (CodeRabbit R3): a contiguous URL's internal ';' are part of the URL.

    'see https://host/a;b?c=1;d=2 now' must redact the WHOLE URL to '[url]'
    with no ';b?c=1;d=2' tail left behind.
    """
    out = MOD._redact_urls("see https://host/a;b?c=1;d=2 now")
    assert out == "see [url] now", out
    assert ";" not in out, out
    assert "https://host/a" not in out, out


def test_semicolon_url_fix_preserves_green_behavior():
    """GREEN: credential-field and prose behavior must survive the URL fix."""
    # A ';' followed by whitespace is NOT consumed by the URL: the field after
    # it stays visible to _SECRET_TAIL_RE and is masked there.
    assert MOD._redact_urls(f"https://host/p; sig={SECRET}") == \
        "[url]; sig=[redacted]"
    # A ';' followed by a double quote is likewise unconsumed: the URL stops
    # at the quote and the credential field after the ';' is masked separately.
    assert MOD._redact_urls(f'https://host/p"; sig={SECRET}') == \
        '[url]"; sig=[redacted]'
    # Unspaced credential field: fully masked as one URL is acceptable.
    unspaced = MOD._redact_urls(f"https://host/p;sig={SECRET}")
    assert SECRET not in unspaced, unspaced
    assert unspaced in ("[url];sig=[redacted]", "[url]"), unspaced
    # Standalone semicolon prose is unchanged; a secret after it still masks.
    assert MOD._redact_urls("; code=404 is common") == "; code=404 is common"
    secret_prose = MOD._redact_urls(f"; code=404; sig={SECRET}")
    assert SECRET not in secret_prose, secret_prose
    assert "[redacted]" in secret_prose, secret_prose
    assert "; code=404" in secret_prose, secret_prose
    # JSON-escaped and unicode-escaped scheme URLs still redact.
    for text in (r"https:\/\/host/a?sig=SECRET",
                 r"\u0068ttps://host/a?sig=SECRET"):
        out = MOD._redact_urls(text)
        assert "SECRET" not in out, (text, out)
        assert "[url]" in out, (text, out)


def test_semicolon_url_regex_has_no_catastrophic_backtracking():
    """A 64KB semicolon/backslash flood must redact in well under 0.5s."""
    flood = "https://" + ";\\" * 32768  # 65536 flood chars
    started = time.monotonic()
    MOD._redact_urls(flood)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, "redaction took %.3fs on a 64KB flood" % elapsed


def test_existing_prose_and_query_secrets_keep_their_behavior():
    """GREEN assertions carried forward from the earlier redaction rounds."""
    for text in ("; code=404 is common", "; key=value in config",
                 "a normal sentence about sigmas and keys",
                 "press ; then type = to continue"):
        assert MOD._redact_urls(text) == text, text
    for text in ("; sig=SECRET", "?sig%3DSECRET", "?sig=%3DSECRET",
                 "&token=SECRET", "host/x?sig=SECRET"):
        out = MOD._redact_urls(text)
        assert "SECRET" not in out, (text, out)
        assert "[redacted]" in out, (text, out)


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as exc:
            failed += 1
            print("FAIL", name, exc)
        except Exception as exc:
            failed += 1
            print("ERROR", name, type(exc).__name__, exc)
    print("\n%d passed, %d failed" % (len(tests) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
