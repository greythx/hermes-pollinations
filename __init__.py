"""Pollinations image generation — live catalog, hybrid free/Quest/paid transport.

Catalog: ``GET https://gen.pollinations.ai/image/models`` (public, no auth), cached under
``$HERMES_HOME/cache/`` with a TTL. Models Pollinations adds or removes appear here with no
plugin edit; per-model specs, prices and the free-model list all come from the live API.
Nothing about the catalog is hardcoded: if the API cannot be reached and nothing is cached,
the plugin offers no model rather than guessing one that may no longer exist. Every outbound
request — catalog, classic, enter and edits — goes through the shared SSRF guard with a
response size ceiling, and every catalog field is shape-checked, so malformed third-party data
degrades instead of raising.

Transport, chosen per request:

* **free classic** — ``GET https://image.pollinations.ai/prompt/{prompt}``. No key and free,
  but every image carries the Pollinations logo, requests are throttled per IP, and output is
  capped at a reduced size. Used only when no key is set.
* **enter** — ``GET https://gen.pollinations.ai/image/{prompt}`` with ``POLLINATIONS_API_KEY``.
  No logo and no per-IP throttle. Spends the account's Pollen — Quest Pollen for Quest-eligible
  models, Paid Pollen for paid-only ones. A model the classic API serves for free is billed this
  way once a key is set, so the same model is watermarked while keyless and clean once a key is
  added.

Editing: ``POST https://gen.pollinations.ai/v1/images/edits``. Only the enter API can edit, so
a key is mandatory; Hermes exposes image inputs for exactly the models whose catalog entry
advertises that endpoint, and caps the reference count at the model's published limit. Source
images may be URLs, base64 data URIs or local paths, each guarded, sized and inlined as data
URIs in a single JSON call. The 1K/2K ``resolution`` creative control is offered only when the
configured model publishes one of those values and the authenticated route can honour it.

API limits are live too: the edit prompt length and the seed range (including the ``-1``
sentinel) are read from the published OpenAPI spec and cached; when the spec cannot be
reached the plugin skips its client-side pre-checks and lets the server enforce its contract.

Selection: ``model`` kwarg → ``POLLINATIONS_IMAGE_MODEL`` → ``image_gen.pollinations.model``
→ ``image_gen.model`` → the free classic model, else the first Quest-eligible one. Catalog ids
*and* aliases resolve case-insensitively (the catalog's own spelling is restored). An unknown
id set explicitly, via ``POLLINATIONS_IMAGE_MODEL`` or ``image_gen.pollinations.model`` passes
through verbatim, so a model added upstream works the day it appears; an unknown shared
``image_gen.model`` value falls back to the live default instead (usually another provider's
leftover id), unless the free classic API serves it.

Picker visibility follows the credentials, so the list never offers a model the current setup
cannot run: no key → the models the free classic API serves (usually one); a key → every
Quest-eligible model; a key plus the ``include_paid_models`` setting (Desktop → Settings →
Plugins) → the paid-only catalog too. The setting also gates invocation: a paid-only catalog
model is refused while it is off, whichever path selected it. The user's saved model is kept
visible whenever the current state can run it (a keyed passthrough id gets a minimal row, and
a retained paid row is marked disabled), so neither change makes it vanish silently. Rows are ordered by estimated per-image cost:
per-image and per-megapixel rates are normalized, per-token models follow them, and unpublished
prices sort last; keyless, the free classic rows lead regardless of any enter price.

Picker labels carry the access requirement, watermark and edit support, none of which the
catalog publishes: the price column reads ``Free · watermark · reduced size · no key needed``
while keyless,
and once a key is set that same model becomes ``<amount> pollen · no watermark · needs key`` —
or states the billing unit when the catalog prices per token or per megapixel rather than per
image. A classic-served model the catalog omits keeps the watermarked classic label even keyed,
because the enter API does not serve it. Edit-capable models add ``· edits``.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import random
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, TypeGuard
from urllib.parse import quote, urlsplit, urlunsplit

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from plugins.image_gen._common import (
    api_key_setup_schema, cache_url_best_effort, catalog_rows, collect_source_images,
    error_factory, load_image_gen_config, record_token_usage)
from hermes_constants import mkdir_under_hermes_home
from utils import atomic_json_write, read_json_or_empty

logger = logging.getLogger(__name__)

# Warn once per Hermes home about an unscoped secret read (see _api_key).
_UNSCOPED_WARNED_HOMES: set = set()

CATALOG_URL = "https://gen.pollinations.ai/image/models"
CLASSIC_MODELS_URL = "https://image.pollinations.ai/models"
# The published API contract: the edit prompt limit and seed range are read from here live.
SPEC_URL = "https://gen.pollinations.ai/openapi.json"
ENTER_IMAGE_URL = "https://gen.pollinations.ai/image"
EDIT_URL = "https://gen.pollinations.ai/v1/images/edits"
EDIT_URL_PATH = "/v1/images/edits"
CLASSIC_IMAGE_URL = "https://image.pollinations.ai/prompt"
# The authenticated host, so a catalog-advertised path can be turned into a URL.
ENTER_BASE_URL = "https://gen.pollinations.ai"
# The only host a catalog-advertised endpoint may name. Catalog data is fetched from the network,
# so it is not trusted to choose where the user's key and reference images are sent.
ENTER_HOST = "gen.pollinations.ai"

API_KEY_ENVS = ("POLLINATIONS_API_KEY",)
MODEL_ENV = "POLLINATIONS_IMAGE_MODEL"
PROVIDER_ID = "pollinations"
# Fallback registry id for a context without ``plugin_id`` (e.g. a hand-rolled stub); the
# admission validator's probe supplies its own plugin_id. The live value comes from ``ctx.plugin_id``.
PLUGIN_ID = "image_gen/pollinations"
INCLUDE_PAID_KEY = "include_paid_models"
CATALOG_TTL = 3600
# The spec changes rarely; a daily refresh still follows upstream without a plugin edit.
SPEC_TTL = 86400
HTTP_TIMEOUT = 120.0

# Source images are inlined into the edits payload as data URIs, so they are capped before the
# base64 blow-up; the format list is a conservative client allowlist of common raster formats
# (the spec publishes no input-format contract). The same raster allowlist gates generated
# output (SVG and other active formats are refused).
MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
# Hermes' resolution creative control can only express these two values.
RESOLUTION_VOCAB = frozenset({"1k", "2k"})
# The reference cap comes from the untrusted catalog, so it is ceilinged here; the bundled edit
# plugin uses the same bound. Caps observed to date range 1-16; the spec publishes no upper
# bound, so the ceiling is a client guard, not an API fact.
MAX_REFERENCE_IMAGES = 16
# Default reference cap for edit models that publish none: a judgment call — community
# endpoints commonly accept a single image, but over-sending surfaces as a server 400 with a
# readable preview rather than failing silently. Published caps are always honored; a
# published 0 means the model accepts no reference images and disables edits for that entry.
DEFAULT_REFERENCE_CAP = 4
# Every response body is bounded before it is buffered, decoded or written to the cache.
MAX_RESPONSE_BYTES = 25 * 1024 * 1024
# Total inlined reference images for one edit, plus the prompt and JSON framing. A conservative
# client guard (the spec publishes no body-size limit); oversize requests are rejected here
# before the base64 blow-up rather than failing server-side.
MAX_EDIT_PAYLOAD_BYTES = 40 * 1024 * 1024
# Data-URI framing ("data:image/jpeg;base64,") and the smallest source worth fetching: the
# remaining payload budget is converted to a raw-byte limit for the NEXT image, so a small
# reference still fits after a large one instead of every image reserving the full 25MB cap.
_DATA_URI_HEADER_BYTES = 64
_MIN_SOURCE_BYTES = 64
# How long a failed free-model lookup is remembered, so a down endpoint is not retried per call.
NEGATIVE_TTL = 60.0
# Fallback entropy bound for a fresh random seed when the live spec range is unknown —
# a safe default within the current spec, not a claim about it. Caller seeds are validated
# against the range read live from the published OpenAPI spec.
_RANDOM_SEED_MAX = 2**31 - 1
# Client sanity ceiling: a hostile/absurd published range must not put 400-digit seeds into
# URLs; the live range stays authoritative within sane bounds.
_SEED_SANITY_MAX = 2**63 - 1
# Picker column widths: strengths is truncated to keep the row readable.
STRENGTHS_LIMIT = 120
# How many edit-capable models an error message names.
EDIT_HINT_MODELS = 3
# Cache schema marker: a cache written by a different plugin version is ignored rather than
# trusted, so a shape change upstream cannot silently mislabel rows.
CACHE_SCHEMA = 1

# Semantic aspect -> pixel size. Verified at authoring time against every per-model size
# constraint the spec publishes in prose (no machine-readable per-model constraints exist);
# re-verify when the spec changes. The free classic route keeps the aspect ratio within its
# own smaller area cap.
SIZES: Dict[str, tuple[int, int]] = {
    "landscape": (1280, 768),
    "square": (1024, 1024),
    "portrait": (768, 1280),
}

# Typical output area in megapixels, derived from the plugin's own sizes — used only to estimate
# a per-image cost for models the catalog bills per megapixel. Not a Pollinations fact.
_TYPICAL_MP = sum(w * h for w, h in SIZES.values()) / len(SIZES) / 1_000_000

# Picker/spec fields, all live from the catalog. Fields Pollinations leaves unpublished for
# most models (resolutions, max_reference_images, capabilities, brand_url, pricing_dimensions)
# are deliberately absent from the row — a field that is usually blank is noise in the picker.
# They are still read where they change behaviour: pricing_units drives the price label,
# required_safety the strengths line, max_reference_images the edit cap.
MODEL_FIELDS = (
    "display", "speed", "strengths", "price", "publisher", "community", "description",
    "aliases", "price_pollen", "price_class", "rpm", "modalities", "endpoints", "health",
    "added", "flat_rate", "free_classic",
)


# --------------------------------------------------------------------------- helpers

def _api_key() -> Optional[str]:
    """The configured Pollinations key, resolved through Hermes' secret scope.

    ``get_secret`` honours the active profile, so a routed profile never picks up another
    profile's key under multiplexing; it falls through to ``os.environ`` for the ordinary
    single-profile deployment, so a key exported in the shell still works. A scope error means
    no key is bound to this context, which is the same as having none — the free route still
    works.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret

    for name in API_KEY_ENVS:
        try:
            secret = get_secret(name)
            value = secret.strip() if isinstance(secret, str) else ""
        except UnscopedSecretError:
            # Deliberate keyless degradation, but under multiplexing this can also mean a
            # spawn-site scoping bug: warn once so it is visible, then stay quiet.
            home = _home_key()
            if home not in _UNSCOPED_WARNED_HOMES:
                _UNSCOPED_WARNED_HOMES.add(home)
                logger.warning("No secret scope bound for %s; treating it as unset. Under "
                               "multiplexing this can indicate a scoping bug at the spawn site.",
                               name)
            else:
                logger.debug("No secret scope bound for %s; treating it as unset.", name)
            continue
        except Exception as exc:
            # A backend failure is not the same as an absent key: warn, so a keyed setup that
            # silently behaves as keyless is at least visible in the log.
            logger.warning("Could not read %s from the secret backend: %s", name, exc)
            continue
        if value:
            return value
    return None


def _model_env() -> str:
    """The profile-scoped model override.

    get_secret honours the routed profile and falls through to os.environ single-profile, so
    a launch-profile POLLINATIONS_IMAGE_MODEL no longer leaks into a routed profile.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        return str(get_secret(MODEL_ENV) or "").strip()
    except UnscopedSecretError:
        return ""
    except Exception as exc:
        logger.debug("Could not read %s from the secret backend: %s", MODEL_ENV, exc)
        return ""


def _hermes_home() -> Path:
    """The active Hermes home, honouring the context-local profile override.

    ``HERMES_HOME`` alone is not enough: under multiplexing a routed profile's override is
    context-local, and a cache written to the launch profile's home would be invisible (or
    shared) from the routed one.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _cache_path() -> Path:
    return _hermes_home() / "cache" / "pollinations_catalog.json"


def _include_paid_models(plugin_id: str) -> bool:
    """The user's ``include_paid_models`` setting (default False): also list paid-only models.

    Plugin-scoped settings live under ``plugins.entries.<plugin_id>.settings``. The id is the
    registry key Hermes actually loaded the plugin under — ``<category>/<name>`` for a directory
    install, the bare ``name`` for a catalog install — so it is supplied by ``ctx`` rather than
    hardcoded. Reading through ``load_config_readonly()`` is the documented path for a backend
    that is not handed ``ctx`` after registration.
    """
    try:
        from hermes_cli.config import load_config_readonly

        entry = (load_config_readonly() or {}).get("plugins", {}).get("entries", {}).get(plugin_id) or {}
        # Hermes stores a real bool here, but a profile imported by hand or written by another
        # version may hold a string; treat the truthy spellings as on and everything else as off.
        return _as_bool((entry.get("settings") or {}).get(INCLUDE_PAID_KEY))
    except Exception as exc:
        logger.debug("Could not read %s: %s", INCLUDE_PAID_KEY, exc)
        return False


def _ssrf_client(timeout: float, *, follow_redirects: bool = True) -> Any:
    """The shared SSRF-guarded client: every outbound request goes through it."""
    from tools.url_safety import create_ssrf_safe_client

    return create_ssrf_safe_client(timeout=timeout, follow_redirects=follow_redirects)


def _unwrap_list(payload: Any, keys: tuple, *,
                 fallback_item_ok: Optional[Callable[[List[Any]], bool]] = None,
                 fallback_key_ok: Optional[Callable[[str], bool]] = None) -> Optional[List[Any]]:
    """The first list a JSON envelope carries, preferring the named keys.

    Third-party envelopes rename their fields, and a rename must not read as an empty response.
    One level of nesting is followed, so ``{"data": {"models": [...]}}`` resolves too.
    The positional fallback is additionally gated by the caller's shape predicates, so a
    name-shaped non-model list (``tags: [{name: "nsfw"}]``) cannot pose as the catalog and a
    classic envelope must name a models-like key.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                inner = _unwrap_list(value, keys, fallback_item_ok=fallback_item_ok,
                                     fallback_key_ok=fallback_key_ok)
                if inner is not None:
                    return inner
        # Positional fallback for a renamed envelope: skip well-known non-model lists and prefer
        # the first model-shaped list, so an "errors" array cannot pose as the catalog.
        fallback: Optional[List[Any]] = None
        candidates = 0
        for key, value in payload.items():
            if key in ("errors", "warnings", "messages") or not isinstance(value, list):
                continue
            if fallback_key_ok is not None and not fallback_key_ok(key):
                continue
            candidates += 1
            if fallback is None:
                fallback = value
            if any(isinstance(item, dict)
                   and (_valid_id(item.get("name")) or _valid_id(item.get("id")))
                   for item in value):
                if fallback_item_ok is None or fallback_item_ok(value):
                    return value
        # Several qualifying lists with no model-shaped items are AMBIGUOUS: choosing by key
        # order could cache a version list as the model list. Treat it as a failed fetch so
        # the caller's last-good cache survives.
        if (fallback is not None and candidates == 1
                and (fallback_item_ok is None or fallback_item_ok(fallback))):
            return fallback
    return None


def _bounded_read(response: Any, url: str, *, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Drain a streaming response, aborting as soon as the body exceeds *limit*.

    The ceiling is also a wall clock: a client that trickles bytes can hold a worker open long
    past any per-read timeout, so a slow upstream is cut off rather than awaited. ``url`` is taken
    for call-site clarity but never interpolated into an error, which could echo signed query
    parameters back to the user.
    """
    chunks: List[bytes] = []
    total = 0
    deadline = time.monotonic() + HTTP_TIMEOUT
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > limit:
            raise ValueError(f"response exceeds the {limit} byte ceiling")
        if time.monotonic() > deadline:
            raise ValueError("response took too long to arrive")
        chunks.append(chunk)
    return b"".join(chunks)


class _HTTPError(Exception):
    """An upstream HTTP response the plugin could not use, with a safe preview of its body."""

    def __init__(self, url: str, status: int, body: str = "",
                 retry_after: Optional[int] = None,
                 error_code: Optional[str] = None,
                 fix_url: Optional[str] = None) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.url = url
        self.status = status
        self.body = body
        self.retry_after = retry_after
        self.error_code = error_code
        self.fix_url = fix_url


def _retry_after(response: Any) -> Optional[int]:
    """The ``Retry-After`` header in seconds, as a delay or an HTTP-date."""
    raw = (response.headers.get("retry-after") or "").strip()
    try:
        # ASCII decimal shape only, mirroring the price parser: float() alone would read
        # '1_0' and non-ASCII digits as delays.
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw, re.ASCII):
            raise ValueError(raw)
        return min(RETRY_AFTER_MAX, max(0, int(float(raw))))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        delta = parsedate_to_datetime(raw) - datetime.now(timezone.utc)
        return min(RETRY_AFTER_MAX, max(0, int(delta.total_seconds())))
    except Exception:
        return None


def _read_prefix(response: Any, limit: int) -> bytes:
    """Read at most *limit* bytes of a body; never raises on an oversize body.

    An error preview must survive large bodies (the classic route can answer an error status
    with a full image): raising here would replace the ``_HTTPError`` with a generic failure
    and lose the status, Retry-After and the route-aware 402/429 guidance. A truncated JSON
    body falls back to the URL-redacted raw text.
    """
    chunks: List[bytes] = []
    total = 0
    deadline = time.monotonic() + HTTP_TIMEOUT
    for chunk in response.iter_bytes():
        piece = chunk[: limit - total]
        chunks.append(piece)
        total += len(piece)
        if total >= limit or time.monotonic() > deadline:
            break
    return b"".join(chunks)


# URLs in raw error text can be JSON-escaped (https:\/\/), double-escaped, or unicode-
# escaped in ANY scheme character (\u0068ttps\u003a//). _redact_urls decodes those spellings
# (bounded passes), strips control characters that could split a URL, then matches http(s)
# and protocol-relative // URLs whose tail allows apostrophes and stops at whitespace,
# double quotes and backslashes; bare credential query tails are masked separately.
# Backslash runs are bounded: an unbounded \\+ backtracks O(n^2) on a hostile 64KB
# backslash flood (~26s). Realistic JSON escape nesting needs only a few per layer.
_ESCAPE_RE = re.compile(r"\\{1,8}u([0-9a-fA-F]{4})|\\{1,8}(/)")
_URL_RE = re.compile(r"(?:https?:)?//[^\s\"\\]*", re.IGNORECASE)
_SECRET_TAIL_RE = re.compile(
    r"([#?&](?:sig|signature|token|access_token|accessToken|access-token|api_key|api-key|apikey|"
    r"key|password|secret|auth|credential|client_secret|refresh_token|policy|key-pair-id|code|"
    r"session|sid|jwt|bearer|x-amz-[a-z0-9-]+|x-goog-[a-z0-9-]+)=)"
    r"[^\s\"&\\]*", re.IGNORECASE)
# Display clamp for an absurd Retry-After, not an API fact.
RETRY_AFTER_MAX = 86400


def _unescape_match(match: Any) -> str:
    if match.group(2) is not None:
        return "/"
    cp = int(match.group(1), 16)
    if 0xD800 <= cp <= 0xDFFF:
        return ""  # a lone surrogate is not displayable text
    return chr(cp)


def _redact_urls(text: str) -> str:
    """Decode escape spellings, then redact URLs and bare credential query tails.

    Shared by the HTTP-error and in-band 200 edit-error paths so no path can leak a signed
    URL. Control characters are removed BEFORE matching (one hidden inside a URL must not
    split it into unrecognizable halves); the preview shows the decoded, redacted text.
    """
    prev = None
    for _ in range(6):
        if text == prev:
            break
        prev = text
        text = _ESCAPE_RE.sub(_unescape_match, text)
    # Browsers normalize a single-slash https:/host to https://host; match that shape too.
    text = re.sub(r"(https?:)/(?!/)", r"\1//", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t")
    text = _URL_RE.sub("[url]", text)
    return _SECRET_TAIL_RE.sub(r"\1[redacted]", text)


def _error_fields(response: Any) -> tuple[Optional[str], str, Optional[str]]:
    """The upstream error code, a short redacted preview, and the published fixUrl (if any).

    The 64KB bounded prefix is parsed before the 200-char preview cut; oversized bodies fall
    back to URL-redacted raw text (a signed URL must never reach the preview on ANY path).
    A JSON error body is reduced to its ``error.message``; anything else has its URLs redacted.
    fixUrl is read live from the envelope (the spec publishes it per error) so guidance can
    point at the upstream fix page instead of a hardcoded constant.
    """
    raw_bytes = _read_prefix(response, 64 * 1024)
    if _sniff_image_mime(raw_bytes) is not None:
        # An error status can carry an image body (the classic route does this when unhealthy);
        # never echo binary garbage into the agent-visible preview.
        return None, "(non-text error body)", None
    raw = raw_bytes.decode("utf-8", "replace")
    code: Optional[str] = None
    message: Optional[str] = None
    fix_url: Optional[str] = None
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            if isinstance(err.get("code"), str):
                code = err["code"]
            if isinstance(err.get("message"), str):
                message = err["message"]
            candidate = err.get("fixUrl")
            if (isinstance(candidate, str) and candidate.startswith("https://")
                    and len(candidate) <= 2048 and candidate.isprintable()
                    and not any(ch.isspace() for ch in candidate)
                    and "\\" not in candidate):  # browsers read \\ as / in the authority
                # Guidance, not payload: drop query/fragment (credential tails) and reject
                # userinfo so nothing sensitive or injectable reaches the agent-visible error.
                base = candidate.split("?", 1)[0].split("#", 1)[0]
                authority = base[len("https://"):].split("/", 1)[0]
                # ASCII-only authority: UTS46 maps fullwidth ＼/﹨/＠ to \/@, which browsers
                # read as authority separators (host spoofing). Legitimate fix hosts are ASCII.
                if "@" not in authority and authority.isascii():
                    fix_url = base
        elif isinstance(err, str):
            message = err
    if message is None:
        message = raw
    # Redact on EVERY path (extracted message included): a signed URL must never reach the
    # agent-visible preview, in plain, JSON-escaped or unicode-escaped form, before the cut.
    preview = _redact_urls(message).strip()[:200]
    return code, preview, fix_url


def _network_error_type(exc: Exception) -> str:
    """A stable error_type for a failed HTTP call, rather than leaking the exception class name."""
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connect" in name or "connection" in name:
        return "connection_error"
    return "api_error"


def _request_failed(exc: Exception, *, action: str, mode: str = "quest",
                    fail: Callable[[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Map a failed Pollinations call to a user-facing error response.

    ``mode`` is the route the call actually used (``free``, ``quest`` or ``paid``), because the
    same status means different things on each: a 401 on the free route is not a rejected key,
    and a 402 there is the per-IP throttle rather than an empty Pollen balance.
    """
    if isinstance(exc, _HTTPError):
        logger.debug("Pollinations %s (%s): HTTP %s", action, mode, exc.status, exc_info=True)
        wait = f", retry in ~{exc.retry_after}s" if exc.retry_after else ""
        # The upstream body is already reduced to a short, control-character-free preview, so it
        # is safe to show. Status semantics differ per route: on the free route 402/429 is the
        # per-IP throttle; keyed, 402 is a balance/payment condition (waiting cannot fix a
        # Paid-Pollen restriction) and 429 alone is the rate limit. The free-route wording must
        # say "rate-limited": the core's guardrail classifier reads the error TEXT, not error_type.
        detail = f": {exc.body}" if exc.body else ""
        if exc.fix_url:
            detail += f" — fix: {exc.fix_url}"
        if exc.status == 401:
            if mode == "free":
                return fail(
                    "Pollinations rejected the anonymous free request (401) — the free classic "
                    "route may now require a key. Add POLLINATIONS_API_KEY, or pick a model the "
                    "free API serves.", "auth_required")
            if (_api_key() or "").lower().startswith("pk_"):
                return fail(
                    f"Pollinations rejected the key (401){detail}. pk_ app keys only identify an "
                    "app and do not authenticate requests. Use a secret "
                    "sk_ key from https://enter.pollinations.ai/keys as POLLINATIONS_API_KEY.",
                    "auth_required")
            return fail(
                f"Pollinations rejected the API key (401){detail}. Check POLLINATIONS_API_KEY.",
                "auth_required")
        if exc.status == 402:
            if mode == "free":
                return fail(
                    f"Pollinations returned HTTP 402{wait}{detail} — the free classic route is "
                    "rate-limited per IP. Wait, or add POLLINATIONS_API_KEY to skip the throttle.",
                    "rate_limited")
            if exc.error_code == "QUEST_POLLEN_ONLY":
                return fail(
                    f"Pollinations returned HTTP 402{detail} — this model needs Paid Pollen. "
                    "Pick a Quest-eligible model, or enable 'Include paid models' with a paid "
                    "balance at https://enter.pollinations.ai.", "payment_required")
            if exc.error_code == "QUEST_POLLEN_RESTRICTED":
                # The spec publishes this code but documents no prose meaning; word neutrally.
                return fail(
                    f"Pollinations returned HTTP 402{detail} — this model is restricted for "
                    "your key; it may need Paid Pollen or a key permission change. Pick a "
                    "Quest-eligible model, or check the account at "
                    "https://enter.pollinations.ai.", "payment_required")
            balance = "Paid Pollen" if mode == "paid" else "Quest Pollen"
            return fail(
                f"Pollinations returned HTTP 402{detail} — usually the {balance} balance or the "
                "key budget is exhausted. Check the account at https://enter.pollinations.ai.",
                "payment_required")
        if exc.status == 429:
            if mode == "free":
                return fail(
                    f"Pollinations returned HTTP 429{wait}{detail} — the free classic route is "
                    "rate-limited per IP. Wait, or add POLLINATIONS_API_KEY to skip the throttle.",
                    "rate_limited")
            return fail(
                f"Pollinations returned HTTP 429{wait}{detail} — rate limit reached. Wait for "
                "the retry window, or check the account at https://enter.pollinations.ai.",
                "rate_limited")
        if exc.status == 403:
            if mode == "free":
                # No key was sent on the classic route: do not blame key permissions.
                return fail(
                    f"Pollinations returned HTTP 403{detail} — the free classic API refused "
                    "this anonymous request. Try again later, or pick another model.",
                    "api_error")
            return fail(
                f"Pollinations returned HTTP 403{detail} — the key lacks permission for this "
                "model or operation. Check the account and key permissions at "
                "https://enter.pollinations.ai, or pick another model.", "auth_required")
        return fail(f"Pollinations returned HTTP {exc.status}{detail}", "api_error")
    logger.debug("Pollinations %s (%s) failed", action, mode, exc_info=True)
    # The raw exception can carry a URL with signed query parameters or a parser's wording, so
    # only the category is surfaced; the log holds the detail.
    kind = _network_error_type(exc)
    return fail(f"Pollinations {action} could not be completed ({kind}). "
                "See the Hermes log for detail.", kind)


def _http_get_bytes(url: str, *, headers: Optional[Dict[str, str]] = None,
                    timeout: float = HTTP_TIMEOUT) -> bytes:
    """GET *url* through the SSRF guard with a hard size ceiling."""
    with _ssrf_client(timeout) as client:
        with client.stream("GET", url, headers=headers or {}) as response:
            if response.status_code >= 400:
                code, preview, fix_url = _error_fields(response)
                raise _HTTPError(url, response.status_code, preview,
                                 _retry_after(response), error_code=code, fix_url=fix_url)
            return _bounded_read(response, url)


def _http_post_json(url: str, *, headers: Dict[str, str], payload: Dict[str, Any],
                    timeout: float = HTTP_TIMEOUT) -> Any:
    """POST JSON to *url* through the SSRF guard with a hard size ceiling.

    Redirects are not followed. This is the edits call, and it carries the user's API key and
    their reference images: the SSRF guard blocks private targets but allows public ones, so a
    redirect would forward the payload to a host the endpoint chose. An edits endpoint answers
    directly, so a 3xx is surfaced as the error it is.
    """
    with _ssrf_client(timeout, follow_redirects=False) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            # Redirects are not followed, so ANY 3xx is terminal: a redirect carrying a JSON
            # body must not parse as a completed edit.
            if response.status_code >= 300:
                code, preview, fix_url = _error_fields(response)
                raise _HTTPError(url, response.status_code, preview,
                                 _retry_after(response), error_code=code, fix_url=fix_url)
            return json.loads(_bounded_read(response, url))


def _http_get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(_http_get_bytes(url, timeout=timeout))


def _parse_image_dimensions(value: Any) -> Optional[tuple[int, int]]:
    """The ``X-Image-Dimensions`` header as (width, height), or None when absent/malformed."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d{1,5})\s*[xX]\s*(\d{1,5})\s*", value)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def _jpeg_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    """Actual pixel size from a JPEG's SOF marker, stdlib-only.

    The classic route clamps output area server-side without announcing it (no
    X-Image-Dimensions header), so the produced size is measured from the bytes themselves.
    A structural scan, not a decode: malformed data returns None and the caller keeps the
    requested dimensions.
    """
    if not raw.startswith(b"\xff\xd8"):
        return None
    pos, n = 2, len(raw)
    while pos + 4 <= n:
        if raw[pos] != 0xFF:
            return None
        marker = raw[pos + 1]
        pos += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue  # standalone markers carry no length
        if marker in (0xDA, 0xD9):
            return None  # scan data / end without a SOF
        if pos + 2 > n:
            return None
        length = int.from_bytes(raw[pos:pos + 2], "big")
        if length < 2:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 7 > n:
                return None
            height = int.from_bytes(raw[pos + 3:pos + 5], "big")
            width = int.from_bytes(raw[pos + 5:pos + 7], "big")
            return (width, height) if width > 0 and height > 0 else None
        pos += length
    return None


def _png_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    """Pixel size from a PNG's IHDR (width/height are the first 8 bytes of chunk data)."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return (width, height) if width > 0 and height > 0 else None


def _gif_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    """Pixel size from a GIF's logical screen descriptor (little-endian, bytes 6..10)."""
    if len(raw) < 10 or raw[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width = int.from_bytes(raw[6:8], "little")
    height = int.from_bytes(raw[8:10], "little")
    return (width, height) if width > 0 and height > 0 else None


def _webp_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    """Pixel size from a WebP's first chunk (VP8X canvas, VP8L packed bits, or VP8 frame)."""
    if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return None
    chunk = raw[12:16]
    if chunk == b"VP8X" and len(raw) >= 30:
        return (int.from_bytes(raw[24:27], "little") + 1,
                int.from_bytes(raw[27:30], "little") + 1)
    if chunk == b"VP8L" and len(raw) >= 25 and raw[20] == 0x2F:
        bits = int.from_bytes(raw[21:25], "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8 " and len(raw) >= 30 and raw[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(raw[26:28], "little") & 0x3FFF
        height = int.from_bytes(raw[28:30], "little") & 0x3FFF
        return (width, height) if width > 0 and height > 0 else None
    return None


def _image_dimensions(raw: bytes, mime: Optional[str]) -> Optional[tuple[int, int]]:
    """Produced pixel size measured structurally from decoded image bytes (no decode).

    Used when the API does not announce the produced size: the classic route clamps silently,
    and an edit may be resized server-side. None when unmeasurable; the caller keeps the
    requested dimensions.
    """
    if mime == "image/jpeg":
        return _jpeg_dimensions(raw)
    if mime == "image/png":
        return _png_dimensions(raw)
    if mime == "image/gif":
        return _gif_dimensions(raw)
    if mime == "image/webp":
        return _webp_dimensions(raw)
    return None


def _http_get_image(url: str, *, headers: Optional[Dict[str, str]] = None,
                    timeout: float = HTTP_TIMEOUT) -> tuple[bytes, Optional[tuple[int, int]]]:
    """GET an image through the SSRF guard, reporting the produced size when the API sends it.

    ``X-Image-Dimensions`` is optional and model-specific; when it is absent the caller keeps
    the requested dimensions (the classic route clamps server-side without announcing it).
    """
    with _ssrf_client(timeout) as client:
        with client.stream("GET", url, headers=headers or {}) as response:
            if response.status_code >= 400:
                code, preview, fix_url = _error_fields(response)
                raise _HTTPError(url, response.status_code, preview,
                                 _retry_after(response), error_code=code, fix_url=fix_url)
            dims = _parse_image_dimensions(response.headers.get("x-image-dimensions"))
            return _bounded_read(response, url), dims


def _seed_from(kwargs: Dict[str, Any]) -> int:
    """A caller-provided seed within the live spec range, else a fresh random one.

    The range (including the -1 sentinel) is read from the published spec, so a widened or
    narrowed range follows upstream without a plugin edit; reproducible generations are
    possible on models that honor seed (direct provider calls only — the image_generate tool
    exposes no seed parameter). A bool is not a seed. Out-of-range or non-integer
    values fall back to random instead of erroring. When the spec range is unknown the
    caller's int passes through verbatim and the server validates it. The -1 sentinel is
    spec-valid on both routes ("random" on GET, the stable compatibility seed on edits) and
    is forwarded verbatim.
    """
    raw = kwargs.get("seed")
    limits = _spec_limits()
    lo, hi = limits["seed_min"], limits["seed_max"]
    if isinstance(raw, int) and not isinstance(raw, bool):
        if (lo is None or hi is None or lo <= raw <= hi) and abs(raw) <= _SEED_SANITY_MAX:
            return raw
    top = min(hi, _SEED_SANITY_MAX) if hi is not None else _RANDOM_SEED_MAX
    bottom = max(0, lo) if lo is not None else 0
    if bottom > top:
        # A negative-only published range cannot host a nonnegative random seed: return the
        # published upper bound (a valid in-range value) rather than raising out of generate().
        return top
    return random.randint(bottom, top)


# Cache state is keyed by Hermes home: under multiplexing a routed profile must not inherit
# another profile's failure window or success token. Locks stay global (cheap, short-held).
# negative_until/last_success use the MONOTONIC clock (immune to wall-clock skew); disk
# "fetched" and classic memory "at" stay wall-clock (they compare across processes).
_CATALOG_STATES: Dict[str, Dict[str, Any]] = {}
_CATALOG_LOCK = threading.Lock()


def _home_key() -> str:
    return str(_hermes_home())


def _catalog_state() -> Dict[str, Any]:
    return _CATALOG_STATES.setdefault(_home_key(), {"negative_until": 0.0, "last_success": 0.0})


def _cache_schema_ok(raw: Dict[str, Any]) -> bool:
    """Strict cache schema gate: an explicit int version equal to CACHE_SCHEMA, nothing else.

    A version-less, boolean or float version is a foreign cache, not ours — ``True == 1`` and
    ``1.0 == 1`` in Python, so a loose comparison would trust a hand-edited marker.
    """
    ver = raw.get("version")
    return isinstance(ver, int) and not isinstance(ver, bool) and ver == CACHE_SCHEMA


def _note_contract_gaps(models: List[Dict[str, Any]]) -> None:
    """Debug-log when a high-impact contract field is absent across the whole catalog.

    The field names are API contract constants; if Pollinations renames one, every row would
    silently mislabel (prices vanish, paid models lose their gate, edits disable). A log line
    makes the rename visible instead of silent.
    """
    for field in ("pricing", "paid_only", "supported_endpoints"):
        if not any(isinstance(m, dict) and field in m for m in models):
            logger.debug("Pollinations catalog publishes no %r field — possible upstream rename.",
                         field)


def _unlink_planted_symlink(path: Path) -> None:
    """Core atomic_replace preserves symlinks; a planted link in the cache dir would turn a
    cache write into an arbitrary-file overwrite. Discard the link (the target is untouched)
    so the write lands on a regular file. Best-effort against a local attacker already inside
    the home-directory trust boundary."""
    try:
        if path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _guarded_cache_write(path: Path, payload: Dict[str, Any], started: float) -> bool:
    """Atomic cache write guarded against cross-process clobber (POSIX flock, best-effort).

    Under the lock, re-read the disk timestamp: if another process wrote NEWER data while
    this fetch was in flight, skip the write rather than overwrite it. The payload is stamped
    with the fetch START, so a later-started fetch wins even when it completes last. On
    platforms without fcntl (or any lock I/O error) the plain atomic write still applies.
    Returns True when this payload was written, False when a newer disk write won.
    """
    try:
        # Create the parent BEFORE the lock open: a missing dir would otherwise drop both
        # first fetches into the unlocked fallback, skipping the ordering check. The
        # tombstone-aware helper never recreates a deleted named profile home.
        mkdir_under_hermes_home(path.parent)
    except Exception:
        return False
    try:
        import fcntl
    except ImportError:
        _unlink_planted_symlink(path)
        atomic_json_write(path, payload)
        return True
    _unlink_planted_symlink(path)
    lock_path = path.with_name(path.name + ".lock")
    try:
        # O_NOFOLLOW: a planted symlink lockfile must not create/lock an arbitrary target.
        fd = os.open(lock_path, os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        fd = None
    if fd is None:
        atomic_json_write(path, payload)  # no lock available: best-effort write
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            disk = read_json_or_empty(path)
            try:
                disk_fetched = float(disk.get("fetched") or 0)
            except (TypeError, ValueError, OverflowError):
                disk_fetched = 0.0  # a corrupt stamp must not immortalize the file
            if not math.isfinite(disk_fetched):
                disk_fetched = 0.0  # Infinity would block every later write
            if disk_fetched > started:
                return False  # a newer write landed mid-fetch
            _unlink_planted_symlink(path)
            atomic_json_write(path, payload)
            return True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        _unlink_planted_symlink(path)
        atomic_json_write(path, payload)
        return True
    finally:
        os.close(fd)


def _catalog() -> List[Dict[str, Any]]:
    """Live catalog: fresh fetch, else a fresh-enough cache, else the stale cache.

    A failed fetch is remembered briefly, so a down endpoint costs one timeout per window
    rather than one per ``list_models()`` / ``capabilities()`` / ``generate()`` call.
    """
    path = _cache_path()
    cached: Optional[List[Dict[str, Any]]] = None
    try:
        raw = read_json_or_empty(path)
        # A cache written by a different plugin version is ignored rather than trusted; entries
        # without a usable id are dropped here too, so a shapeless cache is not a fresh hit.
        models = ([entry for entry in _as_list(raw.get("models"))
                   if isinstance(entry, dict) and _name_of(entry)]
                  if _cache_schema_ok(raw) else [])
        if models:
            cached = models
            age = time.time() - float(raw.get("fetched") or 0)
            # A future timestamp (clock skew or a hostile cache) must not read as permanently fresh.
            if 0 <= age < CATALOG_TTL:
                return models
    except Exception:
        pass
    with _CATALOG_LOCK:
        if time.monotonic() < float(_catalog_state().get("negative_until") or 0.0):
            return cached if cached is not None else []
    # Only a fetch that began after the last successful one may record an outcome; otherwise a
    # slow failure would hide a cache another thread has already refreshed.
    started = time.time()
    started_mono = time.monotonic()
    try:
        models = _unwrap_list(
            _http_get_json(CATALOG_URL), ("models", "data", "items", "results"),
            fallback_item_ok=lambda items: any(isinstance(it, dict) and _is_image_model(it)
                                               for it in items)) or []
        # Drop non-object entries AND entries without a usable id at the source: a renamed
        # envelope can surface a list that is not models at all, and zero usable ids has the
        # same effect as an empty answer — it must not overwrite a good cache.
        models = [entry for entry in models if isinstance(entry, dict) and _name_of(entry)]
        if not models:
            # An empty answer is not a catalog. Treating it as authoritative would overwrite a
            # good cache and empty the picker, so it is handled as a failed fetch instead.
            raise ValueError("Pollinations catalog returned no models")
        _note_contract_gaps(models)
        with _CATALOG_LOCK:
            # Mirror the classic guard on SUCCESS too: a slow stale fetch must not overwrite a
            # newer cache another thread already wrote.
            if started_mono >= float(_catalog_state().get("last_success") or 0.0):
                try:
                    # Atomic replace, cross-process guarded, stamped with the fetch START.
                    _guarded_cache_write(path, {"version": CACHE_SCHEMA, "fetched": started,
                                                "models": models}, started)
                except Exception as exc:
                    logger.debug("Could not cache Pollinations catalog: %s", exc)
                _catalog_state()["negative_until"] = 0.0
                # Record the fetch's START token, not its completion time: a later-started
                # fetch carries newer data and must win even when it completes last.
                _catalog_state()["last_success"] = started_mono
        return models
    except Exception as exc:
        logger.debug("Pollinations catalog fetch failed: %s", exc)
        # Remember the failure briefly, whether or not a stale cache exists: the cache still
        # wins, but a down endpoint must cost one timeout per window rather than one per call.
        with _CATALOG_LOCK:
            if started_mono >= float(_catalog_state().get("last_success") or 0.0):
                _catalog_state()["negative_until"] = time.monotonic() + NEGATIVE_TTL
        # No hardcoded stand-ins: an unreachable catalog means no models to offer.
        return cached if cached is not None else []


_CLASSIC_STATES: Dict[str, Dict[str, Any]] = {}
_CLASSIC_LOCK = threading.Lock()


def _classic_state() -> Dict[str, Any]:
    return _CLASSIC_STATES.setdefault(
        _home_key(), {"at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0})


def _classic_cache_path() -> Path:
    return _hermes_home() / "cache" / "pollinations_classic_models.json"


def _load_classic_cache() -> Optional[List[str]]:
    """Last free-classic model list we managed to fetch, whatever its age."""
    try:
        raw = read_json_or_empty(_classic_cache_path())
        ids = ([item.strip() for item in _as_list(raw.get("ids")) if _valid_id(item)]
               if _cache_schema_ok(raw) else [])
    except Exception as exc:
        # ``read_json_or_empty`` only guards OSError/ValueError; a pathologically nested or
        # otherwise hostile cache file must degrade to "no cache", not escape to the caller.
        logger.debug("Could not read the cached free classic model list: %s", exc)
        return None
    return ids or None


def _store_classic_cache(ids: List[str], started: float) -> List[str]:
    """Cache the fetch result; when a newer cross-process write wins, adopt and return IT.

    Called under _CLASSIC_LOCK by _classic_models, so the skip-path state update is safe.
    """
    try:
        written = _guarded_cache_write(_classic_cache_path(),
                                       {"version": CACHE_SCHEMA, "fetched": started, "ids": ids},
                                       started)
        if not written:
            raw = read_json_or_empty(_classic_cache_path())
            raw_ids = raw.get("ids")
            # An explicit list (empty = withdrawal) only; a malformed ids field must not
            # clear good data.
            if isinstance(raw_ids, list) and _cache_schema_ok(raw):
                valid = [i.strip() for i in raw_ids if _valid_id(i)]
                if not raw_ids or valid:
                    _classic_state()["ids"] = valid
                    _classic_state()["at"] = float(raw.get("fetched") or started)
                    return valid
    except Exception as exc:
        logger.debug("Could not cache the free classic model list: %s", exc)
    return ids


def _adopt_newer_classic_disk(state: Dict[str, Any]) -> None:
    """Adopt a newer cross-process classic write over memory (called under _CLASSIC_LOCK).

    A valid EMPTY list is a withdrawal and wins too; a non-empty list with zero valid ids is
    hostile/corrupt and is ignored. Non-finite stamps never win.
    """
    try:
        raw = read_json_or_empty(_classic_cache_path())
        if not _cache_schema_ok(raw):
            return
        if not isinstance(raw.get("ids"), list):
            return  # a missing/malformed ids field is NOT a withdrawal
        fetched = float(raw.get("fetched") or 0)
        if not math.isfinite(fetched) or fetched <= float(state["at"]):
            return
        raw_ids = _as_list(raw.get("ids"))
        valid = [i.strip() for i in raw_ids if _valid_id(i)]
        if not raw_ids or valid:
            state["ids"] = valid
            state["at"] = fetched
    except Exception:
        pass


def _classic_models() -> List[str]:
    """Model ids the free classic API serves, TTL-cached in memory and on disk.

    No hardcoded guess: when the endpoint is unreachable and nothing is cached the list is
    empty, so the picker offers no model rather than one that may no longer be free. The lock
    covers only the cache read/update; the fetch happens outside it so one slow request never
    blocks another caller. A failed lookup is remembered briefly, so a down endpoint costs one
    timeout per window rather than one per call.
    """
    with _CLASSIC_LOCK:
        state = _classic_state()
        age = time.time() - float(state["at"])
        # A future "at" (backward clock step) must not read as permanently fresh.
        if 0 <= age < CATALOG_TTL:
            # A newer cross-process disk write wins over this process's memory.
            _adopt_newer_classic_disk(state)
            return list(state["ids"])
        if time.monotonic() < float(_classic_state().get("negative_until") or 0.0):
            # Another process may have written a good cache (or a withdrawal) while this one
            # was failing: a newer disk write wins over stale memory here too.
            state = _classic_state()
            _adopt_newer_classic_disk(state)
            return list(state["ids"]) or _load_classic_cache() or []
    cached = _load_classic_cache()
    # Only a fetch that began after the last successful one may record an outcome; otherwise a
    # slow failure would hide (or a slow success would overwrite) what a newer fetch produced.
    started = time.time()
    started_mono = time.monotonic()
    fetched_ids: Optional[List[str]] = None
    try:
        fetched = _unwrap_list(
            _http_get_json(CLASSIC_MODELS_URL), ("models", "data", "ids", "items", "results"),
            fallback_key_ok=lambda k: "model" in k.lower(),
            fallback_item_ok=lambda items: any(_valid_id(it) for it in items))
        if fetched is not None:
            # Only guarded strings can become URL path segments; anything else is discarded.
            valid = [item.strip() for item in fetched if _valid_id(item)]
            if fetched and not valid:
                # A non-empty answer with zero usable ids is a shape change or a decoy
                # envelope, not an authoritative empty — treat it as a failed fetch so a
                # good cache survives (mirrors the catalog's zero-usable-ids guard). A
                # truly empty upstream list stays authoritative below.
                raise ValueError("classic model list returned no usable ids")
            fetched_ids = valid
    except Exception as exc:
        logger.debug("Classic model list fetch failed: %s", exc)
    if fetched_ids is not None:
        # A successful fetch is authoritative, empty or not: the free route may have been
        # withdrawn, and serving a stale id would send the user to a model that now errors.
        with _CLASSIC_LOCK:
            if started_mono >= float(_classic_state().get("last_success") or 0.0):
                winner = _store_classic_cache(fetched_ids, started)
                _classic_state()["ids"] = list(winner)
                if winner is fetched_ids:
                    _classic_state()["at"] = started
                _classic_state()["negative_until"] = 0.0
                # START token, not completion time (see _catalog): later-started wins.
                _classic_state()["last_success"] = started_mono
                return list(winner)
        return list(fetched_ids)
    # Remember the failure whether or not a stale cache exists (mirrors _catalog): a down
    # endpoint must cost one timeout per window, not one per call.
    with _CLASSIC_LOCK:
        if started_mono >= float(_classic_state().get("last_success") or 0.0):
            _classic_state()["negative_until"] = time.monotonic() + NEGATIVE_TTL
    return list(cached) if cached else []


def _classic_set() -> set:
    """The free classic ids lowercased, for case-insensitive matching.

    These ids are matched against catalog keys, so a non-string would raise ``TypeError`` on the
    set membership test rather than simply not matching. Lowercasing here means a catalog or
    config that differs from the endpoint only in case still resolves; ``_classic_id_for``
    restores the endpoint's own spelling for the URL.
    """
    return {_norm_id(key) for key in _classic_models() if isinstance(key, str)}


def _classic_spelling() -> Dict[str, str]:
    """Normalized free classic id -> the endpoint's own spelling, for URL paths."""
    return {_norm_id(key): key for key in _classic_models() if isinstance(key, str)}


def _free_route_unavailable() -> bool:
    """True when the last free-model lookup failed and nothing is cached.

    Distinguishes "this model is not free" from "we could not tell", so a keyless user is not
    told their model is unavailable when the endpoint is simply down.
    """
    with _CLASSIC_LOCK:
        return time.monotonic() < float(_classic_state().get("negative_until") or 0.0)


def _spec_cache_path() -> Path:
    return _hermes_home() / "cache" / "pollinations_spec_limits.json"


_SPEC_STATES: Dict[str, Dict[str, Any]] = {}
_SPEC_LOCK = threading.Lock()


def _spec_state() -> Dict[str, Any]:
    return _SPEC_STATES.setdefault(_home_key(), {"negative_until": 0.0, "last_success": 0.0})


def _is_int(value: Any) -> TypeGuard[int]:
    """A plain int (bool excluded — True is not a limit)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validated_limits(limits: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Shape- and sanity-check the three limits; anything unusable reads as unknown."""
    prompt = limits.get("edit_prompt_max")
    lo, hi = limits.get("seed_min"), limits.get("seed_max")
    lo = lo if _is_int(lo) else None
    hi = hi if _is_int(hi) else None
    if lo is not None and hi is not None and lo > hi:
        lo = hi = None  # an inconsistent range is not a limit
    return {"edit_prompt_max": prompt if _is_int(prompt) and prompt > 0 else None,
            "seed_min": lo, "seed_max": hi}


def _spec_limits_from(payload: Any) -> Dict[str, Optional[int]]:
    """The edit prompt maxLength and the seed range from an OpenAPI document.

    Every read is shape-guarded: the spec is third-party JSON, so a renamed or malformed
    field degrades to "unknown" rather than raising. Unknown limits mean the plugin skips
    its client-side pre-checks and lets the server enforce its own contract.
    """
    out: Dict[str, Any] = {}
    doc = _as_dict(payload)
    paths = _as_dict(doc.get("paths"))
    get = _as_dict(_as_dict(paths.get("/image/{prompt}")).get("get"))
    for param in _as_list(get.get("parameters")):
        if isinstance(param, dict) and param.get("name") == "seed":
            schema = _as_dict(param.get("schema"))
            out["seed_min"], out["seed_max"] = schema.get("minimum"), schema.get("maximum")
    post = _as_dict(_as_dict(paths.get("/v1/images/edits")).get("post"))
    content = _as_dict(_as_dict(post.get("requestBody")).get("content"))
    schema = _as_dict(_as_dict(content.get("application/json")).get("schema"))
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        schemas = _as_dict(_as_dict(doc.get("components")).get("schemas"))
        schema = _as_dict(schemas.get(ref.rsplit("/", 1)[-1]))
    out["edit_prompt_max"] = _as_dict(_as_dict(schema.get("properties")).get("prompt")).get("maxLength")
    return _validated_limits(out)


def _spec_limits() -> Dict[str, Optional[int]]:
    """Live API limits from the published spec: fresh fetch, else fresh-enough cache, else
    stale cache, else all-unknown. Mirrors the catalog's negative-window and start-token
    guards so a down spec endpoint costs one timeout per window, not one per call."""
    path = _spec_cache_path()
    cached: Optional[Dict[str, Optional[int]]] = None
    try:
        raw = read_json_or_empty(path)
        if _cache_schema_ok(raw) and isinstance(raw.get("limits"), dict):
            candidate = _validated_limits(raw["limits"])
            # An all-unknown cache is not a usable limit set (a transient placeholder may
            # have been written by an older version); ignore it and refetch.
            if any(v is not None for v in candidate.values()):
                cached = candidate
                age = time.time() - float(raw.get("fetched") or 0)
                if 0 <= age < SPEC_TTL:
                    return cached
    except Exception:
        pass
    with _SPEC_LOCK:
        if time.monotonic() < float(_spec_state().get("negative_until") or 0.0):
            return cached if cached is not None else _validated_limits({})
    started = time.time()
    started_mono = time.monotonic()
    try:
        limits = _spec_limits_from(_http_get_json(SPEC_URL))
        if not any(v is not None for v in limits.values()):
            # A parseable document carrying none of the limits is not the contract we know
            # (a rename, placeholder or error envelope): treat it as a failed fetch so a good
            # cache survives and the next window retries, rather than caching "unknown".
            raise ValueError("Pollinations spec published no usable limits")
        with _SPEC_LOCK:
            if started_mono >= float(_spec_state().get("last_success") or 0.0):
                try:
                    _guarded_cache_write(path, {"version": CACHE_SCHEMA, "fetched": started,
                                                "limits": limits}, started)
                except Exception as exc:
                    logger.debug("Could not cache Pollinations spec limits: %s", exc)
                _spec_state()["negative_until"] = 0.0
                _spec_state()["last_success"] = started_mono
        return limits
    except Exception as exc:
        logger.debug("Pollinations spec fetch failed: %s", exc)
        with _SPEC_LOCK:
            if started_mono >= float(_spec_state().get("last_success") or 0.0):
                _spec_state()["negative_until"] = time.monotonic() + NEGATIVE_TTL
        return cached if cached is not None else _validated_limits({})


def _as_dict(value: Any) -> Dict[str, Any]:
    """A catalog field that should be an object; anything else reads as empty.

    The catalog is third-party JSON, so a field can arrive with the wrong shape; the plugin must
    degrade to a blank value rather than raise out of ``list_models()``/``capabilities()``.
    """
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    """A catalog field that should be a list; anything else reads as empty."""
    return value if isinstance(value, list) else []


def _as_bool(value: Any) -> bool:
    """A catalog boolean, accepting the string spellings a JSON API may send.

    ``bool("false")`` is ``True`` in Python, so reading one of these fields raw silently inverts
    its meaning: a model marked ``"paid_only": "false"`` would be treated as paid-only and hidden
    behind the toggle, and ``"flat_rate": "false"`` would price it per image instead of per token.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return False


def _paid_flag(entry: Optional[Dict[str, Any]]) -> Optional[bool]:
    """The entry's paid-only class: True/False, or None when the flag is unreadable.

    A non-scalar ``paid_only`` (list/dict) cannot be classified; callers fail CLOSED on None
    rather than letting _as_bool coerce it to False and spend with the toggle off.
    """
    raw = (entry or {}).get("paid_only")
    if isinstance(raw, (list, dict)):
        return None
    return _as_bool(raw)


# Catalog text fields are display metadata. The strengths column shows 120 characters and no
# other consumer reads more than a few hundred, so a field longer than this is truncated before
# the per-character scan — one absurd catalog entry cannot stall the picker.
TEXT_LIMIT = 4096


def _valid_id(value: Any) -> TypeGuard[str]:
    """True for a string usable as a model id: printable, bounded, surrogate-free.

    Shared by catalog names, classic-endpoint ids and passthrough candidates — each can become
    a URL path segment, and a lone surrogate raises UnicodeEncodeError out of urllib's quote()
    while a control character or absurd length builds a hostile URL.
    """
    return (isinstance(value, str) and bool(value.strip()) and value.isprintable()
            and len(value) <= TEXT_LIMIT)


def _name_of(model: Dict[str, Any]) -> Optional[str]:
    """The catalog id when it is a usable string, else None.

    The id becomes a dict key, a URL path segment and part of an error message, so a non-string
    or blank value is discarded rather than coerced into something misleading. ``id`` is accepted
    as a fallback so a catalog that renames the field does not silently empty the picker.
    """
    if not isinstance(model, dict):
        return None
    for field in ("name", "id"):
        value = model.get(field)
        if _valid_id(value):
            return value.strip()
    return None


def _is_number(value: Any) -> TypeGuard[float]:
    """True for a finite int/float. JSON ``NaN``/``Infinity`` are floats but never usable here.

    ``bool`` is a subclass of ``int``, so without the guard a catalog that publishes
    ``"per_user_rpm": true`` would render as a rate limit and ``"completionImageTokens": true``
    would price the model at one pollen.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False  # an int too large for a float


# A published rate is a plain ASCII decimal, optionally signed, with an optional
# exponent. float() alone is too lenient for third-party data: it reads "1_000"
# and non-ASCII digits as amounts, so the shape is gated before conversion.
_PRICE_SHAPE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


def _as_number(value: Any) -> Optional[float]:
    """A catalog price as a finite float, or None when it is not an amount.

    The catalog's OpenAPI contract types every ``pricing`` rate as a string
    (``"0.002"``); older snapshots sent numbers. Both must read as the same
    amount. ``bool`` is excluded explicitly — ``false`` is not a zero price and
    ``true`` is not one pollen — as are NaN/Infinity and strings that are not
    plain decimal numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not _PRICE_SHAPE.fullmatch(value):
            return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        if isinstance(value, str):
            dec = Decimal(value)
            number = float(dec)
            # float() silently underflows a tiny nonzero rate to zero; that is not
            # a free price, and negative zero would also dodge the sign check.
            if dec != 0 and number == 0:
                return None
        else:
            number = float(value)
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    """A catalog string field as printable text; anything else reads as empty.

    Control characters are dropped so a hostile catalog entry cannot smuggle line breaks into
    the picker or an agent-visible error message.
    """
    if not isinstance(value, str):
        return ""
    # Newlines become spaces (dropping them fuses words: "Labs\n\nQuest" -> "LabsQuest");
    # other control characters are dropped so a hostile entry cannot smuggle line breaks.
    cleaned = "".join(ch if ch not in "\n\r" else " "
                      for ch in value[:TEXT_LIMIT] if ch.isprintable() or ch in " \t\n\r")
    return " ".join(cleaned.split())


def _is_video_model(model: Dict[str, Any]) -> bool:
    """True when the catalog positively marks the entry as video.

    Retention and generate refuse positively-video entries; an entry with no modality signal
    at all is ambiguous, not video, and keeps the saved-model visibility convention.
    """
    if not isinstance(model, dict):
        return False
    if str(model.get("category") or "").strip().lower() == "video":
        return True
    return "video" in {str(m).strip().lower() for m in _as_list(model.get("output_modalities"))}


def _is_image_model(model: Dict[str, Any]) -> bool:
    """True when the catalog marks the entry as an image model.

    ``category`` is the primary marker; the spec's stated discriminator is ``output_modalities``,
    so an entry the category omits but that outputs images is still accepted. When neither label
    is present an image endpoint is the last signal, so a catalog that renames its category
    vocabulary does not silently hide every model.
    """
    if not isinstance(model, dict):
        return False
    category = str(model.get("category") or "").strip().lower()
    modalities = {str(m).strip().lower() for m in _as_list(model.get("output_modalities"))}
    if category == "video" or "video" in modalities:
        return False
    if category == "image" or "image" in modalities:
        return True
    return any(isinstance(endpoint, str) and "image" in endpoint.lower()
               for endpoint in _as_list(model.get("supported_endpoints")))


def _classic_keys(model: Dict[str, Any]) -> List[str]:
    """The catalog id and aliases of a model as strings; anything else is discarded.

    These keys become set members and URL path segments, so a non-string ``name`` or alias is
    dropped rather than coerced.
    """
    if not isinstance(model, dict):
        return []
    # Aliases get the same _text cleaning as the index, so a whitespace-padded alias
    # still matches the classic set.
    keys = [_name_of(model)] + [_text(a) for a in _as_list(model.get("aliases"))]
    alt = model.get("id")
    if isinstance(alt, str) and _valid_id(alt):
        keys.append(_text(alt))  # the alternate id can be the spelling classic serves
    return [key for key in keys if isinstance(key, str) and key]


def _is_free_classic(model: Dict[str, Any], classic: set) -> bool:
    """True when the free classic API serves this catalog model (no key, no pollen).

    ``classic`` is the lowercased set from ``_classic_set()``.
    """
    return any(_norm_id(key) in classic for key in _classic_keys(model))


def _catalog_entry(catalog: List[Dict[str, Any]], model: Optional[str]) -> Optional[Dict[str, Any]]:
    """The live catalog entry for a resolved model id, or None when the catalog does not know it."""
    if not model:
        return None
    return next((m for m in catalog if _name_of(m) == model), None)


def _edit_path(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    """The catalog's advertised edits path, matched by shape rather than by exact literal.

    A version bump upstream (``/v2/images/edits``) would otherwise silently disable every edit,
    because an exact ``"/v1/images/edits"`` comparison would stop matching. A query or fragment
    is not part of a path, so it is stripped before matching; otherwise an endpoint advertised
    with one would not match at all and every edit would silently use the built-in default.
    """
    for endpoint in _as_list((entry or {}).get("supported_endpoints")):
        if not isinstance(endpoint, str):
            continue
        candidate = endpoint.strip().split("?", 1)[0].split("#", 1)[0].strip()
        # An IMAGE edits endpoint only: /v1/audio/edits must not enable image editing.
        # The version segment stays flexible (/v2/images/edits follows upstream).
        if re.search(r"/images?/edits$", candidate, re.IGNORECASE):
            return candidate
    return None


def _supports_edits(entry: Optional[Dict[str, Any]]) -> bool:
    """True when the catalog advertises an edits endpoint AND accepts at least one reference."""
    return _edit_path(entry) is not None and _reference_cap(entry) > 0


def _edit_url(entry: Optional[Dict[str, Any]]) -> str:
    """The edits endpoint: a catalog path on Pollinations' own host, else the built-in default.

    Catalog data is fetched from the network, so it is not trusted to choose where the user's API
    key and reference images are sent. An absolute endpoint from it is accepted only when it names
    Pollinations' own host over TLS on the standard port, with no credentials embedded; a relative
    path is joined to the authenticated base. Anything else — another host, plain HTTP, a non
    standard port, userinfo — falls back to ``EDIT_URL`` rather than being requested.
    """
    path = _edit_path(entry)
    if not path:
        return EDIT_URL
    try:
        parsed = urlsplit(path)
        port = parsed.port
    except ValueError:
        # A malformed port or bracket makes urlsplit refuse the value; do not guess at it.
        return EDIT_URL
    if parsed.scheme or parsed.netloc:
        if (parsed.scheme.lower() != "https"
                or (parsed.hostname or "").lower() != ENTER_HOST
                or port not in (None, 443)
                or parsed.username is not None
                or parsed.password is not None):
            return EDIT_URL
        # Rebuild from the host constant rather than echoing the catalog's netloc, so a value that
        # somehow matched above cannot smuggle a second host or a fragment into the request.
        return urlunsplit(("https", ENTER_HOST, parsed.path or EDIT_URL_PATH, "", ""))
    # A relative path may carry its own query or fragment; neither belongs in an endpoint we
    # construct, and a fragment would silently truncate the path for urlsplit-based consumers.
    clean = path.split("?", 1)[0].split("#", 1)[0]
    if any(seg == ".." for seg in clean.split("/")):
        return EDIT_URL  # dot-segments could normalize to another path; do not guess
    return ENTER_BASE_URL + (clean if clean.startswith("/") else "/" + clean)


def _reference_cap(entry: Optional[Dict[str, Any]]) -> int:
    """The catalog's published reference-image cap, ceilinged; a conservative default otherwise.

    Published caps observed to date range 0-16 (0 disables edits). Models that publish none (mostly community
    endpoints) get a conservative default rather than the ceiling: such endpoints commonly
    accept a single image, and over-sending would 400. An explicit cap is honored up to the
    ceiling; a published 0 means no reference images and disables edits for the entry.
    """
    raw = (entry or {}).get("max_reference_images")
    if raw is None or raw == "" or isinstance(raw, bool):
        return DEFAULT_REFERENCE_CAP
    try:
        cap = int(raw)
    except Exception:
        return DEFAULT_REFERENCE_CAP
    if cap <= 0:
        # A published zero (or nonsense negative) means the model accepts no reference
        # images; honor it by disabling edits for the entry rather than clamping to 1.
        return 0
    return min(cap, MAX_REFERENCE_IMAGES)


def _published_resolutions(entry: Optional[Dict[str, Any]]) -> set:
    """Output resolutions the catalog publishes for this model, lowercased."""
    return {r.strip().lower() for r in _as_list((entry or {}).get("resolutions"))
            if isinstance(r, str) and r.strip() and len(r) <= TEXT_LIMIT}


def _supports_resolution(entry: Optional[Dict[str, Any]]) -> bool:
    """True when the model publishes at least one resolution Hermes' 1K/2K control can express."""
    return bool(_published_resolutions(entry) & RESOLUTION_VOCAB)


def _classic_id_for(model: str, catalog: List[Dict[str, Any]]) -> Optional[str]:
    """The free classic API's id for a catalog model, when it serves that model.

    Matching is case-insensitive; the endpoint's own spelling is what goes in the URL.
    """
    spelling = _classic_spelling()
    wanted = _norm_id(model.strip())
    # A model the classic API serves by its own id needs no catalog entry, so a stale catalog
    # cannot strand a keyless user on a model that is in fact free.
    if wanted in spelling:
        return spelling[wanted]
    entry = _catalog_entry(catalog, model)
    for key in _classic_keys(entry) if entry else []:
        if _norm_id(key) in spelling:
            return spelling[_norm_id(key)]
    return None


def _pricing_quantity(model: Dict[str, Any]) -> float:
    """The catalog's ``pricing_units`` quantity for the image rate; 1.0 when unpublished.

    The spec defines the rate per unit as ``pricing[field] × quantity`` (megapixel billing
    publishes quantity 1e6). Guarded like every catalog number: a bool is not a quantity.
    """
    units = _as_dict(_as_dict(model.get("pricing_units")).get("completionImageTokens"))
    quantity = _as_number(units.get("quantity"))
    return quantity if quantity is not None and quantity > 0 else 1.0


def _image_adjustments(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The catalog's per-image pricing adjustments (fees/premiums), shape-guarded.

    Video-kind adjustments belong to video billing and are excluded from image estimates.
    """
    return [a for a in _as_list(model.get("pricing_adjustments"))
            if isinstance(a, dict) and str(a.get("kind") or "image").lower() != "video"]


def _adjustments_per_image(model: Dict[str, Any]) -> float:
    """Sum of adjustments convertible to a per-image pollen figure (generation/image units).

    GPU-second, USD and other unconvertible units are excluded — they surface as a label
    hint instead of a false estimate. Rates are strings per the spec; guarded like all
    catalog numbers.
    """
    total = 0.0
    for adj in _image_adjustments(model):
        if _as_dict(adj.get("option")).get("default") is False:
            continue  # an optional fee that is off by default is not part of the floor
        unit = _text(adj.get("unit")).lower()
        if unit not in ("", "generation", "image", "images"):
            continue
        if _text(adj.get("currency")).lower() not in ("", "pollen"):
            continue
        price = _as_number(adj.get("price"))
        if price is None or price < 0:
            continue
        qty = _as_number(adj.get("quantity"))
        total += price * (qty if qty is not None and qty > 0 else 1.0)
    return total


def _price_class(model: Dict[str, Any]) -> int:
    """Ordering class: 0 = comparable per-image cost, 1 = per-token (or other unconvertible
    unit), 2 = no published price. Per-token rates cannot be compared to per-image costs
    without a token count the catalog does not publish, so they sort after class 0; an
    unpublished price sorts last rather than masquerading as the cheapest."""
    amount = _as_number(_as_dict(model.get("pricing")).get("completionImageTokens"))
    if amount is None or amount < 0:
        pricing = _as_dict(model.get("pricing"))
        if (_as_number(pricing.get("completionTextTokens")) is not None
                or _as_number(pricing.get("promptTextTokens")) is not None):
            return 1  # text-token billing is still per-token, not "unpublished"
        return 2
    if amount == 0:
        return 0
    unit = _pricing_unit(model)
    if unit == "megapixel":
        return 0
    if unit in ("", "image", "images", "generation") and _as_bool(model.get("flat_rate")):
        return 0
    return 1


def _variant_rate(variant: Any) -> Optional[float]:
    """Published rate for one pricing variant.

    The live catalog nests it under ``variant["pricing"]["completionImageTokens"]``; older
    snapshots and test fixtures publish it at the top level. Accept both shapes.
    """
    data = _as_dict(variant)
    rate = _as_number(_as_dict(data.get("pricing")).get("completionImageTokens"))
    if rate is None:
        rate = _as_number(data.get("completionImageTokens"))
    return rate


def _variant_floor(model: Dict[str, Any], base: float) -> float:
    """Lowest published per-image rate across base and variants: a 'from' figure must be a
    true floor (a live variant can be cheaper than the base rate)."""
    floor = base
    for variant in _as_list(model.get("pricing_variants")):
        rate = _variant_rate(variant)
        if rate is not None and 0 <= rate < floor:
            floor = rate
    return floor


def _price_value(model: Dict[str, Any]) -> float:
    """Estimated per-image pollen cost for ordering, within the model's price class.

    Per-image rates scale by ``pricing_units.quantity``; per-megapixel rates additionally
    scale by the plugin's typical output area. Per-token rates keep their raw figure (only
    comparable within class 1). Unpublished prices read as zero but sort last via the class.
    """
    pricing = _as_dict(model.get("pricing"))
    amount = _as_number(pricing.get("completionImageTokens"))
    adjustments = _adjustments_per_image(model)
    if amount is None or amount < 0:
        # Text-token-only models keep a class-1 relative figure (completion rate preferred).
        rate = _as_number(pricing.get("completionTextTokens"))
        if rate is None:
            rate = _as_number(pricing.get("promptTextTokens"))
        return rate if rate is not None and rate >= 0 else 0.0
    if amount == 0:
        return adjustments
    unit = _pricing_unit(model)
    quantity = _pricing_quantity(model)
    if unit == "megapixel":
        return _variant_floor(model, amount) * quantity * _TYPICAL_MP + adjustments
    if unit in ("", "image", "images", "generation") and _as_bool(model.get("flat_rate")):
        return _variant_floor(model, amount) * quantity + adjustments
    return amount


def _format_price(value: Any, currency: str = "pollen") -> str:
    """Plain-decimal amount in the catalog's currency; never scientific notation — except when
    blind truncation would erase every significant digit of a sub-precision rate."""
    amount = _as_number(value)
    if amount is None or amount < 0:
        return "No published price"
    if amount == 0:
        return "Free"
    # Decimal expands the float's shortest repr without scientific notation and
    # without rounding a tiny positive rate to "0 pollen" at a fixed precision.
    # Trim zeros only from a fractional part: an integer like 1e20 has none, and
    # stripping its trailing zeros would render 100000000000000000000 as "1".
    formatted = format(Decimal(str(amount)), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    # An absurd amount would otherwise render as a hundreds-of-digits label.
    if len(formatted) > 24:
        head = formatted[:24]
        if not head.rstrip("0."):
            # The prefix is all zeros: the significant digit lives past the cut. Keep it visible
            # in scientific form rather than rendering an all-zeros "≈0" label.
            return f"≈{amount:.2e} {currency}"
        return head + "… " + currency
    return formatted + " " + currency


def _advertises_free(model: Dict[str, Any]) -> bool:
    """True when the catalog's own price or naming calls the model free.

    A published price of zero is the stronger signal; the name is only a fallback, because a
    title containing "free" can be marketing copy for a model that is in fact billed.
    """
    if _as_number(_as_dict(model.get("pricing")).get("completionImageTokens")) == 0:
        return True
    text = f"{_text(model.get('title'))} {_text(model.get('name'))}"
    return re.search(r"\bfree\b", text, re.IGNORECASE) is not None


def _pricing_unit(model: Dict[str, Any]) -> str:
    """The unit the catalog bills ``completionImageTokens`` in, lowercased; '' for per image.

    The spec's unit enum is megapixel/token/byte; image/images/generation are lenient
    per-image synonyms, and any other unit reads as a generic per-unit charge (class 1).
    Sanitized like every catalog text field: a hostile unit must not smuggle line breaks or an
    absurd label into the picker's price column.
    """
    units = _as_dict(_as_dict(model.get("pricing_units")).get("completionImageTokens"))
    unit = units.get("unit")
    return _text(unit).lower() if isinstance(unit, str) else ""


def _price_label(model: Dict[str, Any], free_classic: bool, has_key: bool,
                 *, classic_route: bool = False) -> str:
    """Access requirement, watermark, edit support and cost, every part read from the live catalog.

    The catalog publishes no "needs a key", watermark or edit-support label, so all three are
    derived. A model the free classic API serves needs no key, but that route is the watermarked
    one, so once a key is set the same model is billed through the enter API instead and comes
    back clean — unless ``classic_route`` marks a catalog-omitted model that must keep using the
    classic route even keyed (the enter API does not serve it). ``supported_endpoints`` decides
    edit support, ``flat_rate`` whether the published price is per image or per token.
    """
    paid_only = _paid_flag(model) is not False
    amount = _as_number(_as_dict(model.get("pricing")).get("completionImageTokens"))
    # The spec guarantees "pollen", but the catalog publishes the currency — read it live.
    currency = _text(_as_dict(model.get("pricing")).get("currency"))[:16] or "pollen"
    if amount is not None and amount < 0:
        amount = None  # a negative rate is not a published price, whatever its type
    if free_classic and (not has_key or classic_route):
        if paid_only:
            return "Free on the classic route · watermark · reduced size · no key needed"
        return "Free · watermark · reduced size · no key needed"
    pricing = _as_dict(model.get("pricing"))
    unit = _pricing_unit(model)
    flat = unit in ("", "image", "images", "generation") and _as_bool(model.get("flat_rate"))
    adjustments = _image_adjustments(model)
    fees = _adjustments_per_image(model)
    folded = False  # whether convertible fees are included in the displayed figure
    if amount is None:
        # No image rate published: text-token rates still mean per-token billing; a name that
        # says "free" is marketing copy, not a published price, so both states agree.
        text_rate = (_as_number(pricing.get("completionTextTokens"))
                     if _as_number(pricing.get("completionTextTokens")) is not None
                     else _as_number(pricing.get("promptTextTokens")))
        if text_rate is not None:
            cost = "charged per token"
        elif adjustments:
            cost = "usage-based fees"  # the fees ARE the price; no base figure is published
            folded = True
        else:
            cost = "No published price"
    elif amount == 0:
        # A published zero image rate is Free only when nothing else is billed: positive
        # text-token rates still charge, and convertible fees still apply.
        text_billed = any(
            (rate := _as_number(pricing.get(field))) is not None and rate > 0
            for field in ("completionTextTokens", "promptTextTokens"))
        if text_billed:
            cost = (_format_price(fees, currency) + " · text billed") if fees else "Free image · text billed"
        else:
            cost = _format_price(fees, currency)
        folded = True
    elif unit == "megapixel":
        # rate x quantity is the published per-unit figure — show it (no area assumption).
        # A zero-rate variant must not erase a paid base: show the base + a free-variant note.
        floor = _variant_floor(model, amount)
        shown = amount if floor == 0 else floor
        cost = f"≈{_format_price(shown * _pricing_quantity(model), currency)} per megapixel"
        if floor == 0:
            cost += f" (variant {_format_price(fees, currency)})" if fees else " (free variant)"
    elif unit and unit not in ("image", "images", "generation", "token", "tokens"):
        # e.g. "byte": a generic per-unit charge, display-bounded like every label field.
        cost = f"charged per {unit[:32]}"
    elif not flat or unit in ("token", "tokens"):
        cost = "charged per token"
    else:
        # The per-image figure includes convertible per-generation fees and the unit quantity
        # so the label and the sort estimate agree. A zero-rate variant must not erase a paid
        # base: show the base figure + a free-variant note (the sort keeps the true floor).
        floor = _variant_floor(model, amount)
        shown = amount if floor == 0 else floor
        cost = _format_price(shown * _pricing_quantity(model) + fees, currency)
        if floor == 0:
            cost += f" (variant {_format_price(fees, currency)})" if fees else " (free variant)"
        folded = True
    # A model priced per variant (a 2K tier, an edit operation) can cost more than the base
    # figure the picker shows: a numeric floor reads "from X"; a per-unit figure says so.
    variants = _as_list(model.get("pricing_variants"))
    if variants:
        cheaper = amount is not None and any(
            (rate := _variant_rate(v)) is not None
            and 0 <= rate < amount for v in variants)
        if cost == "Free":
            cost = "Free base (variants may cost more)"
        elif "(free variant)" in cost or "(variant " in cost:
            # The note explains the cheaper direction; still warn about pricier variants.
            if amount is not None and any(
                    (rate := _variant_rate(v)) is not None
                    and rate > amount for v in variants):
                cost += " (variants may cost more)"
        elif cost.startswith("≈") or cost.startswith("charged per"):
            cost = f"{cost} (variants differ)" if cheaper else f"{cost} (variants may cost more)"
        elif cost not in ("No published price", "usage-based fees"):
            cost = f"from {cost}"
    parts = [cost]
    if adjustments and (not folded or any(
            _text(a.get("unit")).lower() not in ("", "generation", "image", "images")
            or _text(a.get("currency")).lower() not in ("", "pollen")
            for a in adjustments)):
        parts.append("fees extra")
    if has_key:
        input_rate = _as_number(pricing.get("promptImageTokens"))
        if input_rate is not None and input_rate > 0:
            parts.append("input billed")
    if free_classic:
        parts.append("no watermark")
    if has_key and _supports_edits(model):
        parts.append("edits")
    parts.append("paid" if paid_only else "needs key")
    return " · ".join(part for part in parts if part)


def _iso_date(model: Dict[str, Any]) -> str:
    """The catalog's ``added_date`` as a day, whatever unit it is in.

    The spec publishes a number with no unit description; live data is milliseconds, but a
    switch to seconds would otherwise render every model as 1970, so the magnitude decides.
    """
    raw = model.get("added_date")
    if not _is_number(raw):
        return ""
    seconds = float(raw)
    if seconds <= 0:
        return ""
    # A second-based epoch is 10 digits; a millisecond one is 13. The 1e11 boundary is 1973 as
    # milliseconds and year 5138 as seconds, so realistic epochs in either unit classify correctly.
    if abs(seconds) < 10**11:
        pass
    else:
        seconds /= 1000.0
    try:
        tm = time.gmtime(seconds)
    except (OverflowError, OSError, ValueError):
        return ""
    # A hostile or mistaken epoch must not render an absurd year in the picker; degrade to
    # blank like every other display field.
    if not 1970 <= tm.tm_year <= 2100:
        return ""
    return time.strftime("%Y-%m-%d", tm)


def _index(catalog: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every model name and alias to its canonical catalog id; the first entry wins."""
    out: Dict[str, str] = {}
    for model in catalog:
        name = _name_of(model) if isinstance(model, dict) else None
        if not name:
            continue
        keys = [name, *[clean for a in _as_list(model.get("aliases"))
                        if (clean := _text(a))]]
        # The ``id`` fallback is selectable too: a catalog publishing both name and id must
        # resolve an id-addressed model with its metadata, not as an unknown passthrough.
        alt = model.get("id")
        if isinstance(alt, str) and _valid_id(alt) and alt.strip() != name:
            keys.append(alt.strip())
        for key in keys:
            if key in out and out[key] != name:
                logger.debug("Pollinations catalog reuses %r for both %r and %r; keeping %r.",
                             key, out[key], name, out[key])
                continue
            out[key] = name
    return out


# UTS#39-lite confusable guard: NFKC+casefold absorbs ligature/fullwidth spellings; these
# dash variants look identical to '-' but NFKC leaves them alone.
_DASH_CONFUSABLES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D"})


def _norm_id(value: str) -> str:
    """Id-matching normalization: NFKC + casefold + dash confusables, so a ligature or
    fullwidth spelling of a paid id still reaches the paid gate instead of passing through
    as an unknown id. The catalog's own spelling is always what gets sent."""
    return unicodedata.normalize("NFKC", value).casefold().translate(_DASH_CONFUSABLES)


def _folded_index(catalog: List[Dict[str, Any]]) -> Dict[str, str]:
    """Case-folded name/alias/id -> canonical id, canonical names first (first entry wins).

    Catalog matching is case-insensitive like the classic matching, but the catalog's own
    spelling is restored, so a case variant of a paid id still reaches the paid gate instead
    of passing through ungated.
    """
    out: Dict[str, str] = {}
    for model in catalog:
        name = _name_of(model) if isinstance(model, dict) else None
        if name:
            out.setdefault(_norm_id(name), name)
    for model in catalog:
        name = _name_of(model) if isinstance(model, dict) else None
        if not name:
            continue
        keys = [clean for a in _as_list(model.get("aliases")) if (clean := _text(a))]
        alt = model.get("id")
        if isinstance(alt, str) and _valid_id(alt):
            keys.append(alt.strip())
        for key in keys:
            out.setdefault(_norm_id(key), name)
    return out


def _has_t2i_signal(model: Dict[str, Any]) -> bool:
    """True when the entry can run text-to-image: a non-edits image endpoint, or no endpoints
    published at all (the category signal alone). An edits-only entry needs a source image,
    so it must not become the preselected default."""
    endpoints = [e for e in _as_list(model.get("supported_endpoints")) if isinstance(e, str)]
    if not endpoints:
        return True
    return any("/image" in e.lower()
               and not e.lower().split("?", 1)[0].rstrip("/").endswith("/edits")
               for e in endpoints)


def _derived_default(catalog: List[Dict[str, Any]], classic: set,
                     has_key: bool = False) -> Optional[str]:
    """The model to preselect: the one the free classic API serves, else the first Quest-eligible.

    Derived from the live catalog, so a new free model becomes the default the day Pollinations
    adds it; ``None`` when the catalog offers nothing at all.

    The classic endpoint, not the catalog, decides the keyless route: Pollinations can drop a free
    model from the catalog while still serving it, and a stale catalog must not leave a keyless
    user with a model that needs a key. Without a key a Quest-eligible model is not a valid
    fallback, so the classic list is used instead.
    """
    fallback: Optional[str] = None
    free_fallback: Optional[str] = None
    for model in catalog:
        if not _is_image_model(model):
            continue
        name = _name_of(model)
        if not name:
            continue
        if _is_free_classic(model, classic):
            # A free-classic model the catalog also marks paid-only is still usable keyless, but
            # a plainly free one is the better default. Keyed, generate() refuses paid-only
            # models while the toggle is off, so they are no derived default at all.
            if _paid_flag(model) is False:
                # Keyed, a catalog-listed model routes to ENTER even when classic serves it:
                # an SVG or edits-only entry would be refused there, so keep looking. Keyless
                # classic rasterizes to JPEG and runs (the SVG exception).
                if not has_key or (not _is_svg_product(model) and _has_t2i_signal(model)):
                    return name
            if free_fallback is None and not has_key:
                free_fallback = name
        if (fallback is None and has_key and _paid_flag(model) is False
                and _has_t2i_signal(model) and not _is_svg_product(model)):
            fallback = name
    if free_fallback:
        return free_fallback
    if fallback:
        return fallback
    classic_ids = _classic_models()
    if has_key:
        # A classic id the catalog classifies paid-only (or unreadable) is refused by
        # generate() keyed; keep only unclassified ids (classic route) and free ones.
        classic_ids = [cid for cid in classic_ids
                       if _paid_flag(_catalog_entry(catalog, cid)) is False]
    return classic_ids[0] if classic_ids else None


def _resolve_model(explicit: Optional[str], catalog: List[Dict[str, Any]]) -> Optional[str]:
    """First candidate naming a catalog model (canonical id preferred over alias), else passthrough.

    Catalog names/aliases match case-insensitively and restore the catalog's own spelling, so a
    case variant still reaches the paid gate. A classic-served id outranks a paid alias claim.
    An unknown id passes through verbatim, so a model added upstream works the day it appears —
    with one exception. Hermes injects the shared top-level ``image_gen.model`` as the explicit
    ``model`` kwarg on every tool call, so an unknown value there is usually another provider's
    leftover id rather than a new Pollinations model; that value falls back to the derived
    default instead of being sent to Pollinations — unless the free classic endpoint serves it,
    in which case the classic route can still run it. An explicit env/scoped choice passes through
    even when it equals the shared value (the bundled resolve_static_model convention).

    ``None`` when nothing is configured and the catalog offers no default to derive.
    """
    index = _index(catalog)
    folded = _folded_index(catalog)
    config = load_image_gen_config()
    scoped = config.get(PROVIDER_ID)
    shared = config.get("model")
    shared_norm = shared.strip() if isinstance(shared, str) else ""
    has_key = _api_key() is not None
    classic = _classic_set()
    spelling = _classic_spelling()
    candidates = [
        explicit,
        _model_env(),
        scoped.get("model") if isinstance(scoped, dict) else None,
        shared,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            wanted = candidate.strip()
            # A canonical name beats an alias: a hostile catalog entry must not shadow a free
            # model's id with a paid one by claiming it as an alias (the index is first-wins).
            canonical = next((_name_of(m) for m in catalog if _name_of(m) == wanted), None)
            if canonical is not None:
                return canonical
            key = _norm_id(wanted)
            if key in spelling:
                # A classic-served id keeps the endpoint's own spelling unless a readable,
                # non-paid image entry legitimately claims it as an alias. A paid (or
                # unreadable) entry must not hijack the user's free model. A normalized
                # CANONICAL beats an exact alias claim here too (same rule as below).
                folded_target = folded.get(key)
                # A folded value is always SOME entry's canonical name; it is a canonical
                # match for THIS key only when that name itself normalizes to the key. An
                # alias-derived folded value falls through to the exact-alias lookup, so
                # case-variant alias collisions rank exactly like the non-classic path.
                if folded_target is not None and _norm_id(folded_target) == key:
                    target = folded_target
                else:
                    target = index.get(wanted) or folded.get(key)
                entry = _catalog_entry(catalog, target) if target else None
                if entry is not None and _is_image_model(entry) and _paid_flag(entry) is False:
                    return target
                # R6-01: a paid-only (or unreadable) CANONICAL catalog entry that is also
                # classic-served must keep its canonical spelling. Returning the endpoint's
                # case variant here makes _catalog_entry(catalog, model) miss it and skips the
                # paid gate, running the paid model on the free watermarked route. An
                # alias-derived folded value is NOT a canonical match (it normalizes to a
                # different key), so a paid alias still cannot hijack a free classic id.
                if (folded_target is not None and _norm_id(folded_target) == key
                        and entry is not None and _is_image_model(entry)
                        and _paid_flag(entry) is not False):
                    return folded_target
                return spelling[key]
            # A normalized CANONICAL beats another model's exact alias (the folded index is
            # canonical-first), so a case variant of a free id cannot be hijacked by a paid
            # entry claiming the lowercase spelling as an alias. Alias-derived folded values
            # are NOT canonical matches: they fall through to the exact-alias ranking below.
            target = folded.get(key)
            if target is not None and _norm_id(target) == key:
                return target
            if wanted in index:
                return index[wanted]
            if key in folded:
                return folded[key]
    for position, candidate in enumerate(candidates[:3]):
        value = candidate.strip() if isinstance(candidate, str) else ""
        if not value or not _valid_id(value):
            continue
        if position == 0 and value == shared_norm and _norm_id(value) not in classic:
            # The injected shared kwarg carrying an unknown id: likely another provider's leftover.
            # A classic-served id is exempt: the free route can run it (keyed included), so the
            # user's saved free model must not be swapped for a billed Quest default.
            continue
        # An unknown id passes through so a model added upstream works the day it appears.
        # Keyless, though, only the free classic route can run it, and an id the catalog
        # does not know there is far more likely a leftover than a brand-new free model.
        if has_key or _norm_id(value) in classic:
            return value
        logger.debug("Ignoring %r: not a catalog model and not served keyless.", value)
    return _derived_default(catalog, classic, has_key)


def _configured_model(catalog: List[Dict[str, Any]]) -> Optional[str]:
    """The model Hermes will actually use for this call.

    The tool injects the shared ``image_gen.model`` as the ``model`` kwarg on every call, so
    resolving with that value — rather than from the scoped key alone — is what makes
    ``capabilities()`` describe the same model ``generate()`` will use. The two disagreeing is
    how a model ends up advertised with controls it cannot honour.
    """
    return _resolve_model(load_image_gen_config().get("model"), catalog)


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    from agent.image_routing import _sniff_mime_from_bytes

    return _sniff_mime_from_bytes(raw)


def _image_is_complete(raw: bytes, sniffed: Optional[str]) -> bool:
    """True when the payload carries its format's end marker, not just its magic bytes.

    A truncated download keeps its header, so a magic-byte check alone would report success on a
    file a viewer cannot open. These are cheap structural pre-filters, not a decode or an
    integrity check: every marker here (JPEG EOI, GIF trailer, PNG IEND suffix, WebP first
    chunk) can be spoofed by a crafted payload with no decodable image data. They catch
    truncation, not malice.
    """
    if not raw:
        return False
    if sniffed == "image/png":
        # IEND is the final chunk: 4 zero length bytes + tag + its fixed CRC.
        return raw.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
    if sniffed == "image/jpeg":
        # Some encoders append metadata after EOI; allow a short tail.
        return b"\xff\xd9" in raw[-64:]
    if sniffed == "image/gif":
        return raw.rstrip().endswith(b";")
    if sniffed == "image/webp":
        if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
            return False
        declared = int.from_bytes(raw[4:8], "little")
        return (len(raw) >= declared + 8 and declared >= 12
                and raw[12:16] in (b"VP8 ", b"VP8L", b"VP8X"))
    if sniffed == "image/svg+xml":
        return b"</svg>" in raw[-256:].lower()
    return True


def _encode_input_image(raw: bytes) -> str:
    """Size- and MIME-check raw image bytes, then return a canonical ``data:`` URL."""
    if len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"source image exceeds the {MAX_INPUT_IMAGE_BYTES // (1024 * 1024)}MB cap")
    mime = _sniff_image_mime(raw)
    if mime not in ACCEPTED_INPUT_MIME:
        raise ValueError("source image is not a supported PNG, JPEG, GIF or WebP image")
    if not _image_is_complete(raw, mime):
        raise ValueError("source image is incomplete or corrupt")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _remote_image_to_data_url(value: str, limit: int = MAX_INPUT_IMAGE_BYTES) -> str:
    """Fetch a remote source image through the SSRF guard; the edits endpoint takes data URIs.

    ``limit`` bounds the download by the remaining payload budget, not just the per-image cap.
    """
    from tools.url_safety import is_safe_url

    if not is_safe_url(value):
        # The URL can carry a signed query string, so it is logged for the operator rather than
        # echoed into the error the agent sees.
        logger.debug("Source image URL failed the SSRF safety check: %s", value)
        raise ValueError("source image URL failed the SSRF safety check")
    with _ssrf_client(60.0) as client:
        with client.stream("GET", value) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length") or ""
            if declared.isdigit() and int(declared) > limit:
                raise ValueError(f"source image exceeds {limit} bytes")
            raw = _bounded_read(response, value, limit=limit)
    return _encode_input_image(raw)


def _local_image_to_data_url(value: str, limit: int = MAX_INPUT_IMAGE_BYTES) -> str:
    """Read a local source image through the same read guard the file tools use."""
    from agent.file_safety import get_read_block_error

    blocked = get_read_block_error(value)
    if blocked:
        raise ValueError(blocked)
    path = Path(os.path.expanduser(value)).resolve()
    if not path.is_file():
        raise ValueError(f"source image path is not a file: {value}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"source image path is empty: {value}")
    if size > limit:
        raise ValueError(f"source image exceeds {limit} bytes")
    # Read with the same ceiling: a file can change between stat() and read(), and a huge file
    # must not be buffered just because it was small a moment ago.
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"source image exceeds {limit} bytes")
    return _encode_input_image(raw)


def _to_data_url(value: str, limit: int = MAX_INPUT_IMAGE_BYTES) -> str:
    """Inline one source image as a base64 data URI, whatever form Hermes handed us.

    Hermes passes source images as public URLs, base64 data URIs or absolute local paths; the
    edits endpoint accepts a URL or an array of ``{image_url}`` objects, and a data URI keeps the
    request to a single JSON call with no extra upload step. ``limit`` is the raw-byte budget
    remaining for this image, so reading and encoding stay inside the payload cap.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("blank source image")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        return _remote_image_to_data_url(candidate, limit)
    if lowered.startswith("data:"):
        header, _, data = candidate.partition(",")
        if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise ValueError("source image data URL must be a base64 image")
        # Exact decoded length: the naive (len*3)//4 overestimates by the base64 padding
        # (up to 2 bytes), falsely rejecting an exactly-at-cap image that local/URL accept.
        if (len(data) * 3) // 4 - data.count("=") > limit:
            raise ValueError(f"source image exceeds {limit} bytes")
        return _encode_input_image(base64.b64decode(data, validate=True))
    return _local_image_to_data_url(candidate, limit)


_SVG_PRODUCT_RE = re.compile(r"\bsvg\b", re.IGNORECASE)


def _is_svg_product(model: Optional[Dict[str, Any]]) -> bool:
    """True when the live title/description advertises SVG output. This plugin returns raster
    images only, so an SVG-product model is refused BEFORE the request — a paid attempt would
    bill and then fail the output gate. Detection is live-text driven (no hardcoded model
    ids); over-disabling is the safe failure mode."""
    if not isinstance(model, dict):
        return False
    if any(isinstance(m, str) and m.strip().lower() in ("svg", "image/svg+xml")
           for m in _as_list(model.get("output_modalities"))):
        return True
    text = f"{_text(model.get('title'))} {_text(model.get('description'))}"
    return _SVG_PRODUCT_RE.search(text) is not None


def _visible_edit_models(catalog: List[Dict[str, Any]], *, include_paid: bool,
                         limit: int = EDIT_HINT_MODELS) -> List[str]:
    """Edit-capable model names the current credential state would list, for error messages.

    Only called once a key is present, so every listed model is edit-capable in principle.
    """
    names: List[str] = []
    for model in catalog:
        if not _is_image_model(model):
            continue
        if not _supports_edits(model):
            continue
        if _is_svg_product(model):
            continue
        if _paid_flag(model) is None or (_paid_flag(model) is not False and not include_paid):
            continue
        name = _name_of(model)
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _extension_for(content_type: str, url: str) -> str:
    lowered = (content_type or "").lower()
    for marker, ext in (("png", "png"), ("webp", "webp"), ("gif", "gif"),
                        ("svg", "svg"), ("bmp", "bmp"), ("tiff", "tiff"),
                        ("icon", "ico"), ("avif", "avif"), ("heic", "heic"),
                        ("jpeg", "jpg"), ("jpg", "jpg")):
        if marker in lowered:
            return ext
    path = url.split("?", 1)[0].lower()
    for ext in ("png", "jpg", "jpeg", "webp", "gif", "svg"):
        if path.endswith(f".{ext}"):
            return "jpg" if ext == "jpeg" else ext
    return "jpg"


def _specs(model: Dict[str, Any], free_classic: bool, has_key: bool,
           *, classic_route: bool = False) -> Dict[str, Any]:
    """One catalog entry as picker metadata; unpublished fields stay blank.

    ``speed`` carries the catalog's per-user rate limit in plain words, and ``strengths`` leads
    with the catalog's safety filters when it publishes any — otherwise the description's own
    opening words are the most useful summary and a truncated tail can hide a privacy notice.
    """
    rpm = model.get("per_user_rpm")
    health = _as_dict(model.get("health"))
    description = _text(model.get("description"))
    safety = [_text(item) for item in _as_list(model.get("required_safety")) if _text(item)]
    strengths = f"Safety filters: {', '.join(safety)}. {description}" if safety else description
    success_rate = health.get("success_rate")
    # per_user_rpm is the authenticated limit; the keyless classic route is per-IP throttled, so
    # quoting it there would misstate how fast that model actually is.
    keyless_free = free_classic and (not has_key or classic_route)
    speed = ""
    # 10000 is a display sanity bound, not a Pollinations fact: an absurd finite rpm must not
    # render as a hundreds-of-digits or scientific-notation label.
    if not keyless_free and _is_number(rpm) and 0 < rpm <= 10000:
        speed = f"≤{rpm} req/min"
    return {
        "id": _name_of(model),
        "display": _text(model.get("title")) or _name_of(model) or "",
        "publisher": _text(model.get("publisher")),
        "community": _as_bool(model.get("community")),
        "description": description,
        "strengths": strengths[:STRENGTHS_LIMIT],
        "aliases": [clean for a in _as_list(model.get("aliases")) if (clean := _text(a))],
        "price": _price_label(model, free_classic, has_key, classic_route=classic_route),
        "price_pollen": _price_value(model),
        "price_class": _price_class(model),
        "rpm": rpm if _is_number(rpm) else "",
        "speed": speed,
        "modalities": [clean for m in _as_list(model.get("input_modalities"))
                       if (clean := _text(m))],
        "endpoints": [clean for e in _as_list(model.get("supported_endpoints"))
                      if (clean := _text(e))],
        "health": {
            "status": _text(health.get("status")),
            "success_rate": round(success_rate, 1) if _is_number(success_rate) else None,
            "requests": (health.get("requests")
                         if isinstance(health.get("requests"), int)
                         and not isinstance(health.get("requests"), bool) else None),
        },
        "added": _iso_date(model),
        "flat_rate": _as_bool(model.get("flat_rate")),
        "free_classic": free_classic,
    }


# -------------------------------------------------------------------------- provider

class _Route(NamedTuple):
    """A ready text-to-image request, or the reason it cannot be made."""

    url: Optional[str]
    headers: Optional[Dict[str, str]]
    extra: Dict[str, Any]
    error: Optional[str]
    error_type: str = "auth_required"


def _text_to_image_route(
    prompt: str,
    model: str,
    entry: Optional[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    width: int,
    height: int,
    seed: int,
    resolution: str,
) -> _Route:
    """Choose the transport for a text-to-image call and build its URL.

    The free classic API is watermarked and throttled per IP, so it is the no-key route only.
    With a key the same model is billed through the enter API — still Quest-eligible, still
    cheap, but clean and unthrottled. ``error`` is set when the model cannot run under the
    current credentials.
    """
    extra: Dict[str, Any] = {"width": width, "height": height}
    key = _api_key()
    classic_id = _classic_id_for(model, catalog)
    # Keyless, the classic route is the only route. Keyed, a model the classic API serves but
    # the gen catalog does not list (the free model after a key is added) would 400 on the enter
    # API — keep it working on the classic route instead. A catalog-listed model always uses the
    # enter API once a key is set: clean and unthrottled.
    if classic_id is not None and (key is None or entry is None):
        extra["mode"] = "free"
        return _Route(
            url=(
                f"{CLASSIC_IMAGE_URL}/{quote(prompt, safe='')}"
                f"?model={quote(classic_id, safe='')}&width={width}&height={height}"
                f"&seed={seed}"
            ),
            headers={"Accept": "image/*"},
            extra=extra,
            error=None,
        )

    if key is None:
        unreachable = _free_route_unavailable()
        reason = (
            "the free model list could not be reached"
            if unreachable
            else f"model '{model}' is not served by the free classic API"
        )
        return _Route(
            url=None, headers=None, extra=extra,
            error_type=("api_error" if unreachable else "model_unavailable"),
            error=(
                f"Cannot generate '{model}': {reason}, and no POLLINATIONS_API_KEY is set. "
                "Add a key (https://enter.pollinations.ai/keys), or pick a model the free "
                "classic API serves (no key needed)."
            ),
        )

    # The resolved entry already carries paid_only; re-scanning the catalog would be O(n) again.
    paid_only = _as_bool(entry.get("paid_only")) if entry else False
    extra["mode"] = "paid" if paid_only else "quest"
    # Hermes' resolution creative control is honoured only on the authenticated route, and only
    # for a resolution this model actually publishes.
    suffix = f"&resolution={quote(resolution, safe='')}" if resolution in _published_resolutions(entry) else ""
    return _Route(
        url=(
            f"{ENTER_IMAGE_URL}/{quote(prompt, safe='')}"
            f"?model={quote(model, safe='')}&width={width}&height={height}&seed={seed}"
            f"{suffix}"
        ),
        headers={"Authorization": f"Bearer {key}", "Accept": "image/*"},
        extra=extra,
        error=None,
    )


class PollinationsImageGenProvider(ImageGenProvider):
    """Pollinations text-to-image with a live, Quest-eligible model catalog."""

    def __init__(self, plugin_id: str = PLUGIN_ID) -> None:
        # The registry key this plugin was loaded under; see _include_paid_models().
        self._plugin_id = plugin_id

    @property
    def name(self) -> str:
        return PROVIDER_ID

    @property
    def display_name(self) -> str:
        return "Pollinations"

    def is_available(self) -> bool:
        return True  # free classic mode needs no credentials

    def list_models(self) -> List[Dict[str, Any]]:
        """Picker rows for the current credential state, cheapest estimated per-image cost first.

        Per-image and per-megapixel rates are normalized to a per-image estimate; per-token
        models follow them (no published token count to compare with), and unpublished prices
        sort last. No key → the free classic model only. Key → every Quest-eligible model. Key
        plus the ``include_paid_models`` setting → the paid-only catalog too. The user's saved
        model is kept visible whenever the current state can run it (a keyed passthrough id gets
        a minimal row; a retained paid row is marked disabled), so removing the key or turning
        the setting off never makes a runnable configured model silently disappear.
        """
        catalog = _catalog()
        classic = _classic_set()
        has_key = _api_key() is not None
        include_paid = has_key and _include_paid_models(self._plugin_id)

        rows: Dict[str, Dict[str, Any]] = {}
        seen_lower: set = set()
        for model in catalog:
            if not _is_image_model(model):
                continue
            name = _name_of(model)
            if not name:
                continue
            if name in rows or name.lower() in seen_lower:
                logger.debug("Pollinations catalog lists %r more than once; keeping the first.", name)
                continue
            free_classic = _is_free_classic(model, classic)
            if not has_key:
                if not free_classic:
                    continue  # state 1: the free classic API is the only keyless route
            elif _paid_flag(model) is None:
                continue  # unreadable billing: generate() refuses it regardless of the toggle
            elif _paid_flag(model) is not False and not include_paid:
                # state 2: Quest-eligible only (unreadable flags stay hidden). Keyed, a
                # paid-only model is refused by generate() even when classic serves it, so
                # it must not be offered here either.
                continue
            rows[name] = _specs(model, free_classic, has_key)
            if has_key and _is_svg_product(model):
                rows[name]["price"] += " · disabled (SVG output unsupported)"
            seen_lower.add(name.lower())

        if not has_key:
            # The classic endpoint is the source of truth for the keyless route: Pollinations can
            # drop a free model from the catalog while still serving it, and a stale catalog must
            # not leave a keyless user with nothing to pick.
            for key in [item.strip() for item in _classic_models() if _valid_id(item)]:
                if key not in rows and key.lower() not in seen_lower:
                    # The classic endpoint publishes no metadata, so the row is built from the id
                    # alone; ``input_modalities`` states what the route accepts.
                    rows[key] = _specs({"name": key, "input_modalities": ["text"]},
                                      True, has_key)
                    seen_lower.add(key.lower())

        configured = _configured_model(catalog)
        if configured and configured not in rows and configured.lower() not in seen_lower:
            saved = next((m for m in catalog if _name_of(m) == configured), None)
            if saved is not None:
                # Retention keeps the user's saved model visible — but never a video entry:
                # the picker offers image models only, matching capabilities() and generate().
                if not _is_video_model(saved):
                    rows[configured] = _specs(saved, _is_free_classic(saved, classic), has_key)
                    if _is_svg_product(saved) and (has_key or _classic_id_for(configured, catalog) is None):
                        # generate() refuses SVG before the paid gate — except a KEYLESS
                        # classic-served call, which rasterizes to JPEG and runs. Mirror that
                        # exemption so the marker never contradicts what generate() does.
                        rows[configured]["price"] += " · disabled (SVG output unsupported)"
                    elif has_key and _paid_flag(saved) is None:
                        rows[configured]["price"] += " · disabled (unreadable billing flag)"
                    elif has_key and _paid_flag(saved) is not False and not include_paid:
                        # Say plainly that generate() will refuse this row until the toggle
                        # is on, rather than offering an unrunnable choice unexplained.
                        rows[configured]["price"] += " · disabled (paid toggle off)"
                    elif not has_key and not _is_free_classic(saved, classic):
                        rows[configured]["price"] += " · disabled (needs a key)"
                    seen_lower.add(configured.lower())
            elif _valid_id(configured) and _norm_id(configured) in classic:
                # A saved classic-only model the catalog omits: the classic route can still run
                # it (keyed included), so keep the user's choice visible with an honest label.
                spelling = _classic_spelling().get(_norm_id(configured), configured)
                rows[spelling] = _specs({"name": spelling, "input_modalities": ["text"]},
                                        True, has_key, classic_route=True)
                seen_lower.add(spelling.lower())
            elif has_key and _valid_id(configured):
                # A saved passthrough id the catalog does not list: keyed it still runs, so
                # keep the user's choice visible with an honest minimal row. Keyless it
                # cannot run unless classic serves it (the branch above).
                rows[configured] = _specs({"name": configured, "input_modalities": ["text"]},
                                          False, has_key)
                seen_lower.add(configured.lower())

        def _order(row: Dict[str, Any]) -> tuple:
            if not has_key and row["free_classic"]:
                return (0, 0.0)  # keyless: the classic route costs nothing, whatever enter prices
            return (row["price_class"], row["price_pollen"])

        ordered = sorted(rows.values(), key=_order)
        return catalog_rows({row["id"]: row for row in ordered}, fields=MODEL_FIELDS)

    def default_model(self) -> Optional[str]:
        """The resolved configured model when one is set, else the live-derived default.

        Returning the configured model keeps the CLI picker's "currently in use" marker on
        the right row for saved aliases/case variants (the core compares the raw config value
        against row ids and falls back to this method). With nothing configured, the free
        classic model leads, else the first Quest-eligible one — both read live.
        """
        catalog = _catalog()
        configured = _configured_model(catalog)
        if configured:
            return configured
        return _derived_default(catalog, _classic_set(), _api_key() is not None)

    def capabilities(self) -> Dict[str, Any]:
        """What the configured model can do, every part read from the live catalog.

        Editing is advertised only when a key is set — the free classic route is text-to-image
        only — and the configured model lists the edits endpoint, so Hermes never offers an
        edit the current setup cannot perform. The reference cap and the 1K/2K resolution
        control come from the same catalog entry, so a model that gains or loses edit support
        is reflected the next time the catalog is fetched.

        The entry must be an image model, matching what ``list_models`` offers; a video model
        id configured by hand gets the text-only default rather than capabilities it cannot use.
        """
        catalog = _catalog()
        entry = _catalog_entry(catalog, _configured_model(catalog))
        if entry is None or not _is_image_model(entry) or (_api_key() and _is_svg_product(entry)):
            entry = None
        if (entry is not None and (_paid_flag(entry) is None
                or (_paid_flag(entry) is not False and not _include_paid_models(self._plugin_id)))):
            # generate() refuses this model while the toggle is off; the schema must not
            # advertise edits/resolution the call cannot perform.
            entry = None
        capabilities: Dict[str, Any] = {"modalities": ["text"], "max_reference_images": 0}
        if _supports_resolution(entry):
            # The resolution control is honoured only on the authenticated route, so it is
            # offered only when a key is set; the free classic route has no such parameter.
            capabilities["creative_controls"] = ["resolution"] if _api_key() else []
        if _api_key() and _supports_edits(entry):
            capabilities["modalities"] = ["text", "image"]
            capabilities["max_reference_images"] = _reference_cap(entry)
        return capabilities

    def get_setup_schema(self) -> Dict[str, Any]:
        return api_key_setup_schema(
            "Pollinations", "free",
            "Live catalog — free classic API with no key, or the Quest-eligible catalog with a Pollinations key",
            key=API_KEY_ENVS[0],
            prompt="Pollinations secret key (sk_…) — unlocks the Quest-eligible models",
            url="https://enter.pollinations.ai/keys",
        )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = str(prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required", error_type="invalid_input",
                provider=PROVIDER_ID, prompt="", aspect_ratio=aspect)
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate survives JSON tool-call decoding but cannot become a URL path
            # segment; refuse it here instead of letting quote() raise out of generate().
            return error_response(
                error="Prompt contains unsupported characters that cannot be sent to Pollinations.",
                error_type="invalid_input", provider=PROVIDER_ID, prompt="", aspect_ratio=aspect)

        catalog = _catalog()
        model = _resolve_model(kwargs.get("model"), catalog)
        if not model:
            return error_response(
                error="No Pollinations model is available. The live model catalog could not be "
                      "reached and no model is configured; check your connection and try again.",
                error_type="no_model_available", provider=PROVIDER_ID, prompt=prompt,
                aspect_ratio=aspect)
        # A future core aspect ratio degrades to square instead of raising KeyError out of
        # generate() (the bundled SIZES.get convention).
        width, height = SIZES.get(aspect) or SIZES["square"]
        entry = _catalog_entry(catalog, model)
        fail = error_factory(PROVIDER_ID, aspect, model=model, prompt=prompt)

        # A known catalog entry must be an image model: a hand-configured video id is refused
        # here rather than routed to the image endpoint (capabilities already advertises it
        # text-only, and the picker never offers it).
        if entry is not None and _is_video_model(entry):
            return fail(
                f"Model '{model}' is a video model, not an image model. Pick an image model "
                "from the picker.", "model_unavailable")
        if entry is not None and _is_svg_product(entry):
            # The enter route would return SVG, which the raster output gate refuses AFTER
            # billing. Only a KEYLESS classic-served call rasterizes to JPEG: keyed, a
            # catalog-listed model routes to enter even when classic serves it.
            if not (_api_key() is None and _classic_id_for(model, catalog) is not None):
                return fail(
                    f"Model '{model}' produces SVG output, which this plugin cannot return. "
                    "Pick a raster image model from the picker.", "unsupported")
        # The paid toggle gates invocation, not just the picker: a paid-only catalog model must
        # not spend while the setting is off, whichever path (kwarg, env, config) selected it.
        # An unreadable paid_only flag fails CLOSED — the billing class cannot be verified.
        # Keyless calls keep the free-route guidance below; an id the catalog does not know
        # cannot be classified and passes through as before.
        if _api_key() is not None and entry is not None:
            paid = _paid_flag(entry)
            if paid is None:
                return fail(
                    f"Model '{model}' publishes an unreadable paid_only flag; refusing to run "
                    "it while its billing class cannot be verified.", "model_unavailable")
            if paid and not _include_paid_models(self._plugin_id):
                return fail(
                    f"Model '{model}' is paid-only and 'Include paid models' is off. Enable it "
                    "under Desktop → Settings → Plugins → Pollinations, or pick a Quest-eligible "
                    "model.", "paid_model_disabled")

        # A source image means an edit, which only the enter API can do; the free classic route
        # is text-to-image only, so there is nothing to fall back to. Dedupe order-preserving:
        # the agent commonly repeats image_url inside the references, and a duplicate would eat
        # a reference slot and double-count against the payload budget (bundled krea does the same).
        sources = list(dict.fromkeys(collect_source_images(image_url, reference_image_urls)))
        seed = _seed_from(kwargs)
        if sources:
            return self._edit(prompt, aspect, model, entry, sources, width, height,
                              kwargs.get("resolution"), seed, fail)

        try:
            route = _text_to_image_route(
                prompt, model, entry, catalog, width, height, seed,
                str(kwargs.get("resolution") or "").strip().lower())
        except Exception:
            # Defense in depth: every id/prompt is guarded above, but a URL-building failure
            # must degrade to an error response, never escape generate().
            logger.debug("Could not build the Pollinations request", exc_info=True)
            return fail("Could not build the Pollinations request — the prompt or model id "
                        "contains unsupported characters.", "invalid_input")
        if route.error:
            return fail(route.error, route.error_type or "auth_required")
        url, headers, extra = route.url or "", route.headers or {}, route.extra

        try:
            data, produced = _http_get_image(url, headers=headers)
        except Exception as exc:
            return _request_failed(exc, action="image generation",
                                   mode=str(extra.get("mode") or "quest"), fail=fail)
        if produced is None:
            # No announced size (the classic route never sends one): measure the produced
            # image structurally (any raster format) so a server-side clamp is reported honestly.
            produced = _image_dimensions(data, _sniff_image_mime(data))
        if produced is not None:
            # The API announced the produced size (it can clamp oversized requests); report it
            # instead of the requested one, keeping the request alongside for transparency.
            extra["requested_width"], extra["requested_height"] = extra["width"], extra["height"]
            extra["width"], extra["height"] = produced

        if not data:
            return fail("Pollinations returned an empty body", "empty_response")
        # A 200 can still carry an error body; only save it when it really is an image.
        sniffed = _sniff_image_mime(data)
        if sniffed is None or sniffed not in ACCEPTED_INPUT_MIME:
            # Raster only: SVG and other active formats are refused as generated output.
            return fail("Pollinations returned a non-image or non-raster response",
                        "invalid_response")
        if not _image_is_complete(data, sniffed):
            return fail("Pollinations returned an incomplete image", "invalid_response")

        try:
            image_ref = str(save_b64_image(
                base64.b64encode(data).decode(),
                prefix="pollinations",
                extension=_extension_for(sniffed or "", url),
            ))
        except Exception as exc:
            return fail(f"Could not save Pollinations image: {exc}", "io_error")

        return success_response(
            image=image_ref, model=model, prompt=prompt, aspect_ratio=aspect,
            provider=PROVIDER_ID, modality="text", extra=extra)

    def _edit(
        self,
        prompt: str,
        aspect: str,
        model: str,
        entry: Optional[Dict[str, Any]],
        sources: List[str],
        width: int,
        height: int,
        resolution: Any,
        seed: int,
        fail: Callable[[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """``POST /v1/images/edits`` with the source images inlined as data URIs.

        Only the enter API can edit, so a key is mandatory on this path even though the rest of
        the plugin works without one.
        """
        api_key = _api_key()
        if api_key is None:
            return fail(
                "Image editing needs a Pollinations API key — the free classic route is "
                "text-to-image only. Add a key at https://enter.pollinations.ai/keys.",
                "auth_required")
        if entry is not None and not _is_image_model(entry):
            # capabilities() advertises editing only for image entries; generate must agree.
            return fail(f"Model '{model}' is not an image model and cannot edit images.",
                        "unsupported")
        if entry is not None and _is_svg_product(entry):
            return fail(f"Model '{model}' produces SVG output, which this plugin cannot return.",
                        "unsupported")
        if not _supports_edits(entry):
            options = _visible_edit_models(_catalog(), include_paid=_include_paid_models(self._plugin_id))
            hint = f" Edit-capable models: {', '.join(options)}." if options else ""
            return fail(f"Model '{model}' cannot edit images.{hint}", "unsupported")

        prompt_max = _spec_limits()["edit_prompt_max"]
        if prompt_max is not None and len(prompt) > prompt_max:
            return fail(
                f"The edit prompt is {len(prompt)} characters; the API accepts at most "
                f"{prompt_max}. Shorten it and try again.", "invalid_input")
        # Hermes caps the reference count in its schema, but a direct call could over-send.
        cap = _reference_cap(entry)
        dropped = max(0, len(sources) - cap)
        sources = sources[:cap]

        inlined: List[Dict[str, str]] = []
        total = 0
        # Account BEFORE the base64 blow-up: the prompt plus JSON framing count against the same
        # budget, and each source is read/encoded under the REMAINING budget (converted back to
        # raw bytes), so a small reference still fits after a large one. Rejecting early keeps
        # transient memory bounded instead of encoding an image that cannot fit.
        # Budget the SERIALIZED fixed fields: JSON escapes (interior newlines, quotes,
        # control chars) expand the body beyond the raw UTF-8 byte count.
        fixed_fields: Dict[str, Any] = {
            "prompt": prompt, "image": [], "model": model, "size": f"{width}x{height}",
            "seed": seed, "response_format": "b64_json",
        }
        requested = str(resolution or "").strip().lower()
        if requested and requested in _published_resolutions(entry):
            fixed_fields["resolution"] = requested
        overhead = len(json.dumps(fixed_fields, ensure_ascii=False).encode("utf-8"))
        try:
            for source in sources:
                remaining = MAX_EDIT_PAYLOAD_BYTES - total - overhead
                raw_limit = min(MAX_INPUT_IMAGE_BYTES,
                                ((remaining - _DATA_URI_HEADER_BYTES) * 3) // 4)
                if raw_limit < _MIN_SOURCE_BYTES:
                    return fail(
                        "Those reference images are too large to send "
                        f"(limit {MAX_EDIT_PAYLOAD_BYTES // (1024 * 1024)}MB including the "
                        "prompt). Use fewer or smaller images.", "invalid_input")
                data_url = _to_data_url(source, limit=raw_limit)
                total += len(data_url)
                # Belt-and-suspenders: the remaining-budget conversion above already keeps
                # the total inside the cap; this re-check guards any future arithmetic drift.
                if total + overhead > MAX_EDIT_PAYLOAD_BYTES:
                    return fail(
                        f"Those reference images are too large to send "
                        f"({total // (1024 * 1024)}MB so far, limit "
                        f"{MAX_EDIT_PAYLOAD_BYTES // (1024 * 1024)}MB). Use fewer or smaller "
                        "images.", "invalid_input")
                inlined.append({"image_url": data_url})
        except ValueError as exc:
            return fail(f"Could not read the source image: {exc}", "invalid_input")
        except Exception as exc:
            # The raw exception can carry the source URL, which may hold signed query parameters.
            logger.debug("Could not read a source image", exc_info=True)
            return fail(f"Could not read the source image ({_network_error_type(exc)}). "
                        "See the Hermes log for detail.", _network_error_type(exc))
        image = inlined

        payload: Dict[str, Any] = {**fixed_fields, "image": image}
        # Exact check on the serialized body the client will send (escapes included), so the
        # cap holds even when the estimate above and reality drift apart.
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_EDIT_PAYLOAD_BYTES:
            return fail(
                "Those reference images are too large to send "
                f"(limit {MAX_EDIT_PAYLOAD_BYTES // (1024 * 1024)}MB including the prompt). "
                "Use fewer or smaller images.", "invalid_input")

        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        try:
            answered = _http_post_json(_edit_url(entry), headers=headers, payload=payload)
        except Exception as exc:
            return _request_failed(exc, action="image edit",
                                   mode=("paid" if _as_bool((entry or {}).get("paid_only"))
                                         else "quest"), fail=fail)

        if not isinstance(answered, dict):
            return fail("Pollinations returned an unexpected edit payload", "invalid_response")
        # A 200 can still carry an error object; surfacing it beats "no image data".
        upstream_error = answered.get("error")
        if upstream_error:
            if isinstance(upstream_error, dict):
                message = upstream_error.get("message")
            elif isinstance(upstream_error, str):
                message = upstream_error
            else:
                message = None
            # Same redaction as the HTTP-error path: a signed URL echoed in a 200 error
            # message must never reach the agent-visible preview.
            detail = (_text(_redact_urls(message))
                      if isinstance(message, str) else "") or "Pollinations reported an error"
            return fail(f"Pollinations could not edit the image: {detail}", "api_error")
        data = answered.get("data")
        if not isinstance(data, list) or not data:
            return fail("Pollinations returned no image data", "empty_response")
        item = data[0]
        if not isinstance(item, dict):
            return fail("Pollinations returned an unexpected image payload", "invalid_response")

        # Token-billed edit models return a usage block; record it against the session like the
        # bundled token-metered providers do. Record BEFORE local processing: the upstream edit
        # already completed and billed, so a decode/save failure must not lose the accounting.
        # A no-op outside a turn or without tokens; failures stay non-fatal.
        try:
            record_token_usage(answered.get("usage"), model=model, provider=PROVIDER_ID)
        except Exception as exc:
            logger.debug("Could not record Pollinations edit usage: %s", exc)

        # A URL response is cached rather than decoded; only raster payloads are sniffed.
        url = item.get("url")
        encoded = item.get("b64_json")
        remote = False
        produced: Optional[tuple[int, int]] = None
        if isinstance(encoded, str) and encoded:
            if encoded.startswith("data:"):  # tolerate a prefixed payload
                encoded = encoded.partition(",")[2]
            try:
                raw = base64.b64decode(encoded, validate=True)
            except Exception:
                return fail("Pollinations returned unreadable image data", "invalid_response")
            sniffed = _sniff_image_mime(raw)
            if sniffed is None or sniffed not in ACCEPTED_INPUT_MIME:
                return fail("Pollinations returned a non-image or non-raster response",
                            "invalid_response")
            if not _image_is_complete(raw, sniffed):
                return fail("Pollinations returned an incomplete image", "invalid_response")
            produced = _image_dimensions(raw, sniffed)
            try:
                image_ref = str(save_b64_image(
                    encoded, prefix="pollinations", extension=_extension_for(sniffed, "")))
            except Exception as exc:
                return fail(f"Could not save the edited image: {exc}", "io_error")
        elif isinstance(url, str) and url:
            from tools.url_safety import is_safe_url

            if not is_safe_url(url):
                # The response URL is upstream-chosen; never hand an unvetted target (e.g. a
                # link-local metadata address) to the agent as the result image.
                logger.debug("Edit response URL failed the SSRF safety check")
                return fail("Pollinations returned an unsafe image URL", "invalid_response")
            remote = True
            # on_fail keeps the full (possibly signed) URL out of the default-level log.
            image_ref = str(cache_url_best_effort(
                url, prefix="pollinations", label="Pollinations",
                on_fail=lambda exc: logger.debug(
                    "Pollinations edit image URL could not be cached (%s); using the bare URL.",
                    exc)))
        else:
            return fail("Pollinations returned neither image data nor a URL", "empty_response")

        extra = {"width": width, "height": height, "mode": "edit", "sources": len(sources)}
        if produced is not None:
            # The edit may be resized server-side; report the measured size, keeping the
            # request alongside for transparency (URL responses are not fetched: requested).
            extra["requested_width"], extra["requested_height"] = width, height
            extra["width"], extra["height"] = produced
        if remote:
            # The bytes were not fetched, so the file may not exist yet or may be unreachable.
            extra["remote"] = True
        if dropped:
            extra["dropped_references"] = dropped
        return success_response(
            image=image_ref, model=model, prompt=prompt, aspect_ratio=aspect,
            provider=PROVIDER_ID, modality="image", extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire the provider into the registry."""
    ctx.register_image_gen_provider(
        PollinationsImageGenProvider(getattr(ctx, "plugin_id", None) or PLUGIN_ID))
