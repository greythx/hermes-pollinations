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

Selection: ``model`` kwarg → ``POLLINATIONS_IMAGE_MODEL`` → ``image_gen.pollinations.model``
→ ``image_gen.model`` → the free classic model, else the first Quest-eligible one. Catalog ids
*and* aliases resolve; an unknown id passes through verbatim, so a model added upstream works
the day it appears.

Picker visibility follows the credentials, so the list never offers a model the current setup
cannot run: no key → the models the free classic API serves (usually one); a key → every
Quest-eligible model; a key plus the ``include_paid_models`` setting (Desktop → Settings →
Plugins) → the paid-only catalog too. The user's saved model is always kept visible, so
neither change makes it vanish.

Picker labels carry the access requirement, watermark and edit support, none of which the
catalog publishes: the price column reads ``Free · watermark · reduced size · no key needed``
while keyless,
and once a key is set that same model becomes ``<amount> pollen · no watermark · needs key`` —
or states the billing unit when the catalog prices per token or per megapixel rather than per
image. Edit-capable models add ``· edits``.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional
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
    error_factory, load_image_gen_config)
from utils import atomic_json_write, read_json_or_empty

logger = logging.getLogger(__name__)

CATALOG_URL = "https://gen.pollinations.ai/image/models"
CLASSIC_MODELS_URL = "https://image.pollinations.ai/models"
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
# Fallback registry id, used when register() is handed a context without ``plugin_id`` (the
# admission validator's probe stub). The live value comes from ``ctx.plugin_id``.
PLUGIN_ID = "image_gen/pollinations"
INCLUDE_PAID_KEY = "include_paid_models"
CATALOG_TTL = 3600
HTTP_TIMEOUT = 120.0

# Source images are inlined into the edits payload as data URIs, so they are capped before the
# base64 blow-up and restricted to formats every image model accepts.
MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
# Hermes' resolution creative control can only express these two values.
RESOLUTION_VOCAB = frozenset({"1k", "2k"})
# The reference cap comes from the untrusted catalog, so it is ceilinged here; the bundled edit
# plugin uses the same bound.
MAX_REFERENCE_IMAGES = 16
# Every response body is bounded before it is buffered, decoded or written to the cache.
MAX_RESPONSE_BYTES = 25 * 1024 * 1024
# Total inlined reference images for one edit. Each is base64 in a JSON body, so the cap keeps a
# 16-reference request inside a size the enter API will accept rather than failing server-side.
MAX_EDIT_PAYLOAD_BYTES = 40 * 1024 * 1024
# How long a failed free-model lookup is remembered, so a down endpoint is not retried per call.
NEGATIVE_TTL = 60.0
# Pollinations accepts a 32-bit signed seed.
SEED_MAX = 2**31 - 1
# Picker column widths: strengths is truncated to keep the row readable.
STRENGTHS_LIMIT = 120
# How many edit-capable models an error message names.
EDIT_HINT_MODELS = 3
# Cache schema marker: a cache written by a different plugin version is ignored rather than
# trusted, so a shape change upstream cannot silently mislabel rows.
CACHE_SCHEMA = 1

# Semantic aspect -> pixel size; every Pollinations image model accepts these. The free
# classic route keeps the aspect ratio within its own smaller area cap.
SIZES: Dict[str, tuple[int, int]] = {
    "landscape": (1280, 768),
    "square": (1024, 1024),
    "portrait": (768, 1280),
}

# Picker/spec fields, all live from the catalog. Fields Pollinations leaves unpublished for
# most models (resolutions, max_reference_images, capabilities, brand_url, pricing_dimensions)
# are deliberately absent from the row — a field that is usually blank is noise in the picker.
# They are still read where they change behaviour: pricing_units drives the price label,
# required_safety the strengths line, max_reference_images the edit cap.
MODEL_FIELDS = (
    "display", "speed", "strengths", "price", "publisher", "community", "description",
    "aliases", "price_pollen", "rpm", "modalities", "endpoints", "health", "added",
    "flat_rate", "free_classic",
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
            value = str(get_secret(name) or "").strip()
        except UnscopedSecretError:
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


def _unwrap_list(payload: Any, keys: tuple) -> Optional[List[Any]]:
    """The first list a JSON envelope carries, preferring the named keys.

    Third-party envelopes rename their fields, and a rename must not read as an empty response.
    One level of nesting is followed, so ``{"data": {"models": [...]}}`` resolves too.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                inner = _unwrap_list(value, keys)
                if inner is not None:
                    return inner
        for value in payload.values():
            if isinstance(value, list):
                return value
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
                 retry_after: Optional[int] = None) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.url = url
        self.status = status
        self.body = body
        self.retry_after = retry_after


def _retry_after(response: Any) -> Optional[int]:
    """The ``Retry-After`` header in seconds, as a delay or an HTTP-date."""
    raw = (response.headers.get("retry-after") or "").strip()
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        delta = parsedate_to_datetime(raw) - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds()))
    except Exception:
        return None


def _body_preview(response: Any) -> str:
    """A short, control-character-free preview of an error body, safe to show the agent.

    Raw upstream text is never echoed: a JSON error body is reduced to its ``error.message``
    and everything else has its control characters stripped.
    """
    raw = _bounded_read(response, "error response", limit=64 * 1024)[:500].decode("utf-8", "replace")
    payload: Any = None
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            raw = err["message"]
        elif isinstance(err, str):
            raw = err
    return "".join(ch for ch in raw if ch.isprintable() or ch in " \t").strip()[:200]


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
        # The upstream body is already reduced to a short, control-character-free preview, so it is
        # safe to show. A 402 on this API is not always an empty balance and a 402/429 is not always
        # a rate limit — guessing between the two sent a user with an invalid model to check their
        # Pollen balance. The status is named, then the reason Pollinations gave.
        detail = f": {exc.body}" if exc.body else ""
        if exc.status == 401:
            if mode == "free":
                return fail(
                    "Pollinations rejected the anonymous free request (401) — the free classic "
                    "route may now require a key. Add POLLINATIONS_API_KEY, or pick a model the "
                    "free API serves.", "auth_required")
            return fail(
                f"Pollinations rejected the API key (401){detail}. Check POLLINATIONS_API_KEY.",
                "auth_required")
        if exc.status in (402, 429):
            if mode == "free":
                return fail(
                    f"Pollinations returned HTTP {exc.status}{wait}{detail} — the free classic "
                    "route is throttled per IP. Wait, or add POLLINATIONS_API_KEY to skip the "
                    "throttle.", "rate_limited")
            return fail(
                f"Pollinations returned HTTP {exc.status}{wait}{detail} — usually the Quest "
                "Pollen balance or the rate limit. Wait, or check the account at "
                "https://enter.pollinations.ai.", "rate_limited")
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
                raise _HTTPError(url, response.status_code, _body_preview(response),
                                 _retry_after(response))
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
            if response.status_code >= 400:
                raise _HTTPError(url, response.status_code, _body_preview(response),
                                 _retry_after(response))
            return json.loads(_bounded_read(response, url))


def _http_get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(_http_get_bytes(url, timeout=timeout))


_CATALOG_CACHE: Dict[str, Any] = {"negative_until": 0.0, "last_success": 0.0}
_CATALOG_LOCK = threading.Lock()


def _catalog() -> List[Dict[str, Any]]:
    """Live catalog: fresh fetch, else a fresh-enough cache, else the stale cache.

    A failed fetch is remembered briefly, so a down endpoint costs one timeout per window
    rather than one per ``list_models()`` / ``capabilities()`` / ``generate()`` call.
    """
    path = _cache_path()
    cached: Optional[List[Dict[str, Any]]] = None
    try:
        raw = read_json_or_empty(path)
        # A cache written by a different plugin version is ignored rather than trusted.
        models = ([entry for entry in _as_list(raw.get("models")) if isinstance(entry, dict)]
                  if raw.get("version", CACHE_SCHEMA) == CACHE_SCHEMA else [])
        if models:
            cached = models
            if time.time() - float(raw.get("fetched") or 0) < CATALOG_TTL:
                return models
    except Exception:
        pass
    with _CATALOG_LOCK:
        if time.time() < float(_CATALOG_CACHE.get("negative_until") or 0.0):
            return cached if cached is not None else []
    # Only a fetch that began after the last successful one may record an outcome; otherwise a
    # slow failure would hide a cache another thread has already refreshed.
    started = time.time()
    try:
        models = _unwrap_list(_http_get_json(CATALOG_URL),
                              ("models", "data", "items", "results")) or []
        # Drop non-object entries at the source, so every consumer can assume a dict rather than
        # re-checking each element.
        models = [entry for entry in models if isinstance(entry, dict)]
        if not models:
            # An empty answer is not a catalog. Treating it as authoritative would overwrite a
            # good cache and empty the picker, so it is handled as a failed fetch instead.
            raise ValueError("Pollinations catalog returned no models")
        try:
            # Atomic replace: a concurrent reader must never see a half-written catalog.
            atomic_json_write(path, {"version": CACHE_SCHEMA, "fetched": time.time(),
                                     "models": models})
        except Exception as exc:
            logger.debug("Could not cache Pollinations catalog: %s", exc)
        with _CATALOG_LOCK:
            _CATALOG_CACHE["negative_until"] = 0.0
            _CATALOG_CACHE["last_success"] = time.time()
        return models
    except Exception as exc:
        logger.debug("Pollinations catalog fetch failed: %s", exc)
        # Remember the failure briefly, whether or not a stale cache exists: the cache still
        # wins, but a down endpoint must cost one timeout per window rather than one per call.
        with _CATALOG_LOCK:
            if started >= float(_CATALOG_CACHE.get("last_success") or 0.0):
                _CATALOG_CACHE["negative_until"] = time.time() + NEGATIVE_TTL
        # No hardcoded stand-ins: an unreachable catalog means no models to offer.
        return cached if cached is not None else []


_CLASSIC_CACHE: Dict[str, Any] = {
    "at": 0.0, "ids": [], "negative_until": 0.0, "last_success": 0.0}
_CLASSIC_LOCK = threading.Lock()


def _classic_cache_path() -> Path:
    return _hermes_home() / "cache" / "pollinations_classic_models.json"


def _load_classic_cache() -> Optional[List[str]]:
    """Last free-classic model list we managed to fetch, whatever its age."""
    try:
        raw = read_json_or_empty(_classic_cache_path())
        ids = ([item.strip() for item in _as_list(raw.get("ids"))
                if isinstance(item, str) and item.strip()]
               if raw.get("version", CACHE_SCHEMA) == CACHE_SCHEMA else [])
    except Exception as exc:
        # ``read_json_or_empty`` only guards OSError/ValueError; a pathologically nested or
        # otherwise hostile cache file must degrade to "no cache", not escape to the caller.
        logger.debug("Could not read the cached free classic model list: %s", exc)
        return None
    return ids or None


def _store_classic_cache(ids: List[str]) -> None:
    try:
        atomic_json_write(_classic_cache_path(),
                          {"version": CACHE_SCHEMA, "fetched": time.time(), "ids": ids})
    except Exception as exc:
        logger.debug("Could not cache the free classic model list: %s", exc)


def _classic_models() -> List[str]:
    """Model ids the free classic API serves, TTL-cached in memory and on disk.

    No hardcoded guess: when the endpoint is unreachable and nothing is cached the list is
    empty, so the picker offers no model rather than one that may no longer be free. The lock
    covers only the cache read/update; the fetch happens outside it so one slow request never
    blocks another caller. A failed lookup is remembered briefly, so a down endpoint costs one
    timeout per window rather than one per call.
    """
    with _CLASSIC_LOCK:
        if time.time() - float(_CLASSIC_CACHE["at"]) < CATALOG_TTL:
            return list(_CLASSIC_CACHE["ids"])
        if time.time() < float(_CLASSIC_CACHE.get("negative_until") or 0.0):
            # Another process may have written a good cache while this one was failing.
            return list(_CLASSIC_CACHE["ids"]) or _load_classic_cache() or []
    cached = _load_classic_cache()
    # Only a fetch that began after the last successful one may record an outcome; otherwise a
    # slow failure would hide (or a slow success would overwrite) what a newer fetch produced.
    started = time.time()
    fetched_ids: Optional[List[str]] = None
    try:
        fetched = _unwrap_list(_http_get_json(CLASSIC_MODELS_URL),
                               ("models", "data", "ids", "items", "results"))
        if fetched is not None:
            # Only strings can become URL path segments; anything else is discarded.
            fetched_ids = [item.strip() for item in fetched
                           if isinstance(item, str) and item.strip()]
    except Exception as exc:
        logger.debug("Classic model list fetch failed: %s", exc)
    if fetched_ids is not None:
        # A successful fetch is authoritative, empty or not: the free route may have been
        # withdrawn, and serving a stale id would send the user to a model that now errors.
        with _CLASSIC_LOCK:
            if started >= float(_CLASSIC_CACHE.get("last_success") or 0.0):
                _store_classic_cache(fetched_ids)
                _CLASSIC_CACHE["ids"] = list(fetched_ids)
                _CLASSIC_CACHE["at"] = time.time()
                _CLASSIC_CACHE["negative_until"] = 0.0
                _CLASSIC_CACHE["last_success"] = time.time()
        return list(fetched_ids)
    if cached:
        return list(cached)
    with _CLASSIC_LOCK:
        if started >= float(_CLASSIC_CACHE.get("last_success") or 0.0):
            _CLASSIC_CACHE["negative_until"] = time.time() + NEGATIVE_TTL
    return []


def _classic_set() -> set:
    """The free classic ids lowercased, for case-insensitive matching.

    These ids are matched against catalog keys, so a non-string would raise ``TypeError`` on the
    set membership test rather than simply not matching. Lowercasing here means a catalog or
    config that differs from the endpoint only in case still resolves; ``_classic_id_for``
    restores the endpoint's own spelling for the URL.
    """
    return {key.lower() for key in _classic_models() if isinstance(key, str)}


def _classic_spelling() -> Dict[str, str]:
    """Lowercase free classic id -> the endpoint's own spelling, for URL paths."""
    return {key.lower(): key for key in _classic_models() if isinstance(key, str)}


def _free_route_unavailable() -> bool:
    """True when the last free-model lookup failed and nothing is cached.

    Distinguishes "this model is not free" from "we could not tell", so a keyless user is not
    told their model is unavailable when the endpoint is simply down.
    """
    with _CLASSIC_LOCK:
        return time.time() < float(_CLASSIC_CACHE.get("negative_until") or 0.0)


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
        # A surrogate or control character raises out of urllib's quote() when the id becomes a
        # URL path segment, and an absurd length would build an absurd URL, so both are dropped.
        if (isinstance(value, str) and value.strip() and value.isprintable()
                and len(value) <= TEXT_LIMIT):
            return value.strip()
    return None


# Catalog text fields are display metadata. The strengths column shows 120 characters and no
# other consumer reads more than a few hundred, so a field longer than this is truncated before
# the per-character scan — one absurd catalog entry cannot stall the picker.
TEXT_LIMIT = 4096


def _is_number(value: Any) -> bool:
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


def _text(value: Any) -> str:
    """A catalog string field as printable text; anything else reads as empty.

    Control characters are dropped so a hostile catalog entry cannot smuggle line breaks into
    the picker or an agent-visible error message.
    """
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value[:TEXT_LIMIT] if ch.isprintable() or ch in " \t").strip()


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
    return any("image" in str(endpoint).lower()
               for endpoint in _as_list(model.get("supported_endpoints")))


def _classic_keys(model: Dict[str, Any]) -> List[str]:
    """The catalog id and aliases of a model as strings; anything else is discarded.

    These keys become set members and URL path segments, so a non-string ``name`` or alias is
    dropped rather than coerced.
    """
    if not isinstance(model, dict):
        return []
    return [key for key in [_name_of(model), *_as_list(model.get("aliases"))]
            if isinstance(key, str)]


def _is_free_classic(model: Dict[str, Any], classic: set) -> bool:
    """True when the free classic API serves this catalog model (no key, no pollen).

    ``classic`` is the lowercased set from ``_classic_set()``.
    """
    return any(key.lower() in classic for key in _classic_keys(model))


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
        if candidate.lower().endswith("/edits"):
            return candidate
    return None


def _supports_edits(entry: Optional[Dict[str, Any]]) -> bool:
    """True when the catalog advertises an edits endpoint for this model."""
    return _edit_path(entry) is not None


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
    return ENTER_BASE_URL + (clean if clean.startswith("/") else "/" + clean)


def _reference_cap(entry: Optional[Dict[str, Any]]) -> int:
    """The catalog's published reference-image cap, ceilinged; the ceiling when it publishes none.

    The edit models that publish a cap say 10-16, and the ones that publish nothing still accept
    several, so defaulting to 1 would silently discard references the model could have used.
    """
    raw = (entry or {}).get("max_reference_images")
    if raw is None or raw == "" or isinstance(raw, bool):
        return MAX_REFERENCE_IMAGES
    try:
        return max(1, min(int(raw), MAX_REFERENCE_IMAGES))
    except Exception:
        return MAX_REFERENCE_IMAGES


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
    wanted = model.strip().lower()
    # A model the classic API serves by its own id needs no catalog entry, so a stale catalog
    # cannot strand a keyless user on a model that is in fact free.
    if wanted in spelling:
        return spelling[wanted]
    entry = _catalog_entry(catalog, model)
    for key in _classic_keys(entry) if entry else []:
        if key.lower() in spelling:
            return spelling[key.lower()]
    return None


def _price_value(model: Dict[str, Any]) -> float:
    """Numeric pollen price for ordering; unpublished prices sort as zero (cheapest)."""
    raw = _as_dict(model.get("pricing")).get("completionImageTokens")
    if not _is_number(raw):
        return 0.0
    amount = float(raw)
    return amount if amount >= 0 else 0.0


def _format_price(value: Any) -> str:
    """Plain-decimal pollen amount; never scientific notation."""
    # ``bool`` is a subclass of ``int``, so it is excluded explicitly: a catalog that publishes
    # ``true`` as a price is not charging one pollen.
    if isinstance(value, bool) or not _is_number(value):
        return "No published price"
    amount = float(value)
    if amount < 0:
        return "No published price"
    if amount == 0:
        return "Free"
    formatted = f"{amount:.12f}".rstrip("0").rstrip(".")
    # An absurd amount would otherwise render as a hundreds-of-digits label.
    return (formatted[:24] + "…" if len(formatted) > 24 else formatted) + " pollen"


def _advertises_free(model: Dict[str, Any]) -> bool:
    """True when the catalog's own price or naming calls the model free.

    A published price of zero is the stronger signal; the name is only a fallback, because a
    title containing "free" can be marketing copy for a model that is in fact billed.
    """
    raw = _as_dict(model.get("pricing")).get("completionImageTokens")
    try:
        if float(raw) == 0:
            return True
    except (TypeError, ValueError, OverflowError):
        pass
    text = f"{model.get('title') or ''} {model.get('name') or ''}"
    return re.search(r"\bfree\b", text, re.IGNORECASE) is not None


def _pricing_unit(model: Dict[str, Any]) -> str:
    """The unit the catalog bills ``completionImageTokens`` in, lowercased; '' for per image."""
    units = _as_dict(_as_dict(model.get("pricing_units")).get("completionImageTokens"))
    unit = units.get("unit")
    return unit.strip().lower() if isinstance(unit, str) else ""


def _price_label(model: Dict[str, Any], free_classic: bool, has_key: bool) -> str:
    """Access requirement, watermark, edit support and cost, every part read from the live catalog.

    The catalog publishes no "needs a key", watermark or edit-support label, so all three are
    derived. A model the free classic API serves needs no key, but that route is the watermarked
    one, so once a key is set the same model is billed through the enter API instead and comes
    back clean. ``supported_endpoints`` decides edit support, ``flat_rate`` whether the published
    price is per image or per token.
    """
    paid_only = _as_bool(model.get("paid_only"))
    raw = _as_dict(model.get("pricing")).get("completionImageTokens")
    if free_classic and not has_key:
        if paid_only:
            return "Free on the classic route · watermark · reduced size · no key needed"
        return "Free · watermark · reduced size · no key needed"
    if not _is_number(raw):
        # A title that says "free" describes the free classic route; once a key is set this model
        # is billed through the enter API, so the label must not imply it is free.
        cost = ("No published price" if has_key
                else ("Free pollen" if _advertises_free(model) else "No published price"))
    else:
        unit = _pricing_unit(model)
        if unit and unit not in ("image", "images", "generation", "token", "tokens"):
            # e.g. "megapixel": the published figure is per unit, not per image.
            cost = f"charged per {unit}"
        elif not _as_bool(model.get("flat_rate")) or unit in ("token", "tokens"):
            cost = "charged per token"
        else:
            cost = _format_price(raw)
    # A model priced per variant (a 2K tier, an edit operation) can cost more than the base
    # figure the picker shows, so mark that the figure is a floor.
    if _as_list(model.get("pricing_variants")) and cost.endswith("pollen"):
        cost = f"from {cost}"
    parts = [cost]
    if free_classic:
        parts.append("no watermark")
    if has_key and _supports_edits(model):
        parts.append("edits")
    parts.append("paid" if paid_only else "needs key")
    return " · ".join(part for part in parts if part)


def _iso_date(model: Dict[str, Any]) -> str:
    """The catalog's ``added_date`` as a day, whatever unit it is in.

    The field is documented as milliseconds, but a switch to seconds would otherwise render every
    model as 1970, so the magnitude decides.
    """
    raw = model.get("added_date")
    if not _is_number(raw):
        return ""
    seconds = float(raw)
    if seconds <= 0:
        return ""
    # A second-based epoch is 10 digits; a millisecond one is 13. Anything older than the year
    # 2286 in milliseconds is almost certainly seconds.
    if abs(seconds) < 10**11:
        pass
    else:
        seconds /= 1000.0
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(seconds))
    except (OverflowError, OSError, ValueError):
        return ""


def _index(catalog: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every model name and alias to its canonical catalog id; the first entry wins."""
    out: Dict[str, str] = {}
    for model in catalog:
        name = _name_of(model) if isinstance(model, dict) else None
        if not name:
            continue
        for key in [name, *[clean for a in _as_list(model.get("aliases"))
                             if (clean := _text(a))]]:
            if key in out and out[key] != name:
                logger.debug("Pollinations catalog reuses %r for both %r and %r; keeping %r.",
                             key, out[key], name, out[key])
                continue
            out[key] = name
    return out


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
            # a plainly free one is the better default.
            if not _as_bool(model.get("paid_only")):
                return name
            if free_fallback is None:
                free_fallback = name
        if fallback is None and has_key and not _as_bool(model.get("paid_only")):
            fallback = name
    if free_fallback:
        return free_fallback
    if fallback:
        return fallback
    classic_ids = _classic_models()
    return classic_ids[0] if classic_ids else None


def _resolve_model(explicit: Optional[str], catalog: List[Dict[str, Any]]) -> Optional[str]:
    """First candidate naming a catalog model (id or alias), else the first non-empty one.

    An unknown id passes through verbatim, so a model added upstream works the day it appears —
    with one exception. Hermes injects the shared top-level ``image_gen.model`` as the explicit
    ``model`` kwarg on every tool call, so an unknown value there is usually another provider's
    leftover id rather than a new Pollinations model; that value falls back to the derived
    default instead of being sent to Pollinations.

    ``None`` when nothing is configured and the catalog offers no default to derive.
    """
    index = _index(catalog)
    config = load_image_gen_config()
    scoped = config.get(PROVIDER_ID)
    shared = config.get("model")
    has_key = _api_key() is not None
    classic = _classic_set()
    candidates = [
        explicit,
        os.environ.get(MODEL_ENV, "").strip(),
        scoped.get("model") if isinstance(scoped, dict) else None,
        shared,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() in index:
            return index[candidate.strip()]
    for candidate in candidates[:3]:
        value = candidate.strip() if isinstance(candidate, str) else ""
        if value and value != shared:
            # An unknown id passes through so a model added upstream works the day it appears.
            # Keyless, though, only the free classic route can run it, and an id the catalog
            # does not know there is far more likely a leftover than a brand-new free model.
            if has_key or value.lower() in classic:
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
    file a viewer cannot open.
    """
    if not raw:
        return False
    if sniffed == "image/png":
        return b"IEND" in raw[-16:]
    if sniffed == "image/jpeg":
        return b"\xff\xd9" in raw[-4:]
    if sniffed == "image/gif":
        return raw.rstrip().endswith(b";")
    if sniffed == "image/webp":
        if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
            return False
        declared = int.from_bytes(raw[4:8], "little")
        return len(raw) >= declared + 8
    return True


def _encode_input_image(raw: bytes) -> str:
    """Size- and MIME-check raw image bytes, then return a canonical ``data:`` URL."""
    if len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"source image exceeds the {MAX_INPUT_IMAGE_BYTES // (1024 * 1024)}MB cap")
    mime = _sniff_image_mime(raw)
    if mime not in ACCEPTED_INPUT_MIME:
        raise ValueError("source image is not a supported PNG, JPEG, GIF or WebP image")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _remote_image_to_data_url(value: str) -> str:
    """Fetch a remote source image through the SSRF guard; the edits endpoint takes data URIs."""
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
            if declared.isdigit() and int(declared) > MAX_INPUT_IMAGE_BYTES:
                raise ValueError(f"source image exceeds {MAX_INPUT_IMAGE_BYTES} bytes")
            raw = _bounded_read(response, value)
    return _encode_input_image(raw)


def _local_image_to_data_url(value: str) -> str:
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
    if size > MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"source image exceeds {MAX_INPUT_IMAGE_BYTES} bytes")
    # Read with the same ceiling: a file can change between stat() and read(), and a huge file
    # must not be buffered just because it was small a moment ago.
    with path.open("rb") as handle:
        raw = handle.read(MAX_INPUT_IMAGE_BYTES + 1)
    if len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"source image exceeds {MAX_INPUT_IMAGE_BYTES} bytes")
    return _encode_input_image(raw)


def _to_data_url(value: str) -> str:
    """Inline one source image as a base64 data URI, whatever form Hermes handed us.

    Hermes passes source images as public URLs, base64 data URIs or absolute local paths; the
    edits endpoint accepts a URL or an array of ``{image_url}`` objects, and a data URI keeps the
    request to a single JSON call with no extra upload step.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("blank source image")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        return _remote_image_to_data_url(candidate)
    if lowered.startswith("data:"):
        header, _, data = candidate.partition(",")
        if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise ValueError("source image data URL must be a base64 image")
        if (len(data) * 3) // 4 > MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"source image exceeds {MAX_INPUT_IMAGE_BYTES} bytes")
        return _encode_input_image(base64.b64decode(data, validate=True))
    return _local_image_to_data_url(candidate)


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
        if _as_bool(model.get("paid_only")) and not include_paid:
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


def _specs(model: Dict[str, Any], free_classic: bool, has_key: bool) -> Dict[str, Any]:
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
    keyless_free = free_classic and not has_key
    speed = ""
    if not keyless_free and _is_number(rpm) and rpm > 0:
        speed = f"≤{rpm} req/min"
    return {
        "id": _name_of(model),
        "display": _text(model.get("title")) or _name_of(model) or "",
        "publisher": _text(model.get("publisher")),
        "community": _as_bool(model.get("community")),
        "description": description,
        "strengths": strengths[:STRENGTHS_LIMIT],
        "aliases": [clean for a in _as_list(model.get("aliases")) if (clean := _text(a))],
        "price": _price_label(model, free_classic, has_key),
        "price_pollen": _price_value(model),
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
    classic_id = _classic_id_for(model, catalog) if _api_key() is None else None
    if classic_id is not None:
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

    key = _api_key()
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
        """Picker rows for the current credential state, cheapest first.

        No key → the free classic model only. Key → every Quest-eligible model. Key plus the
        ``include_paid_models`` setting → the paid-only catalog too. The user's saved model is
        always kept visible, so removing the key or turning the setting off never makes a
        configured model silently disappear.
        """
        catalog = _catalog()
        classic = _classic_set()
        has_key = _api_key() is not None
        include_paid = has_key and _include_paid_models(self._plugin_id)

        rows: Dict[str, Dict[str, Any]] = {}
        for model in catalog:
            if not _is_image_model(model):
                continue
            name = _name_of(model)
            if not name:
                continue
            if name in rows:
                logger.debug("Pollinations catalog lists %r more than once; keeping the first.", name)
                continue
            free_classic = _is_free_classic(model, classic)
            if not free_classic:
                if not has_key:
                    continue  # state 1: the free classic API is the only keyless route
                if _as_bool(model.get("paid_only")) and not include_paid:
                    continue  # state 2: Quest-eligible only
            rows[name] = _specs(model, free_classic, has_key)

        if not has_key:
            # The classic endpoint is the source of truth for the keyless route: Pollinations can
            # drop a free model from the catalog while still serving it, and a stale catalog must
            # not leave a keyless user with nothing to pick.
            for key in [item.strip() for item in _classic_models()
                        if isinstance(item, str) and item.strip()]:
                if key not in rows:
                    # The classic endpoint publishes no metadata, so the row is built from the id
                    # alone; ``input_modalities`` states what the route accepts.
                    rows[key] = _specs({"name": key, "input_modalities": ["text"]},
                                      True, has_key)

        configured = _configured_model(catalog)
        if configured and configured not in rows:
            saved = next((m for m in catalog if _name_of(m) == configured), None)
            if saved is not None:
                rows[configured] = _specs(saved, _is_free_classic(saved, classic), has_key)

        ordered = sorted(rows.values(), key=lambda row: row["price_pollen"])
        return catalog_rows({row["id"]: row for row in ordered}, fields=MODEL_FIELDS)

    def default_model(self) -> Optional[str]:
        """The free classic model, else the first Quest-eligible one — both read live."""
        return _derived_default(_catalog(), _classic_set(), _api_key() is not None)

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
        if not _is_image_model(entry):
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

        catalog = _catalog()
        model = _resolve_model(kwargs.get("model"), catalog)
        if not model:
            return error_response(
                error="No Pollinations model is available. The live model catalog could not be "
                      "reached and no model is configured; check your connection and try again.",
                error_type="no_model_available", provider=PROVIDER_ID, prompt=prompt,
                aspect_ratio=aspect)
        width, height = SIZES[aspect]
        entry = _catalog_entry(catalog, model)
        fail = error_factory(PROVIDER_ID, aspect, model=model, prompt=prompt)

        # A source image means an edit, which only the enter API can do; the free classic route
        # is text-to-image only, so there is nothing to fall back to.
        sources = collect_source_images(image_url, reference_image_urls)
        if sources:
            return self._edit(prompt, aspect, model, entry, sources, width, height,
                              kwargs.get("resolution"), fail)

        seed = random.randint(0, SEED_MAX)
        route = _text_to_image_route(
            prompt, model, entry, catalog, width, height, seed,
            str(kwargs.get("resolution") or "").strip().lower())
        if route.error:
            return fail(route.error, route.error_type or "auth_required")
        url, headers, extra = route.url or "", route.headers or {}, route.extra

        try:
            data = _http_get_bytes(url, headers=headers)
        except Exception as exc:
            return _request_failed(exc, action="image generation",
                                   mode=str(extra.get("mode") or "quest"), fail=fail)

        if not data:
            return fail("Pollinations returned an empty body", "empty_response")
        # A 200 can still carry an error body; only save it when it really is an image.
        sniffed = _sniff_image_mime(data)
        if sniffed is None:
            return fail("Pollinations returned a non-image response", "invalid_response")
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
        if not _supports_edits(entry):
            options = _visible_edit_models(_catalog(), include_paid=_include_paid_models(self._plugin_id))
            hint = f" Edit-capable models: {', '.join(options)}." if options else ""
            return fail(f"Model '{model}' cannot edit images.{hint}", "unsupported")

        # Hermes caps the reference count in its schema, but a direct call could over-send.
        cap = _reference_cap(entry)
        dropped = max(0, len(sources) - cap)
        sources = sources[:cap]

        inlined: List[Dict[str, str]] = []
        total = 0
        try:
            for source in sources:
                data_url = _to_data_url(source)
                total += len(data_url)
                if total > MAX_EDIT_PAYLOAD_BYTES:
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

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "model": model,
            "size": f"{width}x{height}",
            "seed": random.randint(0, SEED_MAX),
            "response_format": "b64_json",
        }
        # Hermes' resolution creative control is only meaningful for models that price by it.
        requested = str(resolution or "").strip().lower()
        if requested and requested in _published_resolutions(entry):
            payload["resolution"] = requested

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
            message = upstream_error.get("message") if isinstance(upstream_error, dict) else None
            detail = _text(message) or "Pollinations reported an error"
            return fail(f"Pollinations could not edit the image: {detail}", "api_error")
        data = answered.get("data")
        if not isinstance(data, list) or not data:
            return fail("Pollinations returned no image data", "empty_response")
        item = data[0]
        if not isinstance(item, dict):
            return fail("Pollinations returned an unexpected image payload", "invalid_response")

        # A URL response is cached rather than decoded; only raster payloads are sniffed.
        url = item.get("url")
        encoded = item.get("b64_json")
        remote = False
        if isinstance(encoded, str) and encoded:
            if encoded.startswith("data:"):  # tolerate a prefixed payload
                encoded = encoded.partition(",")[2]
            try:
                raw = base64.b64decode(encoded, validate=True)
            except Exception:
                return fail("Pollinations returned unreadable image data", "invalid_response")
            sniffed = _sniff_image_mime(raw)
            if sniffed is None:
                return fail("Pollinations returned a non-image response", "invalid_response")
            if not _image_is_complete(raw, sniffed):
                return fail("Pollinations returned an incomplete image", "invalid_response")
            try:
                image_ref = str(save_b64_image(
                    encoded, prefix="pollinations", extension=_extension_for(sniffed, "")))
            except Exception as exc:
                return fail(f"Could not save the edited image: {exc}", "io_error")
        elif isinstance(url, str) and url:
            remote = True
            image_ref = str(cache_url_best_effort(url, prefix="pollinations", label="Pollinations"))
        else:
            return fail("Pollinations returned neither image data nor a URL", "empty_response")

        extra = {"width": width, "height": height, "mode": "edit", "sources": len(sources)}
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
