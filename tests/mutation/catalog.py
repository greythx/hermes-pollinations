"""Mutation catalog for the v1.7.0 permanent mutation gate (PR #5).

Rebuilt against the SHIPPED v1.7.0 source (branch test-coverage-hardening,
``__init__.py`` 146,742 bytes). Structure is ported from the v1 harness at
``f8270ce``; every ANCHOR is new: ``old_string`` is sliced from the current
source, occurs exactly once, and ``new_string`` inverts a real v1.7.0 fix so
the mutant behaves like the pre-fix bug. ``validate()`` re-checks uniqueness,
compilation and non-no-op at call time. No model/API claims, no network.

Non-mutable fixes (documented, not mutated):
* V1 test-fixture placement -- a test-quality fix (a vacuous test was
  re-fixtured so the signed URL sits inside both the 64KiB read prefix and
  the 200-char preview with a positive ``[url]`` assert). It changes no
  plugin behavior, so no source transform represents it; its behavior is
  guarded by M2_redaction_off / M2_* and the preview tests.
"""

from typing import Any, Dict, List

MUTATIONS: List[Dict[str, Any]] = [
    {
        "name": "M1_alias_shadow",
        "description": "Canonical-id priority removed: a paid alias can shadow a canonical/free id.",
        "finding": "M1: canonical id must beat an alias claim in _resolve_model.",
        "old_string": "            canonical = next((_name_of(m) for m in catalog if _name_of(m) == wanted), None)\n            if canonical is not None:\n                return canonical\n",
        "new_string": "            pass  # M1_alias_shadow\n",
    },
    {
        "name": "F1_case_sensitive_catalog",
        "description": "Folded catalog index made case-sensitive: case variant of a paid id passes through ungated.",
        "finding": "F1: catalog name/alias/id matching is case-insensitive with canonical spelling restore.",
        "old_string": "        if name:\n            out.setdefault(_norm_id(name), name)\n",
        "new_string": "        if name:\n            out.setdefault(name, name)  # F1_case_sensitive_catalog\n",
    },
    {
        "name": "FUZZ_R3_paid_failopen",
        "description": "Non-scalar paid_only coerced to False: paid gate fails OPEN regardless of toggle.",
        "finding": "FUZZ-R3: unreadable paid_only must fail CLOSED across invocation/picker/capabilities.",
        "old_string": "    raw = (entry or {}).get(\"paid_only\")\n    if isinstance(raw, (list, dict)):\n        return None\n    return _as_bool(raw)\n",
        "new_string": "    raw = (entry or {}).get(\"paid_only\")\n    return _as_bool(raw)  # FUZZ_R3_paid_failopen: non-scalar fails OPEN\n",
    },
    {
        "name": "F2_caps_no_paid_gate",
        "description": "capabilities() paid/credential gate removed: edits/resolution advertised while toggle is off.",
        "finding": "F2: capabilities must apply the same paid gate as generate/list_models.",
        "old_string": "        if (entry is not None and (_paid_flag(entry) is None\n                or (_paid_flag(entry) is not False and not _include_paid_models(self._plugin_id)))):\n",
        "new_string": "        if False:  # F2_caps_no_paid_gate\n",
    },
    {
        "name": "F10_post_3xx_accepted",
        "description": "Edits POST status guard disabled: a 3xx with a JSON body parses as a completed edit.",
        "finding": "F10: every non-2xx (including 3xx) on the no-redirect POST must raise _HTTPError.",
        "old_string": "            if response.status_code >= 300:\n",
        "new_string": "            if False:  # F10_post_3xx_accepted\n",
    },
    {
        "name": "M2_redaction_off",
        "description": "URL redaction removed from the agent-visible error preview.",
        "finding": "M2: signed URLs must never reach the error preview (plain spelling).",
        "old_string": "    text = _URL_RE.sub(\"[url]\", text)\n",
        "new_string": "    text = text  # M2_redaction_off\n",
    },
    {
        "name": "M2_unicode_scheme",
        "description": "Escape decoding disabled: unicode-escaped scheme characters hide a signed URL.",
        "finding": "M2: bounded JSON-escape decode closes \\uXXXX scheme/colon evasion.",
        "old_string": "    for _ in range(6):\n        if text == prev:\n            break\n        prev = text\n        text = _ESCAPE_RE.sub(_unescape_match, text)\n",
        "new_string": "    for _ in range(0):  # M2_unicode_scheme: no escape decoding\n        if text == prev:\n            break\n        prev = text\n        text = _ESCAPE_RE.sub(_unescape_match, text)\n",
    },
    {
        "name": "M2_apostrophe_tail",
        "description": "Apostrophe excluded from the URL tail: an apostrophe splits a signed URL and leaks its tail.",
        "finding": "M2: apostrophes are legal URL characters in the redactor.",
        "old_string": "_URL_RE = re.compile(r\"(?:https?:)?//[^\\s\\\"\\\\;]*\", re.IGNORECASE)\n",
        "new_string": "_URL_RE = re.compile(r\"(?:https?:)?//[^\\s\\\"\\\\;']*\", re.IGNORECASE)  # M2_apostrophe_tail\n",
    },
    {
        "name": "M2_protocol_relative",
        "description": "Redactor requires an explicit scheme: protocol-relative // URLs are no longer redacted.",
        "finding": "M2: protocol-relative //host URLs must be redacted.",
        "old_string": "_URL_RE = re.compile(r\"(?:https?:)?//[^\\s\\\"\\\\;]*\", re.IGNORECASE)\n",
        "new_string": "_URL_RE = re.compile(r\"https?://[^\\s\\\"\\\\;]*\", re.IGNORECASE)  # M2_protocol_relative\n",
    },
    {
        "name": "M2_single_slash",
        "description": "Single-slash https:/ URL normalization disabled.",
        "finding": "M2/B1: browsers normalize https:/host to https://host; redact that shape too.",
        "old_string": "    text = re.sub(r\"(https?:)/(?!/)\", r\"\\1//\", text)\n",
        "new_string": "    text = text  # M2_single_slash\n",
    },
    {
        "name": "M2_control_strip",
        "description": "Control characters no longer stripped before matching: one inside a URL splits the redactor.",
        "finding": "M2: control chars stripped BEFORE matching so a URL cannot be split into unrecognizable halves.",
        "old_string": "    text = \"\".join(ch for ch in text if ch.isprintable() or ch in \" \\t\")\n",
        "new_string": "    text = text  # M2_control_strip\n",
    },
    {
        "name": "L2_adjustments_off",
        "description": "Per-image pricing adjustments dropped from every estimate and label.",
        "finding": "L2: convertible per-generation pollen fees must be folded into the per-image cost.",
        "old_string": "    return [a for a in _as_list(model.get(\"pricing_adjustments\"))\n            if isinstance(a, dict) and str(a.get(\"kind\") or \"image\").lower() != \"video\"]\n",
        "new_string": "    return []  # L2_adjustments_off\n",
    },
    {
        "name": "L3_from_marker_off",
        "description": "The \"from\" floor marker removed from variant-priced labels.",
        "finding": "L3/F8: a numeric variant floor must read \"from X\" so a higher base is not advertised.",
        "old_string": "            cost = f\"from {cost}\"\n",
        "new_string": "            cost = cost  # L3_from_marker_off\n",
    },
    {
        "name": "L4_text_token_rates",
        "description": "Text-token-only models priced as 0/unpublished in the sort estimate.",
        "finding": "L4: text-token-only models keep their completion rate as a per-token figure.",
        "old_string": "        rate = _as_number(pricing.get(\"completionTextTokens\"))\n        if rate is None:\n            rate = _as_number(pricing.get(\"promptTextTokens\"))\n        return rate if rate is not None and rate >= 0 else 0.0\n",
        "new_string": "        rate = 0.0  # L4_text_token_rates: text-token models read as unpublished\n",
    },
    {
        "name": "L4_text_token_class",
        "description": "Text-token-only models classed as unpublished instead of per-token.",
        "finding": "L4: text-token billing is per-token, not an unpublished price.",
        "old_string": "        if (_as_number(pricing.get(\"completionTextTokens\")) is not None\n                or _as_number(pricing.get(\"promptTextTokens\")) is not None):\n            return 1  # text-token billing is still per-token, not \"unpublished\"\n        return 2\n",
        "new_string": "        return 2  # L4_text_token_class: text-token models classed unpublished\n",
    },
    {
        "name": "L12_unwrap_fallback",
        "description": "Positional fallback for renamed envelopes removed: a renamed list reads as empty.",
        "finding": "L12: _unwrap_list must keep the shape-predicated positional fallback return.",
        "old_string": "        if (fallback is not None and candidates == 1\n                and (fallback_item_ok is None or fallback_item_ok(fallback))):\n            return fallback\n",
        "new_string": "        if False:  # L12_unwrap_fallback: positional fallback removed\n            return fallback\n",
    },
    {
        "name": "F5_future_at_catalog",
        "description": "Future catalog cache timestamp treated as permanently fresh.",
        "finding": "F5: a future/absurd disk \"fetched\" stamp must not read as fresh.",
        "old_string": "            # A future timestamp (clock skew or a hostile cache) must not read as permanently fresh.\n            if 0 <= age < CATALOG_TTL:\n                return models\n",
        "new_string": "            # A future timestamp (clock skew or a hostile cache) guard removed\n            if age < CATALOG_TTL:\n                return models\n",
    },
    {
        "name": "F5_future_at_classic",
        "description": "Future classic memory \"at\" timestamp treated as permanently fresh.",
        "finding": "F5: a backward clock step must not make the classic memory cache immortal.",
        "old_string": "        # A future \"at\" (backward clock step) must not read as permanently fresh.\n        if 0 <= age < CATALOG_TTL:\n            # A newer cross-process disk write wins over this process's memory.\n            _adopt_newer_classic_disk(state)\n",
        "new_string": "        # F5_future_at_classic: guard removed\n        if age < CATALOG_TTL:\n            # A newer cross-process disk write wins over this process\"s memory.\n            _adopt_newer_classic_disk(state)\n",
    },
    {
        "name": "F6_dup_rows",
        "description": "Case-variant dedupe removed: duplicate catalog ids produce duplicate picker rows.",
        "finding": "F6: keyless picker dedupes catalog/classic case variants (first wins).",
        "old_string": "            if name in rows or name.lower() in seen_lower:\n                logger.debug(\"Pollinations catalog lists %r more than once; keeping the first.\", name)\n                continue\n",
        "new_string": "            if name in rows:  # F6_dup_rows\n                logger.debug(\"Pollinations catalog lists %r more than once; keeping the first.\", name)\n                continue\n",
    },
    {
        "name": "F16_video_capabilities",
        "description": "capabilities() gives image capabilities to a video-only configured model.",
        "finding": "F16/R2-P1: a video id must not be routed through the image generator; capabilities agree.",
        "old_string": "        if entry is None or not _is_image_model(entry) or (_api_key() and _is_svg_product(entry)):\n",
        "new_string": "        if entry is None or (_api_key() and _is_svg_product(entry)):  # F16_video_capabilities\n",
    },
    {
        "name": "F16_video_generate",
        "description": "generate() no longer refuses a positively-video catalog entry.",
        "finding": "F16: generate() must refuse video models before the paid gate.",
        "old_string": "        if entry is not None and _is_video_model(entry):\n            return fail(\n",
        "new_string": "        if False:  # F16_video_generate: refusal removed\n            return fail(\n",
    },
    {
        "name": "R2P1_video_retention",
        "description": "Picker main loop no longer filters non-image entries out of the rows.",
        "finding": "R2-P1: video/non-image entries must be retained-free in list_models.",
        "old_string": "        for model in catalog:\n            if not _is_image_model(model):\n                continue\n            name = _name_of(model)\n",
        "new_string": "        for model in catalog:\n            # R2P1_video_retention: image-model filter removed\n            name = _name_of(model)\n",
    },
    {
        "name": "R2P1_video_retention_saved",
        "description": "Retention keeps a saved video model visible in the picker.",
        "finding": "R2-P1: saved-model retention must never offer a video entry.",
        "old_string": "                if not _is_video_model(saved):\n",
        "new_string": "                if True:  # R2P1_video_retention_saved\n",
    },
    {
        "name": "R2P4_free_name_lie",
        "description": "Name-based free claim uses loose substring: \"freebie\" reads as free.",
        "finding": "R2-P4: keyless name-free fallback must use a word boundary, not a substring.",
        "old_string": "    return re.search(r\"\\bfree\\b\", text, re.IGNORECASE) is not None\n",
        "new_string": "    return \"free\" in text.lower()  # R2P4_free_name_lie\n",
    },
    {
        "name": "R2P5_newline_fusion",
        "description": "Text sanitizer deletes newlines instead of spacing them, fusing adjacent words.",
        "finding": "R2-P5: newline/whitespace must normalize to a space, not be deleted.",
        "old_string": "    cleaned = \"\".join(ch if ch not in \"\\n\\r\" else \" \"\n                      for ch in value[:TEXT_LIMIT] if ch.isprintable() or ch in \" \\t\\n\\r\")\n    return \" \".join(cleaned.split())\n",
        "new_string": "    cleaned = \"\".join(ch for ch in value[:TEXT_LIMIT] if ch.isprintable() or ch in \" \\t\")  # R2P5_newline_fusion\n    return \" \".join(cleaned.split())\n",
    },
    {
        "name": "R2E1_datauri_exact_cap",
        "description": "Data-URI decoded length overestimates by padding: a payload exactly at the cap is rejected.",
        "finding": "R2-E1: exact decoded length must accept a payload exactly at the budget cap.",
        "old_string": "        if (len(data) * 3) // 4 - data.count(\"=\") > limit:\n",
        "new_string": "        if (len(data) * 3) // 4 > limit:  # R2E1_datauri_exact_cap\n",
    },
    {
        "name": "L13_retry_clamp",
        "description": "Retry-After decimal delay no longer clamped to >=0 / <=86400.",
        "finding": "L13: negative/absurd Retry-After must be clamped for display.",
        "old_string": "        return min(RETRY_AFTER_MAX, max(0, int(float(raw))))\n",
        "new_string": "        return int(float(raw))  # L13_retry_clamp\n",
    },
    {
        "name": "L13_retry_date_clamp",
        "description": "Retry-After HTTP-date delay no longer clamped.",
        "finding": "L13: an HTTP-date far in the past/future must be clamped too.",
        "old_string": "        return min(RETRY_AFTER_MAX, max(0, int(delta.total_seconds())))\n",
        "new_string": "        return int(delta.total_seconds())  # L13_retry_date_clamp\n",
    },
    {
        "name": "MP1_cross_process_guard",
        "description": "Cross-process write guard removed: a newer cache write can be clobbered mid-fetch.",
        "finding": "MP-1: _guarded_cache_write must skip when a newer disk write landed mid-fetch.",
        "old_string": "            if disk_fetched > started:\n                return False  # a newer write landed mid-fetch\n",
        "new_string": "            if False:  # MP1_cross_process_guard: newer-write skip removed\n                return False\n",
    },
    {
        "name": "MP2_classic_disk_first",
        "description": "Classic memory TTL hit no longer checks disk first: a newer cross-process write loses.",
        "finding": "MP-2: classic memory TTL hit must adopt a newer cross-process disk write.",
        "old_string": "        if 0 <= age < CATALOG_TTL:\n            # A newer cross-process disk write wins over this process's memory.\n            _adopt_newer_classic_disk(state)\n            return list(state[\"ids\"])\n",
        "new_string": "        if 0 <= age < CATALOG_TTL:\n            # MP2_classic_disk_first: disk adoption removed\n            return list(state[\"ids\"])\n",
    },
    {
        "name": "MP3_monotonic_catalog",
        "description": "Catalog failure window read with the wall clock instead of the monotonic clock.",
        "finding": "MP-3: negative_until/last_success use the monotonic clock (skew-proof).",
        "old_string": "        if time.monotonic() < float(_catalog_state().get(\"negative_until\") or 0.0):\n",
        "new_string": "        if time.time() < float(_catalog_state().get(\"negative_until\") or 0.0):  # MP3_monotonic_catalog\n",
    },
    {
        "name": "MP3_monotonic_classic",
        "description": "Classic failure window read with the wall clock instead of the monotonic clock.",
        "finding": "MP-3: classic negative_until uses the monotonic clock.",
        "old_string": "        if time.monotonic() < float(_classic_state().get(\"negative_until\") or 0.0):\n",
        "new_string": "        if time.time() < float(_classic_state().get(\"negative_until\") or 0.0):  # MP3_monotonic_classic\n",
    },
    {
        "name": "MP4_home_keyed_catalog",
        "description": "Catalog cache state no longer keyed by Hermes home (profiles share failure windows).",
        "finding": "MP-4: cache state is home-keyed so a routed profile cannot inherit another profile.",
        "old_string": "    return _CATALOG_STATES.setdefault(_home_key(), {\"negative_until\": 0.0, \"last_success\": 0.0})\n",
        "new_string": "    return _CATALOG_STATES.setdefault(\"shared\", {\"negative_until\": 0.0, \"last_success\": 0.0})  # MP4_home_keyed_catalog\n",
    },
    {
        "name": "MP4_home_keyed_classic",
        "description": "Classic cache state no longer keyed by Hermes home.",
        "finding": "MP-4: classic state is home-keyed for routed-profile isolation.",
        "old_string": "    return _CLASSIC_STATES.setdefault(\n        _home_key(), {\"at\": 0.0, \"ids\": [], \"negative_until\": 0.0, \"last_success\": 0.0})\n",
        "new_string": "    return _CLASSIC_STATES.setdefault(\n        \"shared\", {\"at\": 0.0, \"ids\": [], \"negative_until\": 0.0, \"last_success\": 0.0})  # MP4_home_keyed_classic\n",
    },
    {
        "name": "MP6_env_scope_bypass",
        "description": "Model override read from a raw os.environ key, bypassing profile scope.",
        "finding": "MP-6: MODEL_ENV must be read via get_secret (scoped, env fallback only).",
        "old_string": "        return str(get_secret(MODEL_ENV) or \"\").strip()\n",
        "new_string": "        return (os.environ.get(MODEL_ENV) or \"\").strip()  # MP6_env_scope_bypass\n",
    },
    {
        "name": "CG_api_key_non_str",
        "description": "Non-string secret coerced with str(): a bytes secret becomes a garbage bearer key.",
        "finding": "PR#5 coverage gap: non-str secrets must be ignored, not str()-coerced.",
        "old_string": "            value = secret.strip() if isinstance(secret, str) else \"\"\n",
        "new_string": "            value = str(secret).strip()  # CG_api_key_non_str\n",
    },
    {
        "name": "CG_include_paid_hostile",
        "description": "include_paid_models setting read as always-on regardless of config shape.",
        "finding": "PR#5 coverage gap: hostile config shapes degrade to False.",
        "old_string": "        return _as_bool((entry.get(\"settings\") or {}).get(INCLUDE_PAID_KEY))\n",
        "new_string": "        return True  # CG_include_paid_hostile\n",
    },
    {
        "name": "CG_capabilities_resolution",
        "description": "Resolution creative control advertised keyless (no authenticated route).",
        "finding": "PR#5 coverage gap: resolution control is offered only with a key.",
        "old_string": "            capabilities[\"creative_controls\"] = [\"resolution\"] if _api_key() else []\n",
        "new_string": "            capabilities[\"creative_controls\"] = [\"resolution\"]  # CG_capabilities_resolution\n",
    },
    {
        "name": "CG_edit_hint_bounds",
        "description": "Edit-hint model list no longer bounded by EDIT_HINT_MODELS.",
        "finding": "PR#5 coverage gap: edit hints are bounded.",
        "old_string": "        if len(names) >= limit:\n",
        "new_string": "        if False:  # CG_edit_hint_bounds\n",
    },
    {
        "name": "CG_route_modes",
        "description": "Free classic route condition disabled: keyless calls build an enter URL with no key.",
        "finding": "PR#5 coverage gap: keyless route modes (free/quest/paid) are locked.",
        "old_string": "    if classic_id is not None and (key is None or entry is None):\n",
        "new_string": "    if False:  # CG_route_modes: free classic route disabled\n",
    },
    {
        "name": "CG_webp_vp8x_dimensions",
        "description": "WebP VP8X canvas dimensions off by one (uses raw value instead of +1).",
        "finding": "PR#5 coverage gap: WebP VP8X/VP8L/VP8 variants measured exactly.",
        "old_string": "        return (int.from_bytes(raw[24:27], \"little\") + 1,\n",
        "new_string": "        return (int.from_bytes(raw[24:27], \"little\"),\n",
    },
    {
        "name": "CG_cache_write_fallback",
        "description": "No-fcntl cache-write fallback stops reporting success (and skips the write).",
        "finding": "PR#5 coverage gap: _guarded_cache_write degrades to a plain atomic write without fcntl.",
        "old_string": "    except ImportError:\n        _unlink_planted_symlink(path)\n        atomic_json_write(path, payload)\n        return True\n",
        "new_string": "    except ImportError:\n        _unlink_planted_symlink(path)\n        return False  # CG_cache_write_fallback\n",
    },
    {
        "name": 'R6_1_variant_floor_legacy_only',
        "description": 'Variant floor reads only the legacy top-level rate: live nested pricing_variants are ignored.',
        "finding": 'R6-1: live catalog nests the variant rate under variant.pricing.completionImageTokens.',
        "old_string": '        rate = _variant_rate(variant)\n',
        "new_string": '        rate = _as_number(_as_dict(variant).get("completionImageTokens"))  # R6_1_variant_floor_legacy_only\n',
    },
    {
        "name": 'R6_01_classic_paid_passthrough',
        "description": 'Paid canonical classic id regains its endpoint spelling: the paid gate cannot see it.',
        "finding": 'R6-01: a paid-only canonical classic-served id must keep canonical spelling so the gate fires.',
        "old_string": '                if (folded_target is not None and _norm_id(folded_target) == key\n                        and entry is not None and _is_image_model(entry)\n                        and _paid_flag(entry) is not False):\n                    return folded_target\n',
        "new_string": '                if False:  # R6_01_classic_paid_passthrough\n                    return folded_target\n',
    },
    {
        "name": 'R6_2_secret_tail_no_semicolon',
        "description": 'Semicolon-separated credential tails are no longer redacted.',
        "finding": "R6-2: bare '; sig=SECRET' credential tails must be masked.",
        "old_string": '    r"([#?&;][ \\t]*(?:sig|signature|token|',
        "new_string": '    r"([#?&][ \\t]*(?:sig|signature|token|',
    },
]

NON_MUTABLE = [
    "V1 test-fixture placement - test-only fix (vacuous test re-fixtured); no source behavior to invert.",
]


def validate(source: str) -> None:
    """Raise AssertionError unless every entry is a precise, compiling, non-no-op edit."""
    for entry in MUTATIONS:
        name = entry["name"]
        old = entry["old_string"]
        count = source.count(old)
        if count != 1:
            raise AssertionError(
                "%s: old_string occurs %d times (need exactly 1)" % (name, count))
        mutated = source.replace(old, entry["new_string"], 1)
        try:
            compile(mutated, "__init__.py (mutated: %s)" % name, "exec")
        except SyntaxError as exc:
            raise AssertionError("%s: mutation does not compile: %s" % (name, exc))
        if mutated == source:
            raise AssertionError("%s: mutation is a no-op" % name)


def by_name(name: str):
    for entry in MUTATIONS:
        if entry["name"] == name:
            return entry
    return None


def names():
    return [entry["name"] for entry in MUTATIONS]
