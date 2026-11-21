# Pollinations

Image generation and editing through [Pollinations](https://pollinations.ai), as a Hermes
image-gen backend. It works with no account. A key unlocks the full catalog.

## Three states

The picker lists only models your current setup can run — except your saved model,
which stays visible even when the current state cannot run it, marked disabled so the
picker never offers an unexplained dead choice. Generation still refuses what the current
state forbids.

| State | What you get |
| --- | --- |
| No API key | The model the free route serves today. Watermarked, throttled per IP, and capped at a reduced size. |
| API key added | The rest of the catalog Pollinations publishes. No watermark, no per-IP throttle — except a classic-served model the catalog omits, which keeps the watermarked classic route even keyed. |
| Key plus "Include paid models" | The full catalog, paid-only models included — and permitted to run. |

The toggle sits under Desktop → Settings → Plugins → Pollinations. It ships off. While it is
off, paid-only models are both hidden from the picker and refused at generation time. A model
id the catalog does not list cannot be classified and is sent as-is — except an unknown shared
`image_gen.model` value, which falls back to the live default (it is usually another provider's
leftover id), unless the free classic API serves it. Keyless, an unknown id the free classic
API does not serve is likewise ignored in favor of the live default.

## Setup

```
hermes plugins install pollinations
hermes plugins enable pollinations
hermes config set image_gen.provider pollinations
```

To unlock the catalog, get a key at https://enter.pollinations.ai/keys and save it as
`POLLINATIONS_API_KEY`. The free route does not need it.

## Editing

Models that support it get `image_url` and `reference_image_urls` in the tool schema. Hermes
shows those only for the model you have selected, so every option in front of you works.
Reference images can be a URL, a base64 data URI, or an absolute path on your machine.

## Behaviour worth knowing

The model list is fetched live from Pollinations and cached for an hour; if Pollinations is
unreachable, the last good list keeps serving rather than guessing. Models Pollinations adds or
removes show up without a plugin update. Prices and rate limits come from the catalog; the
free-model list is read live from the classic API (`image.pollinations.ai/models`). The edit
prompt length limit and the seed range are read live from the published API spec.

Rows are ordered by estimated per-image cost: per-image and per-megapixel prices are
normalized to a per-image figure, per-token models follow them (the catalog publishes no token
count to compare with), and models with no published price sort last.

A caller-provided `seed` is honored for reproducible images on models that support it, when
passed programmatically (the `image_generate` tool does not expose one); otherwise each call
gets a fresh random seed.

If neither the catalog nor the free-model list is reachable and nothing is cached, the plugin
offers no model rather than guessing at one that may no longer exist. Keyless, a reachable or
cached free-model list still works while the catalog is down.

The resolution control appears only with a key set, for models that publish a tier Hermes
can express (1K/2K).

## Disclosure

- Generation and catalog requests go to `gen.pollinations.ai` or `image.pollinations.ai`.
  Edit results returned as URLs are cached from whatever public host the response names
  (often `media.pollinations.ai`), through the SSRF guard with no credentials sent;
  reference-image URLs you supply are fetched from their own hosts the same way.
- The edits endpoint is pinned to `gen.pollinations.ai`: a model entry that advertises an edits
  URL on another host, over plain HTTP, or on a non-standard port is ignored and the Pollinations
  endpoint is used instead, so your key and reference images only ever leave for Pollinations.
- Prompts and reference images are sent to Pollinations. Generation prompts travel in the
  request URL path (they may appear in intermediary access logs); edit prompts travel in the
  request body.
- The free route watermarks images and caps output at a reduced size.
- Models whose live listing advertises SVG output are shown disabled and refused before any
  paid request: this plugin returns raster images only, and a paid SVG attempt would bill then
  fail. Exception: a keyless user can still run a classic-served SVG model on the free classic
  route, which always returns a raster JPEG.
- Generated images are written to `$HERMES_HOME/cache/generated/images/`. The model catalog,
  the free-model list and the API spec limits are cached under `$HERMES_HOME/cache/`
  (`pollinations_catalog.json`, `pollinations_classic_models.json`, `pollinations_spec_limits.json`).
- The free classic route's model list (`image.pollinations.ai/models`) and its `model` parameter
  are live surfaces the classic docs do not fully document; the plugin reads them defensively
  and degrades when they are unreachable.
- Paid-only models spend real money from your Pollinations balance. They stay hidden and are
  refused at generation time until you turn the toggle on. A model id the catalog does not
  list cannot be classified and is sent as-is (with the shared-config exception above).
- There is no telemetry or analytics. Outbound traffic is limited to the Pollinations API
  calls listed above, the public hosts named by edit-result URLs, and the reference-image
  hosts you supply — all public-only through the SSRF guard.

## License

MIT. See [LICENSE](LICENSE).
