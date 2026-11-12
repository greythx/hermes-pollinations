# Pollinations

Image generation and editing through [Pollinations](https://pollinations.ai), as a Hermes
image-gen backend. It works with no account. A key unlocks the full catalog.

## Three states

The picker lists only models your current setup can run.

| State | What you get |
| --- | --- |
| No API key | The model the free route serves today. Watermarked, throttled per IP, and capped at a reduced size. |
| API key added | The rest of the catalog Pollinations publishes. No watermark, no per-IP throttle. |
| Key plus "Include paid models" | The full catalog, paid-only models included. |

The toggle sits under Desktop → Settings → Plugins → Pollinations. It ships off.

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

The model list is fetched from Pollinations each time the picker opens, and cached briefly.
Models Pollinations adds or removes show up without a plugin update. Prices, rate limits, and
the free-model list come from the same catalog.

If the catalog is unreachable and nothing is cached, the plugin offers no model rather than
guessing at one that may no longer exist.

The resolution control appears only for models that publish a tier Hermes can express.

## Disclosure

- Every request goes to `gen.pollinations.ai` or `image.pollinations.ai`.
- The edits endpoint is pinned to `gen.pollinations.ai`: a model entry that advertises an edits
  URL on another host, over plain HTTP, or on a non-standard port is ignored and the Pollinations
  endpoint is used instead, so your key and reference images only ever leave for Pollinations.
- Prompts and reference images are sent to Pollinations.
- The free route watermarks images and caps output at a reduced size.
- Generated images are written to `$HERMES_HOME/cache/generated/images/`. The model catalog
  and the free-model list are cached under `$HERMES_HOME/cache/`.
- Paid-only models spend real money from your Pollinations balance. They stay hidden until you
  turn the toggle on.
- There is no telemetry or analytics. The only outbound traffic is the Pollinations API calls
  listed above.

## License

MIT. See [LICENSE](LICENSE).
