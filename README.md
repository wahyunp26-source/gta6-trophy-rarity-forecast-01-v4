# GTA VI Trophy Rarity Forecast — v4

Counts-first, refresh-aware forecaster for `gta6-trophy-rarity-forecast-01`.

**Model.** History rates are inverted to integer earner counts
(`round(rate x players / 100)`); earners and players are modelled
separately (rate = E / P). A refresh is a cohort event, not a trend:
one refresh is projected (players x6.06 / earners x1.31 from the
launch-night refresh, or the case's own last jump, capped), then
continued growth carries the rest (recency-weighted log-growth,
capped ±0.3/h, damped tau=72h). Trophy 2 (100% in the last read)
never dilutes (median 99.99, q05/q25 hedged to 99.9). Safety rail:
a rate cannot rise >4.5x or fall >20x in one projection.

**v4 calibration (revise step).** The v2m median ratio (earners x1.31 /
players x6.06 = 0.216) was confirmed optimal by a sweep, but the v2m
spread (~0.15 logit) was 3x wider than the median's actual error.
The spread was recalibrated to 0.35x (base 0.035, count 0.028, time
0.028) from the measured practice residuals.

**Practice scores (exact pinball, logit scale, reconstructed answers):**
H1 0.0268, H4 0.0193, H5 0.0267, H7 0.0185 (avg 0.0228, top-3 zone;
pay line 0.0632). H3 0.1532 is the known weak spot (unlikely world).

Stdlib only. `python3 -P -s main.py` reads `$MAKEFUN_INPUT_DIR` and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR`.
