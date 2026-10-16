# Hydrological extreme-flow metrics and inundation CSI

These metrics are intended for CoLM river-routing D (reservoir) and B
(bifurcation) experiments. They appear in the Metrics GUI and configuration
under **Hydrology** or **Inundation**.

## Flow quantiles

- **Q95_Bias**, **Q99_Bias** (%) compare the 0.95 and 0.99 *non-exceedance*
  percentiles of modeled vs. observed discharge:
  100 * (quantile(sim, p) - quantile(obs, p)) / quantile(obs, p).
- They are **high-flow quantiles** and must not be confused with the Q95
  exceedance flow-duration notation sometimes used for low flow.
- Only dates with finite values on *both sides* are used. If the observed
  quantile is zero, or fewer than two pairs remain, the value is NaN.
- These are different from PBIAS_HF, which sums biases **on observed high-flow
  dates** rather than comparing two marginal flow percentiles.

## Event-based flood peak timing

- **PeakTimingError** (days, ideal 0): mean absolute difference between
  matched observed/simulated local flood-peak dates.
- **PeakTimingMatchRate** (fraction 0-1): fraction of independently detected
  observed flood peaks matched to a model peak. Always examine alongside
  PeakTimingError: unmatched events are not included in the timing mean.
- Defaults: observed Q95 threshold; at least 15 days between observed
  independent events; match the largest available simulated local maximum
  within +/- 7 days, one-to-one, prioritizing the strongest observed floods.
  The method requires >= 30 common finite daily/subdaily samples. All matching
  occurs after OpenBench's ordinary time alignment.
- Monthly and annual flow series are *not* suitable for this event metric and
  yield NaN; use KGE, flow quantiles, or an appropriate seasonal metric there.
- Defaults are adjustable through the programmatic peak_timing_stats helper;
  the current GUI uses the documented defaults.

## Inundation Fraction: distinguish two CSI meanings

- **Inundation_CSI** is the **per-cell temporal CSI**. On each grid cell,
  across valid paired months, it is wet/wet count divided by wet-on-either
  count. This yields the normal (lat, lon) NetCDF evaluation map.
- When Inundation_CSI is evaluated on a gridded **Inundation_Fraction** item,
  OpenBench **also writes**:
  - metrics/<item>_ref_<ref>_sim_<sim>_Inundation_CSI_spatial_timeseries.csv
  - metrics/<item>_ref_<ref>_sim_<sim>_Inundation_CSI_spatial_summary.csv
- The first CSV has the **per-month spatial CSI**, formed from wet/wet area
  divided by the union of wet areas over the evaluated domain. The summary
  contains time-mean CSI and area-time pooled CSI. They are NOT the same
  quantity as per-cell temporal CSI.
- Default threshold: **fraction > 0.05 is wet**, applied identically to
  modeled and observed inundation fractions. Inputs must be in 0-1 fraction,
  NOT 0-100 percent or flooded area in m2. Paired missing cells are excluded.
  All-dry pixels/months have undefined (NaN) CSI, not an artificial perfect 1.
- Spatial CSI uses cos(latitude) cell-area weights, appropriate to a
  regularly spaced rectilinear lat/lon grid. It is not appropriate for a
  projected/equal-area source grid before that source is correctly regridded.
- Match GIEMS vs. model spatial footprints, land/water definitions,
  resolution and monthly time labeling before interpreting CSI.

## Study comparisons

For CTL vs. D or CTL vs. B, first use identical station IDs and **common
valid timestamps**. Positive delta-KGE indicates better skill; positive
difference in absolute quantile bias or absolute timing error means worse
skill. Compare changes in errors, not signed PBIAS differences alone.

