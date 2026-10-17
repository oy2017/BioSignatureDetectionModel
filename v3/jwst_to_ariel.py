"""Convert a JWST transmission spectrum to the consortium classifier's input (REVISION_PLAN.md, A2).

The classifier reads 104 transit depths: the three Ariel photometric bands (0.50-0.60, 0.60-0.80,
0.80-1.10 um) and the Tier-3 spectrometer bins (NIRSpec R = 20, AIRS-CH0 R = 100, AIRS-CH1 R = 30),
per-spectrum normalized to zero mean and unit dispersion. A JWST spectrum is a set of segments
(one per instrument mode) of transit depths on their own wavelength grids. The conversion:

  1. quantity    JWST products and the training spectra are both transit depth (Rp/Rs)^2;
                 a segment in ppm or percent is converted on reading
  2. binning     each Ariel bin takes the inverse-variance-weighted mean of the JWST points whose
                 wavelength intervals overlap it, each point weighted by its overlap fraction;
                 the error is propagated; the same is done per mode, and modes are then combined
                 by inverse variance, so a bin covered by two modes uses both
  3. coverage    per bin: number of points, modes used, and the fraction of the bin's width that
                 JWST points cover; bins below a coverage threshold are "empty"
  4. offsets     a per-mode additive offset (the fitted inter-mode offsets and their errors) can be
                 redrawn so a verdict becomes a distribution over offset realizations; the global
                 offset is irrelevant after normalization, relative offsets are not
  5. gap filling empty bins are filled from a model spectrum binned by integration and flagged
  6. noise       (i) JWST precision as delivered; (ii) Ariel Tier-1 noise at the target's MCS
                 transit count (noise.radiometric_sigma) added as Gaussian draws; (iii) other tiers
  7. round trip  a model spectrum sampled at the JWST grids with the JWST errors, pushed through
                 2-6, against the same model binned by integration: the conversion's own error

Only the arithmetic lives here; readers for each archive are in jwst_readers.py, and the classifier
is applied in alfnoor_jwst.py. Nothing here touches the classifier.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ariel_bins as AB  # noqa: E402
from bin_spectra import bin_native  # noqa: E402

PHOT = {"VISPhot", "FGS1", "FGS2"}


def tier3_layout():
    """(edges, channel label per bin) of the classifier's 104-bin input."""
    edges, labels = AB.ariel_edges(AB.tier_channels(20, 100, 30))
    return np.asarray(edges, float), np.asarray(labels)


def point_intervals(wl, width=None):
    """[lo, hi] of each JWST point. With no stated width, midpoints between neighbours, extended
    half a step at the ends (the convention used for TauREx's native grid in shift_opacity)."""
    wl = np.asarray(wl, float)
    if width is not None:
        w = np.broadcast_to(np.asarray(width, float), wl.shape)
        return wl - w / 2, wl + w / 2
    step = np.diff(wl)
    # half-width = half the SMALLER neighbouring step, so a point at the edge of a gap in the
    # grid (e.g. PRISM's saturated region) does not claim the gap
    half = np.empty_like(wl)
    half[0], half[-1] = step[0] / 2, step[-1] / 2
    half[1:-1] = np.minimum(step[:-1], step[1:]) / 2
    return wl - half, wl + half


def bin_segment(wl, depth, err, edges, width=None):
    """One mode onto the Ariel bins. Returns depth, error, n points, covered fraction per bin;
    NaN where nothing overlaps. Weight of a point in a bin: overlap fraction / err^2."""
    wl, depth, err = (np.asarray(a, float) for a in (wl, depth, err))
    lo, hi = point_intervals(wl, width)
    nb = len(edges) - 1
    out_d, out_e, n_pts, cov = np.full(nb, np.nan), np.full(nb, np.nan), np.zeros(nb, int), np.zeros(nb)
    for b in range(nb):
        a, c = edges[b], edges[b + 1]
        ov = np.clip(np.minimum(hi, c) - np.maximum(lo, a), 0, None)
        m = (ov > 0) & np.isfinite(depth) & np.isfinite(err) & (err > 0)
        if not m.any():
            continue
        frac = ov[m] / (hi[m] - lo[m])              # share of each point inside the bin
        w = frac / err[m] ** 2
        out_d[b] = np.sum(w * depth[m]) / np.sum(w)
        out_e[b] = np.sqrt(np.sum((w * err[m]) ** 2)) / np.sum(w)   # propagated through the weighted mean
        n_pts[b] = int(m.sum()); cov[b] = min(1.0, ov[m].sum() / (c - a))
    return out_d, out_e, n_pts, cov


def combine(segments, edges, offsets=None):
    """segments: {mode: dict(wl, depth, err, width=None)}. offsets: {mode: additive depth offset}.
    Returns (depth, err) per bin and a coverage table."""
    labels = tier3_layout()[1]
    nb = len(edges) - 1
    num, den = np.zeros(nb), np.zeros(nb)
    rows = [dict(bin=b, lo=edges[b], hi=edges[b + 1], channel=labels[b], modes="", n_points=0, covered=0.0) for b in range(nb)]
    for mode, s in segments.items():
        off = 0.0 if offsets is None else offsets.get(mode, 0.0)
        d, e, n, cov = bin_segment(s["wl"], np.asarray(s["depth"], float) + off, s["err"], edges, s.get("width"))
        ok = np.isfinite(d)
        w = np.where(ok, 1.0 / np.where(ok, e, 1.0) ** 2, 0.0)
        num += np.where(ok, w * np.where(ok, d, 0.0), 0.0); den += w
        for b in np.where(ok)[0]:
            rows[b]["modes"] = (rows[b]["modes"] + "+" + mode).strip("+"); rows[b]["n_points"] += int(n[b])
            rows[b]["covered"] = min(1.0, rows[b]["covered"] + cov[b])
    depth = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    err = np.where(den > 0, 1.0 / np.sqrt(np.where(den > 0, den, 1.0)), np.nan)
    return depth, err, pd.DataFrame(rows)


def fill_gaps(depth, err, model_wl, model_depth, edges, min_cover=0.5, coverage=None):
    """Fill bins with no data (or covered below min_cover) from a model binned by integration, after
    shifting the model by the inverse-variance-weighted mean offset between data and model over the
    measured bins, so the fill continues the data's level rather than the model's absolute depth.
    Returns filled depth, filled error (NaN where filled: no measurement), and the filled mask."""
    filled = ~np.isfinite(depth)
    if coverage is not None:
        filled |= coverage.covered.to_numpy() < min_cover
    model_b = bin_native(np.asarray(model_depth, float)[None, :], np.asarray(model_wl, float), edges)[0]
    meas = ~filled & np.isfinite(err) & (err > 0)
    if meas.any():
        w = 1.0 / err[meas] ** 2
        model_b = model_b + np.sum(w * (depth[meas] - model_b[meas])) / np.sum(w)
    out = depth.copy(); out[filled] = model_b[filled]
    e = err.copy(); e[filled] = np.nan
    return out, e, filled


def align_offsets(segments, reference, min_overlap_points=5):
    """Additive offsets that bring each mode onto the reference mode's level, estimated from the
    inverse-variance-weighted mean difference over the wavelengths the modes share, propagating
    outward from the reference through the chain of overlaps. Returns {mode: offset} (reference: 0).
    This is the step Carter & May et al. 2024 did at the light-curve level; for products reduced
    separately it has to be done here, and the fitted values are reported as part of the procedure."""
    off = {reference: 0.0}
    pending = [m for m in segments if m != reference]
    while pending:
        progressed = False
        for m in list(pending):
            s = segments[m]
            best = None
            for r, o in off.items():
                t = segments[r]
                lo, hi = max(s["wl"].min(), t["wl"].min()), min(s["wl"].max(), t["wl"].max())
                if hi <= lo:
                    continue
                ms, mt = (s["wl"] >= lo) & (s["wl"] <= hi), (t["wl"] >= lo) & (t["wl"] <= hi)
                if ms.sum() < min_overlap_points or mt.sum() < min_overlap_points:
                    continue
                # compare on the coarser of the two grids: interpolate the finer onto it
                if ms.sum() <= mt.sum():
                    d_t = np.interp(s["wl"][ms], t["wl"], t["depth"]); d_s = s["depth"][ms]; w = 1.0 / s["err"][ms] ** 2
                else:
                    d_s = np.interp(t["wl"][mt], s["wl"], s["depth"]); d_t = t["depth"][mt]; w = 1.0 / t["err"][mt] ** 2
                delta = np.sum(w * (d_t + o - d_s)) / np.sum(w); n = int(min(ms.sum(), mt.sum()))
                if best is None or n > best[1]:
                    best = (delta, n)
            if best is not None:
                off[m] = float(best[0]); pending.remove(m); progressed = True
        if not progressed:
            for m in pending:
                off[m] = 0.0                      # no overlap with anything aligned: left as published
            break
    return off


def ariel_sigma(name, edges, transits="tier1_transits"):
    """Per-bin Tier-1 (or other tier) sigma for a named MCS target, from the payload noise model."""
    import noise as NZ
    s, _ = NZ.radiometric_sigma(pd.DataFrame({"name": [name]}), edges, transits)
    return s[0]


def noise_draws(depth, sigma, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return depth[None, :] + rng.normal(0, 1, (n, len(depth))) * sigma[None, :]


def normalize(X):
    X = np.atleast_2d(X)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def sample_clean(model_wl, model_depth, segments):
    """A model spectrum observed like the JWST segments, noise-free: the model averaged over each
    point's wavelength interval (interpolated where the interval holds no model node)."""
    model_wl, model_depth = np.asarray(model_wl, float), np.asarray(model_depth, float)
    out = {}
    for mode, s in segments.items():
        lo, hi = point_intervals(s["wl"], s.get("width"))
        i0, i1 = np.searchsorted(model_wl, lo), np.searchsorted(model_wl, hi)
        csum = np.concatenate([[0.0], np.cumsum(model_depth)])
        n = i1 - i0
        d = np.where(n > 0, (csum[i1] - csum[i0]) / np.maximum(n, 1), np.interp(0.5 * (lo + hi), model_wl, model_depth))
        out[mode] = dict(wl=np.asarray(s["wl"], float), depth=d, err=np.asarray(s["err"], float), width=s.get("width"))
    return out


def round_trip(model_wl, model_depth, segments, edges, n=200, seed=0, offsets_sigma=None):
    """Distribution of the converted model spectrum against the directly binned model (step 7).
    Each realization scatters the sampled model with the JWST errors and, if offsets_sigma is given,
    redraws each mode's additive offset. Returns the directly binned model and (n, nbins) realizations."""
    rng = np.random.default_rng(seed)
    direct = bin_native(np.asarray(model_depth, float)[None, :], np.asarray(model_wl, float), edges)[0]
    clean = sample_clean(model_wl, model_depth, segments)
    reals = []
    for _ in range(n):
        obs = {m: dict(s, depth=s["depth"] + rng.normal(0, 1, len(s["depth"])) * s["err"]) for m, s in clean.items()}
        off = None if offsets_sigma is None else {m: rng.normal(0, s) for m, s in offsets_sigma.items()}
        d, e, cov = combine(obs, edges, off)
        d, _, _ = fill_gaps(d, e, model_wl, model_depth, edges, coverage=cov)
        reals.append(d)
    return direct, np.vstack(reals)


def band_metric(depth, edges, molecule):
    """Mugnai et al. 2021, Eq. 3, with the bands of their Table 3: the mean over feature bands of
    (band mean - normalization-band mean) / sqrt(band dispersion^2 + normalization dispersion^2)."""
    bands = {"H2O": [(1.2, 1.6), (1.7, 2.1), (2.6, 3.0), (5.4, 6.1), (6.5, 7.0)],
             "CH4": [(1.5, 1.8), (2.1, 2.6), (3.1, 3.7)],
             "CO2": [(1.9, 2.3), (2.6, 3.2), (4.2, 4.8)]}
    norm = {"H2O": (3.6, 4.2), "CH4": (4.0, 5.0), "CO2": (5.0, 6.0)}
    cen = 0.5 * (edges[1:] + edges[:-1])

    def stats(lo, hi):
        m = (cen >= lo) & (cen < hi) & np.isfinite(depth)
        return (np.mean(depth[m]), np.std(depth[m])) if m.sum() else (np.nan, np.nan)
    sn, dn = stats(*norm[molecule])
    vals = []
    for lo, hi in bands[molecule]:
        sb, db = stats(lo, hi)
        vals.append((sb - sn) / np.sqrt(db ** 2 + dn ** 2))
    return float(np.nanmean(vals))


def align_to_model(segments, offsets, model_wl, model_depth, modes):
    """Offsets for modes that share no wavelengths with the aligned ones (MIRI against the near-infrared
    modes: the overlap is one bin). The model is first levelled onto the already-aligned modes by the
    inverse-variance-weighted mean residual over all their points, then each listed mode gets the
    weighted mean of (levelled model - mode) over its points inside the model's range. The model fixes
    only the level across the gap; its shape inside the mode is not used. Returns offsets with the
    listed modes added, plus the model level applied."""
    clean = sample_clean(model_wl, model_depth, segments)
    num = den = 0.0
    for m, o in offsets.items():
        if m in modes:
            continue
        s = segments[m]; w = 1.0 / np.asarray(s["err"], float) ** 2
        num += np.sum(w * (np.asarray(s["depth"], float) + o - clean[m]["depth"])); den += np.sum(w)
    level = num / den if den > 0 else 0.0
    out = dict(offsets)
    for m in modes:
        s = segments[m]; inside = (np.asarray(s["wl"], float) >= model_wl.min()) & (np.asarray(s["wl"], float) <= model_wl.max())
        w = 1.0 / np.asarray(s["err"], float)[inside] ** 2
        out[m] = float(np.sum(w * (clean[m]["depth"][inside] + level - np.asarray(s["depth"], float)[inside])) / np.sum(w))
    return out, level
