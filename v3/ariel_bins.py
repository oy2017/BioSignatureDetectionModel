"""Wavelength bin edges for the three spectral configurations used in paper 1.

  ariel   Ariel's delivered Tier 3 configuration: three photometric bands below
          1.1 um (VISPhot 0.50-0.60, FGS-1 0.60-0.80, FGS-2 0.80-1.10), then
          NIRSpec at R = 15 over 1.10-1.95 um, AIRS-CH0 at R = 100 over
          1.95-3.90 um and AIRS-CH1 at R = 30 over 3.90-7.80 um. Channel bounds
          follow the Ariel Definition Study Report (Tinetti et al. 2018) and the
          resolving powers are the delivered (binned) values used by ArielRad
          (Mugnai et al. 2020), not the pixel sampling of the wavelength
          solutions in ../ariel_noise_model/. Bins are log-spaced within each
          channel at the channel's R, so a bin of width d(lambda) satisfies
          lambda / d(lambda) = R.
  r100    uniform R = 100 over 0.5-7.8 um (the old manuscript's "Tier 3 modeled
          at R = 100" comparator)
  r200    550 points over 0.5-7.8 um, MultiREx's own log-spaced grid, identical
          to the old study's grid (R ~ 200)

Run as a script to write ariel_bins.json (edges and centres in microns) and
print the bin counts.
"""
import json
import numpy as np

WL_MIN, WL_MAX = 0.5, 7.8

ARIEL_CHANNELS = [
    # (name, lo, hi, R or None for a single photometric band)
    ("VISPhot", 0.50, 0.60, None),
    ("FGS1", 0.60, 0.80, None),
    ("FGS2", 0.80, 1.10, None),
    ("NIRSpec", 1.10, 1.95, 15),
    ("AIRS-CH0", 1.95, 3.90, 100),
    ("AIRS-CH1", 3.90, 7.80, 30),
]


def log_edges(lo, hi, R):
    """Edges log-spaced so that each bin has lambda/dlambda = R (rounded to an
    integer number of bins across the channel)."""
    n = max(1, int(round(R * np.log(hi / lo))))
    return np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))


def tier_channels(r_nir, r_ch0, r_ch1):
    """The same six channels at another tier's binning. ArielRad's tier prescriptions
    (Mugnai et al. 2020; Edwards & Tinetti 2022): Tier 1 R ~ 1 / 3 / 1 and Tier 2
    R ~ 10 / 50 / 10 for NIRSpec / AIRS-CH0 / AIRS-CH1; photometers unchanged."""
    return [(n, lo, hi, {"NIRSpec": r_nir, "AIRS-CH0": r_ch0, "AIRS-CH1": r_ch1}.get(n, R))
            for n, lo, hi, R in ARIEL_CHANNELS]


def ariel_edges(channels=None):
    edges = []
    channel = []
    for name, lo, hi, R in (channels or ARIEL_CHANNELS):
        e = np.array([lo, hi]) if R is None else log_edges(lo, hi, R)
        if edges:
            e = e[1:]           # share the boundary with the previous channel
        edges.extend(e.tolist())
        channel.extend([name] * (len(e) if not channel else len(e)))
    edges = np.array(edges)
    # channel label per bin
    labels = []
    for name, lo, hi, R in (channels or ARIEL_CHANNELS):
        n = 1 if R is None else len(log_edges(lo, hi, R)) - 1
        labels.extend([name] * n)
    assert len(labels) == len(edges) - 1
    return edges, labels


def uniform_R_edges(R, lo=WL_MIN, hi=WL_MAX):
    return log_edges(lo, hi, R)


def multirex_550_centres():
    """MultiREx's Physics.wavenumber_grid(0.5, 7.8, 550) in microns, ascending.
    Reproduced here without importing multirex: it is 550 points log-spaced in
    wavenumber between 10000/7.8 and 10000/0.5."""
    wn = np.logspace(np.log10(1e4 / WL_MAX), np.log10(1e4 / WL_MIN), 550)
    return np.sort(1e4 / wn)


def centres_to_edges(c):
    lg = np.log(c)
    e = np.empty(len(c) + 1)
    e[1:-1] = np.exp(0.5 * (lg[:-1] + lg[1:]))
    e[0] = np.exp(lg[0] - 0.5 * (lg[1] - lg[0]))
    e[-1] = np.exp(lg[-1] + 0.5 * (lg[-1] - lg[-2]))
    return e


def configurations():
    ae, al = ariel_edges()
    r100 = uniform_R_edges(100)
    r200 = centres_to_edges(multirex_550_centres())
    t1e, t1l = ariel_edges(tier_channels(1, 3, 1))
    t2e, t2l = ariel_edges(tier_channels(10, 50, 10))
    out = {
        "ariel": {"edges": ae.tolist(), "channel": al},
        "r100": {"edges": r100.tolist()},
        "r200": {"edges": r200.tolist()},
        "tier1": {"edges": t1e.tolist(), "channel": t1l},
        "tier2": {"edges": t2e.tolist(), "channel": t2l},
    }
    for k, v in out.items():
        e = np.array(v["edges"])
        v["centres"] = np.sqrt(e[:-1] * e[1:]).tolist()
        v["n_bins"] = len(e) - 1
    return out


if __name__ == "__main__":
    cfg = configurations()
    with open("ariel_bins.json", "w") as f:
        json.dump(cfg, f, indent=1)
    for k, v in cfg.items():
        print(f"{k:6s} {v['n_bins']:4d} bins")
    from collections import Counter
    print("ariel per channel:", dict(Counter(cfg["ariel"]["channel"])))
