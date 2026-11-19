#!/usr/bin/env python3
"""
HW5 Part 2 - Photon escape time from the Sun (extends mc_photon_slab.py).

Idea
----
1. Re-use the slab random walk from Part 1 (isotropic Thomson scattering,
   exponential free paths) in units of the mean free path l.  Run it
   ~1000 times for a slab ~10 mfp wide and store, for each run, the travel time
   (total path length, in units of l/c), which face the photon left through
   (+1 forward / -1 backward) and how far it drifted sideways.
   Because everything is in units of l, the table is valid at ANY density.

2. Walk a photon through the Sun, where n_e(r) = 2.5e26 exp(-r / 0.096 R_sun).
   At radius r the local mean free path is l(r) = 1/(n_e(r) sigma_T).  Each
   "stage" draws one row of the table, i.e. a travel time (in units of l(r)/c),
   an exit face and a sideways drift, and moves the photon accordingly.
   Stages are repeated until the photon reaches 0.9 R_sun.

Examples
--------
    python mc_photon_sun.py
    python mc_photon_sun.py --n-photons 2000 --eps 0.02 --seed 1
    python mc_photon_sun.py --save sun_escape.png --no-show
"""

import argparse
import time

import numpy as np

from mc_photon_slab import SIGMA_T  # Thomson cross section [cm^2] from Part 1

C_LIGHT = 2.99792458e10      # cm/s
R_SUN = 696340.0 * 1e5       # 696,340 km in cm
YEAR = 3.15576e7             # s
Z0 = 0.7104                  # extrapolation length (in mfp) for isotropic scattering


# --------------------------------------------------------------------------
# Part A: slab table (vectorised version of the Part 1 random walk)
# --------------------------------------------------------------------------
def slab_exit_table(rng, n_runs, width, max_steps=10_000_000):
    """Run n_runs photons through a slab `width` mean free paths wide.

    Each photon starts in the middle of the slab (just after a scattering, so
    its first direction is isotropic) and walks until it leaves through the
    front (-) or back (+) face.  Everything is in units of the mean free path.

    Returns arrays (T, side, lat):
        T    total path length = travel time in units of l/c
        side +1 if the photon left through the forward face, -1 otherwise
        lat  sideways distance between start and exit points [mfp]
    """
    half = 0.5 * width
    z = np.zeros(n_runs)
    mu = 2.0 * rng.random(n_runs) - 1.0
    path = np.zeros(n_runs)
    lx = np.zeros(n_runs)
    ly = np.zeros(n_runs)
    T = np.zeros(n_runs)
    side = np.zeros(n_runs)
    lat = np.zeros(n_runs)

    active = np.arange(n_runs)
    steps = 0
    while active.size and steps < max_steps:
        steps += 1
        m = active.size
        s = -np.log1p(-rng.random(m))                 # free path ~ Exp(1) [mfp]
        mui = mu[active]
        sin_t = np.sqrt(1.0 - mui * mui)
        phi = 2.0 * np.pi * rng.random(m)
        dz = s * mui
        znew = z[active] + dz
        out = np.abs(znew) > half

        # clip the last step at the face the photon leaves through
        frac = np.ones(m)
        bound = np.sign(znew[out]) * half
        frac[out] = (bound - z[active][out]) / dz[out]
        seff = s * frac

        path[active] += seff
        lx[active] += seff * sin_t * np.cos(phi)
        ly[active] += seff * sin_t * np.sin(phi)

        done = active[out]
        T[done] = path[done]
        side[done] = np.sign(znew[out])
        lat[done] = np.hypot(lx[done], ly[done])

        keep = ~out
        active = active[keep]
        z[active] = znew[keep]
        mu[active] = 2.0 * rng.random(active.size) - 1.0   # isotropic scattering

    if active.size:
        raise RuntimeError("some slab photons did not exit; raise --max-steps")
    return T, side, lat


# --------------------------------------------------------------------------
# Part B: the Sun
# --------------------------------------------------------------------------
class Sun:
    """n_e(r) = n0 exp(-r/H), valid out to r_out."""

    def __init__(self, n0, sigma, scale_height_frac, r_out_frac):
        self.n0 = n0
        self.sigma = sigma
        self.H = scale_height_frac * R_SUN
        self.r_out = r_out_frac * R_SUN
        self.S = sigma * n0 * self.H          # optical depth from r=0 to infinity

    def n_e(self, r):
        return self.n0 * np.exp(-r / self.H)

    def mfp(self, r):
        return 1.0 / (self.n_e(r) * self.sigma)

    def tau(self, r):
        """Radial optical depth from the centre out to r."""
        return -self.S * np.expm1(-r / self.H)

    def r_of_tau(self, t):
        """Inverse of tau(r) (returns negative r for negative t)."""
        return -self.H * np.log1p(-t / self.S)

    def diffusion_escape_time(self):
        """Analytic diffusion-limit escape time from the centre [s].

        Solve (1/r^2) d/dr (r^2 D dT/dr) = -1 with D = c l/3 and T(r_out)=0:
            T(0) = (sigma n0 / c) * int_0^r_out r exp(-r/H) dr
        """
        x = self.r_out / self.H
        return self.sigma * self.n0 * self.H ** 2 / C_LIGHT * (1.0 - (1.0 + x) * np.exp(-x))


def simulate_sun(rng, sun, table, half0, n_photons, eps, max_stages):
    """Escape time of n_photons from r=0 to r_out, built from slab-table draws.

    One stage, for a photon at radius r with local mean free path l:
      * choose the stage half-width X [mfp] = min(eps*H/l, distance to r_out)
      * draw a random row (T, lat) of the slab table and a random exit face (+/-)
      * rescale it from the table's half-width to X (time ~ (X+z0)^2,
        sideways drift ~ (X+z0), where z0 accounts for the slab boundary)
      * time += T * scale^2 * l / c
      * move +/-X along the radial optical-depth coordinate (this is exact
        for a plane-parallel medium and automatically includes the drift
        caused by the density gradient), and add the sideways drift, which
        is what produces the spherical-geometry (2D/r) effect.
    """
    T, _side, lat = table
    n_tab = T.size
    tau_out = sun.tau(sun.r_out)

    r = np.zeros(n_photons)
    t = np.zeros(n_photons)
    n_stages = np.zeros(n_photons, dtype=np.int64)
    active = np.arange(n_photons)
    stage = 0

    while active.size:
        stage += 1
        if stage > max_stages:
            raise RuntimeError("hit --max-stages; increase it or raise --eps")

        ri = r[active]
        l = sun.mfp(ri)
        tau_i = sun.tau(ri)
        X = np.minimum(eps * sun.H / l, tau_out - tau_i)

        # closer to the edge than one table slab: treat as escaped
        go = X >= half0
        idx = active[go]
        if idx.size == 0:
            break
        X, l, tau_i = X[go], l[go], tau_i[go]

        k = rng.integers(n_tab, size=idx.size)
        # The slab is symmetric, so T and lat do not depend on the exit face.
        # Draw the face fresh (+/-1) instead of using the table's finite-sample
        # forward fraction: a table that is 52/48 instead of 50/50 would act as
        # a constant drift at every one of the ~1e4 stages and swamp the result.
        sgn = np.where(rng.random(idx.size) < 0.5, -1.0, 1.0)
        a = (X + Z0) / (half0 + Z0)
        t[idx] += T[k] * a * a * l / C_LIGHT
        r_signed = sun.r_of_tau(tau_i + sgn * X)
        r[idx] = np.hypot(r_signed, lat[k] * a * l)
        n_stages[idx] += 1

        active = idx[r[idx] < sun.r_out]

    return t, n_stages


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo photon escape time from the Sun, built from "
                    "slab random walks (HW5 part 2).")
    p.add_argument("--n-slab", type=int, default=1000,
                   help="number of slab runs used to build the time table (default: 1000)")
    p.add_argument("--slab-width", type=float, default=10.0,
                   help="slab width for the table in mean free paths (default: 10)")
    p.add_argument("--n-photons", type=int, default=1000,
                   help="number of photons sent through the Sun (default: 1000)")
    p.add_argument("--eps", type=float, default=0.05,
                   help="max stage size as a fraction of the density scale height; "
                        "smaller = more accurate but slower (default: 0.05)")
    p.add_argument("--density0", type=float, default=2.5e26,
                   help="central electron density [cm^-3] (default: 2.5e26)")
    p.add_argument("--scale-height", type=float, default=0.096,
                   help="density scale height in units of R_sun (default: 0.096)")
    p.add_argument("--r-out", type=float, default=0.9,
                   help="outer edge in units of R_sun (default: 0.9)")
    p.add_argument("--sigma", type=float, default=SIGMA_T,
                   help="cross section [cm^2] (default: Thomson)")
    p.add_argument("--max-stages", type=int, default=5_000_000,
                   help="safety cap on stages per photon (default: 5e6)")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--no-plot", action="store_true", help="skip the figure")
    p.add_argument("--save", type=str, default=None,
                   help="save the figure to this file (e.g. sun_escape.png)")
    p.add_argument("--no-show", action="store_true",
                   help="do not open an interactive window")
    return p.parse_args()


def make_figure(sun, table, half0, t_sec, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, side, _ = table
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].hist(T, bins=60, color="tab:blue", alpha=0.8)
    ax[0].axvline(T.mean(), color="k", ls="--", label=f"mean = {T.mean():.1f}")
    ax[0].set_xlabel("travel time through slab  [l / c]")
    ax[0].set_ylabel("number of runs")
    ax[0].set_title(f"Slab table ({2 * half0:g} mfp wide, {T.size} runs)")
    ax[0].legend()

    r = np.linspace(0, sun.r_out, 400)
    ax[1].semilogy(r / R_SUN, sun.n_e(r), color="tab:red", label="$n_e$ [cm$^{-3}$]")
    ax[1].set_xlabel("r / R$_{sun}$")
    ax[1].set_ylabel("$n_e$ [cm$^{-3}$]", color="tab:red")
    ax1b = ax[1].twinx()
    ax1b.semilogy(r / R_SUN, sun.mfp(r), color="tab:green")
    ax1b.set_ylabel("mean free path [cm]", color="tab:green")
    ax[1].set_title("Solar model")

    yrs = t_sec / YEAR
    ax[2].hist(yrs, bins=40, color="tab:orange", alpha=0.85)
    ax[2].axvline(yrs.mean(), color="k", ls="--", label=f"mean = {yrs.mean():.0f} yr")
    ax[2].axvline(sun.diffusion_escape_time() / YEAR, color="tab:purple", ls=":",
                  label="diffusion estimate")
    ax[2].set_xlabel("photon escape time  [years]")
    ax[2].set_ylabel("number of photons")
    ax[2].set_title(f"Escape time from the Sun ({yrs.size} photons)")
    ax[2].legend()

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved figure to {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sun = Sun(args.density0, args.sigma, args.scale_height, args.r_out)
    half0 = 0.5 * args.slab_width

    tau_out = sun.tau(sun.r_out)
    print(f"Centre: n_e = {sun.n0:.2e} cm^-3, l = {sun.mfp(0.0):.3e} cm")
    print(f"Edge (r = {args.r_out} R_sun): n_e = {sun.n_e(sun.r_out):.2e} cm^-3, "
          f"l = {sun.mfp(sun.r_out):.3e} cm")
    print(f"Optical depth centre -> edge: {tau_out:.3e} mean free paths")
    print(f"Brute force would need ~tau^2 = {tau_out ** 2:.1e} scatterings per photon "
          f"(~{tau_out ** 2 / 1e8 / YEAR:.0e} yr at 1e8 scatterings/s)\n")

    # ---- Part A: slab table --------------------------------------------
    t0 = time.perf_counter()
    table = slab_exit_table(rng, args.n_slab, args.slab_width)
    t_slab = time.perf_counter() - t0
    T, side, lat = table
    print(f"[timing] slab table: {args.n_slab} runs in {t_slab:.3f} s")
    print(f"  travel time: mean = {T.mean():.1f}, median = {np.median(T):.1f}, "
          f"std = {T.std():.1f}  [l/c]")
    print(f"  fraction leaving through the forward face: {(side > 0).mean():.3f} "
          f"(should be 0.5; the Sun run draws the face at random)\n")

    # ---- Part B: Sun ----------------------------------------------------
    t0 = time.perf_counter()
    t_esc, n_stages = simulate_sun(rng, sun, table, half0, args.n_photons,
                                   args.eps, args.max_stages)
    t_sun = time.perf_counter() - t0
    print(f"[timing] Sun: {args.n_photons} photons, mean {n_stages.mean():.0f} stages each "
          f"(max {n_stages.max()}), {t_sun:.2f} s")

    yrs = t_esc / YEAR
    print("\nEscape time to 0.9 R_sun:")
    print(f"  mean   = {yrs.mean():.4g} yr  (std {yrs.std():.3g}, "
          f"standard error {yrs.std() / np.sqrt(yrs.size):.2g})")
    print(f"  median = {np.median(yrs):.4g} yr")
    print(f"  min / max = {yrs.min():.4g} / {yrs.max():.4g} yr")
    print(f"  analytic diffusion estimate = {sun.diffusion_escape_time() / YEAR:.4g} yr")
    print(f"  (free-streaming time to cross the Sun, for comparison: "
          f"{R_SUN / C_LIGHT:.2f} s)")

    if not args.no_plot:
        make_figure(sun, table, half0, t_esc, args.save, not args.no_show)


if __name__ == "__main__":
    main()
