
"""
HW5 Part 1 - Monte Carlo radiative transfer through a slab of electrons.

A photon enters a slab at depth z = 0 travelling in the +z direction.
Between scatterings it travels a distance s drawn from the exponential
distribution  p(s) = exp(-s/l)/l  with mean free path l = 1/(n_e * sigma_T).
At each scattering (isotropic, low-energy Thomson regime) the new direction
cosine mu = cos(theta) is uniform on [-1, 1] and the azimuth phi is uniform
on [0, 2*pi).  The photon is followed until it leaves the slab either out the
back (z > L, "transmitted") or back out the front (z < 0, "reflected").

Examples
--------
    python mc_photon_slab.py                       # defaults, shows animation
    python mc_photon_slab.py --seed 3 --save photon.gif --no-show
    python mc_photon_slab.py --n-photons 20000 --no-animation
"""

import argparse

import numpy as np

SIGMA_T = 6.652e-25  # Thomson cross section [cm^2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo photon random walk through a slab of free electrons."
    )
    p.add_argument("--density", type=float, default=1e20,
                   help="electron number density n_e [cm^-3] (default: 1e20)")
    p.add_argument("--width", type=float, default=1.0,
                   help="slab width [km] (default: 1.0)")
    p.add_argument("--sigma", type=float, default=SIGMA_T,
                   help="interaction cross section [cm^2] (default: Thomson)")
    p.add_argument("--n-photons", type=int, default=10000,
                   help="number of photons for the statistics run (default: 10000)")
    p.add_argument("--max-steps", type=int, default=1_000_000,
                   help="safety cap on scatterings per photon (default: 1e6)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducibility")
    p.add_argument("--no-animation", action="store_true",
                   help="skip the animation and only print statistics")
    p.add_argument("--save", type=str, default=None,
                   help="save animation to this file (e.g. photon.gif or photon.mp4)")
    p.add_argument("--no-show", action="store_true",
                   help="do not open an interactive window")
    p.add_argument("--fps", type=int, default=15,
                   help="animation frames per second (default: 15)")
    p.add_argument("--min-scatters", type=int, default=5,
                   help="when picking the photon to animate, require at least this many "
                        "scatterings so the path is interesting (default: 5)")
    return p.parse_args()


def follow_photon(rng, mfp, L, max_steps=1_000_000, record=False):
    """Follow one photon through the slab.

    Returns (outcome, n_scatter, path) where outcome is 'transmitted',
    'reflected' or 'lost' (hit max_steps), and path is an (N, 3) array of
    (x, y, z) positions in cm if record=True, otherwise None.
    """
    pos = np.zeros(3)             # start at the front face
    mu, phi = 1.0, 0.0            # moving straight into the slab (+z)
    path = [pos.copy()] if record else None
    n_scatter = 0

    for _ in range(max_steps):
        # distance to next interaction: s = -l ln(xi),  xi in (0, 1]
        s = -mfp * np.log(1.0 - rng.random())
        sin_t = np.sqrt(max(0.0, 1.0 - mu * mu))
        pos = pos + s * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), mu])

        # Did the photon leave the slab?  (clip the last step at the boundary)
        if pos[2] < 0.0 or pos[2] > L:
            if record:
                prev = path[-1]
                boundary = 0.0 if pos[2] < 0.0 else L
                frac = (boundary - prev[2]) / (pos[2] - prev[2])
                path.append(prev + frac * (pos - prev))
            outcome = "reflected" if pos[2] < 0.0 else "transmitted"
            return outcome, n_scatter, (np.array(path) if record else None)

        if record:
            path.append(pos.copy())

        # isotropic scattering: cos(theta) uniform in [-1, 1], phi uniform
        n_scatter += 1
        mu = 2.0 * rng.random() - 1.0
        phi = 2.0 * np.pi * rng.random()

    return "lost", n_scatter, (np.array(path) if record else None)


def run_statistics(rng, n_photons, mfp, L, max_steps):
    counts = {"transmitted": 0, "reflected": 0, "lost": 0}
    scatters = []
    for _ in range(n_photons):
        outcome, n, _ = follow_photon(rng, mfp, L, max_steps)
        counts[outcome] += 1
        scatters.append(n)
    return counts, np.array(scatters)


def pick_photon_to_animate(rng, mfp, L, max_steps, min_scatters):
    """Keep drawing photons until one has enough scatterings to be interesting."""
    for _ in range(10000):
        outcome, n, path = follow_photon(rng, mfp, L, max_steps, record=True)
        if n >= min_scatters and outcome != "lost":
            return outcome, n, path
    return outcome, n, path  # fall back to the last one


def animate_path(path, L, outcome, n_scatter, fps, save, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    km = 1e5  # cm per km
    z = path[:, 2] / km
    # transverse coordinate: use x (the slab is symmetric in x and y)
    t = path[:, 0] / km
    L_km = L / km

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(0, L_km, color="gold", alpha=0.15, label="electron slab")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(L_km, color="k", lw=1)

    pad = 0.1 * L_km
    ax.set_xlim(min(z.min(), 0) - pad, max(z.max(), L_km) + pad)
    ty = max(abs(t).max(), 0.1 * L_km) * 1.15
    ax.set_ylim(-ty, ty)
    ax.set_xlabel("depth into slab z [km]")
    ax.set_ylabel("transverse position x [km]")
    ax.set_title(f"Photon random walk ({outcome}, {n_scatter} scatterings)")

    (line,) = ax.plot([], [], "-", color="tab:blue", lw=1.2, alpha=0.8)
    (dots,) = ax.plot([], [], "o", color="tab:red", ms=4)
    (head,) = ax.plot([], [], "*", color="k", ms=14, label="photon")
    ax.legend(loc="upper right")

    def update(i):
        line.set_data(z[: i + 1], t[: i + 1])
        dots.set_data(z[1: i + 1], t[1: i + 1])
        head.set_data([z[i]], [t[i]])
        return line, dots, head

    anim = FuncAnimation(fig, update, frames=len(z), interval=1000 / fps,
                         blit=True, repeat=False)

    if save:
        writer = "pillow" if save.lower().endswith(".gif") else None
        anim.save(save, writer=writer, fps=fps)
        print(f"Saved animation to {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mfp = 1.0 / (args.density * args.sigma)   # [cm]
    L = args.width * 1e5                      # km -> cm

    print(f"n_e = {args.density:.3e} cm^-3,  sigma = {args.sigma:.3e} cm^2")
    print(f"mean free path = {mfp:.4e} cm = {mfp / 100:.1f} m")
    print(f"slab width     = {L:.4e} cm = {args.width} km  "
          f"({L / mfp:.2f} mean free paths)")

    counts, scatters = run_statistics(rng, args.n_photons, mfp, L, args.max_steps)
    N = args.n_photons
    print(f"\nResults for {N} photons:")
    print(f"  transmitted: {counts['transmitted'] / N:.4f}")
    print(f"  reflected  : {counts['reflected'] / N:.4f}")
    if counts["lost"]:
        print(f"  lost (hit max-steps): {counts['lost'] / N:.4f}")
    print(f"  mean scatterings per photon: {scatters.mean():.2f}")

    if not args.no_animation:
        outcome, n, path = pick_photon_to_animate(
            rng, mfp, L, args.max_steps, args.min_scatters)
        print(f"\nAnimating one photon: {outcome} after {n} scatterings")
        animate_path(path, L, outcome, n, args.fps, args.save, not args.no_show)


if __name__ == "__main__":
    main()
