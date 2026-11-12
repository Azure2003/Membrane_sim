"""
3D Brownian dynamics of ions near a lipid-bilayer membrane with a cylindrical channel.

Forces (and only these):
  1. Coulomb          - ion-ion, ion-membrane (plain 1/r, water dielectric via Bjerrum length)
  2. van der Waals    - Lennard-Jones, ion-ion and ion-membrane
  3. Solvent drag     - overdamped Langevin: friction + random kicks (D = kT/gamma)

Units: length = nm, energy = kT, time = ns, charge = e.
Overdamped update (Ermak-McCammon):
    r(t+dt) = r(t) + (D/kT) * F * dt + sqrt(2 D dt) * N(0,1)   (per Cartesian component)

Geometry: the membrane lies in the x-y plane at z = 0 (z is the membrane normal).
x and y are periodic; z has reflecting walls at +/- LZ/2.

Parallelism: the force loop is compiled with numba and split across CPU threads (one ion
per task, GIL released). In the live window the simulation also runs in a background
thread so drawing never stalls the physics. Without numba it falls back to NumPy.

Usage: edit the SETTINGS block below, then run:  python ion_membrane_bd_3d.py
"""
import numpy as np

# =============================================================================
# SETTINGS - edit here
# =============================================================================
CHANNEL = False          # False = intact membrane
PORE_RADIUS = 0.9       # nm, radius of the cylindrical region carved out of the bilayer
VOLTAGE = 0.0           # transmembrane bias in kT/e (1 = 25.7 mV); +ve pushes cations toward -z
DT = 2e-4               # time step, ns
SEED = 1                # random-number seed (Brownian kicks)
FRAMES = 300            # animation frames
STEPS_PER_FRAME = 20    # simulation steps per frame
SAVE_PATH = None        # e.g. "out.gif" to render to file; None = live window
CUTAWAY = False          # draw only the back half of the membrane so the pore is visible
VIEW = (20, -60)        # camera (elevation, azimuth) in degrees
THREADS = None          # CPU threads for the force calculation; None = all cores
BACKGROUND_SIM = True   # live window only: run physics in its own thread while the GUI draws

# Ions: (species, x, y, z) in nm. Species: "Na", "K", "Cl".
# Box: x, y in [0, 6] (periodic), z in [-6, 6]. Membrane occupies |z| < ~2.1 (z>0 = top).
# Pore axis is at x = y = 3. Keep ions >= 0.4 nm from each other and from membrane beads.
# The membrane's total charge is printed at start-up; balance it with extra cations
# (a warning is printed if the whole system is charged).
# Set IONS = None to use random placement instead (RANDOM_* below).
IONS = [
    ("Na",  1.5,  1.5,  4.0),
    ("Na",  2.5,  3.5,  3.0),
    ("Na",  4.5,  2.0,  4.8),
    ("K",   3.0,  3.0, -3.0),
    ("K",   4.0,  4.5, -5.0),
    ("Cl",  1.0,  4.0, 3.0),
    ("Cl",  4.0,  2.0, 4.0),
    ("Cl",  2.0,  5.0, 5.0),
]

RANDOM_NA = 30
RANDOM_K = 30
RANDOM_TOP_FRACTION = 0.8   # fraction of random ions starting on the top side
# =============================================================================

# ------------------------------------------------------------------- parameters
LB = 0.70            # Bjerrum length in water at 300 K (nm): Coulomb = LB*qi*qj/r  [kT]   This is e^2k/KT
LX, LY, LZ = 6.0, 6.0, 12.0
BOX = np.array([LX, LY, 0.0])   # periodic wrap in x, y only
CUTOFF_LJ = 1.2      # LJ cutoff (nm)
R_MIN = 0.12         # inner distance clamp to avoid singularities (nm)
MAX_STEP = 0.05      # max displacement per step (nm), numerical safety
Z_CROSS = 2.6        # |z| an ion must reach on the far side to count as a crossing

# species: name -> (charge, sigma [nm], epsilon [kT], D [nm^2/ns], colour) ##Charge, size, energy for Van der Wals, Diffusivity constant
SPECIES = {
    "Na": (+1, 0.28, 0.6, 1.33, "tab:red"),
    "K":  (+1, 0.34, 0.5, 1.96, "tab:orange"),
    "Cl": (-1, 0.36, 0.5, 2.03, "tab:blue"),
}
# membrane bead types: (sigma, epsilon)
HEAD = (0.45, 1.0)
TAIL = (0.40, 0.3)


def min_image(dr):
    """Minimum-image convention in x and y only."""
    dr[..., :2] -= BOX[:2] * np.round(dr[..., :2] / BOX[:2])
    return dr


# ------------------------------------------------------------------- membrane
def build_membrane(channel=True, head_charge=-0.05, lining_charge=-0.15,
                   pore_radius=PORE_RADIUS, spacing=0.50):
    """Bilayer at z=0: per leaflet 1 head layer + 4 tail layers on a square lattice."""
    xs = np.arange(spacing / 2, LX, spacing)
    ys = np.arange(spacing / 2, LY, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    r = np.hypot(X - LX / 2, Y - LY / 2)
    keep = ~(r < pore_radius) if channel else np.ones_like(r, bool)
    lining = (r < pore_radius + 0.4) & channel
    pos, q, sig, eps, kind = [], [], [], [], []
    for side in (+1, -1):
        for layer in range(5):                    # layers 0..3 tails, 4 heads
            z = side * (0.40 + 0.40 * layer)
            is_head = layer == 4
            s, e = HEAD if is_head else TAIL
            for x, y, k, lin in zip(X, Y, keep, lining):
                if not k:
                    continue
                charge = head_charge if is_head else 0.0
                ee = e
                if lin:
                    charge, ee = lining_charge, HEAD[1]
                pos.append((x, y, z)); q.append(charge); sig.append(s); eps.append(ee)
                kind.append("head" if is_head else ("lining" if lin else "tail"))
    return (np.array(pos), np.array(q), np.array(sig), np.array(eps), np.array(kind))


# ----------------------------------------------------------------------- ions
def ions_from_list(ions):
    """Build ion arrays from an explicit list of (species, x, y, z)."""
    names = [i[0] for i in ions]
    for n in names:
        if n not in SPECIES:
            raise ValueError(f"unknown species {n!r}; choose from {list(SPECIES)}")
    pos = np.array([[i[1], i[2], i[3]] for i in ions], float)
    q = np.array([SPECIES[n][0] for n in names], float)
    sig = np.array([SPECIES[n][1] for n in names])
    eps = np.array([SPECIES[n][2] for n in names])
    D = np.array([SPECIES[n][3] for n in names])
    return names, pos, q, sig, eps, D


def random_ions(mem_pos, mem_q, n_na, n_k, top_fraction, rng):
    """Fallback: random non-overlapping placement, electroneutral overall."""
    n_cl = int(round(n_na + n_k + mem_q.sum()))
    names = ["Na"] * n_na + ["K"] * n_k + ["Cl"] * max(n_cl, 0)
    placed = []
    for sp in names:
        for _ in range(10000):
            top = rng.random() < top_fraction
            z = rng.uniform(Z_CROSS, LZ / 2 - 0.2) * (1 if top else -1)
            c = np.array([rng.uniform(0, LX), rng.uniform(0, LY), z])
            ok = np.all(np.linalg.norm(min_image(mem_pos - c), axis=1) > 0.5)
            if placed:
                ok = ok and np.all(np.linalg.norm(min_image(np.array(placed) - c), axis=1) > 0.5)
            if ok:
                placed.append(c)
                break
    return [(sp, *p) for sp, p in zip(names, placed)]


# --------------------------------------------------------- parallel force kernel
try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                               # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def _force_kernel(pos, q, sig, eps, mpos, mq, msig, meps, field,
                      lb, lx, ly, cut2, rmin2):
        n = pos.shape[0]
        m = mpos.shape[0]
        F = np.zeros((n, 3))
        for i in prange(n):                       # each ion handled by one thread
            fx = 0.0; fy = 0.0; fz = 0.0
            xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
            qi = q[i]; si = sig[i]; ei = eps[i]
            for j in range(n + m):
                if j < n:                         # ion-ion
                    if j == i:
                        continue
                    dx = xi - pos[j, 0]; dy = yi - pos[j, 1]; dz = zi - pos[j, 2]
                    qq = qi * q[j]; s = 0.5 * (si + sig[j]); e = np.sqrt(ei * eps[j])
                else:                             # ion-membrane bead
                    k = j - n
                    dx = xi - mpos[k, 0]; dy = yi - mpos[k, 1]; dz = zi - mpos[k, 2]
                    qq = qi * mq[k]; s = 0.5 * (si + msig[k]); e = np.sqrt(ei * meps[k])
                dx -= lx * np.floor(dx / lx + 0.5)
                dy -= ly * np.floor(dy / ly + 0.5)
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < rmin2:
                    r2 = rmin2
                inv = 1.0 / r2
                f = lb * qq * inv * np.sqrt(inv)  # Coulomb
                if r2 < cut2:                     # Lennard-Jones
                    sr6 = (s * s * inv) ** 3
                    f += 24.0 * e * (2.0 * sr6 * sr6 - sr6) * inv
                fx += f * dx; fy += f * dy; fz += f * dz
            F[i, 0] = fx; F[i, 1] = fy; F[i, 2] = fz + qi * field
        return F


# --------------------------------------------------------------------- forces
def pair_force(dr, qq, sigma, eps):
    """Force on particle i from j, dr = r_i - r_j (..., 3). Coulomb + Lennard-Jones."""
    ##Squared distance
    r2 = np.maximum(np.einsum("...k,...k->...", dr, dr), R_MIN ** 2)
    inv_r2 = 1.0 / r2
    f = LB * qq * inv_r2 * np.sqrt(inv_r2)        # Coulomb: (F/r) factor
    sr6 = (sigma ** 2 * inv_r2) ** 3
    f_lj = 24.0 * eps * (2 * sr6 * sr6 - sr6) * inv_r2
    f += np.where(r2 < CUTOFF_LJ ** 2, f_lj, 0.0)
    return f[..., None] * dr


def make_pair_params(q, sig, eps, mem_q, mem_sig, mem_eps):
    """Species-dependent pair parameters never change, so compute them once."""
    ii = (q[:, None] * q[None, :], 0.5 * (sig[:, None] + sig[None, :]),
          np.sqrt(eps[:, None] * eps[None, :]))
    im = (q[:, None] * mem_q[None, :], 0.5 * (sig[:, None] + mem_sig[None, :]),
          np.sqrt(eps[:, None] * mem_eps[None, :]))
    return ii, im


def total_force(pos, q, mem_pos, ii, im, field):
    n = len(pos)
    # ion-ion (minimum image in x, y)
    dr = min_image(pos[:, None, :] - pos[None, :, :])
    f = pair_force(dr, *ii)
    f[np.arange(n), np.arange(n)] = 0.0
    F = f.sum(1)
    # ion-membrane (fixed beads)
    dr = min_image(pos[:, None, :] - mem_pos[None, :, :])
    F += pair_force(dr, *im).sum(1)
    # optional uniform transmembrane field along z (kT per e per nm)
    F[:, 2] += q * field
    return F


# ----------------------------------------------------------------- simulation
class Sim:
    def __init__(self, ions=None, channel=True, voltage=0.0, dt=2e-4, seed=1,
                 n_na=20, n_k=20, top_fraction=0.8):
        """ions: list of (species, x, y, z). If None, random placement."""
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.field = voltage / LZ                 # voltage in kT/e across box
        (self.mem_pos, self.mem_q, self.mem_sig,
         self.mem_eps, self.mem_kind) = build_membrane(channel)
        print(f"membrane: {len(self.mem_pos)} beads, total charge {self.mem_q.sum():+.1f} e")
        if ions is None:
            ions = random_ions(self.mem_pos, self.mem_q, n_na, n_k, top_fraction, self.rng)
        (self.names, self.pos, self.q, self.sig,
         self.eps, self.D) = ions_from_list(ions)
        self.pos[:, :2] %= [LX, LY]
        self.pp_ii, self.pp_im = make_pair_params(self.q, self.sig, self.eps,
                                                  self.mem_q, self.mem_sig, self.mem_eps)
        net = self.q.sum() + self.mem_q.sum()
        print(f"{len(self.names)} ions, system net charge {net:+.1f} e")
        if abs(net) > 1e-6:
            print("warning: system is not electroneutral")
        self.side = np.sign(self.pos[:, 2]).astype(int)
        self.cross = {s: 0 for s in SPECIES}      # net top->bottom crossings
        self.t = 0.0

    def step(self):
        if HAVE_NUMBA:
            F = _force_kernel(self.pos, self.q, self.sig, self.eps, self.mem_pos, self.mem_q,
                              self.mem_sig, self.mem_eps, self.field, LB, LX, LY,
                              CUTOFF_LJ ** 2, R_MIN ** 2)
        else:
            F = total_force(self.pos, self.q, self.mem_pos, self.pp_ii, self.pp_im, self.field)
        D = self.D[:, None]
        disp = D * F * self.dt + np.sqrt(2 * D * self.dt) * self.rng.standard_normal(self.pos.shape)
        mag = np.linalg.norm(disp, axis=1, keepdims=True)
        disp *= np.minimum(1.0, MAX_STEP / np.maximum(mag, 1e-12))
        self.pos += disp
        self.pos[:, :2] %= [LX, LY]
        # reflecting walls in z
        lo, hi = -LZ / 2, LZ / 2
        z = self.pos[:, 2]
        z = np.where(z > hi, 2 * hi - z, z)
        z = np.where(z < lo, 2 * lo - z, z)
        self.pos[:, 2] = z
        # crossing bookkeeping (hysteresis: must reach |z| > Z_CROSS on the other side)
        new = np.where(z > Z_CROSS, 1, np.where(z < -Z_CROSS, -1, self.side))
        for i in np.where(new != self.side)[0]:
            self.cross[self.names[i]] += int(self.side[i] - new[i]) // 2  # +1 = downward
        self.side = new
        self.t += self.dt

    def run(self, n):
        for _ in range(n):
            self.step()


# ---------------------------------------------------------------------- plots
def make_figure(sim):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.set_zlim(-LZ / 2, LZ / 2)
    ax.set_box_aspect((LX, LY, LZ))
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    ax.view_init(*VIEW)

    colours = {"head": "#e6b800", "tail": "#bbbbbb", "lining": "#2ca02c"}
    m = np.ones(len(sim.mem_pos), bool)
    if CUTAWAY:
        m = sim.mem_pos[:, 1] >= LY / 2
    p = sim.mem_pos[m]
    mc = [colours[k] for k in sim.mem_kind[m]]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=mc, s=34, alpha=0.55, edgecolors="none",
               depthshade=False)

    ic = [SPECIES[n][4] for n in sim.names]
    size = 220 * (sim.sig / 0.3) ** 2 / 3
    ions = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], sim.pos[:, 2], c=ic, s=size,
                      edgecolors="k", linewidths=0.4, depthshade=False)
    txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.8, lw=0))
    h = [Line2D([], [], marker="o", ls="", color=SPECIES[s][4], label=s) for s in SPECIES]
    h += [Line2D([], [], marker="o", ls="", color=colours["head"], label="lipid head"),
          Line2D([], [], marker="o", ls="", color=colours["tail"], label="lipid tail"),
          Line2D([], [], marker="o", ls="", color=colours["lining"], label="pore lining")]
    ax.legend(handles=h, loc="lower right", fontsize=7, ncol=2)
    return fig, ax, ions, txt


def update_text(sim):
    c = sim.cross
    return (f"t = {sim.t:6.3f} ns\nnet crossings (top->bottom)\n"
            f"Na {c['Na']:+d}  K {c['K']:+d}  Cl {c['Cl']:+d}")


class SimThread:
    """Runs the physics in a worker thread; the GUI reads the latest snapshot."""
    def __init__(self, sim, steps_per_frame):
        import threading
        self.sim, self.n = sim, steps_per_frame
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.snap = self._snapshot()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _snapshot(self):
        s = self.sim
        return s.pos.copy(), s.t, dict(s.cross)

    def _loop(self):
        while not self.stop.is_set():
            self.sim.run(self.n)                  # numba releases the GIL in here
            snap = self._snapshot()
            with self.lock:
                self.snap = snap

    def latest(self):
        with self.lock:
            return self.snap

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def main():
    import matplotlib
    if SAVE_PATH:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if HAVE_NUMBA:
        if THREADS:
            numba.set_num_threads(THREADS)
        print(f"numba force kernel on {numba.get_num_threads()} thread(s)")
    else:
        print("numba not found - using single-threaded NumPy (pip install numba)")

    sim = Sim(IONS, CHANNEL, VOLTAGE, DT, SEED, RANDOM_NA, RANDOM_K, RANDOM_TOP_FRACTION)
    fig, ax, ions, txt = make_figure(sim)

    worker = None
    if BACKGROUND_SIM and not SAVE_PATH:          # saved movies stay deterministic/synchronous
        worker = SimThread(sim, STEPS_PER_FRAME)
        worker.start()
        fig.canvas.mpl_connect("close_event", lambda _e: worker.close())

    def frame(_):
        if worker:
            pos, t, cross = worker.latest()
            view = type("V", (), {"t": t, "cross": cross})
        else:
            sim.run(STEPS_PER_FRAME)
            pos, view = sim.pos, sim
        ions._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        txt.set_text(update_text(view))
        return ions, txt

    anim = FuncAnimation(fig, frame, frames=FRAMES if SAVE_PATH else None,
                         interval=30, blit=False, cache_frame_data=False)
    if SAVE_PATH:
        anim.save(SAVE_PATH, fps=25)
        print("saved", SAVE_PATH, "|", update_text(sim).replace("\n", " | "))
    else:
        plt.show()
        if worker:
            worker.close()


if __name__ == "__main__":
    main()