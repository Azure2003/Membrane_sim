"""
3D underdamped (inertial) Langevin dynamics of ions near a lipid-bilayer membrane with a
cylindrical channel. No explicit water; momentum is kept.

Each ion has a mass and a velocity and obeys
    m dv/dt = F - m*gamma*v + sqrt(2 m gamma kT) * xi(t)
with gamma = kT / (m D) so the long-time diffusion constant is still the tabulated D.
Because ions carry momentum they now collide: Lennard-Jones contact with other ions and with
membrane beads transfers momentum, ions overshoot, rebound off the membrane, and keep a
Maxwell-Boltzmann velocity distribution (not instantaneous overdamped relaxation).

Forces:
  1. Coulomb        - ion-ion, ion-membrane; water dielectric via Bjerrum length LB, and water/ion
                      SHIELDING via a Debye (Yukawa) screening factor exp(-r/lambda_D) with a
                      shifted-force cutoff COULOMB_CUTOFF (interactions vanish beyond it)
  2. Lennard-Jones  - ion-ion and ion-membrane (contact / excluded volume)
  3. Solvent        - implicit: friction -m*gamma*v plus random kicks (Langevin thermostat)
  4. Optional uniform transmembrane field

Integrator: BAOAB (Leimkuhler-Matthews), which stays accurate even when gamma*dt is O(1).
Units: length = nm, energy = kT (300 K), time = ns, charge = e.
Mass unit = kT/(nm/ns)^2 = 2.49e6 amu.

Geometry: membrane in the x-y plane at z = 0. x, y periodic; z has reflecting walls at +/- LZ/2
(the normal velocity component is reversed on reflection).

Usage: edit SETTINGS, then run:  python ion_membrane_langevin_3d.py
(numba is used for the force loop if installed; otherwise NumPy is used.)
"""
import numpy as np

# =============================================================================
# SETTINGS - edit here
# =============================================================================
CHANNEL = False          # False = intact membrane
PORE_RADIUS = 0.9        # nm
VOLTAGE = 0.0            # kT/e across the box (1 = 25.7 mV); +ve pushes cations toward -z
DT = 1e-5                # ns (10 fs). Ion velocity relaxation time is ~12 fs (see GAMMA_SCALE)
SEED = 1
FRAMES = 300
STEPS_PER_FRAME = 100    # 100 steps = 1 ps per frame
SAVE_PATH = None         # e.g. "out.gif"; None = live window
CUTAWAY = False
VIEW = (20, -60)
THREADS = None
BACKGROUND_SIM = True

# Friction: gamma_i = GAMMA_SCALE * kT / (m_i * D_i).  1.0 reproduces the tabulated diffusion
# constant. Smaller values = less friction = more ballistic motion / bouncier collisions.
GAMMA_SCALE = 1.0

# Water shielding of electrostatics (applies to ion-ion AND ion-membrane charges):
#   screened Coulomb  U = LB*qi*qj*exp(-r/DEBYE_LENGTH)/r  (Debye-Hueckel / Yukawa),
#   truncated at COULOMB_CUTOFF with a shifted force so force and energy go smoothly to zero.
# Debye length ~ 0.304/sqrt(c[M]) nm: 0.15 M -> 0.78 nm, 0.5 M -> 0.43 nm, 0.05 M -> 1.4 nm.
# Set DEBYE_LENGTH = None for unscreened (but still cut off) Coulomb.
DEBYE_LENGTH = 0.8       # nm
COULOMB_CUTOFF = 2.4     # nm (~3 Debye lengths; must be < LX/2 = 3.0 nm)

# Ions: (species, x, y, z) in nm. Species: "Na", "K", "Cl". Set IONS = None for random placement.
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
RANDOM_TOP_FRACTION = 0.8
# =============================================================================

# ------------------------------------------------------------------- parameters
LB = 0.70            # Bjerrum length in water at 300 K (nm): Coulomb = LB*qi*qj/r  [kT]
LX, LY, LZ = 6.0, 6.0, 12.0
BOX = np.array([LX, LY, 0.0])
CUTOFF_LJ = 1.2
KAPPA = 1.0 / DEBYE_LENGTH if DEBYE_LENGTH else 0.0     # inverse Debye length (1/nm)
R_MIN = 0.12
Z_CROSS = 2.6
if COULOMB_CUTOFF >= min(LX, LY) / 2:
    raise ValueError("COULOMB_CUTOFF must be < half the periodic box width (minimum image)")
T_KELVIN = 300.0

KT_J = 1.380649e-23 * T_KELVIN
AMU = 1.66054e-27
MASS_UNIT_AMU = KT_J / AMU       # kT/(nm/ns)^2 in amu (~2.49e6)

# species: name -> (charge, sigma [nm], epsilon [kT], D [nm^2/ns], mass [amu], colour)
SPECIES = {
    "Na": (+1, 0.28, 0.6, 1.33, 22.99, "tab:red"),
    "K":  (+1, 0.34, 0.5, 1.96, 39.10, "tab:orange"),
    "Cl": (-1, 0.36, 0.5, 2.03, 35.45, "tab:blue"),
}
HEAD = (0.45, 1.0)
TAIL = (0.40, 0.3)


def min_image(dr):
    dr[..., :2] -= BOX[:2] * np.round(dr[..., :2] / BOX[:2])
    return dr


# ------------------------------------------------------------------- membrane
def build_membrane(channel=True, head_charge=-0.05, lining_charge=-0.15,
                   pore_radius=PORE_RADIUS, spacing=0.50):
    """Frozen bilayer at z=0: per leaflet 1 head layer + 4 tail layers on a square lattice."""
    xs = np.arange(spacing / 2, LX, spacing)
    ys = np.arange(spacing / 2, LY, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    r = np.hypot(X - LX / 2, Y - LY / 2)
    keep = ~(r < pore_radius) if channel else np.ones_like(r, bool)
    lining = (r < pore_radius + 0.4) & channel
    pos, q, sig, eps, kind = [], [], [], [], []
    for side in (+1, -1):
        for layer in range(5):
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
    names = [i[0] for i in ions]
    for n in names:
        if n not in SPECIES:
            raise ValueError(f"unknown species {n!r}; choose from {list(SPECIES)}")
    pos = np.array([[i[1], i[2], i[3]] for i in ions], float)
    q = np.array([SPECIES[n][0] for n in names], float)
    sig = np.array([SPECIES[n][1] for n in names])
    eps = np.array([SPECIES[n][2] for n in names])
    D = np.array([SPECIES[n][3] for n in names])
    mass = np.array([SPECIES[n][4] for n in names]) / MASS_UNIT_AMU
    return names, pos, q, sig, eps, D, mass


def random_ions(mem_pos, mem_q, n_na, n_k, top_fraction, rng):
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
                      lb, lx, ly, cut2, rmin2, kappa, ccut2):
        n = pos.shape[0]
        rc = np.sqrt(ccut2)
        fshift = np.exp(-kappa * rc) * (1.0 / ccut2 + kappa / rc)   # force at the cutoff
        m = mpos.shape[0]
        F = np.zeros((n, 3))
        for i in prange(n):
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
                f = 0.0
                if r2 < ccut2 and qq != 0.0:      # screened, shifted-force Coulomb
                    r = np.sqrt(r2)
                    f = lb * qq * (np.exp(-kappa * r) * (inv + kappa / r) - fshift) / r
                if r2 < cut2:                     # Lennard-Jones
                    sr6 = (s * s * inv) ** 3
                    f += 24.0 * e * (2.0 * sr6 * sr6 - sr6) * inv
                fx += f * dx; fy += f * dy; fz += f * dz
            F[i, 0] = fx; F[i, 1] = fy; F[i, 2] = fz - qi * field   # +V pushes cations to -z
        return F


# --------------------------------------------------------------- NumPy fallback
def pair_force(dr, qq, sigma, eps):
    r2 = np.maximum(np.einsum("...k,...k->...", dr, dr), R_MIN ** 2)
    inv_r2 = 1.0 / r2
    r = np.sqrt(r2)
    rc = COULOMB_CUTOFF
    fshift = np.exp(-KAPPA * rc) * (1.0 / rc ** 2 + KAPPA / rc)
    f_c = LB * qq * (np.exp(-KAPPA * r) * (inv_r2 + KAPPA / r) - fshift) / r
    f = np.where(r2 < rc ** 2, f_c, 0.0)
    sr6 = (sigma ** 2 * inv_r2) ** 3
    f_lj = 24.0 * eps * (2 * sr6 * sr6 - sr6) * inv_r2
    f += np.where(r2 < CUTOFF_LJ ** 2, f_lj, 0.0)
    return f[..., None] * dr


def make_pair_params(q, sig, eps, mem_q, mem_sig, mem_eps):
    ii = (q[:, None] * q[None, :], 0.5 * (sig[:, None] + sig[None, :]),
          np.sqrt(eps[:, None] * eps[None, :]))
    im = (q[:, None] * mem_q[None, :], 0.5 * (sig[:, None] + mem_sig[None, :]),
          np.sqrt(eps[:, None] * mem_eps[None, :]))
    return ii, im


def total_force(pos, q, mem_pos, ii, im, field):
    n = len(pos)
    dr = min_image(pos[:, None, :] - pos[None, :, :])
    f = pair_force(dr, *ii)
    f[np.arange(n), np.arange(n)] = 0.0
    F = f.sum(1)
    dr = min_image(pos[:, None, :] - mem_pos[None, :, :])
    F += pair_force(dr, *im).sum(1)
    F[:, 2] -= q * field
    return F


# ----------------------------------------------------------------- simulation
class Sim:
    def __init__(self, ions=None, channel=True, voltage=0.0, dt=1e-5, seed=1,
                 n_na=20, n_k=20, top_fraction=0.8):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.field = voltage / LZ
        (self.mem_pos, self.mem_q, self.mem_sig,
         self.mem_eps, self.mem_kind) = build_membrane(channel)
        print(f"membrane: {len(self.mem_pos)} beads, total charge {self.mem_q.sum():+.1f} e")
        if ions is None:
            ions = random_ions(self.mem_pos, self.mem_q, n_na, n_k, top_fraction, self.rng)
        (self.names, self.pos, self.q, self.sig,
         self.eps, self.D, self.mass) = ions_from_list(ions)
        self.pos[:, :2] %= [LX, LY]
        self.pp_ii, self.pp_im = make_pair_params(self.q, self.sig, self.eps,
                                                  self.mem_q, self.mem_sig, self.mem_eps)
        net = self.q.sum() + self.mem_q.sum()
        print(f"{len(self.names)} ions, system net charge {net:+.1f} e")
        if abs(net) > 1e-6:
            print("warning: system is not electroneutral")

        # friction from the Einstein relation D = kT/(m*gamma)  (kT = 1)
        self.gamma = GAMMA_SCALE / (self.mass * self.D)
        print("velocity relaxation time 1/gamma: " +
              ", ".join(f"{n} {1e6 / self.gamma[self.names.index(n)]:.0f} fs"
                        for n in dict.fromkeys(self.names)))
        self.inv_m = (1.0 / self.mass)[:, None]
        self.vth = (1.0 / np.sqrt(self.mass))[:, None]          # sqrt(kT/m)
        self.vel = self.rng.standard_normal(self.pos.shape) * self.vth   # Maxwell-Boltzmann
        self.vmax = 8.0 * self.vth                               # safety clamp only
        self.c1 = np.exp(-self.gamma * dt)[:, None]               # OU decay per step
        self.c2 = np.sqrt(1.0 - self.c1 ** 2) * self.vth          # OU noise amplitude

        self.F = self.forces()
        self.side = np.sign(self.pos[:, 2]).astype(int)
        self.cross = {s: 0 for s in SPECIES}
        self.t = 0.0

    def forces(self):
        if HAVE_NUMBA:
            return _force_kernel(self.pos, self.q, self.sig, self.eps, self.mem_pos, self.mem_q,
                                 self.mem_sig, self.mem_eps, self.field, LB, LX, LY,
                                 CUTOFF_LJ ** 2, R_MIN ** 2, KAPPA, COULOMB_CUTOFF ** 2)
        return total_force(self.pos, self.q, self.mem_pos, self.pp_ii, self.pp_im, self.field)

    def _drift(self, h):
        """Move positions by h*v; periodic in x,y; elastic reflection (v_z reversed) in z."""
        self.pos += h * self.vel
        self.pos[:, :2] %= [LX, LY]
        hi, lo = LZ / 2, -LZ / 2
        z, vz = self.pos[:, 2], self.vel[:, 2]
        m = z > hi
        z[m] = 2 * hi - z[m]; vz[m] = -vz[m]
        m = z < lo
        z[m] = 2 * lo - z[m]; vz[m] = -vz[m]

    def step(self):
        """BAOAB: kick, drift, Ornstein-Uhlenbeck (friction + noise), drift, kick."""
        dt = self.dt
        self.vel += 0.5 * dt * self.F * self.inv_m                       # B
        self._drift(0.5 * dt)                                           # A
        self.vel = self.c1 * self.vel + self.c2 * self.rng.standard_normal(self.vel.shape)  # O
        self._drift(0.5 * dt)                                           # A
        self.F = self.forces()
        self.vel += 0.5 * dt * self.F * self.inv_m                       # B
        np.clip(self.vel, -self.vmax, self.vmax, out=self.vel)

        z = self.pos[:, 2]                      # crossing bookkeeping (hysteresis)
        new = np.where(z > Z_CROSS, 1, np.where(z < -Z_CROSS, -1, self.side))
        for i in np.where(new != self.side)[0]:
            self.cross[self.names[i]] += int(self.side[i] - new[i]) // 2
        self.side = new
        self.t += dt

    def run(self, n):
        for _ in range(n):
            self.step()

    def temperature(self):
        """Instantaneous kinetic temperature (K); fluctuates strongly for a few ions."""
        return T_KELVIN * (self.mass[:, None] * self.vel ** 2).sum() / (3 * len(self.mass))


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
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[colours[k] for k in sim.mem_kind[m]],
               s=34, alpha=0.55, edgecolors="none", depthshade=False)

    ic = [SPECIES[n][5] for n in sim.names]
    size = 220 * (sim.sig / 0.3) ** 2 / 3
    ions = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], sim.pos[:, 2], c=ic, s=size,
                      edgecolors="k", linewidths=0.4, depthshade=False)
    txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.8, lw=0))
    h = [Line2D([], [], marker="o", ls="", color=SPECIES[s][5], label=s) for s in SPECIES]
    h += [Line2D([], [], marker="o", ls="", color=colours["head"], label="lipid head"),
          Line2D([], [], marker="o", ls="", color=colours["tail"], label="lipid tail"),
          Line2D([], [], marker="o", ls="", color=colours["lining"], label="pore lining")]
    ax.legend(handles=h, loc="lower right", fontsize=7, ncol=2)
    return fig, ax, ions, txt


def update_text(t, cross, temp):
    return (f"t = {t * 1000:7.2f} ps   T_kin = {temp:5.0f} K\nnet crossings (top->bottom)\n"
            f"Na {cross['Na']:+d}  K {cross['K']:+d}  Cl {cross['Cl']:+d}")


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
        return s.pos.copy(), s.t, dict(s.cross), s.temperature()

    def _loop(self):
        while not self.stop.is_set():
            self.sim.run(self.n)
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
    if BACKGROUND_SIM and not SAVE_PATH:
        worker = SimThread(sim, STEPS_PER_FRAME)
        worker.start()
        fig.canvas.mpl_connect("close_event", lambda _e: worker.close())

    def frame(_):
        if worker:
            pos, t, cross, temp = worker.latest()
        else:
            sim.run(STEPS_PER_FRAME)
            pos, t, cross, temp = sim.pos, sim.t, sim.cross, sim.temperature()
        ions._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        txt.set_text(update_text(t, cross, temp))
        return ions, txt

    anim = FuncAnimation(fig, frame, frames=FRAMES if SAVE_PATH else None,
                         interval=30, blit=False, cache_frame_data=False)
    if SAVE_PATH:
        anim.save(SAVE_PATH, fps=25)
        print("saved", SAVE_PATH, "|", update_text(sim.t, sim.cross, sim.temperature()).replace("\n", " | "))
    else:
        plt.show()
        if worker:
            worker.close()


if __name__ == "__main__":
    main()