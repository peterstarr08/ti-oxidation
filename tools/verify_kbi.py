"""Finite-volume weighted coordination numbers for the anatase slab partitions.

Uses the finite-volume Kirkwood-Buff machinery of Kruger & Vlugt,
PRE 97, 051301(R) (2018) -- the INTEGRAL METHOD ONLY, not the KBI itself:

    w(r) = r^2 * T(r) / V                                   [Eq. (7)]

with T(r) the analytic piecewise cuboid expression of Table I, built from

    P(r)      = 4*pi*a*b*c - 2*pi*(ab+ac+bc)*r
                + (8/3)*(a+b+c)*r^2 - r^3                   [Eq. (9)]
    Q_pqs(r)  = 4*pi*p*q*s - 2*pi*p*q*r + (8/3)*s*r^2 - r^3
                - [4*arccos(s/r)*(p+q)*s + 2*s^2]*r
                + (s^4/3 - 2*pi*p*q*s^2)/r
                + (4/3)*(p+q)*(s^2 + 2*r^2)*sqrt(1-s^2/r^2) [Eq. (10)]

    Table I (a >= b >= c):
        0 < r < c                       T = P
        c < r < b                       T = P - Q_abc
        b < r < min(a, sqrt(b^2+c^2))   T = P - Q_abc - Q_cab
        a < r < sqrt(b^2+c^2)           T = P - Q_abc - Q_cab - Q_bca
        (beyond that: no analytic form -> we raise; our r_max never
         reaches it since r_max <= min(z_part, Lx, Ly)/2 < b)

We integrate g(r) DIRECTLY (not h = g - 1):

    N_AB = rho_B * INT_0^{r_max} g_AB(r) * w(r) dr          (RK4)

with rho_B the layer-corrected average local density saved by the
reference g(r) calculation, and V ALWAYS the SLAB volume
Lx * Ly * z_part (never the full cell volume).

GROUND TRUTH: pbc turned off entirely on the cropped partition, a fresh
ASE neighbor list is built, and for each reference atom of type A we
simply count type-B neighbors within r_max and average over all A atoms.
Edge deficits at the open faces are exactly what the analytic w(r)
encodes, so the two numbers should agree.

Reads the pickles in calculated_gr/ produced by calculate_gr_reference.py.
"""
import pickle
from pathlib import Path
from scipy.integrate import quad, trapezoid

import numpy as np
from ase.neighborlist import neighbor_list

DATA_DIR = Path(r"D:\ti-oxidation\temp") / "anatase"
GR_DIR = DATA_DIR / "calculated_gr"
OUT_DIR = DATA_DIR / "kbi_coordination"
OUT_DIR.mkdir(exist_ok=True)

PAIRS = [("Ti", "Ti"), ("Ti", "O"), ("O", "O")]


# ---------------------------------------------------------------------------
# analytic cuboid w(r)  [Kruger & Vlugt, Eqs. (7), (9), (10), Table I]
# ---------------------------------------------------------------------------

def P_func(r, a, b, c):
    """Eq. (9)."""
    return (4 * np.pi * a * b * c
            - 2 * np.pi * (a * b + a * c + b * c) * r
            + (8.0 / 3.0) * (a + b + c) * r ** 2
            - r ** 3)


# def Q_func(r, p, q, s):
#     """Eq. (10) with 'special' side s (the subscript-last side).

#     Q_abc = Q_func(r, a, b, c); Q_cab = Q_func(r, c, a, b);
#     Q_bca = Q_func(r, b, c, a). Vanishes at r = s (continuity of T).
#     Only ever evaluated for r > s.
#     """
#     x = np.clip(s / r, -1.0, 1.0)
#     root = np.sqrt(max(0.0, 1.0 - x * x))
#     return (4 * np.pi * p * q * s
#             - 2 * np.pi * p * q * r
#             + (8.0 / 3.0) * s * r ** 2
#             - r ** 3
#             - (4 * np.arccos(x) * (p + q) * s + 2 * s ** 2) * r
#             + (s ** 4 / 3.0 - 2 * np.pi * p * q * s ** 2) / r
#             + (4.0 / 3.0) * (p + q) * (s ** 2 + 2 * r ** 2) * root)


def Q_func(r, p, q, s):
    """Eq. (10) with 'special' side s. Valid ONLY for r >= s (Table I).
    Raises if called outside its domain; tolerates only ulp-level
    roundoff at the r = s breakpoint, where Q = 0 exactly.
    """
    ratio = s / r
    if ratio > 1.0:
        if ratio > 1.0 + 1e-12:
            raise ValueError(
                f"Q_func evaluated at r={r:.6f} < s={s:.6f}: "
                f"outside its Table I domain -- caller bug"
            )
        ratio = 1.0  # r == s breakpoint hit with float roundoff
    root = np.sqrt(1.0 - ratio * ratio)
    return (4 * np.pi * p * q * s
            - 2 * np.pi * p * q * r
            + (8.0 / 3.0) * s * r ** 2
            - r ** 3
            - (4 * np.arccos(ratio) * (p + q) * s + 2 * s ** 2) * r
            + (s ** 4 / 3.0 - 2 * np.pi * p * q * s ** 2) / r
            + (4.0 / 3.0) * (p + q) * (s ** 2 + 2 * r ** 2) * root)


def T_func(r, a, b, c):
    """Table I, piecewise. Requires a >= b >= c. Raises if r enters the
    region with no analytic expression (never happens for our r_max)."""
    assert a >= b >= c > 0, "sides must be sorted a >= b >= c"
    if r <= 0.0:
        return 4 * np.pi * a * b * c  # tau(0) = V, T(0) = 4*pi*V
    d_bc = np.hypot(b, c)
    if r < c:
        return P_func(r, a, b, c)
    if r < b:
        return P_func(r, a, b, c) - Q_func(r, a, b, c)
    if r < min(a, d_bc):
        return (P_func(r, a, b, c) - Q_func(r, a, b, c)
                - Q_func(r, c, a, b))
    if a < d_bc and r < d_bc:
        return (P_func(r, a, b, c) - Q_func(r, a, b, c)
                - Q_func(r, c, a, b) - Q_func(r, b, c, a))
    raise ValueError(
        f"r={r:.4f} is in the analytically unknown region of T(r) "
        f"(sides a={a:.3f}, b={b:.3f}, c={c:.3f}); r_max should never "
        f"reach here."
    )


def w_func(r, a, b, c, V):
    """w(r) = r^2 T(r) / V, Eq. (7). V is the SLAB volume a*b*c."""
    return r * r * T_func(r, a, b, c) / V


def _self_test_w():
    """Check the implementation against two exact results from the paper:
    (1) the cube closed form, Eq. (11):
        w = 4*pi*r^2 * (1 - (3/2)x + (2/pi)x^2 - x^3/(4*pi)),  x = r/a
    (2) the universal small-r expansion, Eq. (12):
        w = 4*pi*r^2 * (1 - (A/4V) r + O(r^2))
    """
    a = 3.7
    V = a ** 3
    for x in (0.05, 0.3, 0.7, 0.99):
        r = x * a
        w_num = w_func(r, a, a, a, V)
        w_ref = 4 * np.pi * r * r * (
            1 - 1.5 * x + (2 / np.pi) * x ** 2 - x ** 3 / (4 * np.pi)
        )
        assert abs(w_num - w_ref) < 1e-9 * max(1.0, abs(w_ref)), \
            f"cube w(r) self-test failed at x={x}"
    # small-r surface-term expansion on a cuboid
    a, b, c = 30.0, 20.0, 10.0
    V = a * b * c
    A = 2 * (a * b + a * c + b * c)
    r = 1e-3 * c
    w_num = w_func(r, a, b, c, V)
    w_lead = 4 * np.pi * r * r * (1 - A / (4 * V) * r)
    assert abs(w_num - w_lead) < 1e-6 * w_lead, "A/4V expansion self-test failed"
    print("w(r) self-tests passed (cube closed form + A/4V expansion)")


# ---------------------------------------------------------------------------
# RK4 integration of g(r) * w(r) up to r_max
# ---------------------------------------------------------------------------

# def rk4_integrate(f, r_lo, r_hi, step):
#     """Classic RK4 on y'(r) = f(r), y(r_lo) = 0, returning y(r_hi)."""
#     if r_hi <= r_lo:
#         return 0.0
#     n = max(1, int(np.ceil((r_hi - r_lo) / step)))
#     h = (r_hi - r_lo) / n
#     y = 0.0
#     r = r_lo
#     for _ in range(n):
#         k1 = f(r)
#         k2 = f(r + h / 2.0)
#         k3 = f(r + h / 2.0)
#         k4 = f(r + h)
#         y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#         r += h
#     return y


# def weighted_g_integral(r_tab, g_tab, r_max, a, b, c, V):
#     """INT_0^{r_max} g(r) w(r) dr via RK4.

#     g(r) is interpolated linearly from the tabulated (lower-bin-edge)
#     values; the integration is split at the analytic breakpoints of T(r)
#     so RK4 never steps across a kink.
#     """
#     def f(r):
#         g = np.interp(r, r_tab, g_tab)  # clamped at both ends
#         return g * w_func(r, a, b, c, V)

#     dr_tab = r_tab[1] - r_tab[0]
#     step = dr_tab / 2.0

#     d_bc = np.hypot(b, c)
#     breaks = [x for x in (c, b, min(a, d_bc), a, d_bc)
#               if 0.0 < x < r_max]
#     edges = [0.0] + sorted(set(breaks)) + [r_max]

#     total = 0.0
#     for lo, hi in zip(edges[:-1], edges[1:]):
#         total += rk4_integrate(f, lo, hi, step)
#     return total

# def weighted_g_integral(r_tab, g_tab, r_max, a, b, c, V):
#     def f(r):
#         return np.interp(r, r_tab, g_tab) * w_func(r, a, b, c, V)

#     d_bc = np.hypot(b, c)
#     breaks = sorted({x for x in (c, b, min(a, d_bc), a, d_bc)
#                      if 0.0 < x < r_max})
#     total, abserr = quad(f, 0.0, r_max, points=breaks, limit=400)
#     return total

def weighted_g_integral(r_tab, g_tab, r_max, a, b, c, V):
    def f(r):
        return np.interp(r, r_tab, g_tab) * w_func(r, a, b, c, V)

    dr_tab = r_tab[1] - r_tab[0]
    d_bc = np.hypot(b, c)
    breaks = [x for x in (c, b, min(a, d_bc), a, d_bc) if 0.0 < x < r_max]

    # nodes: tabulated points + midpoints + T(r) breakpoints + endpoints
    r = np.union1d(
        np.arange(0.0, r_max, dr_tab / 2.0),
        np.array([0.0, r_max] + breaks),
    )
    r = r[(r >= 0.0) & (r <= r_max)]
    y = np.array([f(x) for x in r])
    return trapezoid(y, x=r)

# ---------------------------------------------------------------------------
# ground truth: no PBC at all, fresh NL, plain neighbor counting
# ---------------------------------------------------------------------------

def ground_truth_avg_count(atoms, r_max, A, B):
    """Average number of B atoms within r_max of each A atom, with PBC
    turned off entirely (open box in x, y, AND z). Fresh ASE NL."""
    system = atoms.copy()
    system.wrap()
    system.set_pbc([False, False, False])
    i, j, d = neighbor_list('ijd', system, r_max)

    symbols = np.array(system.get_chemical_symbols())
    A_index = np.flatnonzero(symbols == A)
    if len(A_index) == 0:
        raise ValueError(f"no {A} atoms in this config")

    count_B = np.zeros(len(system))
    sel = symbols[j] == B
    np.add.at(count_B, i[sel], 1.0)

    return float(count_B[A_index].mean()), len(A_index)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def process_gr_pickle(pkl_path):
    print(f"\n=== {pkl_path.name} ===")
    with open(pkl_path, "rb") as f:
        entry = pickle.load(f)

    atoms = entry["atoms"]
    r_max = float(entry["r_max"])
    z_part = float(entry["z_part"])

    cell = np.diag(np.asarray(atoms.cell))
    Lx, Ly = float(cell[0]), float(cell[1])

    # slab cuboid: xy spans the cell, height is the partition height.
    # V is ALWAYS the slab volume, never the full cell volume.
    a, b, c = sorted((Lx, Ly, z_part), reverse=True)
    V = a * b * c
    print(f"slab sides a={a:.3f} b={b:.3f} c={c:.3f}  V={V:.3f}  "
          f"r_max={r_max:.3f}")

    results = {}
    for A, B in PAIRS:
        pair_name = f"{A}-{B}"
        pair = entry["gr"][pair_name]
        r_tab = np.asarray(pair["r"], dtype=float)
        g_tab = np.asarray(pair["g"], dtype=float)
        rho_B = float(pair["rho_B"])

        integral = weighted_g_integral(r_tab, g_tab, r_max, a, b, c, V)
        n_integral = rho_B * integral

        n_truth, n_A = ground_truth_avg_count(atoms, r_max, A, B)

        rel = (n_integral - n_truth) / n_truth if n_truth != 0 else np.nan
        print(f"  {pair_name:6s}  rho_B={rho_B:.6f}  "
              f"integral={integral:10.4f}  "
              f"N_int={n_integral:8.4f}  N_truth={n_truth:8.4f} "
              f"({n_A} {A} atoms)  rel_diff={rel:+.3%}")

        results[pair_name] = {
            "rho_B": rho_B,
            "g_w_integral": integral,
            "N_integral": n_integral,
            "N_ground_truth": n_truth,
            "n_ref_atoms": n_A,
            "rel_diff": rel,
        }

    out = {
        "source_gr_pickle": pkl_path.name,
        "config_file": entry["config_file"],
        "r_max": r_max,
        "slab_sides_abc": (a, b, c),
        "slab_volume": V,
        "z_part": z_part,
        "z_cut": entry["z_cut"],
        "results": results,
    }
    out_name = f"{pkl_path.stem}_kbi_coord.pkl"
    with open(OUT_DIR / out_name, "wb") as f:
        pickle.dump(out, f)
    print(f"  saved -> kbi_coordination/{out_name}")


if __name__ == "__main__":
    _self_test_w()
    pkls = sorted(GR_DIR.glob("*_gr.pkl"))
    if not pkls:
        raise SystemExit(f"No g(r) pickles found in {GR_DIR}")
    for p in pkls:
        process_gr_pickle(p)
    print("\nAll done.")