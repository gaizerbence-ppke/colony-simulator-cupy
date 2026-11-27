"""Step 2 validation: nutrient diffusion through agar and colony, and Michaelis-Menten uptake.

Run from anywhere once the package is installed (`pip install -e .`):
    python validation/step2_nutrient_transport.py
Prints numbers to inspect and writes figures to validation_outputs/step2/:

1. vertical_cosine_mode.png   vertical diffusion in the agar against the exact decay of a cosine
                              mode, for Crank-Nicolson (theta = 0.5) and backward Euler (theta = 1)
2. horizontal_point_source.png horizontal spread of a filled column against the exact solution,
                              plus mass conservation on a circular dish
3. uptake_mass_balance.png    two strains eating from a small, well-mixed plate against the
                              Michaelis-Menten ODE, and the mass balance error
4. region_conservation.png    nutrient stays in its own ECM region when regions are close together
5. thin_layer_theta.png       a realistic colony (thin layers, dense cells): Crank-Nicolson against
                              backward Euler against a fine time-step reference
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import colonysimulator  # noqa: F401  (installed with `pip install -e .`)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colonysimulator import CellStrain, ColonyState, NutrientField, PlateGeometry
from colonysimulator.nutrient import label_regions, masked_gaussian_spread

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation_outputs" / "step2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLUCOSE_MOLECULAR_WEIGHT = 180e-3   # g/mmol
CELL_DRY_WEIGHT = 1e-12             # g
AGAR_DIFFUSION = 2.4                # mm^2/h, glucose in agar (6.7e-6 cm^2/s)
ECM_DIFFUSION = 0.72                # mm^2/h, glucose in the colony (2e-6 cm^2/s, literature 1-3e-6)
KM = 1.0 * GLUCOSE_MOLECULAR_WEIGHT * 1e-3   # g/ml, 1 mM (placeholder)
M_MAX = 15 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT          # g/cell/h
MAINTENANCE = 0.3 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT   # g/cell/h
GLUCOSE = 0.1e-3                    # g/ml in the agar at t = 0


def strain(name: str, m_max: float = M_MAX, km: float = KM) -> CellStrain:
    return CellStrain(name, division_time=1.75, m_max=m_max, km=km,
                      nutrient_consumption=MAINTENANCE, volume=42e-9)


def empty_colony(geometry: PlateGeometry, strains: list[CellStrain]) -> ColonyState:
    return ColonyState(geometry, strains, bracket_count=2, pinned=False, chunk_layers=1)


# ---- 1. vertical diffusion against an exact solution ------------------------------------

def vertical_cosine_mode() -> None:
    print("--- 1. vertical diffusion of a cosine mode in the agar")
    h, depth = 0.1, 4.0
    geometry = PlateGeometry.rectangular(h, h, depth, h, layer_height=0.005, max_colony_height=0.01)
    colony = empty_colony(geometry, [strain("s")])
    z = (np.arange(geometry.agar_depth) + 0.5) * h
    wavenumber = np.pi / (geometry.agar_depth * h)
    decay_rate = AGAR_DIFFUSION * wavenumber ** 2

    def initial() -> np.ndarray:
        return 1.0 + 0.5 * np.cos(wavenumber * z)

    def exact(t: float) -> np.ndarray:
        return 1.0 + 0.5 * np.cos(wavenumber * z) * np.exp(-decay_rate * t)

    snapshots = (0.0, 0.2, 0.5, 1.0)
    runs = {}
    for label, theta, dt in (("theta 0.5, dt 0.1 h", 0.5, 0.1), ("theta 1, dt 0.1 h", 1.0, 0.1),
                             ("theta 1, dt 0.01 h", 1.0, 0.01)):
        field = NutrientField(geometry, AGAR_DIFFUSION, ECM_DIFFUSION, theta=theta)
        field.agar[:, 0, 0] = cp.asarray(initial(), dtype=cp.float32)
        mass0 = field.total_mass()
        steps = int(round(1.0 / dt))
        times, amplitudes, profiles = [0.0], [0.5], {0.0: initial()}
        for k in range(1, steps + 1):
            field.step(colony, dt)
            t = k * dt
            profile = cp.asnumpy(field.agar[:, 0, 0]).astype(np.float64)
            times.append(t)
            # Project on the cosine mode to get its amplitude.
            amplitudes.append(2 * np.mean((profile - profile.mean()) * np.cos(wavenumber * z)))
            for s in snapshots:
                if abs(t - s) < dt / 2:
                    profiles[s] = profile
        runs[label] = (np.array(times), np.array(amplitudes), profiles)
        error = np.abs(profiles[1.0] - exact(1.0)).max()
        print(f"{label}: max error at 1 h {error:.2e} (mode amplitude {0.5 * np.exp(-decay_rate):.3f}), "
              f"relative mass change {(field.total_mass() - mass0) / mass0:.1e}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in snapshots:
        axes[0].plot(z, exact(s), color="0.6", lw=3)
        axes[0].plot(z, runs["theta 0.5, dt 0.1 h"][2][s], "o", ms=3, label=f"t = {s} h")
    axes[0].set_xlabel("height above dish bottom (mm)")
    axes[0].set_ylabel("concentration (relative)")
    axes[0].set_title("profiles, theta 0.5, dt 0.1 h (grey: exact)")
    axes[0].legend()
    t_fine = np.linspace(0, 1, 200)
    axes[1].semilogy(t_fine, 0.5 * np.exp(-decay_rate * t_fine), color="0.6", lw=3, label="exact")
    for label, (times, amplitudes, _) in runs.items():
        axes[1].semilogy(times, amplitudes, marker=".", label=label)
    axes[1].set_xlabel("time (h)")
    axes[1].set_ylabel("cosine mode amplitude")
    axes[1].set_title("decay of the mode")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vertical_cosine_mode.png", dpi=120)
    plt.close(figure)


# ---- 2. horizontal diffusion against an exact solution ----------------------------------

def horizontal_point_source() -> None:
    print("--- 2. horizontal spread of one filled column")
    h, size, dt = 0.1, 16.0, 0.1
    geometry = PlateGeometry.rectangular(size, size, 1.0, h, 0.005, 0.01)
    colony = empty_colony(geometry, [strain("s")])
    field = NutrientField(geometry, AGAR_DIFFUSION, ECM_DIFFUSION)
    ci, cj = geometry.centre_voxel()
    field.agar[:, ci, cj] = 1.0
    x = (np.arange(geometry.length) - ci) * h

    def exact(t: float) -> np.ndarray:
        # A square of side h filled at t = 0, spread in 2D; this is the value through its centre.
        spread = np.sqrt(4 * AGAR_DIFFUSION * t)
        f = 0.5 * (erf((x + h / 2) / spread) - erf((x - h / 2) / spread))
        return f * 0.5 * (erf((h / 2) / spread) - erf((-h / 2) / spread))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mass0 = field.total_mass()
    for k in range(1, 6):
        field.step(colony, dt)
        if k in (1, 3, 5):
            t = k * dt
            profile = cp.asnumpy(field.agar[geometry.agar_depth // 2, :, cj])
            axes[0].plot(x, exact(t), color="0.6", lw=3)
            axes[0].plot(x, profile, ".", label=f"t = {t:.1f} h")
            peak = exact(t).max()
            print(f"t = {t:.1f} h: max abs error {np.abs(profile - exact(t)).max() / peak:.2e} of the peak")
    print(f"rectangular plate relative mass change: {(field.total_mass() - mass0) / mass0:.1e}")
    axes[0].set_xlim(-4, 4)
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("concentration (relative)")
    axes[0].set_title("profile through the source (grey: exact)")
    axes[0].legend()

    # A source next to the wall of a circular dish: nothing may leak through the wall.
    dish = PlateGeometry.circular(10.0, 1.0, h, 0.005, 0.01)
    dish_colony = empty_colony(dish, [strain("s")])
    dish_field = NutrientField(dish, AGAR_DIFFUSION, ECM_DIFFUSION)
    i, j = dish.to_voxel(5.0, 0.3)
    dish_field.agar[:, i, j] = 1.0
    mass0 = dish_field.total_mass()
    times, change = [0.0], [0.0]
    for k in range(1, 41):
        dish_field.step(dish_colony, dt)
        times.append(k * dt)
        change.append((dish_field.total_mass() - mass0) / mass0)
    print(f"circular dish, source at the wall: max relative mass change {np.abs(change).max():.1e} over 4 h")
    uniform_field = NutrientField(dish, AGAR_DIFFUSION, ECM_DIFFUSION, initial_concentration=1.0)
    for _ in range(10):
        uniform_field.step(dish_colony, dt)
    inside = cp.asnumpy(uniform_field.agar)[:, cp.asnumpy(dish.mask)]
    print(f"circular dish, uniform field after 10 steps: max deviation from uniform {np.abs(inside - 1).max():.1e}")
    axes[1].plot(times, change, marker=".")
    axes[1].set_xlabel("time (h)")
    axes[1].set_ylabel("relative change of total mass")
    axes[1].set_title("source at the wall of a 10 mm dish")
    inset = axes[1].inset_axes([0.55, 0.5, 0.4, 0.45])
    inset.imshow(cp.asnumpy(dish_field.agar[0]).T, origin="lower")
    inset.set_xticks([])
    inset.set_yticks([])
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "horizontal_point_source.png", dpi=120)
    plt.close(figure)


# ---- 3. uptake by two strains and the mass balance --------------------------------------

def uptake_mass_balance() -> None:
    print("--- 3. Michaelis-Menten uptake by two strains on a small plate")
    h, dt, hours = 0.1, 0.1, 12.0
    geometry = PlateGeometry.rectangular(2.0, 2.0, 1.0, h, 0.005, 0.01)
    strains = [strain("low affinity", M_MAX, KM), strain("high affinity", 0.5 * M_MAX, 0.1 * KM)]
    colony = empty_colony(geometry, strains)
    cells_per_voxel = (30.0, 20.0)
    for s, cells in enumerate(cells_per_voxel):
        for i in range(geometry.length):
            for j in range(geometry.width):
                colony.seed_voxel(s, i, j, cells)
    field = NutrientField(geometry, AGAR_DIFFUSION, ECM_DIFFUSION, initial_concentration=GLUCOSE)
    mass0 = field.total_mass()
    volume_ml = float(field.row_volume_ml[:field.agar_depth + 1].sum()) * geometry.length * geometry.width

    totals = np.array(cells_per_voxel) * geometry.length * geometry.width
    m_max = np.array([s.m_max for s in strains])
    km = np.array([s.km for s in strains])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        c = max(y[0], 0.0) / volume_ml
        rates = totals * m_max * c / (km + c)
        return np.concatenate([[-rates.sum()], rates])

    steps = int(round(hours / dt))
    times = np.arange(steps + 1) * dt
    reference = solve_ivp(rhs, (0, hours), [mass0, 0.0, 0.0], t_eval=times, rtol=1e-10, atol=1e-20)

    mass = [mass0]
    eaten = [np.zeros(2)]
    for _ in range(steps):
        uptake = field.step(colony, dt)
        eaten.append(eaten[-1] + cp.asnumpy(uptake.sum(axis=(0, 2, 3), dtype=cp.float64)))
        mass.append(field.total_mass())
    mass = np.array(mass)
    eaten = np.array(eaten)
    balance = (mass + eaten.sum(axis=1) - mass0) / mass0
    print(f"after {hours:.0f} h: {mass[-1] / mass0:.3f} of the nutrient left (ODE {reference.y[0, -1] / mass0:.3f}); "
          f"eaten per strain {eaten[-1] / mass0} (ODE {reference.y[1:, -1] / mass0})")
    print(f"mass balance: max |left + eaten - initial| / initial = {np.abs(balance).max():.1e}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(reference.t, reference.y[0] / mass0, color="0.6", lw=3, label="well-mixed ODE")
    axes[0].plot(times, mass / mass0, ".", ms=3, label="nutrient left")
    for s, name in enumerate(s.name for s in strains):
        axes[0].plot(reference.t, reference.y[1 + s] / mass0, color="0.6", lw=3)
        axes[0].plot(times, eaten[:, s] / mass0, ".", ms=3, label=f"eaten by {name}")
    axes[0].set_xlabel("time (h)")
    axes[0].set_ylabel("fraction of the initial nutrient")
    axes[0].legend()
    axes[0].set_title("simulation (dots) vs ODE (grey)")
    axes[1].plot(times, balance, marker=".")
    axes[1].set_xlabel("time (h)")
    axes[1].set_ylabel("(left + eaten - initial) / initial")
    axes[1].set_title("mass balance")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "uptake_mass_balance.png", dpi=120)
    plt.close(figure)


# ---- 4. mass stays within each connected ECM region -------------------------------------

def region_conservation() -> None:
    print("--- 4. per-region conservation of the horizontal spread")
    n = 60
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    medium = np.zeros((1, n, n), dtype=bool)
    medium[0] |= (ii - 20) ** 2 + (jj - 20) ** 2 <= 10 ** 2          # region A
    medium[0] |= (ii - 20) ** 2 + (jj - 42) ** 2 <= 10 ** 2          # region B, 2 voxels from A
    medium[0] |= (ii - 48) ** 2 + (jj - 30) ** 2 <= 6 ** 2           # region C, far away
    block = np.zeros((1, n, n), dtype=np.float32)
    block[0, 15:20, 24:29] = 10.0     # all nutrient of A sits next to the gap to B
    block[0, 46:50, 28:32] = 1.0
    medium_d = cp.asarray(medium)
    labels, count = label_regions(medium_d)
    labels_h = cp.asnumpy(labels)
    # For comparison: the same spread treating the whole layer as one region, so only the
    # layer total is conserved and nutrient can hop the gap between A and B.
    one_region = cp.asarray(medium.astype(np.int32))
    sigma = np.sqrt(2 * ECM_DIFFUSION * 0.1) / 0.084

    def masses_of(values: np.ndarray) -> list[float]:
        return [float(values[labels_h == k].sum()) for k in range(1, count + 1)]

    per_region, layer_only = cp.asarray(block), cp.asarray(block)
    masses, leaky = [masses_of(block)], [masses_of(block)]
    for _ in range(20):
        per_region = masked_gaussian_spread(per_region, medium_d, sigma, labels, count)
        layer_only = masked_gaussian_spread(layer_only, medium_d, sigma, one_region, 1)
        masses.append(masses_of(cp.asnumpy(per_region)))
        leaky.append(masses_of(cp.asnumpy(layer_only)))
    masses, leaky = np.array(masses), np.array(leaky)
    result = cp.asnumpy(per_region)
    print(f"{count} regions; masses at start {masses[0]}, after 20 spreads {masses[-1]}; "
          f"without the per-region rescale {leaky[-1]}")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(np.where(medium[0], block[0], np.nan).T, origin="lower")
    axes[0].set_title("start (white: no ECM)")
    axes[1].imshow(np.where(medium[0], result[0], np.nan).T, origin="lower")
    axes[1].set_title(f"after 20 spreads, sigma {sigma:.1f} voxels")
    for k in range(count):
        line, = axes[2].plot(masses[:, k], marker=".", label=f"region {k + 1}")
        axes[2].plot(leaky[:, k], "--", color=line.get_color(),
                     label=f"region {k + 1}, layer total only")
    axes[2].set_xlabel("spread steps")
    axes[2].set_ylabel("nutrient in region (relative)")
    axes[2].legend()
    axes[2].set_title("mass per region")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "region_conservation.png", dpi=120)
    plt.close(figure)


# ---- 5. realistic thin colony layers: Crank-Nicolson vs backward Euler ------------------

def thin_layer_theta() -> None:
    print("--- 5. thin colony layers packed with cells: theta 0.5 vs theta 1")
    h, layer = 0.1, 0.005
    geometry = PlateGeometry.rectangular(h, h, 4.0, h, layer, 0.2)
    colony = empty_colony(geometry, [strain("s")])
    capacity = colony.strains[0].capacity_per_voxel(geometry)
    for l in range(geometry.n_layers):
        colony.seed_voxel(0, 0, 0, capacity, layer=l)
    mesh_ratio = ECM_DIFFUSION * 0.1 / layer ** 2
    print(f"{geometry.n_layers} layers of {layer * 1e3:.0f} um, {capacity:.0f} cells each; "
          f"D dt / dz^2 in the colony = {mesh_ratio:.0f} for dt = 0.1 h")

    def run(theta: float, dt: float, hours: float) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        field = NutrientField(geometry, AGAR_DIFFUSION, ECM_DIFFUSION, GLUCOSE, theta=theta)
        steps = int(round(hours / dt))
        record = max(1, int(round(0.1 / dt)))
        minima, profiles, eaten = [], [], [0.0]
        for k in range(1, steps + 1):
            uptake = field.step(colony, dt)
            eaten.append(eaten[-1] + float(uptake.sum(dtype=cp.float64)))
            if k % record == 0:
                profile = cp.asnumpy(field.colony[:, 0, 0]).astype(np.float64)
                minima.append(profile.min())
                profiles.append(profile)
        return np.array(minima), profiles, np.array(eaten[::record])

    hours = 1.0
    runs = {"theta 0.5, dt 0.1 h": run(0.5, 0.1, hours),
            "theta 1, dt 0.1 h": run(1.0, 0.1, hours),
            "reference: theta 1, dt 0.001 h": run(1.0, 0.001, hours)}
    times = np.arange(1, int(round(hours / 0.1)) + 1) * 0.1
    reference_eaten = runs["reference: theta 1, dt 0.001 h"][2]
    for label, (minima, _, eaten) in runs.items():
        error = np.abs(eaten - reference_eaten).max() / reference_eaten[-1]
        print(f"{label}: lowest colony concentration {minima.min():.2e} g/ml, "
              f"uptake differs from reference by up to {error:.1e} of the total")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    heights = (np.arange(geometry.n_layers) + 0.5) * layer * 1e3
    for label, (_, profiles, _) in runs.items():
        for index, style in ((0, "-"), (4, "--")):
            axes[0].plot(heights, profiles[index] * 1e3, style, marker=".", ms=3,
                         label=f"{label}, t = {times[index]:.1f} h")
    axes[0].set_xlabel("height in colony (um)")
    axes[0].set_ylabel("glucose (mg/ml)")
    axes[0].set_title("profile in the colony")
    axes[0].legend(fontsize=7)
    for label, (minima, _, _) in runs.items():
        axes[1].plot(times, minima * 1e3, marker=".", label=label)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("time (h)")
    axes[1].set_ylabel("lowest colony glucose (mg/ml)")
    axes[1].set_title("negative values are unphysical")
    axes[1].legend(fontsize=7)
    for label, (_, _, eaten) in runs.items():
        axes[2].plot(np.concatenate([[0], times]), eaten * 1e6, marker=".", label=label)
    axes[2].set_xlabel("time (h)")
    axes[2].set_ylabel("cumulative uptake (ug)")
    axes[2].set_title("uptake")
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "thin_layer_theta.png", dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    vertical_cosine_mode()
    horizontal_point_source()
    uptake_mass_balance()
    region_conservation()
    thin_layer_theta()
    print(f"figures written to {OUTPUT_DIR}")
