"""Dynamic layer count: the colony arrays grow as the colony rises.

Run from anywhere once the package is installed (`pip install -e .`):
    python validation/step1_dynamic_layers.py
Prints numbers to inspect and writes validation_outputs/step1/dynamic_layers.png:

left   allocated layers against occupied layers while cells are added one layer higher at a
       time, with the host memory the allocation takes
right  nutrient mass balance across the same run: nutrient left + eaten - initial, which
       must stay at rounding level when the arrays are reallocated
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colonysimulator import CellStrain, ColonyState, NutrientField, PlateGeometry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation_outputs" / "step1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLUCOSE_MOLECULAR_WEIGHT = 180e-3   # g/mmol
CELL_DRY_WEIGHT = 1e-12             # g

STRAIN = CellStrain("s", division_time=1.75, m_max=15 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT,
                    km=GLUCOSE_MOLECULAR_WEIGHT * 1e-3,
                    nutrient_consumption=0.3 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT,
                    volume=42e-9)


def rising_colony() -> None:
    print("--- arrays growing under a rising colony")
    geometry = PlateGeometry.rectangular(4.0, 3.0, 1.0, spatial_resolution=0.1, layer_height=0.02)
    print(geometry.describe())
    colony = ColonyState(geometry, STRAIN, bracket_count=10, pinned=False, chunk_layers=4,
                         initial_layers=4, headroom=2, growth_layers=4)
    field = NutrientField(geometry, agar_diffusion=2.4, ecm_diffusion=0.72, initial_concentration=1e-4)
    initial_mass = field.total_mass()

    rng = np.random.default_rng(0)
    top_layer = 60
    seeded = np.zeros(top_layer)       # cells put into every layer, for checking the copies
    allocated, active, host_mb, balance = [], [], [], []
    eaten = 0.0
    for layer in range(top_layer):
        for _ in range(5):
            i, j = int(rng.integers(geometry.length)), int(rng.integers(geometry.width))
            cells = float(rng.uniform(1, 50))
            colony.seed_voxel(0, i, j, cells, layer=layer, bracket=int(rng.integers(10)))
            seeded[layer] += np.float32(cells)
        uptake = field.step(colony, 0.1)
        eaten += float(uptake.sum(dtype=cp.float64))
        allocated.append(colony.n_layers)
        active.append(colony.active_layers)
        host_mb.append(colony.host_bytes / 1e6)
        balance.append((field.total_mass() + eaten - initial_mass) / initial_mass)

    growths = int(np.count_nonzero(np.diff(allocated)))
    host_per_layer = cp.asnumpy(cp.asarray(colony.living.sum(axis=(1, 2, 3, 4))))[:top_layer]
    device_per_layer = cp.asnumpy(colony.living_total.sum(axis=(1, 2, 3)))[:top_layer]
    print(f"{top_layer} occupied layers, {colony.n_layers} allocated after {growths} growths")
    print(f"cells per layer after all copies, max relative difference to what was seeded: "
          f"host {np.abs(host_per_layer - seeded).max() / seeded.max():.1e}, "
          f"GPU totals {np.abs(device_per_layer - seeded).max() / seeded.max():.1e}")
    print(f"nutrient rows follow the colony: {field.n_layers} colony rows; "
          f"mass balance max |error| {np.abs(balance).max():.1e}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    steps = np.arange(1, top_layer + 1)
    axes[0].step(steps, allocated, where="post", label="allocated layers")
    axes[0].plot(steps, active, label="occupied layers")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("layers")
    axes[0].legend(loc="upper left")
    memory_axis = axes[0].twinx()
    memory_axis.plot(steps, host_mb, color="0.6", ls=":", label="host memory")
    memory_axis.set_ylabel("host memory of the cell distribution (MB)")
    memory_axis.legend(loc="lower right")
    axes[0].set_title("layers are added as the colony rises")
    axes[1].plot(steps, balance, marker=".")
    for step, (before, after) in enumerate(zip(allocated[:-1], allocated[1:]), start=2):
        if after != before:
            axes[1].axvline(step, color="0.8", lw=0.8)
    axes[1].set_xlabel("step (grey: arrays grown)")
    axes[1].set_ylabel("(left + eaten - initial) / initial")
    axes[1].set_title("nutrient mass balance")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dynamic_layers.png", dpi=120)
    plt.close(figure)


def height_limit() -> None:
    print("--- optional height limit")
    geometry = PlateGeometry.rectangular(1.0, 1.0, 1.0, 0.1, layer_height=0.02, max_colony_height=0.2)
    colony = ColonyState(geometry, STRAIN, bracket_count=2, pinned=False, chunk_layers=1)
    print(f"max_colony_height 0.2 mm allows {geometry.max_layers} layers; allocated at start: {colony.n_layers}")
    colony.seed_voxel(0, 0, 0, 1.0, layer=geometry.max_layers - 1)
    print(f"seeding the top allowed layer works, allocated: {colony.n_layers}")
    try:
        colony.seed_voxel(0, 0, 0, 1.0, layer=geometry.max_layers)
    except ValueError as error:
        print(f"one layer higher is refused: {error}")


if __name__ == "__main__":
    rising_colony()
    height_limit()
    print(f"figure written to {OUTPUT_DIR}")
