"""Single cell placed at the centre of a small square plate, grown and recorded as height maps.

Run from anywhere:
    python scenarios/singleCellExperiment/main.py
    %run scenarios/singleCellExperiment/main.py      (Colab, from the repository root)

Outputs go to scenarios/singleCellExperiment/output/.

Status: the data layout (step 1) and the nutrient field (step 2) are in place. The cell is
fed every step, but the colony does not change between frames until the growth kernel
(step 3) and spreading (step 4) are plugged in where marked.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import colonysimulator  # noqa: F401  (installed with `pip install -e .`)
except ImportError:
    # Fallback for a plain clone without installation, e.g. a fresh Colab runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cupy as cp
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colonysimulator import CellStrain, ColonyState, NutrientField, PlateGeometry

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---- plate -------------------------------------------------------------------------------
MM_LENGTH = 8.0              # mm
MM_WIDTH = 8.0               # mm
MM_DEPTH = 4.0               # mm of agar
SPATIAL_RESOLUTION = 0.1     # mm, horizontal voxel size and agar voxel height
LAYER_HEIGHT = 0.005         # mm, one colony layer (about one cell)
MAX_COLONY_HEIGHT = None     # mm, optional upper limit; layers are added as the colony rises

# ---- nutrient ----------------------------------------------------------------------------
GLUCOSE_CONCENTRATION = 0.1 / 1000   # g/ml in the agar at t = 0
AGAR_DIFFUSION = 2.4                 # mm^2/h, glucose in agar (6.7e-6 cm^2/s)
ECM_DIFFUSION = 0.72                 # mm^2/h, glucose in the colony (2e-6 cm^2/s; literature 1-3e-6)
NUTRIENT_THETA = 1.0                 # 1 = backward Euler, 0.5 = Crank-Nicolson (see validation/step2)

# ---- strain ------------------------------------------------------------------------------
GLUCOSE_MOLECULAR_WEIGHT = 180e-3    # g/mmol
CELL_DRY_WEIGHT = 1e-12              # g
MAX_UPTAKE = 15                      # mmol/gDW/h
MAINTENANCE_UPTAKE = 0.3             # mmol/gDW/h
KM_GLUCOSE = 1.0                     # mM, high-affinity hexose transport

strain = CellStrain(
    name="test",
    division_time=1.75,                                                         # h
    m_max=MAX_UPTAKE * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT,              # g/cell/h
    km=KM_GLUCOSE * GLUCOSE_MOLECULAR_WEIGHT * 1e-3,                            # g/ml
    nutrient_consumption=MAINTENANCE_UPTAKE * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT,
    volume=42e-9,                                                               # mm^3
)

# ---- run ---------------------------------------------------------------------------------
BRACKET_COUNT = 50
TIME_RESOLUTION = 0.1        # h
N_STEPS = 2000
RECORD_EVERY = 100           # steps between recorded height maps


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 200) -> None:
    """Grey-scale animation, normalised to the tallest point over all frames."""
    peak = max(float(frame.max()) for frame in frames)
    scale = 255.0 / peak if peak > 0 else 0.0
    images = [Image.fromarray((frame.T[::-1] * scale).astype(np.uint8)) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def save_height_map(height: np.ndarray, geometry: PlateGeometry, path: Path, title: str) -> None:
    extent = (0, geometry.length * geometry.spatial_resolution,
              0, geometry.width * geometry.spatial_resolution)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(height.T * 1e3, origin="lower", extent=extent)
    figure.colorbar(image, ax=axis, label="colony height (um)")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title(title)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_profiles(height: np.ndarray, geometry: PlateGeometry, path: Path) -> None:
    """Height along the axis and along the 45 degree diagonal, from the centre outward."""
    ci, cj = geometry.centre_voxel()
    reach = min(geometry.length - ci, geometry.width - cj)
    axial = height[ci:, cj]
    diagonal = np.array([height[ci + k, cj + k] for k in range(reach)])
    step = geometry.spatial_resolution

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(np.arange(len(axial)) * step, axial * 1e3, marker="o", label="axial")
    axis.plot(np.arange(len(diagonal)) * step * np.sqrt(2), diagonal * 1e3, marker="s",
              label="45 degree diagonal")
    axis.set_xlabel("distance from centre (mm)")
    axis.set_ylabel("colony height (um)")
    axis.grid(True)
    axis.legend()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    geometry = PlateGeometry.rectangular(MM_LENGTH, MM_WIDTH, MM_DEPTH, SPATIAL_RESOLUTION,
                                         LAYER_HEIGHT, MAX_COLONY_HEIGHT)
    colony = ColonyState(geometry, [strain], BRACKET_COUNT)
    print(geometry.describe())
    print(colony.describe())
    print(f"capacity per voxel: {strain.capacity_per_voxel(geometry):.0f} cells, "
          f"maintenance food ratio: {strain.maintenance_ratio:.3f}")

    nutrient = NutrientField(geometry, AGAR_DIFFUSION, ECM_DIFFUSION,
                             initial_concentration=GLUCOSE_CONCENTRATION, theta=NUTRIENT_THETA)
    colony.seed_single_cell_at_centre(strain.name)
    initial_mass = nutrient.total_mass()
    eaten = 0.0

    frames: list[np.ndarray] = []
    for step in range(N_STEPS):
        # step 2: nutrient column solve with Michaelis-Menten sinks + horizontal spread
        uptake = nutrient.step(colony, TIME_RESOLUTION)
        eaten += float(uptake.sum(dtype=cp.float64))
        # step 3: growth kernel over the streamed bracket distribution (turns `uptake` into cells)
        # step 4: tension-driven spreading of living and dead cells
        if step % RECORD_EVERY == RECORD_EVERY - 1:
            living, dead = colony.cell_counts()
            hours = (step + 1) * TIME_RESOLUTION
            print(f"step {step + 1:5d} ({hours:6.1f} h): living {living.sum():.4g}, "
                  f"dead {dead.sum():.4g}, active layers {colony.active_layers}, "
                  f"glucose eaten {eaten:.3g} g, mass balance error "
                  f"{(nutrient.total_mass() + eaten - initial_mass) / initial_mass:.1e}")
            frames.append(colony.height_map().get())

    final_height = colony.height_map().get()
    save_gif(frames, OUTPUT_DIR / "colony_growth.gif")
    save_height_map(final_height, geometry, OUTPUT_DIR / "colony_result.png",
                    f"height map after {N_STEPS * TIME_RESOLUTION:.0f} h")
    save_profiles(final_height, geometry, OUTPUT_DIR / "height_profiles.png")
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
