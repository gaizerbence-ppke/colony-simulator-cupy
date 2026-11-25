"""Step 1 validation: plate geometry, colony data layout and host->GPU layer streaming.

Run from anywhere once the package is installed (`pip install -e .`):
    python validation/step1_data_layout.py
Prints numbers to inspect and writes figures to validation_outputs/step1/.
"""
import time
from pathlib import Path

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colonysimulator import CellStrain, ColonyState, PlateGeometry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "validation_outputs" / "step1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLUCOSE_MOLECULAR_WEIGHT = 180e-3  # g / mmol
CELL_DRY_WEIGHT = 1e-12            # g


def make_strains():
    m_max = 15 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT
    consumption = 0.3 * CELL_DRY_WEIGHT * GLUCOSE_MOLECULAR_WEIGHT
    fast = CellStrain("fast", division_time=1.75, m_max=m_max, km=0.5e-3,
                      nutrient_consumption=consumption, volume=42e-9)
    slow = CellStrain("slow", division_time=2.5, m_max=0.8 * m_max, km=0.5e-3,
                      nutrient_consumption=consumption, volume=60e-9)
    return [fast, slow]


def geometry_report():
    print("--- geometry")
    omnitray = PlateGeometry.rectangular(128, 86, 4, spatial_resolution=0.084,
                                         layer_height=0.1, max_colony_height=3)
    print("omnitray:", omnitray.describe())
    dish = PlateGeometry.circular(35, 4, spatial_resolution=0.084, layer_height=0.1,
                                  max_colony_height=3)
    print("dish:    ", dish.describe())
    print(f"dish mask fills {dish.plate_columns / (dish.length * dish.width):.3f} of the grid "
          f"(pi/4 = {np.pi / 4:.3f})")

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(omnitray.mask.get().T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("rectangular mask")
    axes[1].imshow(dish.mask.get().T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("circular mask")
    figure.savefig(OUTPUT_DIR / "plate_masks.png", dpi=120, bbox_inches="tight")
    plt.close(figure)

    strains = make_strains()
    for layer_height in (0.005, 0.02, 0.1):
        geometry = PlateGeometry.rectangular(128, 86, 4, 0.084, layer_height, 3)
        host_gb = geometry.n_layers * len(strains) * 50 * geometry.length * geometry.width * 4 / 1e9
        caps = [strain.capacity_per_voxel(geometry) for strain in strains]
        print(f"layer {layer_height * 1e3:5.0f} um: {geometry.n_layers:4d} layers, "
              f"host distribution {host_gb:7.1f} GB for 2 strains x 50 brackets, "
              f"capacity per voxel {caps[0]:.0f} / {caps[1]:.0f} cells")


def layout_report():
    print("--- colony state and totals")
    geometry = PlateGeometry.rectangular(8, 6, 2, spatial_resolution=0.1, layer_height=0.02,
                                         max_colony_height=0.5)
    strains = make_strains()
    colony = ColonyState(geometry, strains, bracket_count=50, chunk_layers=4)
    print(colony.describe())

    colony.seed_cells("fast", 2.0, 3.0, 1.0)
    colony.seed_cells("slow", 6.0, 3.0, 1.0)
    colony.seed_voxel("fast", 40, 10, 3.0, layer=2, bracket=7)
    print(f"active layers after seeding: {colony.active_layers}")

    expected_total = colony.living.sum(axis=2)
    difference = np.abs(colony.living_total.get() - expected_total).max()
    print(f"living_total vs host sum after seeding, max abs difference: {difference:.3g}")

    rng = np.random.default_rng(0)
    colony.living[:] = rng.random(colony.living.shape, dtype=np.float32) * geometry.mask.get()
    colony.refresh_totals(all_layers=True)
    expected_total = colony.living.sum(axis=2)
    difference = np.abs(colony.living_total.get() - expected_total).max()
    print(f"living_total vs host sum after refresh, max abs difference: {difference:.3g} "
          f"(float32 sum over {colony.bracket_count} brackets)")
    print(f"active layers after refresh: {colony.active_layers} of {colony.n_layers}")
    living, dead = colony.cell_counts()
    print(f"cell counts per strain: living {living}, dead {dead}")

    occupancy = colony.occupancy()
    height = colony.height_map()
    print(f"occupancy range {float(occupancy.min()):.3g}..{float(occupancy.max()):.3g}, "
          f"height map range {float(height.min()):.3g}..{float(height.max()):.3g} mm "
          f"(max possible {geometry.n_layers * geometry.layer_height} mm)")

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(height.get().T, origin="lower")
    figure.colorbar(image, ax=axis, label="height (mm)")
    axis.set_title("height map of a random distribution")
    figure.savefig(OUTPUT_DIR / "height_map_random.png", dpi=120, bbox_inches="tight")
    plt.close(figure)


def streaming_round_trip():
    print("--- streaming round trip")
    geometry = PlateGeometry.rectangular(8, 6, 2, spatial_resolution=0.1, layer_height=0.02,
                                         max_colony_height=0.5)
    colony = ColonyState(geometry, make_strains(), bracket_count=50, chunk_layers=3, n_buffers=2)
    rng = np.random.default_rng(1)
    colony.living[:] = rng.random(colony.living.shape, dtype=np.float32)
    colony.active_layers = colony.n_layers
    expected = colony.living * 2 + np.arange(colony.n_layers, dtype=np.float32)[:, None, None, None, None]

    def transform(chunk, layers):
        chunk *= 2
        chunk += cp.arange(layers.start, layers.stop, dtype=cp.float32)[:, None, None, None, None]
        return float(chunk.sum())

    chunk_sums = colony.map_layers(transform)
    difference = np.abs(colony.living - expected).max()
    print(f"chunks processed: {len(chunk_sums)}, host pinned: {colony.streamer.pinned}")
    print(f"max abs difference host vs expected after write-back: {difference:.3g}")
    print(f"sum of chunk sums vs expected sum: {sum(chunk_sums):.6g} vs {expected.sum():.6g}")


def streaming_timing():
    print("--- streaming throughput")
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    geometry = PlateGeometry.rectangular(40, 30, 2, spatial_resolution=0.1, layer_height=0.1,
                                         max_colony_height=4.0)
    strains = make_strains()[:1]
    bracket_count = 50
    layer_bytes = len(strains) * bracket_count * geometry.length * geometry.width * 4
    print(f"grid {geometry.length} x {geometry.width}, {geometry.n_layers} layers, "
          f"{layer_bytes / 1e6:.1f} MB per layer, {free_bytes / 1e9:.2f} GB free on the device")

    def work(chunk, layers):
        # A few passes over the data so compute is comparable to the copies.
        for _ in range(4):
            chunk *= 1.0001
            chunk += 1e-3
        return None

    results = []
    for n_buffers in (1, 2):
        for chunk_layers in (1, 2, 5, 10):
            colony = ColonyState(geometry, strains, bracket_count, chunk_layers=chunk_layers,
                                 n_buffers=n_buffers)
            colony.living[:] = 1.0
            colony.active_layers = colony.n_layers
            colony.map_layers(work)  # warm-up (kernel compilation, pool allocation)
            repeats = 3
            start = time.perf_counter()
            for _ in range(repeats):
                colony.map_layers(work)
            elapsed = (time.perf_counter() - start) / repeats
            moved_gb = 2 * colony.host_bytes / 1e9  # upload + download
            results.append((n_buffers, chunk_layers, elapsed, moved_gb / elapsed))
            print(f"buffers {n_buffers}, chunk {chunk_layers:2d} layers: {elapsed * 1e3:7.1f} ms per pass, "
                  f"{moved_gb / elapsed:5.2f} GB/s moved")
            del colony
            cp.get_default_memory_pool().free_all_blocks()

    figure, axis = plt.subplots(figsize=(6, 4))
    for n_buffers, marker in ((1, "o"), (2, "s")):
        rows = [row for row in results if row[0] == n_buffers]
        axis.plot([row[1] for row in rows], [row[3] for row in rows], marker=marker,
                  label=f"{n_buffers} buffer(s)")
    axis.set_xlabel("layers per chunk")
    axis.set_ylabel("host<->device throughput (GB/s)")
    axis.set_title("streaming pass over the cell distribution")
    axis.grid(True)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "streaming_throughput.png", dpi=120, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    geometry_report()
    layout_report()
    streaming_round_trip()
    streaming_timing()
    print(f"figures written to {OUTPUT_DIR}")
