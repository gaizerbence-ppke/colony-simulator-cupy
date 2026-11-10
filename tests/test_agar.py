"""Checks of the spectral agar model (bugs B1, B2 and nutrient uptake).

Run from the repository root:  python tests/test_agar.py   (or %run in Colab)
Outputs: test_outputs/agar/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CheckRun, gpuAvailable, finish  # also puts the repository root on sys.path

import cupy as cp
from matplotlib.figure import Figure

import colonysimulator.simulator as cosim

TOLERANCE = 1e-4


def randomAgar(depthMm, seed=0):
    """Small non-square agar with a random concentration field (arbitrary units)."""
    agar = cosim.AgarModel(1.2, 0.8, depthMm, 0.1, 0.1, 2.4)
    rng = cp.random.default_rng(seed)
    agar._concentrationMap[:] = rng.random(agar._concentrationMap.shape, dtype=cp.float32)
    original = agar._concentrationMap.copy()
    agar.initiateModel()
    return agar, original


def maxError(a, b):
    return float(cp.max(cp.abs(a - b)))


def checkRefreshRoundTrip(run):
    """B1: forward and inverse DCT must use the same (orthonormal) normalisation."""
    agar, original = randomAgar(0.4)
    agar.refreshConcentrationMap()
    error = maxError(agar._concentrationMap, original)
    run.check("refreshConcentrationMap returns the original field", error < TOLERANCE, f"max error {error:.2e}")

    figure = Figure(figsize=(9, 3))
    panels = [(original[:, :, 0], "original, z = 0"),
              (agar._concentrationMap[:, :, 0], "after refresh, z = 0"),
              (agar._concentrationMap[:, :, 0] - original[:, :, 0], "difference")]
    for i, (image, title) in enumerate(panels):
        axis = figure.add_subplot(1, 3, i + 1)
        shown = axis.imshow(cp.asnumpy(image))
        figure.colorbar(shown, ax=axis, shrink=0.8)
        axis.set_title(title)
    run.saveFigure(figure, "refresh_round_trip.png")


def checkTopLayerReadout(run):
    """B2: the top-layer readout must equal layer z = 0 for any agar depth."""
    depths, errors = [0.4, 0.9, 4.0], []
    for depthMm in depths:
        agar, original = randomAgar(depthMm)
        top = agar._topLayerInverseTransform()
        errors.append(maxError(top, original[:, :, 0]))
        run.check(f"top-layer readout is exact at {depthMm} mm depth ({agar.depth} voxels)",
                  errors[-1] < TOLERANCE, f"max error {errors[-1]:.2e}")

    figure = Figure(figsize=(9, 3))
    axis = figure.add_subplot(1, 3, 1)
    axis.bar([f"{d} mm" for d in depths], errors)
    axis.set_yscale("log")
    axis.axhline(TOLERANCE, color="k", linestyle="--", linewidth=0.8)
    axis.set_title("max readout error")
    for i, (image, title) in enumerate([(original[:, :, 0], "true z = 0 (4 mm)"), (top, "readout (4 mm)")]):
        axis = figure.add_subplot(1, 3, i + 2)
        axis.imshow(cp.asnumpy(image))
        axis.set_title(title)
    run.saveFigure(figure, "top_layer_readout.png")


def checkUptake(run):
    """Uptake removes nutrient from the top layer only, and exactly the amount taken."""
    agar, original = randomAgar(0.9)
    required = cp.full((agar.length, agar.width), 0.25, dtype=cp.float32)
    taken = agar.nutrientUptakeStep(required)
    agar.refreshConcentrationMap()
    topError = maxError(agar._concentrationMap[:, :, 0], original[:, :, 0] - taken)
    restError = maxError(agar._concentrationMap[:, :, 1:], original[:, :, 1:])
    run.check("uptake is removed from the top layer", topError < TOLERANCE, f"max error {topError:.2e}")
    run.check("deeper layers are unchanged by uptake", restError < TOLERANCE, f"max error {restError:.2e}")
    run.check("uptake never exceeds what was available", bool(cp.all(taken <= original[:, :, 0] + TOLERANCE)))


def checkDiffusionConservesMass(run):
    """No-flux boundaries: diffusion redistributes nutrient but keeps the total."""
    agar, original = randomAgar(0.9)
    totals = [float(original.sum())]
    for _ in range(50):
        agar.diffusionStep()
        agar.refreshConcentrationMap()
        totals.append(float(agar._concentrationMap.sum()))
    drift = abs(totals[-1] - totals[0]) / totals[0]
    run.check("diffusion conserves total nutrient", drift < 1e-5, f"relative drift {drift:.2e} after 50 steps")

    figure = Figure(figsize=(9, 3))
    axis = figure.add_subplot(1, 2, 1)
    axis.plot(totals)
    axis.set_xlabel("step")
    axis.set_title("total nutrient")
    axis = figure.add_subplot(1, 2, 2)
    axis.imshow(cp.asnumpy(agar._concentrationMap[:, :, 0]))
    axis.set_title("z = 0 after 50 steps")
    run.saveFigure(figure, "diffusion_mass.png")


def main():
    if not gpuAvailable():
        print("No CUDA device found: these checks need CuPy with a GPU.")
        return False
    run = CheckRun("agar")
    checkRefreshRoundTrip(run)
    checkTopLayerReadout(run)
    checkUptake(run)
    checkDiffusionConservesMass(run)
    return run.summary()


if __name__ == "__main__":
    finish(main())
