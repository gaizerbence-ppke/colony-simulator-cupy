"""Checks of the colony growth model (bugs B3-B6) plus a short single-colony run.

Run from the repository root:  python tests/test_growth.py   (or %run in Colab)
Outputs: test_outputs/growth/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CheckRun, gpuAvailable, finish  # also puts the repository root on sys.path

import cupy as cp
import numpy as np
from matplotlib.figure import Figure

import colonysimulator.simulator as cosim

# Values as in experiments/singleCellExperiment/main.py
DRY_WEIGHT = 1e-12   # g
GLUCOSE_MW = 180e-3  # g / mmol
UPTAKE = 15 * DRY_WEIGHT * GLUCOSE_MW       # g / h / cell (maximum uptake)
CONSUMPTION = 0.3 * DRY_WEIGHT * GLUCOSE_MW  # g / h / cell (maintenance)
VOLUME = 42e-9                               # mm^3


class FixedFoodAgar:
    """Minimal stand-in for AgarModel that offers a fixed amount of food per voxel."""

    def __init__(self, length, width, available, spatialResolution=0.1, timeResolution=0.1):
        self.length, self.width = length, width
        self.spatialResolution = spatialResolution
        self.timeResolution = timeResolution
        self.available = available

    def nutrientUptakeStep(self, required):
        return cp.minimum(required, self.available)


def makeModel(agar, bracketCount=50):
    strain = cosim.CellStrain("test", 1.75, UPTAKE, CONSUMPTION, VOLUME)
    return cosim.ColonyModel(agar, bracketCount, strain)


def cellMap(model):
    return cp.asnumpy(model.growingMatrix.sum(axis=0) + model.deadMatrix)


def totalCells(model):
    return float(model.growingMatrix.sum() + model.deadMatrix.sum())


def showMaps(run, maps, fileName):
    figure = Figure(figsize=(4 * len(maps), 3.4))
    for i, (image, title) in enumerate(maps):
        axis = figure.add_subplot(1, len(maps), i + 1)
        shown = axis.imshow(image, origin="upper")
        figure.colorbar(shown, ax=axis, shrink=0.8)
        axis.set_title(title)
        axis.set_xlabel("width index")
        axis.set_ylabel("length index")
    run.saveFigure(figure, fileName)


def checkKernelVoxelIdentity(run):
    """B3: on a non-square grid every voxel is processed exactly once, in place."""
    length, width = 12, 8
    model = makeModel(FixedFoodAgar(length, width, available=cp.float32(1e9)))
    counts = {(0, 0): 1.0, (length - 1, width - 1): 2.0, (3, 6): 3.0, (10, 1): 4.0}
    for (x, y), n in counts.items():
        model.growingMatrix[0, x, y] = n
    before = cellMap(model)
    model.step()  # unlimited food: every cell advances by exactly maxGrowth states
    stayed = all(abs(float(model.growingMatrix[model.maxGrowth, x, y]) - n) < 1e-5 for (x, y), n in counts.items())
    run.check("each voxel's cells advance in place on a 12x8 grid", stayed)
    run.check("no cells created or lost by the kernel", abs(totalCells(model) - sum(counts.values())) < 1e-4,
              f"total {totalCells(model):.4f}, expected {sum(counts.values()):.4f}")
    showMaps(run, [(before, "before step"), (cellMap(model), "after step")], "kernel_voxel_identity.png")


def checkSpreading(run):
    """B4 + B6: spreading neither creates cells nor moves them across the plate edge."""
    length, width = 12, 8
    model = makeModel(FixedFoodAgar(length, width, available=cp.float32(1e9)))
    capacity = model.maximumCellsPerVoxel
    model.growingMatrix[-1, 5, 4] = 2 * capacity   # interior voxel about to divide
    model.growingMatrix[-1, 0, 3] = 2 * capacity   # edge voxel about to divide
    model.growingMatrix[0, 6, 4] = 2 * capacity    # overfull neighbour that blocks one direction
    before, totalBefore = cellMap(model), totalCells(model)
    model.step()
    divisions = float(model.postGrowthTemporal[model.maxPerish + model.bracketCount:].sum())
    expected = totalBefore + divisions
    run.check("cells divided in the test setup", divisions > 0, f"{divisions:.0f} divisions")
    run.check("spreading conserves cells", abs(totalCells(model) - expected) <= 1e-5 * expected,
              f"total {totalCells(model):.1f}, expected {expected:.1f}")
    oppositeEdge = float(model.growingMatrix[:, length - 1, :].sum())
    run.check("no cells wrap to the opposite plate edge", oppositeEdge == 0.0, f"{oppositeEdge:.1f} cells there")
    showMaps(run, [(before / capacity, "before (fraction of capacity)"),
                   (cellMap(model) / capacity, "after one step")], "spreading.png")


def checkDemandUsesMaximumUptake(run):
    """B5: food covering only maintenance gives foodRatio = consumption / uptake, not 1."""
    cells, state, dt = 10.0, 25, 0.1
    model = makeModel(FixedFoodAgar(4, 4, available=cp.float32(cells * CONSUMPTION * dt), timeResolution=dt))
    model.growingMatrix[state, 1, 2] = cells
    model.step()
    center = round((model.maxGrowth + model.maxPerish) * CONSUMPTION / UPTAKE)
    expectedMean = state - model.maxPerish + center   # the kernel is symmetric around `center`
    column = cp.asnumpy(model.growingMatrix[:, 1, 2])
    mean = float((np.arange(model.bracketCount) * column).sum() / column.sum())
    run.check("maintenance-only food gives the expected maturity shift", abs(mean - expectedMean) < 1e-4,
              f"mean state {mean:.3f}, expected {expectedMean} (start {state})")

    figure = Figure(figsize=(5, 3))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(np.arange(model.bracketCount), column)
    axis.axvline(state, color="k", linestyle="--", linewidth=0.8, label="start state")
    axis.set_xlabel("maturity state")
    axis.set_ylabel("cells")
    axis.set_title("one step on maintenance-only food")
    axis.legend()
    run.saveFigure(figure, "demand_maintenance_only.png")


def checkSingleColonyRun(run, steps=200):
    """Short single-cell run on real agar: no NaNs, cells never decrease, colony spreads."""
    agar = cosim.AgarModel(mmLength=2.0, mmWidth=2.0, mmDepth=1.0, spatialResolution=0.1,
                           timeResolution=0.1, diffusionCoefficient=2.4)
    agar.setConcentration(2.0 / 100)  # g / ml
    agar.initiateModel()
    model = makeModel(agar)
    model.initiateSingleCellAtCenter()
    totals = [totalCells(model)]
    for _ in range(steps):
        model.step()
        agar.diffusionStep()
        totals.append(totalCells(model))
    finite = bool(cp.all(cp.isfinite(model.growingMatrix))) and bool(cp.all(cp.isfinite(model.deadMatrix)))
    run.check("single-colony run stays finite", finite)
    run.check("cell count never decreases", all(b >= a - 1e-3 for a, b in zip(totals, totals[1:])),
              f"{totals[0]:.0f} -> {totals[-1]:.3g} cells in {steps * agar.timeResolution:.0f} h")

    figure = Figure(figsize=(9, 3.4))
    axis = figure.add_subplot(1, 2, 1)
    axis.semilogy(np.arange(len(totals)) * agar.timeResolution, totals)
    axis.set_xlabel("time (h)")
    axis.set_ylabel("live + dead cells")
    axis.set_title("growth curve")
    axis = figure.add_subplot(1, 2, 2)
    shown = axis.imshow(cellMap(model) / model.maximumCellsPerVoxel)
    figure.colorbar(shown, ax=axis, shrink=0.8)
    axis.set_title("colony (fraction of capacity)")
    run.saveFigure(figure, "single_colony.png")


def main():
    if not gpuAvailable():
        print("No CUDA device found: these checks need CuPy with a GPU.")
        return False
    run = CheckRun("growth")
    checkKernelVoxelIdentity(run)
    checkSpreading(run)
    checkDemandUsesMaximumUptake(run)
    checkSingleColonyRun(run)
    return run.summary()


if __name__ == "__main__":
    finish(main())
