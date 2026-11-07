import pytest

cp = pytest.importorskip("cupy")

import colonysimulator.simulator as cosim

# Values as in experiments/singleCellExperiment/main.py
DRY_WEIGHT = 1e-12   # g
GLUCOSE_MW = 180e-3  # g / mmol
UPTAKE = 15 * DRY_WEIGHT * GLUCOSE_MW       # g / h / cell (maximum)
CONSUMPTION = 0.3 * DRY_WEIGHT * GLUCOSE_MW  # g / h / cell (maintenance)
VOLUME = 42e-9


class FixedFoodAgar:
    """Minimal stand-in for AgarModel that offers a fixed amount of food per voxel."""

    def __init__(self, length, width, available, spatialResolution=0.1, timeResolution=0.1):
        self.length, self.width = length, width
        self.spatialResolution = spatialResolution
        self.timeResolution = timeResolution
        self.available = available

    def nutrientUptakeStep(self, required):
        return cp.minimum(required, self.available)


def _model(agar, bracketCount=50):
    strain = cosim.CellStrain("test", 1.75, UPTAKE, CONSUMPTION, VOLUME)
    return cosim.ColonyModel(agar, bracketCount, strain)


def _total_cells(model):
    return float(model.growingMatrix.sum() + model.deadMatrix.sum())


def test_colonyModel():
    # Smoke test. Time step must be small enough that maxGrowth < bracketCount.
    agarModel = cosim.AgarModel(.51, .51, .31, 0.1, 0.1, 2.4)
    agarModel.setConcentration(2.0 / 100)
    agarModel.initiateModel()
    colonyModel = _model(agarModel, bracketCount=40)
    colonyModel.initiateSingleCellAtCenter()
    for _ in range(60):
        colonyModel.step()
        agarModel.diffusionStep()


def test_kernel_keeps_voxel_identity_on_non_square_grid():
    """B3: every voxel of a non-square grid is processed exactly once, in place."""
    length, width = 12, 8
    model = _model(FixedFoodAgar(length, width, available=cp.float32(1e9)))
    counts = {(0, 0): 1.0, (length - 1, width - 1): 2.0, (3, 6): 3.0, (10, 1): 4.0}
    for (x, y), n in counts.items():
        model.growingMatrix[0, x, y] = n
    model.step()  # unlimited food: every cell advances by exactly maxGrowth states
    for (x, y), n in counts.items():
        assert float(model.growingMatrix[model.maxGrowth, x, y]) == pytest.approx(n)
    assert _total_cells(model) == pytest.approx(sum(counts.values()))


def test_spreading_conserves_cells_and_does_not_wrap():
    """B4 + B6: spreading neither creates cells nor moves them across the plate edge."""
    length, width = 12, 8
    model = _model(FixedFoodAgar(length, width, available=cp.float32(1e9)))
    cap = model.maximumCellsPerVoxel
    model.growingMatrix[-1, 5, 4] = 2 * cap   # interior voxel about to divide
    model.growingMatrix[-1, 0, 3] = 2 * cap   # edge voxel about to divide
    model.growingMatrix[0, 6, 4] = 2 * cap    # overfull neighbour that blocks one direction
    before = _total_cells(model)
    model.step()
    divisions = float(model.postGrowthTemporal[model.maxPerish + model.bracketCount:].sum())
    assert divisions > 0
    assert _total_cells(model) == pytest.approx(before + divisions, rel=1e-5)
    assert float(model.growingMatrix[:, length - 1, :].sum()) == 0.0  # opposite edge stays empty


def test_demand_is_based_on_maximum_uptake():
    """B5: food covering only maintenance gives foodRatio = consumption / uptake, not 1."""
    length, width, cells, state = 4, 4, 10.0, 25
    dt = 0.1
    maintenanceOnly = cp.float32(cells * CONSUMPTION * dt)
    model = _model(FixedFoodAgar(length, width, available=maintenanceOnly, timeResolution=dt))
    model.growingMatrix[state, 1, 2] = cells
    model.step()
    width_ = model.maxGrowth + model.maxPerish
    center = round(width_ * CONSUMPTION / UPTAKE)
    expectedMeanState = state - model.maxPerish + center   # the kernel is symmetric around `center`
    states = cp.arange(model.bracketCount, dtype=cp.float32)
    column = model.growingMatrix[:, 1, 2]
    assert float((states * column).sum() / column.sum()) == pytest.approx(expectedMeanState, abs=1e-4)
