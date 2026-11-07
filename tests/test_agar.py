import pytest

cp = pytest.importorskip("cupy")

import colonysimulator.simulator as cosim


def _random_agar(depth_mm, seed=0):
    agar = cosim.AgarModel(1.2, 0.8, depth_mm, 0.1, 0.1, 2.4)
    rng = cp.random.default_rng(seed)
    agar._concentrationMap[:] = rng.random(agar._concentrationMap.shape, dtype=cp.float32)
    original = agar._concentrationMap.copy()
    agar.initiateModel()
    return agar, original


def test_agarModel():
    agarModel = cosim.AgarModel(20, 20, 20, 0.1, 500, 6e-4)
    agarModel.setConcentration(1.0)
    agarModel.initiateModel()

    nutrientRequired = cp.zeros((agarModel.length, agarModel.width))
    nutrientRequired[int(round(agarModel.length / 2)), :] = 1
    nutrientRequired[:, int(round(agarModel.width / 2))] = 1

    nutrientUptake = agarModel.nutrientUptakeStep(nutrientRequired)
    assert nutrientUptake.shape == (agarModel.length, agarModel.width)

    agarModel.diffusionStep()
    agarModel.refreshConcentrationMap()
    agarModel.getConcentrationMapSlice(1, 20)


def test_refresh_round_trip():
    """B1: forward and inverse transforms must use the same (orthonormal) normalisation."""
    agar, original = _random_agar(0.4)
    agar.refreshConcentrationMap()
    cp.testing.assert_allclose(agar._concentrationMap, original, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth_mm", [0.4, 0.9, 4.0])
def test_top_layer_readout_is_exact(depth_mm):
    """B2: the top-layer readout must equal layer z = 0 for any agar depth."""
    agar, original = _random_agar(depth_mm)
    top = agar._topLayerInverseTransform()
    cp.testing.assert_allclose(top, original[:, :, 0], rtol=1e-4, atol=1e-5)


def test_uptake_removes_only_from_top_layer():
    agar, original = _random_agar(0.9)
    required = cp.full((agar.length, agar.width), 0.25, dtype=cp.float32)
    taken = agar.nutrientUptakeStep(required)
    agar.refreshConcentrationMap()
    expectedTop = original[:, :, 0] - taken
    cp.testing.assert_allclose(agar._concentrationMap[:, :, 0], expectedTop, rtol=1e-4, atol=1e-5)
    cp.testing.assert_allclose(agar._concentrationMap[:, :, 1:], original[:, :, 1:], rtol=1e-4, atol=1e-5)
