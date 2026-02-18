import colonysimulator.simulator as cosim
from colonysimulator.utility import setup_array_backend

xp, xfft = setup_array_backend()


def test_agarModel():
    agarModel = cosim.AgarModel(20, 20, 20, 0.1, 500, 6e-4)
    agarModel.setConcentration(1.0)
    agarModel.initiateModel()
    
    nutrientRequired = xp.zeros((agarModel.length, agarModel.width))
    nutrientRequired[int(xp.round(agarModel.length / 2)), :] = 1
    nutrientRequired[:, int(xp.round(agarModel.width / 2))] = 1

    nutrientUptake = agarModel.nutrientUptakeStep(nutrientRequired)
    assert nutrientUptake.shape == (agarModel.length, agarModel.width)

    agarModel.diffusionStep()
    agarModel.refreshConcentrationMap()
    slice = agarModel.getConcentrationMapSlice(1, 20)

    assert True