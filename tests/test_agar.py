import colonysimulator.simulator as cosim
import numpy as np

def test_agarModel():
    agarModel = cosim.AgarModel(20, 20, 20, 0.1, 6e-4)
    agarModel.setConcentration(1.0)
    agarModel.initiateModel(500)
    
    nutrientRequired = np.zeros((agarModel.length, agarModel.width))
    nutrientRequired[int(np.round(agarModel.length / 2)), :] = 1
    nutrientRequired[:, int(np.round(agarModel.width / 2))] = 1

    nutrientUptake = agarModel.nutrientUptakeStep(nutrientRequired)

    agarModel.diffusionStep()

    assert True