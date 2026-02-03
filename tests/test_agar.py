import colonysimulator.simulator as cosim

def test_agarModel_creation():
    agarModel = cosim.AgarModel(20, 20, 20, 0.1, 6e-4)
    assert True