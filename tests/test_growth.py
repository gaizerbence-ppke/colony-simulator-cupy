import colonysimulator.simulator as cosim
from colonysimulator.utility import setup_array_backend

xp, xfft = setup_array_backend()

def test_colonyModel():
    agarModel = cosim.AgarModel(1, 1, .3, 0.1, 500, 6e-4)
    agarModel.setConcentration(10000.0)
    agarModel.initiateModel()

    bracketCount = 40
    divisionTime = 40

    strain = cosim.CellStrain("test", divisionTime, 5.0, 1.0)
    colonyModel = cosim.ColonyModel(agarModel, bracketCount, strain)

    colonyModel.initiateSingleCellAtCenter()
    colonyModel.step()

    print(xp.sum(colonyModel.growingMatrix, axis=2))

    assert True