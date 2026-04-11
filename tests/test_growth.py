import colonysimulator.simulator as cosim
from colonysimulator.utility import setup_array_backend

xp, xfft = setup_array_backend()

def test_colonyModel():
    agarModel = cosim.AgarModel(.51, .51, .31, 0.1, 500, 6e-4)
    agarModel.setConcentration(100000.0)
    agarModel.initiateModel()

    bracketCount = 40
    divisionTime = 40

    strain = cosim.CellStrain("test", divisionTime, 5.0, 1.0, 42e-9)
    colonyModel = cosim.ColonyModel(agarModel, bracketCount, strain)

    colonyModel.initiateSingleCellAtCenter()
    for _ in range(60):
        colonyModel.step()
        agarModel.diffusionStep()
        print(xp.sum(colonyModel.growingMatrix, axis=0) + colonyModel.deadMatrix)

    assert True