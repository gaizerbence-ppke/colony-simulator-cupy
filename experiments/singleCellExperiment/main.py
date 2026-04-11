import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import colonysimulator.simulator as cosim
from colonysimulator.utility import setup_array_backend

from PIL import Image

xp, xfft = setup_array_backend()

meanwhileImages = []
meanWhileFrequency = 100

if __name__ == "__main__":
    agarModel = cosim.AgarModel(10, 10, 3, 0.1, 500, 6e-4)
    agarModel.setConcentration(100000.0)
    agarModel.initiateModel()

    bracketCount = 40
    divisionTime = 40

    strain = cosim.CellStrain("test", divisionTime, 5.0, 1.0, 42e-9)
    colonyModel = cosim.ColonyModel(agarModel, bracketCount, strain)

    colonyModel.initiateSingleCellAtCenter()
    for i in range(4000):
        colonyModel.step()
        agarModel.diffusionStep()
        if i % meanWhileFrequency == meanWhileFrequency - 1:
            print(f"Step {i+1}, total cells: {xp.sum(colonyModel.growingMatrix) + xp.sum(colonyModel.deadMatrix)}")
            cellMass = xp.sum(colonyModel.growingMatrix, axis=0) + colonyModel.deadMatrix
            cellMass = xp.clip(cellMass / colonyModel.maximumCellsPerVoxel, 0, 1)
            result = Image.fromarray((cellMass * 255).get().astype(xp.uint8))
            #result.save(f"colony_step_{i+1}.png")
            meanwhileImages.append(result)

    cellMass = xp.sum(colonyModel.growingMatrix, axis=0) + colonyModel.deadMatrix
    cellMass = xp.clip(cellMass / colonyModel.maximumCellsPerVoxel, 0, 1)
    result = Image.fromarray((cellMass * 255).get().astype(xp.uint8))
    result.save("colony_result.png")
    meanwhileImages[0].save("colony_growth.gif", save_all=True, append_images=meanwhileImages[1:], duration=200, loop=0)
    assert True