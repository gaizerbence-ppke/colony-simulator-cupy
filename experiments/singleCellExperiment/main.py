import sys
sys.path.append('/content/colony-simulator-cupy')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import colonysimulator.simulator as cosim
from colonysimulator.utility import setup_array_backend

from PIL import Image
import numpy as np

xp, xfft = setup_array_backend()

meanwhileImages = []
meanWhileFrequency = 100

if __name__ == "__main__":
    agarModel = cosim.AgarModel( mmLength= 8, \
                                 mmWidth= 8, \
                                 mmDepth= 4, \
                                 spatialResolution= 0.1, \
                                 timeResolution= 0.1, \
                                 diffusionCoefficient= 2.4) # mm^2 / h 
    agarModel.setConcentration(2.0 / 100) # g / ml
    agarModel.initiateModel()

    idealCellUptake = 15 # mmol / gDW / h
    minimumUptake = 0.3 # mmol / gDW / h
    singleCellDryWeight = 1e-12 # g
    glucoseMolecularWeight = 180e-3 # g / mmol

    bracketCount = 50
    divisionTime = 1.75 # h
    nutrientUpake = idealCellUptake * singleCellDryWeight * glucoseMolecularWeight # g / h / cell
    nutrientConsumption = minimumUptake * singleCellDryWeight * glucoseMolecularWeight # g / h / cell
    volume = 42e-9

    strain = cosim.CellStrain("test", divisionTime, nutrientUpake, nutrientConsumption, volume)
    colonyModel = cosim.ColonyModel(agarModel, bracketCount, strain)

    colonyModel.initiateSingleCellAtCenter()
    for i in range(2000):
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
    import matplotlib.pyplot as plt
    cellMass = cellMass.get()
    center = cellMass.shape[0] // 2

    # Vertical line (through center column)
    vertical_profile = cellMass[cellMass.shape[0] // 2:, center]

    # 45° diagonal line (from center outward)
    diagonal_profile = np.array([cellMass[center + i, center + i] for i in range(min(cellMass.shape) - center)])

    distances_vertical = np.arange(len(vertical_profile))
    distances_diagonal = np.arange(len(diagonal_profile)) * np.sqrt(2)  # Diagonal distance is sqrt(2) times the index

    plt.figure(figsize=(10, 5))
    plt.plot(distances_vertical, vertical_profile, label='Vertical', marker='o')
    plt.plot(distances_diagonal, diagonal_profile, label='45° Diagonal', marker='s')
    plt.xlabel('Distance (voxels)')
    plt.ylabel('Cell Mass')
    plt.legend()
    plt.grid(True)
    plt.savefig('cell_mass_profiles.png')
    plt.close()