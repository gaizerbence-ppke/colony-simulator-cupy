from colonysimulator.utility import initialize_binomial_distribution_matrix
import os

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import scipy.fft as spfft


def _neighbour(array, dx, dy, fill):
    """Return b with b[i, j] = array[i + dx, j + dy]; positions outside the plate get `fill`.

    Unlike cp.roll, nothing wraps around the plate edges.
    """
    length, width = array.shape
    padded = cp.pad(array, 1, mode="constant", constant_values=fill)
    return padded[1 + dx:1 + dx + length, 1 + dy:1 + dy + width]

class CellSimulationModel:
    def __init__(self):
        pass

class AgarModel:
    def __init__(self, mmLength, mmWidth, mmDepth, spatialResolution, timeResolution, diffusionCoefficient):
        self.mmLength = mmLength
        self.mmWidth = mmWidth
        self.mmDepth = mmDepth
        self.spatialResolution = spatialResolution
        self.timeResolution = timeResolution
        self.diffusionCoefficient = diffusionCoefficient

        self.length = int(mmLength // spatialResolution)
        self.width = int(mmWidth // spatialResolution)
        self.depth = int(mmDepth // spatialResolution)

        self._concentrationMap = cp.zeros((self.length, self.width, self.depth), dtype=cp.float32)
        
        self._spectralMap = None
        self._diffusionKernel = None

    def setConcentration(self, concentration):
        self._concentrationMap[:, :, :] = concentration * self.spatialResolution**3 * 1e-3

    def initiateModel(self):
        self._spectralMap = cufft.dctn(self._concentrationMap, norm="ortho")

        x = cp.linspace(0, self.length - 1, self.length) / self.length
        y = cp.linspace(0, self.width - 1, self.width) / self.width
        z = cp.linspace(0, self.depth - 1, self.depth) / self.depth
        xx, yy, zz = cp.meshgrid(x, y, z, indexing='ij')
        
        self._diffusionKernel = cp.exp(-(xx**2 + yy**2 + zz**2) 
                                      * (self.diffusionCoefficient * self.timeResolution * cp.pi**2) 
                                      / (self.spatialResolution**2))

    def diffusionStep(self):
        self._spectralMap *= self._diffusionKernel


    def _topLayerBasis(self):
        # Row z = 0 of the orthonormal DCT-II along depth: s_k * cos(pi * k / (2 * depth)),
        # with s_0 = sqrt(1 / depth) and s_k = sqrt(2 / depth) for k > 0.
        freqIndices = cp.arange(self.depth)
        basisVector = cp.cos(cp.pi * freqIndices / 2 / self.depth)
        scaling = cp.full(self.depth, cp.sqrt(2.0 / self.depth))
        scaling[0] = cp.sqrt(1.0 / self.depth)
        return basisVector * scaling

    def _topLayerInverseTransform(self):
        # Exact reconstruction of the top agar layer (z = 0) from the 3D spectrum.
        reduced_2d = cp.tensordot(self._spectralMap, self._topLayerBasis(), axes=([2], [0]))
        return cufft.idctn(reduced_2d, norm="ortho")

    def _topLayerSparseTransform(self, layer):
        transformedLayer = cufft.dctn(layer, norm="ortho")
        basisVector = self._topLayerBasis()
        return transformedLayer[:, :, cp.newaxis] * basisVector[cp.newaxis, cp.newaxis, :]

    def nutrientUptakeStep(self, nutrientRequired):
        top_layer = self._topLayerInverseTransform()
        nutrientTakenMap = cp.min(cp.stack((top_layer, nutrientRequired), axis=2), axis=2)
        nutrientTakenMap = cp.max(cp.stack((nutrientTakenMap, nutrientTakenMap * 0), axis=2), axis=2)
        nutrientTakenSpectrum = self._topLayerSparseTransform(nutrientTakenMap)
        self._spectralMap -= nutrientTakenSpectrum
        return nutrientTakenMap
    
    def refreshConcentrationMap(self):
        self._concentrationMap = cufft.idctn(self._spectralMap, norm="ortho")

    def getConcentrationMapSlice(self, axis, index):
        axisInt = int(axis)
        if axisInt < 0 or axisInt > 2:
            raise Exception("Axis must be 0, 1 or 2")
        indexInt = int(index)
        if indexInt < 0 or indexInt > self._concentrationMap.shape[axis]:
            raise Exception(f"index on axis {axisInt} must be between 0 and {self._concentrationMap.shape[axis]}")
        if axisInt == 0:
            return self._concentrationMap[index, :, :]
        if axisInt == 1:
            return self._concentrationMap[:, index, :]
        if axisInt == 2:
            return self._concentrationMap[:, :, index]
class CellStrain:
    def __init__(self, name, divisionTime, nutrientUptake, nutrientConsumption, volume):
        self.name = name
        self.divisionTime = divisionTime
        self.nutrientUptake = nutrientUptake
        self.nutrientConsumption = nutrientConsumption
        self.volume = volume
class ColonyModel:
    def __init__(self, agarModel, bracketCount, cellStrain):
        self.agarModel = agarModel
        self.bracketCount = bracketCount
        self.divisionTime = cellStrain.divisionTime
        self.nutrientUptake = cellStrain.nutrientUptake
        self.nutrientConsumption = cellStrain.nutrientConsumption
        self.maximumCellsPerVoxel = self.agarModel.spatialResolution**2 * 5e-3 / cellStrain.volume

        idealMaxGrowth = bracketCount / (self.divisionTime / agarModel.timeResolution)
        idealMaxPerish = (idealMaxGrowth + bracketCount) * (self.nutrientConsumption / (self.nutrientUptake - self.nutrientConsumption))

        self.maxGrowth = round(idealMaxGrowth)
        self.maxPerish = round(idealMaxPerish)

        print(f"Model for {cellStrain.name} initialized with max growth {self.maxGrowth} and max perish {self.maxPerish}")
        print(f"Deviation from ideal growth: {(self.maxGrowth - idealMaxGrowth) / idealMaxGrowth}, Deviation from ideal perish: {(self.maxPerish - idealMaxPerish) / idealMaxPerish}")

        self.deadMatrix = cp.zeros((agarModel.length, agarModel.width), dtype=cp.float32)
        self.growingMatrix = cp.zeros((self.bracketCount, agarModel.length, agarModel.width), dtype=cp.float32)

        growthWidth = self.maxGrowth + self.maxPerish + 1

        self.binomialDistributionMatrix = initialize_binomial_distribution_matrix(growthWidth)
        self.postGrowthTemporal = cp.zeros((self.bracketCount + self.maxGrowth + self.maxPerish, agarModel.length, agarModel.width), dtype=cp.float32)

        if cp.__name__ == 'cupy':
            kernel_code = open(os.path.join(os.path.dirname(__file__), "growthKernel.cu"), "r").read()
            self.growthKernel = cp.RawKernel(kernel_code, "growthKernel")
        else:
            self.growthKernel = None
            # TODO: Implement CPU mock

    def initiateSingleCellAtCenter(self):
        centerX = self.agarModel.length // 2
        centerY = self.agarModel.width // 2
        self.growingMatrix[0, centerX, centerY] = 1

    def step(self):
        # Demand is based on maximum uptake, so foodRatio is the fraction of maximal uptake a cell
        # receives (1 = full growth; nutrientConsumption / nutrientUptake = maintenance only).
        foodRequired = self.growingMatrix.sum(axis=0) * self.nutrientUptake * self.agarModel.timeResolution
        foodAvailable = self.agarModel.nutrientUptakeStep(foodRequired).astype(cp.float32)
        foodRatio = (foodAvailable / foodRequired).astype(cp.float32)
        foodRatio = cp.nan_to_num(foodRatio, nan=0.0, posinf=0.0, neginf=0.0)
        diagonal = 0.7071
        if self.growthKernel is not None:
            blockSize = (16, 16)
            gridSize = ((self.agarModel.length + blockSize[0] - 1) // blockSize[0], (self.agarModel.width + blockSize[1] - 1) // blockSize[1])
            
            self.postGrowthTemporal.fill(0)  # Clear the post-growth temporal array before the kernel call
            self.growthKernel(gridSize, blockSize,
                              (self.growingMatrix, self.postGrowthTemporal, foodRatio, self.binomialDistributionMatrix, self.maxGrowth, self.maxPerish, self.bracketCount, self.agarModel.length, self.agarModel.width))

            self.deadMatrix += cp.sum(self.postGrowthTemporal[:self.maxPerish, :, :], axis=0)
            self.growingMatrix = self.postGrowthTemporal[self.maxPerish:self.maxPerish + self.bracketCount, :, :].copy()
            self.growingMatrix[:self.maxGrowth, :, :] += self.postGrowthTemporal[self.maxPerish + self.bracketCount:, :, :]           

            overFlowMask = cp.sum(self.growingMatrix, axis=0) + self.deadMatrix > self.maximumCellsPerVoxel
            newCells = cp.sum(self.postGrowthTemporal[self.maxPerish + self.bracketCount:, :, :], axis=0)

            # Directions as (offset along length, offset along width, weight).
            # A direction is blocked if the target voxel overflows or lies outside the plate.
            directions = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                          (-1, -1, diagonal), (1, 1, diagonal), (-1, 1, diagonal), (1, -1, diagonal))
            openFlags = [(1 - _neighbour(overFlowMask, dx, dy, fill=True)).astype(cp.float32)
                         for dx, dy, _ in directions]

            flowDirections = cp.zeros_like(newCells)
            for (_, _, weight), isOpen in zip(directions, openFlags):
                flowDirections += isOpen * weight
            flowDirections *= overFlowMask.astype(cp.float32)

            outFlowPerDirection = newCells / flowDirections
            outFlowPerDirection = cp.nan_to_num(outFlowPerDirection, nan=0.0, posinf=0.0, neginf=0.0)

            # Each overflowing source sends to its open neighbours; the amount sent is
            # computed at the source and then moved to the target, so cells are conserved.
            for (dx, dy, weight), isOpen in zip(directions, openFlags):
                sent = outFlowPerDirection * isOpen * weight
                self.growingMatrix[0, :, :] += _neighbour(sent, -dx, -dy, fill=0)

            self.growingMatrix[0, :, :] += newCells * (flowDirections == 0)

        else:
            print("CPU growth step not implemented yet")