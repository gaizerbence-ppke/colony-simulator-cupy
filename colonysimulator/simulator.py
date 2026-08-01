from colonysimulator.utility import initialize_binomial_distribution_matrix
import os

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import scipy.fft as spfft

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


    def _topLayerInverseTransform(self):
        n = cp.arange(self.depth)
        basisVector = cp.cos(cp.pi * n / 2 / self.depth)
        basisVector[0] = 1 / cp.sqrt(2)
        reduced_2d = cp.tensordot(self._spectralMap, basisVector, axes=([2], [0]))
        return cufft.idctn(reduced_2d, norm="ortho") / cp.sqrt(5)

    def _topLayerSparseTransform(self, layer):
        transformedLayer = cufft.dctn(layer, norm="ortho")
        freqIndices = cp.arange(self.depth)
        basisVector = cp.cos(cp.pi * freqIndices / 2 / self.depth)
        scaling = cp.full(self.depth, cp.sqrt(2.0 / self.depth))
        scaling[0] = cp.sqrt(1.0 / self.depth)
        basisVector *= scaling
        return transformedLayer[:, :, cp.newaxis] * basisVector[cp.newaxis, cp.newaxis, :]

    def nutrientUptakeStep(self, nutrientRequired):
        top_layer = self._topLayerInverseTransform()
        nutrientTakenMap = cp.min(cp.stack((top_layer, nutrientRequired), axis=2), axis=2)
        nutrientTakenMap = cp.max(cp.stack((nutrientTakenMap, nutrientTakenMap * 0), axis=2), axis=2)
        nutrientTakenSpectrum = self._topLayerSparseTransform(nutrientTakenMap)
        self._spectralMap -= nutrientTakenSpectrum
        return nutrientTakenMap
    
    def refreshConcentrationMap(self):
        self._concentrationMap = cufft.idctn(self._spectralMap)

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
        foodRequired = self.growingMatrix.sum(axis=0) * self.nutrientConsumption * self.agarModel.timeResolution
        foodAvailable = self.agarModel.nutrientUptakeStep(foodRequired).astype(cp.float32)
        foodRatio = foodAvailable / foodRequired
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
            blockedNorth = cp.roll(overFlowMask, -1, axis=0)
            blockedSouth = cp.roll(overFlowMask, 1, axis=0)
            blockedEast = cp.roll(overFlowMask, -1, axis=1)
            blockedWest = cp.roll(overFlowMask, 1, axis=1)
            blockedNorthWest = cp.roll(overFlowMask, (1, 1), axis=(0, 1))
            blockedSouthEast = cp.roll(overFlowMask, (-1, -1), axis=(0, 1))
            blockedSouthWest = cp.roll(overFlowMask, (1, -1), axis=(0, 1))
            blockedNorthEast = cp.roll(overFlowMask, (-1, 1), axis=(0, 1))

            flowDirections =  (1 - blockedNorth).astype(cp.float32) \
                            + (1 - blockedSouth).astype(cp.float32) \
                            + (1 - blockedEast).astype(cp.float32) \
                            + (1 - blockedWest).astype(cp.float32) \
                            + (1 - blockedNorthWest).astype(cp.float32) * diagonal \
                            + (1 - blockedSouthEast).astype(cp.float32) * diagonal \
                            + (1 - blockedSouthWest).astype(cp.float32) * diagonal \
                            + (1 - blockedNorthEast).astype(cp.float32) * diagonal

            flowDirections *= overFlowMask.astype(cp.float32)

            newCells = cp.sum(self.postGrowthTemporal[self.maxPerish + self.bracketCount:, :, :], axis=0)

            outFlowPerDirection = newCells / flowDirections
            outFlowPerDirection = cp.nan_to_num(outFlowPerDirection, nan=0.0, posinf=0.0, neginf=0.0)

            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, 1, axis=0) * (1 - blockedNorth)
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, -1, axis=0) * (1 - blockedSouth)
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, 1, axis=1) * (1 - blockedEast)
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, -1, axis=1) * (1 - blockedWest)
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, (-1, -1), axis=(0, 1)) * (1 - blockedNorthWest) * diagonal
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, (1, 1), axis=(0, 1)) * (1 - blockedSouthEast) * diagonal
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, (-1, 1), axis=(0, 1)) * (1 - blockedSouthWest) * diagonal
            self.growingMatrix[0, :, :] += cp.roll(outFlowPerDirection, (1, -1), axis=(0, 1)) * (1 - blockedNorthEast) * diagonal

            self.growingMatrix[0, :, :] += newCells * (flowDirections == 0)

        else:
            print("CPU growth step not implemented yet")