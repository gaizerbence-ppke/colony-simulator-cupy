from colonysimulator.utility import initialize_binomial_distribution_matrix, setup_array_backend

xp, xfft = setup_array_backend()

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

        self._concentrationMap = xp.zeros((self.length, self.width, self.depth), dtype=xp.float32)
        
        self._spectralMap = None
        self._diffusionKernel = None

    def setConcentration(self, concentration):
        self._concentrationMap[:, :, :] = concentration

    def initiateModel(self):
        self._spectralMap = xfft.dctn(self._concentrationMap, norm="ortho")

        x = xp.linspace(0, self.length - 1, self.length) / self.length
        y = xp.linspace(0, self.width - 1, self.width) / self.width
        z = xp.linspace(0, self.depth - 1, self.depth) / self.depth
        xx, yy, zz = xp.meshgrid(x, y, z, indexing='ij')
        
        self._diffusionKernel = xp.exp(-(xx**2 + yy**2 + zz**2) 
                                      * (self.diffusionCoefficient * self.timeResolution * xp.pi**2) 
                                      / (self.spatialResolution**2))

    def diffusionStep(self):
        self._spectralMap *= self._diffusionKernel


    def _topLayerInverseTransform(self):
        n = xp.arange(self.depth)
        basisVector = xp.cos(xp.pi * n / 2 / self.depth)
        basisVector[0] = 1 / xp.sqrt(2)
        reduced_2d = xp.tensordot(self._spectralMap, basisVector, axes=([2], [0]))
        return xfft.idctn(reduced_2d, norm="ortho") / xp.sqrt(5)

    def _topLayerSparseTransform(self, layer):
        transformedLayer = xfft.dctn(layer, norm="ortho")
        freqIndices = xp.arange(self.depth)
        basisVector = xp.cos(xp.pi * freqIndices / 2 / self.depth)
        scaling = xp.full(self.depth, xp.sqrt(2.0 / self.depth))
        scaling[0] = xp.sqrt(1.0 / self.depth)
        basisVector *= scaling
        return transformedLayer[:, :, xp.newaxis] * basisVector[xp.newaxis, xp.newaxis, :]

    def nutrientUptakeStep(self, nutrientRequired):
        top_layer = self._topLayerInverseTransform()
        nutrientTakenMap = xp.min(xp.stack((top_layer, nutrientRequired), axis=2), axis=2)
        nutrientTakenMap = xp.max(xp.stack((nutrientTakenMap, nutrientTakenMap * 0), axis=2), axis=2)
        nutrientTakenSpectrum = self._topLayerSparseTransform(nutrientTakenMap)
        self._spectralMap -= nutrientTakenSpectrum
        return nutrientTakenMap
    
    def refreshConcentrationMap(self):
        self._concentrationMap = xfft.idctn(self._spectralMap)

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
    def __init__(self, name, divisionTime, nutrientUptake, nutrientConsumption):
        self.name = name
        self.divisionTime = divisionTime
        self.nutrientUptake = nutrientUptake
        self.nutrientConsumption = nutrientConsumption
class ColonyModel:
    def __init__(self, agarModel, bracketCount, cellStrain):
        self.agarModel = agarModel
        self.bracketCount = bracketCount
        self.divisionTime = cellStrain.divisionTime
        self.nutrientUptake = cellStrain.nutrientUptake
        self.nutrientConsumption = cellStrain.nutrientConsumption

        idealMaxGrowth = bracketCount * self.divisionTime / agarModel.timeResolution
        idealMaxPerish = idealMaxGrowth * (self.nutrientUptake / self.nutrientConsumption - 1)

        self.maxGrowth = round(idealMaxGrowth)
        self.maxPerish = round(idealMaxPerish)

        print(f"Model for {cellStrain.name} initialized with max growth {self.maxGrowth} and max perish {self.maxPerish}")
        print(f"Deviation from ideal growth: {(self.maxGrowth - idealMaxGrowth) / idealMaxGrowth}, Deviation from ideal perish: {(self.maxPerish - idealMaxPerish) / idealMaxPerish}")

        self.deadMatrix = xp.zeros((agarModel.length, agarModel.width), dtype=xp.int32)
        self.growingMatrix = xp.zeros((agarModel.length, agarModel.width, self.bracketCount), dtype=xp.int32)

        growthWidth = self.maxGrowth + self.maxPerish + 1

        self.binomialDistributionMatrix = initialize_binomial_distribution_matrix(growthWidth)
        print(self.binomialDistributionMatrix)
    def initiateSingleCellAtCenter(self):
        centerX = self.agarModel.length // 2
        centerY = self.agarModel.width // 2
        self.growingMatrix[centerX, centerY, 0] = 1

    def step(self):
        foodRequired = self.growingMatrix.sum(axis=2) * self.nutrientConsumption * self.agarModel.timeResolution
        foodAvailable = self.agarModel.nutrientUptakeStep(foodRequired)
        foodRatio = foodAvailable / foodRequired
        foodRatio = xp.nan_to_num(foodRatio, nan=0.0, posinf=0.0, neginf=0.0)