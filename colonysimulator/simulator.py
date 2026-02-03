import cupy as cp
import numpy as np
import cupyx.scipy.fft as cufft

class CellSimulationModel:
    def __init__():
        pass

class AgarModel:
    def __init__(self, mmLength, mmWidth, mmDepth, spatialResolution, diffusionCoefficient):
        self.mmLength = mmLength
        self.mmWidth = mmWidth
        self.mmDepth = mmDepth
        self.spatialResolution = spatialResolution
        self.diffusionCoefficient = diffusionCoefficient

        self.length = int(mmLength // spatialResolution)
        self.width = int(mmWidth // spatialResolution)
        self.depth = int(mmDepth // spatialResolution)

        self._concentrationMap = np.zeros((self.length, self.width, self.depth), dtype=np.float32)