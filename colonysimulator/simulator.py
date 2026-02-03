import numpy as np
import scipy.fft
try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    if cp.is_available():
        print("Cuda available")
        xp = cp
        xfft = cufft
    else:
        xp = np
        xfft = scipy.fft
        print("Cuda NOT available, fallback to CPU")
except (ImportError, Exception):
    xp = np
    cp = None
    fft = scipy.fft
    print("Cuda NOT available, fallback to CPU")

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

        self._concentrationMap = xp.zeros((self.length, self.width, self.depth), dtype=np.float32)

    def setConcentration(self, concentration):
        self._concentrationMap[:, :, :] = concentration