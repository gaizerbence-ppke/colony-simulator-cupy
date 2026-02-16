import numpy as np
import scipy.fft


def setup_array_backend():
    try:
        import cupy as cp
        import cupyx.scipy.fft as cufft
        
        if cp.is_available():
            print("Cuda available")
            return cp, cufft
        else:
            print("Cuda NOT available, fallback to CPU")
            return np, scipy.fft
    except (ImportError, Exception):
        print("Cuda NOT available, fallback to CPU")
        return np, scipy.fft
